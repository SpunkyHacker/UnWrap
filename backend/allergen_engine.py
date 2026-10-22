"""Deterministic allergen-checking engine for the Unwrap app.

This module is purely deterministic — no AI model is involved in
allergen decisions. It cross-references matched ingredient data and raw
label text against the user's allergen profile and the allergen hierarchy
defined in ``allergens.json``.

Unknown ingredients are never treated as having no allergens; they are
flagged as ``UNCERTAIN``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

from backend.models import (
    AllergenAlert,
    AllergenResult,
    AllergenStatus,
    MatchResult,
    ParsedResult,
    TranscriptionResult,
    UserProfile,
)

logger = logging.getLogger(__name__)

# Default path to the allergen hierarchy data file
_ALLERGENS_JSON_PATH = Path(__file__).resolve().parent / "data" / "allergens.json"

# Severity ranking — higher index = more severe. Used for overall_status.
_SEVERITY_RANK: dict[AllergenStatus, int] = {
    AllergenStatus.CLEAR: 0,
    AllergenStatus.UNCERTAIN: 1,
    AllergenStatus.MIGHT_CONTAIN: 2,
    AllergenStatus.CONTAINS: 3,
}

# Cross-contact phrasing regular expressions
_RE_CROSS_CONTACT = re.compile(
    r"(?:may\s+contain(?:\s+traces?\s+of)?|traces?\s+of|"
    r"manufactured\s+in\s+a\s+facility\s+that\s+also\s+(?:handles|processes)|"
    r"facility\s+that\s+(?:also\s+)?(?:handles|processes)|"
    r"shared\s+equipment|processed\s+on\s+shared\s+equipment|"
    r"made\s+on\s+the\s+same\s+line)[^.;\n]*",
    re.IGNORECASE,
)

# "Free" / "Non-" claims that must NEVER downgrade allergen detection
_RE_FREE_CLAIMS = re.compile(
    r"\b(lactose[- ]free|gluten[- ]free|dairy[- ]free|non[- ]dairy|milk[- ]free)\b",
    re.IGNORECASE,
)

# Known non-dairy / safe false-positive compound phrases to protect against naive keyword matches
_FALSE_POSITIVE_BUTTERS = re.compile(
    r"\b(?:cocoa|cacao|shea|peanut|apple|almond)\s+butter\b",
    re.IGNORECASE,
)

# Plant-based milks: coconut milk, almond milk, oat milk, soy milk, etc.
_RE_PLANT_MILKS = re.compile(
    r"\b(coconut|almond|oat|soy|soya|cashew|rice|hemp|flax|pea)\s+milk\b",
    re.IGNORECASE,
)

# Cream of tartar (potassium bitartrate) is not dairy cream
_RE_CREAM_OF_TARTAR = re.compile(
    r"\bcream\s+of\s+tartar\b",
    re.IGNORECASE,
)

# False nuts: butternut squash, nutmeg, water chestnut
_RE_FALSE_NUTS = re.compile(
    r"\b(?:butternut\s+squash|nutmeg|water\s+chestnut)\b",
    re.IGNORECASE,
)

# Lactic acid / lactates are fermentation products and not milk/lactose
_RE_LACTATES = re.compile(
    r"\b(?:lactic\s+acid|sodium\s+lactate|calcium\s+lactate|potassium\s+lactate|lactate)\b",
    re.IGNORECASE,
)

# Derived additive codes with potential allergen origins (FDA, EFSA, Codex)
# These produce MIGHT_CONTAIN (not CONTAINS) because the source may be synthetic or alternative
DERIVED_ADDITIVE_ALLERGENS: dict[str, list[tuple[str, AllergenStatus, str]]] = {
    "322": [("soy", AllergenStatus.MIGHT_CONTAIN, "INS 322 (lecithin) may be derived from soy; verify source on packaging")],
    "476": [("soy", AllergenStatus.MIGHT_CONTAIN, "INS 476 (polyglycerol polyricinoleate) may be derived from or processed with soybean derivatives")],
    "1105": [("egg", AllergenStatus.MIGHT_CONTAIN, "INS 1105 (lysozyme) is derived from egg whites")],
    "150b": [("sulphites", AllergenStatus.MIGHT_CONTAIN, "INS 150b (caustic sulphite caramel) is prepared using sulphite compounds")],
    "150d": [("sulphites", AllergenStatus.MIGHT_CONTAIN, "INS 150d (sulphite ammonia caramel) is prepared using sulphite compounds")],
}

_RE_DERIVED_ADDITIVE_CODES = re.compile(
    r"\b(?:INS\s*[-]?\s*|E\s*[-]?\s*)?(322|476|1105|150b|150d)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------


def _load_allergen_hierarchy(
    path: Path | None = None,
) -> dict[str, list[str]]:
    """Load the ``hierarchy`` section from ``allergens.json``."""
    if path is None:
        path = _ALLERGENS_JSON_PATH

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    return data.get("hierarchy", {})


def _expand_allergens(
    user_allergens: list[str],
    hierarchy: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Expand each user allergen into the full set of matching terms."""
    expanded: dict[str, set[str]] = {}
    for allergen in user_allergens:
        key = allergen.strip().lower()
        terms: set[str] = {key}
        if key in hierarchy:
            for child in hierarchy[key]:
                terms.add(child.strip().lower())
        # Cross-alias: shellfish & crustaceans
        if key == "shellfish" and "crustaceans" in hierarchy:
            for child in hierarchy["crustaceans"]:
                terms.add(child.strip().lower())
        elif key == "crustaceans" and "shellfish" in hierarchy:
            for child in hierarchy["shellfish"]:
                terms.add(child.strip().lower())
        expanded[key] = terms
    return expanded


# ---------------------------------------------------------------------------
# Result Confidence Scorer
# ---------------------------------------------------------------------------


def allergen_result_confidence_with_reason(
    matches: list[MatchResult],
    transcription: TranscriptionResult | None = None,
    engine_used: str = "",
    unreadable_parts: list[str] | None = None,
    parsed: ParsedResult | None = None,
) -> tuple[str, str]:
    """Return (confidence, confidence_reason) for the analysis.

    Confidence levels: 'high' | 'medium' | 'low'.
    Per spec: Stop counting KB-unknown ingredients toward 'low'.
    Low must come from photo/transcription quality only:
    - unreadable_parts
    - OCR fallback (fallback_occurred is True or fallback engine used)
    - many corrections (>= 3 corrections repaired)
    - short text (< 30 characters of ingredient text)
    """
    # 1. Unreadable parts
    unreadable = unreadable_parts if unreadable_parts is not None else (
        transcription.unreadable_parts if transcription else []
    )
    if unreadable:
        return "low", f"Part of label could not be read: {', '.join(unreadable)}"

    # 2. Vision OCR fallback
    engine = engine_used or (transcription.engine_used if transcription else "")
    fallback_occ = transcription.fallback_occurred if transcription else False
    if fallback_occ or engine.lower() in ("failed", "unknown", "apple_vision") or "fallback" in engine.lower():
        return "low", "Vision fallback engine used"

    # 3. Many OCR corrections (>= 3)
    corrections_count = 0
    if transcription and transcription.corrections:
        corrections_count += len(transcription.corrections)
    if parsed and getattr(parsed, "corrections", None):
        corrections_count += len(parsed.corrections)
    if corrections_count >= 3:
        return "low", f"Multiple OCR misreads repaired ({corrections_count} corrections)"

    # 4. Very short text (< 15 characters for transcription, < 12 for raw text)
    if transcription and transcription.ingredients_text:
        tx_len = len(transcription.ingredients_text.strip())
        if 0 < tx_len < 15:
            return "low", "Ingredients text is very short"
    elif parsed and parsed.raw_text:
        tx_len = len(parsed.raw_text.strip())
        if 0 < tx_len < 12:
            return "low", "Ingredients text is very short"

    # Medium confidence conditions
    has_fuzzy = any(m.match_method == "fuzzy" for m in matches)
    if has_fuzzy:
        return "medium", "Some ingredient names matched fuzzily"
    if corrections_count > 0:
        return "medium", "OCR corrections applied"

    return "high", ""


def allergen_result_confidence(
    matches: list[MatchResult],
    transcription: TranscriptionResult | None = None,
    engine_used: str = "",
    unreadable_parts: list[str] | None = None,
    parsed: ParsedResult | None = None,
) -> str:
    """Return overall confidence level ('high' | 'medium' | 'low') for backward compatibility."""
    level, _ = allergen_result_confidence_with_reason(
        matches=matches,
        transcription=transcription,
        engine_used=engine_used,
        unreadable_parts=unreadable_parts,
        parsed=parsed,
    )
    return level


# ---------------------------------------------------------------------------
# Independent Raw-Text Safety Scan
# ---------------------------------------------------------------------------


def _gather_raw_text(parsed: ParsedResult, transcription: TranscriptionResult | None = None) -> str:
    """Combine all available raw text representations into one string for safety scanning."""
    parts: list[str] = []
    if parsed.raw_text:
        parts.append(parsed.raw_text)
    if getattr(parsed, "corrections", None):
        for c in parsed.corrections:
            if c.get("original"):
                parts.append(c["original"])
            if c.get("corrected"):
                parts.append(c["corrected"])
    if transcription:
        if transcription.ingredients_text:
            parts.append(transcription.ingredients_text)
        if transcription.contains_statement:
            parts.append(f"Contains: {transcription.contains_statement}")
        if transcription.may_contain_statement:
            parts.append(f"May contain: {transcription.may_contain_statement}")
        # Rule 4: Include both original and corrected tokens from OCR spell-correction
        if transcription.corrections:
            for c in transcription.corrections:
                if c.get("original"):
                    parts.append(c["original"])
                if c.get("corrected"):
                    parts.append(c["corrected"])
    for ing in parsed.ingredients:
        if ing.raw_text:
            parts.append(ing.raw_text)
        elif ing.name:
            parts.append(ing.name)
    if parsed.contains_allergens:
        parts.append(f"Contains: {', '.join(parsed.contains_allergens)}")
    if parsed.may_contain_allergens:
        parts.append(f"May contain: {', '.join(parsed.may_contain_allergens)}")
    return " . ".join(parts)


def _scan_raw_text(
    full_text: str,
    user_allergen_keys: set[str],
    expanded: dict[str, set[str]],
    custom_terms: set[str],
) -> list[AllergenAlert]:
    """Perform an independent regex scan over raw label text for all allergen terms."""
    if not full_text or not full_text.strip():
        return []

    alerts: list[AllergenAlert] = []
    seen: set[tuple[str, str, AllergenStatus]] = set()

    # Find cross-contact spans
    cross_contact_spans: list[tuple[int, int, str]] = []
    for m in _RE_CROSS_CONTACT.finditer(full_text):
        cross_contact_spans.append((m.start(), m.end(), m.group(0)))

    # Find "free" claim spans
    free_claim_spans: list[tuple[int, int, str]] = []
    for m in _RE_FREE_CLAIMS.finditer(full_text):
        free_claim_spans.append((m.start(), m.end(), m.group(0)))

    # Find non-dairy butter spans
    false_butter_spans: list[tuple[int, int]] = []
    for m in _FALSE_POSITIVE_BUTTERS.finditer(full_text):
        false_butter_spans.append((m.start(), m.end()))

    # Find plant milk spans
    plant_milk_spans: list[tuple[int, int, str]] = []
    for m in _RE_PLANT_MILKS.finditer(full_text):
        plant_milk_spans.append((m.start(), m.end(), m.group(0)))

    # Find cream of tartar spans
    cream_tartar_spans: list[tuple[int, int]] = []
    for m in _RE_CREAM_OF_TARTAR.finditer(full_text):
        cream_tartar_spans.append((m.start(), m.end()))

    # Find false nut spans (butternut squash, nutmeg, water chestnut)
    false_nut_spans: list[tuple[int, int, str]] = []
    for m in _RE_FALSE_NUTS.finditer(full_text):
        false_nut_spans.append((m.start(), m.end(), m.group(0)))

    # Find lactate spans
    lactate_spans: list[tuple[int, int, str]] = []
    for m in _RE_LACTATES.finditer(full_text):
        lactate_spans.append((m.start(), m.end(), m.group(0)))

    def is_in_spans(spans: list, start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e, *_ in spans)

    target_allergens: dict[str, set[str]] = {}
    for allergen in user_allergen_keys:
        target_allergens[allergen] = expanded.get(allergen, {allergen})
    for custom in custom_terms:
        target_allergens[custom] = {custom}

    for allergen_key, terms in target_allergens.items():
        sorted_terms = sorted(terms, key=len, reverse=True)

        for term in sorted_terms:
            t = term.strip().lower()
            if not t:
                continue

            if t == "wheat":
                pattern = r"(?<!buck[\s-])\bwheat\b"
            elif t == "butter":
                pattern = r"\bbutter\b"
            else:
                pattern = r"\b" + re.escape(t) + r"\b"

            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue

            for match in rx.finditer(full_text):
                m_start, m_end = match.start(), match.end()
                matched_word = match.group(0)

                # Rule 5: False butter protection (cocoa butter, peanut butter != dairy butter)
                if allergen_key == "milk" and t == "butter" and is_in_spans(false_butter_spans, m_start, m_end):
                    continue

                # Rule 3: Cream of tartar is not milk cream
                if allergen_key == "milk" and t in ("cream", "dairy") and is_in_spans(cream_tartar_spans, m_start, m_end):
                    continue

                # Rule 4: Lactic acid / lactates are not milk lactose/protein
                if allergen_key == "milk" and t in ("lactose", "milk", "dairy") and is_in_spans(lactate_spans, m_start, m_end):
                    continue

                # Rule 6: False nuts (butternut squash, nutmeg, water chestnut) must NOT trigger tree nut
                if allergen_key in ("tree_nuts", "tree nut") and is_in_spans(false_nut_spans, m_start, m_end):
                    continue

                # Rule 3: Plant-based milks for milk profile
                in_plant_milk = is_in_spans(plant_milk_spans, m_start, m_end)
                if allergen_key == "milk" and in_plant_milk:
                    status = AllergenStatus.MIGHT_CONTAIN
                    reason = f"'{matched_word}' is part of plant-based milk; verify no dairy added"
                    dedup_key = (allergen_key, t, status)
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        alerts.append(AllergenAlert(
                            status=status,
                            allergen=allergen_key,
                            ingredient_name=matched_word,
                            reason=reason,
                        ))
                    continue

                # Rule 6: Coconut under tree nut profile -> MIGHT_CONTAIN with note (FDA considers coconut tree nut, botany differs)
                if allergen_key in ("tree_nuts", "tree nut") and t == "coconut":
                    status = AllergenStatus.MIGHT_CONTAIN
                    reason = "Coconut is botanically a drupe and distinct from tree nuts; FDA classifies it as a tree nut, verify tolerance"
                    dedup_key = (allergen_key, t, status)
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        alerts.append(AllergenAlert(
                            status=status,
                            allergen=allergen_key,
                            ingredient_name=matched_word,
                            reason=reason,
                        ))
                    continue

                # Rule 7: Oats under gluten profile -> MIGHT_CONTAIN (cross-contact risk during milling)
                if allergen_key == "gluten" and t in ("oats", "oat flour", "oat"):
                    status = AllergenStatus.MIGHT_CONTAIN
                    reason = "'oats' presents cross-contact risk with gluten-containing grains during harvesting and milling; check for certified gluten-free label"
                    dedup_key = (allergen_key, t, status)
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        alerts.append(AllergenAlert(
                            status=status,
                            allergen=allergen_key,
                            ingredient_name=matched_word,
                            reason=reason,
                        ))
                    continue

                # Check cross-contact
                in_cross_contact = False
                cross_contact_stmt = ""
                for cc_start, cc_end, cc_text in cross_contact_spans:
                    if cc_start <= m_start and m_end <= cc_end:
                        in_cross_contact = True
                        cross_contact_stmt = cc_text
                        break

                # Check near free claim
                near_free_claim = False
                free_claim_text = ""
                for fc_start, fc_end, fc_text in free_claim_spans:
                    if abs(m_start - fc_end) <= 45 or abs(fc_start - m_end) <= 45:
                        near_free_claim = True
                        free_claim_text = fc_text
                        break

                if in_cross_contact:
                    status = AllergenStatus.MIGHT_CONTAIN
                    reason = (
                        f"Raw text safety scan found cross-contact advisory '{cross_contact_stmt.strip()}' "
                        f"mentioning '{matched_word}' ({allergen_key})"
                    )
                elif near_free_claim:
                    status = AllergenStatus.MIGHT_CONTAIN
                    reason = (
                        f"Raw text safety scan found '{matched_word}' ({allergen_key}) near claim '{free_claim_text}'; "
                        f"claim does not eliminate allergen risk (requires label verification)"
                    )
                else:
                    status = AllergenStatus.CONTAINS
                    reason = (
                        f"Raw text safety scan detected '{matched_word}' ({allergen_key}) directly in product text"
                    )

                dedup_key = (allergen_key, t, status)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    alerts.append(AllergenAlert(
                        status=status,
                        allergen=allergen_key,
                        ingredient_name=matched_word,
                        reason=reason,
                    ))
                break

    # Derived additive allergen code scan in raw text (e.g. INS 322, 476, 1105, 150b, 150d)
    for m in _RE_DERIVED_ADDITIVE_CODES.finditer(full_text):
        c_key = m.group(1).lower()
        if c_key in DERIVED_ADDITIVE_ALLERGENS:
            for allergen, status, reason in DERIVED_ADDITIVE_ALLERGENS[c_key]:
                if allergen in user_allergen_keys or allergen in custom_terms or any(allergen in expanded.get(u, set()) for u in user_allergen_keys):
                    dedup_key = (allergen, c_key, status)
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        alerts.append(AllergenAlert(
                            status=status,
                            allergen=allergen,
                            ingredient_name=m.group(0),
                            reason=reason,
                        ))

    return alerts


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------


def check_allergens(
    matches: list[MatchResult],
    profile: UserProfile,
    parsed: ParsedResult,
    transcription: TranscriptionResult | None = None,
) -> AllergenResult:
    """Run a deterministic allergen check against the user's profile."""
    # Rule 1: Empty string, whitespace-only, or < 10 chars ingredients_text -> UNCERTAIN
    raw_clean = (parsed.raw_text or "").strip()
    if transcription and transcription.ingredients_text:
        raw_clean = transcription.ingredients_text.strip() or raw_clean

    hierarchy = _load_allergen_hierarchy()
    expanded = _expand_allergens(profile.allergens, hierarchy)
    user_allergen_keys: set[str] = {a.strip().lower() for a in profile.allergens}

    custom_terms: set[str] = set()
    if profile.custom_allergens:
        for term in profile.custom_allergens.split(","):
            stripped = term.strip().lower()
            if stripped:
                custom_terms.add(stripped)

    kb_alerts: list[AllergenAlert] = []
    seen_kb: dict[tuple[str, str], AllergenStatus] = {}

    def add_kb_alert(status: AllergenStatus, allergen: str, ingredient_name: str, reason: str) -> None:
        key = (ingredient_name.strip().lower(), allergen.strip().lower())
        prev_status = seen_kb.get(key)
        if prev_status is not None:
            if _SEVERITY_RANK[status] <= _SEVERITY_RANK[prev_status]:
                return
            nonlocal kb_alerts
            kb_alerts = [a for a in kb_alerts if not (a.ingredient_name.strip().lower() == key[0] and a.allergen.strip().lower() == key[1])]
        seen_kb[key] = status
        kb_alerts.append(AllergenAlert(
            status=status,
            allergen=allergen,
            ingredient_name=ingredient_name,
            reason=reason,
        ))

    if not raw_clean or len(raw_clean) < 10:
        kb_alerts.append(AllergenAlert(
            status=AllergenStatus.UNCERTAIN,
            allergen="unknown",
            ingredient_name="",
            reason="No ingredients could be read",
        ))

    # Phase 2.5 Item 3c: Possible merged ingredients stay UNCERTAIN until confirmed
    for ing in parsed.ingredients:
        if getattr(ing, "possible_merged", False):
            split_str = ", ".join(ing.suggested_split) if ing.suggested_split else "multiple items"
            kb_alerts.append(AllergenAlert(
                status=AllergenStatus.UNCERTAIN,
                allergen="unknown",
                ingredient_name=ing.name,
                reason=f"'{ing.name}' appears to contain merged ingredients ({split_str}); status remains uncertain until confirmed",
            ))

    for match in matches:
        if match.kb_entry is not None:
            entry = match.kb_entry

            entry_names_lower: set[str] = {entry.canonical_name.strip().lower()}
            for n in entry.names:
                entry_names_lower.add(n.strip().lower())

            # Rule 3: Cream of tartar is not dairy
            if entry.id in ("cream_of_tartar", "potassium_bitartrate"):
                continue

            # Rule 4: Lactic acid is not dairy
            if entry.id in ("lactic_acid", "sodium_lactate"):
                continue

            entry_direct = {a.strip().lower() for a in entry.allergens_direct}
            entry_possible = {a.strip().lower() for a in entry.allergens_possible}

            direct_hits = entry_direct & user_allergen_keys
            possible_hits = entry_possible & user_allergen_keys

            is_fuzzy = match.match_method == "fuzzy"

            for allergen in sorted(direct_hits):
                # Rule 6: Coconut under tree_nuts -> MIGHT_CONTAIN
                if allergen in ("tree_nuts", "tree nut") and entry.id == "coconut_oil":
                    add_kb_alert(
                        status=AllergenStatus.MIGHT_CONTAIN,
                        allergen=allergen,
                        ingredient_name=match.parsed_name,
                        reason="Coconut is botanically a drupe and distinct from tree nuts; FDA classifies it as a tree nut, verify tolerance",
                    )
                    continue

                reason = f"'{match.parsed_name}' directly contains {allergen}"
                if is_fuzzy:
                    reason += "; matched approximately; verify on the label"
                add_kb_alert(
                    status=AllergenStatus.CONTAINS,
                    allergen=allergen,
                    ingredient_name=match.parsed_name,
                    reason=reason,
                )

            for allergen in sorted(possible_hits):
                # Rule 7: Oats for gluten -> MIGHT_CONTAIN cross-contact
                if allergen == "gluten" and "oat" in entry.canonical_name.lower():
                    reason = f"'{match.parsed_name}' presents cross-contact risk with gluten-containing grains; verify certified gluten-free"
                else:
                    reason = f"'{match.parsed_name}' may contain {allergen}"
                if is_fuzzy:
                    reason += "; matched approximately; verify on the label"
                add_kb_alert(
                    status=AllergenStatus.MIGHT_CONTAIN,
                    allergen=allergen,
                    ingredient_name=match.parsed_name,
                    reason=reason,
                )

            if is_fuzzy and (entry_direct or entry_possible):
                for allergen, terms in expanded.items():
                    if allergen in direct_hits or allergen in possible_hits:
                        continue
                    if any(t in entry_direct or t in entry_possible for t in terms):
                        add_kb_alert(
                            status=AllergenStatus.MIGHT_CONTAIN,
                            allergen=allergen,
                            ingredient_name=match.parsed_name,
                            reason=f"'{match.parsed_name}' matched approximately; verify on the label",
                        )

            if entry.source_ambiguous and entry.ambiguity_note:
                ambiguity_lower = entry.ambiguity_note.strip().lower()
                for allergen, terms in expanded.items():
                    for term in terms:
                        if term in ambiguity_lower:
                            add_kb_alert(
                                status=AllergenStatus.MIGHT_CONTAIN,
                                allergen=allergen,
                                ingredient_name=match.parsed_name,
                                reason=(
                                    f"'{match.parsed_name}' has ambiguous sourcing "
                                    f"that may relate to {allergen}"
                                ),
                            )
                            break

            for allergen, terms in expanded.items():
                if allergen in direct_hits:
                    continue
                for term in terms:
                    if term in entry_names_lower:
                        # Rule 6: coconut
                        if allergen in ("tree_nuts", "tree nut") and term == "coconut":
                            add_kb_alert(
                                status=AllergenStatus.MIGHT_CONTAIN,
                                allergen=allergen,
                                ingredient_name=match.parsed_name,
                                reason="Coconut is botanically a drupe and distinct from tree nuts; FDA classifies it as a tree nut, verify tolerance",
                            )
                            break
                        # Rule 7: oats
                        if allergen == "gluten" and term in ("oats", "oat flour", "oat"):
                            add_kb_alert(
                                status=AllergenStatus.MIGHT_CONTAIN,
                                allergen=allergen,
                                ingredient_name=match.parsed_name,
                                reason=f"'{match.parsed_name}' presents cross-contact risk with gluten-containing grains; verify certified gluten-free",
                            )
                            break

                        reason = f"'{match.parsed_name}' is a known form of {allergen}"
                        if is_fuzzy:
                            reason += "; matched approximately; verify on the label"
                        add_kb_alert(
                            status=AllergenStatus.CONTAINS,
                            allergen=allergen,
                            ingredient_name=match.parsed_name,
                            reason=reason,
                        )
                        break

            for custom in sorted(custom_terms):
                matched = False
                if custom in entry_names_lower:
                    matched = True
                else:
                    for name in entry_names_lower:
                        if custom in name or name in custom or fuzz.ratio(custom, name) >= 85:
                            matched = True
                            break
                if matched:
                    add_kb_alert(
                        status=AllergenStatus.CONTAINS,
                        allergen=custom,
                        ingredient_name=match.parsed_name,
                        reason=f"'{match.parsed_name}' matches custom allergen '{custom}'",
                    )

        if match.match_method == "unknown":
            parsed_lower = match.parsed_name.strip().lower()
            detected_allergen = None
            is_cross_contact = bool(_RE_CROSS_CONTACT.search(parsed_lower))

            # Rule 3, 5, 6 false positive checks for unknown strings
            is_plant_milk = bool(_RE_PLANT_MILKS.search(parsed_lower))
            is_cream_tartar = bool(_RE_CREAM_OF_TARTAR.search(parsed_lower))
            is_false_nut = bool(_RE_FALSE_NUTS.search(parsed_lower))
            is_false_butter = bool(_FALSE_POSITIVE_BUTTERS.search(parsed_lower))
            is_lactate = bool(_RE_LACTATES.search(parsed_lower))

            for allergen, terms in expanded.items():
                if allergen == "milk" and (is_cream_tartar or is_false_butter or is_lactate):
                    continue
                if allergen in ("tree_nuts", "tree nut") and is_false_nut:
                    continue

                for t in terms:
                    if t == "wheat":
                        pattern = r"(?<!buck[\s-])\bwheat\b"
                    elif t == "butter":
                        if is_false_butter:
                            continue
                        pattern = r"\bbutter\b"
                    else:
                        pattern = r"\b" + re.escape(t) + r"\b"

                    if re.search(pattern, parsed_lower):
                        detected_allergen = allergen
                        break
                if detected_allergen:
                    break

            if detected_allergen:
                if detected_allergen == "milk" and is_plant_milk:
                    add_kb_alert(
                        status=AllergenStatus.MIGHT_CONTAIN,
                        allergen=detected_allergen,
                        ingredient_name=match.parsed_name,
                        reason=f"'{match.parsed_name}' is a plant-based milk; verify no dairy added",
                    )
                elif detected_allergen in ("tree_nuts", "tree nut") and "coconut" in parsed_lower:
                    add_kb_alert(
                        status=AllergenStatus.MIGHT_CONTAIN,
                        allergen=detected_allergen,
                        ingredient_name=match.parsed_name,
                        reason="Coconut is botanically a drupe and distinct from tree nuts; FDA classifies it as a tree nut, verify tolerance",
                    )
                elif detected_allergen == "gluten" and any(w in parsed_lower for w in ("oat", "oats")):
                    add_kb_alert(
                        status=AllergenStatus.MIGHT_CONTAIN,
                        allergen=detected_allergen,
                        ingredient_name=match.parsed_name,
                        reason=f"'{match.parsed_name}' presents cross-contact risk with gluten-containing grains; verify certified gluten-free",
                    )
                elif is_cross_contact:
                    add_kb_alert(
                        status=AllergenStatus.MIGHT_CONTAIN,
                        allergen=detected_allergen,
                        ingredient_name=match.parsed_name,
                        reason=f"Unmatched label text '{match.parsed_name}' contains cross-contact advisory for '{detected_allergen}'",
                    )
                else:
                    add_kb_alert(
                        status=AllergenStatus.CONTAINS,
                        allergen=detected_allergen,
                        ingredient_name=match.parsed_name,
                        reason=f"Unmatched ingredient '{match.parsed_name}' contains allergen reference to '{detected_allergen}'",
                    )

    for stated_allergen in parsed.contains_allergens:
        stated_lower = stated_allergen.strip().lower()
        matched_allergen = None
        if stated_lower in user_allergen_keys or stated_lower in custom_terms:
            matched_allergen = stated_lower
        else:
            for u_allergen, terms in expanded.items():
                if stated_lower in terms or any(t in stated_lower for t in terms):
                    matched_allergen = u_allergen
                    break
        if matched_allergen:
            add_kb_alert(
                status=AllergenStatus.CONTAINS,
                allergen=matched_allergen,
                ingredient_name=stated_allergen,
                reason=f"Listed in 'Contains' statement: '{stated_allergen}'",
            )

    # Rule 9: "May contain" statement must give MIGHT_CONTAIN, never CONTAINS
    for stated_allergen in parsed.may_contain_allergens:
        stated_lower = stated_allergen.strip().lower()
        matched_allergen = None
        if stated_lower in user_allergen_keys or stated_lower in custom_terms:
            matched_allergen = stated_lower
        else:
            for u_allergen, terms in expanded.items():
                if stated_lower in terms or any(t in stated_lower for t in terms):
                    matched_allergen = u_allergen
                    break
        if matched_allergen:
            add_kb_alert(
                status=AllergenStatus.MIGHT_CONTAIN,
                allergen=matched_allergen,
                ingredient_name=stated_allergen,
                reason=f"Listed in 'May contain' statement: '{stated_allergen}'",
            )

    # Rule 2: Non-empty unreadable_parts -> UNCERTAIN
    if transcription is not None:
        for part in transcription.unreadable_parts:
            kb_alerts.append(AllergenAlert(
                status=AllergenStatus.UNCERTAIN,
                allergen="unknown",
                ingredient_name="",
                reason=f"Part of the label could not be read: '{part}'",
            ))

    # Check derived additive allergen codes for all parsed items and matches
    for match in matches:
        code_matches = _RE_DERIVED_ADDITIVE_CODES.findall(match.parsed_name)
        for code_match in code_matches:
            c_key = code_match.lower()
            if c_key in DERIVED_ADDITIVE_ALLERGENS:
                for allergen, status, reason in DERIVED_ADDITIVE_ALLERGENS[c_key]:
                    if allergen in user_allergen_keys or allergen in custom_terms or any(allergen in expanded.get(u, set()) for u in user_allergen_keys):
                        add_kb_alert(
                            status=status,
                            allergen=allergen,
                            ingredient_name=match.parsed_name,
                            reason=reason,
                        )

    for ing in parsed.ingredients:
        code_matches = _RE_DERIVED_ADDITIVE_CODES.findall(ing.name)
        for code_match in code_matches:
            c_key = code_match.lower()
            if c_key in DERIVED_ADDITIVE_ALLERGENS:
                for allergen, status, reason in DERIVED_ADDITIVE_ALLERGENS[c_key]:
                    if allergen in user_allergen_keys or allergen in custom_terms or any(allergen in expanded.get(u, set()) for u in user_allergen_keys):
                        add_kb_alert(
                            status=status,
                            allergen=allergen,
                            ingredient_name=ing.name,
                            reason=reason,
                        )

    full_raw_text = _gather_raw_text(parsed, transcription)
    raw_alerts = _scan_raw_text(full_raw_text, user_allergen_keys, expanded, custom_terms)

    merged_alerts: list[AllergenAlert] = []
    all_allergens = set(user_allergen_keys) | set(custom_terms) | {"unknown"}

    uncertain_alerts = [a for a in kb_alerts + raw_alerts if a.status == AllergenStatus.UNCERTAIN]
    merged_alerts.extend(uncertain_alerts)

    for allergen in sorted(all_allergens):
        if allergen == "unknown":
            continue

        kb_allergen_alerts = [a for a in kb_alerts if a.allergen.strip().lower() == allergen]
        raw_allergen_alerts = [a for a in raw_alerts if a.allergen.strip().lower() == allergen]

        kb_status = max((a.status for a in kb_allergen_alerts), default=AllergenStatus.CLEAR, key=lambda s: _SEVERITY_RANK[s])
        raw_status = max((a.status for a in raw_allergen_alerts), default=AllergenStatus.CLEAR, key=lambda s: _SEVERITY_RANK[s])

        target_status = max(kb_status, raw_status, key=lambda s: _SEVERITY_RANK[s])

        if target_status != AllergenStatus.CLEAR:
            matching_kb = [a for a in kb_allergen_alerts if a.status == target_status]
            matching_raw = [a for a in raw_allergen_alerts if a.status == target_status]

            if matching_kb:
                merged_alerts.extend(matching_kb)
            if matching_raw:
                kb_reasons = {a.reason.lower() for a in matching_kb}
                for ra in matching_raw:
                    if ra.reason.lower() not in kb_reasons:
                        merged_alerts.append(ra)
            if not matching_kb and not matching_raw:
                merged_alerts.extend(kb_allergen_alerts)
                merged_alerts.extend(raw_allergen_alerts)

    if not merged_alerts:
        overall = AllergenStatus.CLEAR
    else:
        overall = max(
            (alert.status for alert in merged_alerts),
            key=lambda s: _SEVERITY_RANK[s],
        )

    # Compute result confidence
    engine_name = transcription.engine_used if transcription else ""
    unreadable_p = transcription.unreadable_parts if transcription else None
    confidence, confidence_reason = allergen_result_confidence_with_reason(
        matches=matches,
        transcription=transcription,
        engine_used=engine_name,
        unreadable_parts=unreadable_p,
        parsed=parsed,
    )

    return AllergenResult(
        overall_status=overall,
        alerts=merged_alerts,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )
