"""Ingredient matching ladder for the Unwrap app.

Resolves each :class:`ParsedIngredient` against the knowledge-base via a
deterministic matching ladder:

1. **Exact** — normalised name lookup (O(1))
2. **E-number** — e.g. ``E322`` lookup (O(1))
3. **Fuzzy** — ``rapidfuzz.fuzz.WRatio`` with configurable threshold
4. **LLM pick** — optional, disabled by default
5. **Unknown** — ingredient could not be identified; logged to JSONL
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from backend.config import Settings, get_settings
from backend.models import IngredientKBEntry, MatchResult, ParsedIngredient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip, and collapse multiple whitespace to a single space."""
    return _MULTI_SPACE.sub(" ", text.strip().lower())


def normalize_e_lookup_keys(e_num: str) -> list[str]:
    """Generate standard format variations (E, INS, bare) for an additive code."""
    clean = re.sub(r"[\s\-]", "", e_num.lower())
    m = re.match(r"^(?:ins|e)?(\d{3,4})([a-z]?(?:\([a-z0-9]+\))?)?$", clean)
    if not m:
        return [clean]
    digits, suffix = m.group(1), m.group(2) or ""
    return [f"e{digits}{suffix}", f"ins{digits}{suffix}", f"{digits}{suffix}"]


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Loads ``ingredients.json`` and builds fast lookup indices.

    Indices built on construction:

    * ``_name_index``  — normalised name ➜ :class:`IngredientKBEntry`
    * ``_e_number_index`` — e.g. ``"e322"`` ➜ :class:`IngredientKBEntry`
    * ``_all_names`` — flat list of every normalised name (for fuzzy search)
    * ``_name_to_entry`` — maps each name in ``_all_names`` to its entry
    """

    def __init__(self, data_path: Path | None = None) -> None:
        if data_path is None:
            data_path = Path(__file__).resolve().parent / "data" / "ingredients.json"

        self._data_path = data_path

        # Primary lookup tables
        self._entries: dict[str, IngredientKBEntry] = {}
        self._name_index: dict[str, IngredientKBEntry] = {}
        self._e_number_index: dict[str, IngredientKBEntry] = {}

        # Fuzzy-match support
        self._all_names: list[str] = []
        self._name_to_entry: dict[str, IngredientKBEntry] = {}

        self._load(data_path)

    # -- loading -----------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Read the JSON array of ingredient entries and populate indices."""
        if not path.exists():
            logger.warning("Ingredient KB file not found at %s — KB will be empty.", path)
            return

        with open(path, encoding="utf-8") as fh:
            raw_entries: list[dict] = json.load(fh)

        for raw in raw_entries:
            entry = IngredientKBEntry(**raw)
            self._index_entry(entry)

        logger.info(
            "KnowledgeBase loaded: %d entries, %d names indexed.",
            len(raw_entries),
            len(self._all_names),
        )

    def _index_entry(self, entry: IngredientKBEntry) -> None:
        """Index a single KB entry under all its known names & e-number."""
        self._entries[entry.id] = entry

        # Collect canonical_name + every alias in entry.names
        all_names: list[str] = [entry.canonical_name] + list(entry.names)

        for name in all_names:
            norm = _normalise(name)
            if not norm:
                continue
            self._name_index[norm] = entry
            # Only add to the fuzzy list once per unique normalised form
            if norm not in self._name_to_entry:
                self._all_names.append(norm)
                self._name_to_entry[norm] = entry

        # E-number index (stored with standard prefix key variations: e322, ins322, 322)
        if entry.e_number:
            for k in normalize_e_lookup_keys(entry.e_number):
                self._e_number_index[k] = entry

    # -- public queries ----------------------------------------------------

    @property
    def entries(self) -> list[IngredientKBEntry]:
        """Return all unique ingredient entries in the knowledge base."""
        return list(self._entries.values())

    def get(self, ingredient_id: str) -> Optional[IngredientKBEntry]:
        """Return an entry by its unique ID, or None."""
        return self._entries.get(ingredient_id)

    def lookup_name(self, name: str) -> Optional[IngredientKBEntry]:
        """Return an exact name-index hit, or ``None``."""
        return self._name_index.get(_normalise(name))

    def lookup_e_number(self, e_number: str) -> Optional[IngredientKBEntry]:
        """Return an e-number hit, or ``None``."""
        for k in normalize_e_lookup_keys(e_number):
            hit = self._e_number_index.get(k)
            if hit is not None:
                return hit
        return self._e_number_index.get(e_number.strip().lower())


# ---------------------------------------------------------------------------
# Unknown-ingredient logging
# ---------------------------------------------------------------------------

_UNKNOWN_LOG_PATH = Path(__file__).resolve().parent / "data" / "unknown_ingredients.jsonl"


def _log_unknown(ingredient_name: str) -> None:
    """Append a JSON line to the unknown-ingredients log file."""
    record = {
        "name": ingredient_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(_UNKNOWN_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("Failed to write to unknown-ingredients log.")


# ---------------------------------------------------------------------------
# Matching ladder
# ---------------------------------------------------------------------------

# Confidence lookup by match method
_CONFIDENCE: dict[str, str] = {
    "exact": "high",
    "e_number": "high",
    "fuzzy": "medium",
    "llm_pick": "medium",
    "unknown": "low",
}


def detect_merged_aliases(name: str, kb: KnowledgeBase) -> list[str]:
    """Check if an unknown token contains two or more known non-overlapping ingredient aliases.

    E.g. 'Palm Oil Cocoa Powder' -> ['Palm Oil', 'Cocoa Powder']
         'Salt Raising Agent' -> ['Salt', 'Raising Agent']
    """
    clean = _normalise(name)
    if len(clean) < 6:
        return []

    hits: list[tuple[int, int, str, str]] = []  # (start, end, alias, canonical_name)
    # Check aliases sorted by length descending
    for alias in sorted(kb._all_names, key=len, reverse=True):
        if len(alias) < 3:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, clean):
            s, e = m.start(), m.end()
            # Ensure no overlap with previously matched longer aliases
            if not any(max(s, hs) < min(e, he) for hs, he, *_ in hits):
                entry = kb._name_to_entry[alias]
                hits.append((s, e, alias, entry.canonical_name))

    hits.sort(key=lambda x: x[0])
    if len(hits) >= 2:
        return [h[3] for h in hits]
    return []


def _match_single(
    parsed: ParsedIngredient,
    kb: Optional[KnowledgeBase],
    config: Settings,
) -> MatchResult:
    """Run the matching ladder for a single parsed ingredient.

    Ladder order:
    1. Exact name match (if KB provided)
    2. E-number match
    3. Fuzzy match (if KB provided)
    4. LLM pick (skipped unless enabled)
    5. Unknown
    """
    norm_name = _normalise(parsed.name)

    # 1. Exact name match (if KB loaded) ---------------------------------
    if kb is not None:
        entry = kb.lookup_name(norm_name)
        if entry is not None:
            return MatchResult(
                parsed_name=parsed.name,
                kb_entry=entry,
                match_method="exact",
                match_score=100.0,
                confidence=_CONFIDENCE["exact"],
            )

    # 2. E-number match ---------------------------------------------------
    from backend.parser import normalize_additive_code
    e_code = parsed.e_number
    if not e_code:
        e_code = normalize_additive_code(parsed.name)
        if not e_code:
            m_code = re.search(r"\(([^)]+)\)", parsed.name)
            if m_code:
                e_code = normalize_additive_code(m_code.group(1))

    if e_code:
        entry = kb.lookup_e_number(e_code) if kb is not None else None
        if entry is not None:
            return MatchResult(
                parsed_name=parsed.name,
                kb_entry=entry,
                match_method="e_number",
                match_score=100.0,
                confidence=_CONFIDENCE["e_number"],
            )
        else:
            # Recognized additive code
            return MatchResult(
                parsed_name=parsed.name,
                kb_entry=None,
                match_method="e_number",
                match_score=100.0,
                confidence="medium",
            )

    # 3. Fuzzy match (requires at least 5 characters for safety, if KB loaded)
    if kb is not None and len(norm_name) >= 5 and kb._all_names:
        best = process.extractOne(
            norm_name,
            kb._all_names,
            scorer=fuzz.WRatio,
            score_cutoff=config.FUZZY_MATCH_THRESHOLD,
        )
        if best is not None:
            matched_name, score, _idx = best
            entry = kb._name_to_entry[matched_name]
            return MatchResult(
                parsed_name=parsed.name,
                kb_entry=entry,
                match_method="fuzzy",
                match_score=round(score, 2),
                confidence=_CONFIDENCE["fuzzy"],
            )

    # 4. LLM pick (placeholder — disabled by default) --------------------
    if config.LLM_PICK_ENABLED:
        logger.debug("LLM pick is enabled but not yet implemented for '%s'.", parsed.name)

    # 5. Unknown ----------------------------------------------------------
    suggested = detect_merged_aliases(parsed.name, kb) if kb is not None else []
    if suggested:
        parsed.possible_merged = True
        parsed.suggested_split = suggested
        logger.info("Ingredient '%s' marked possible_merged with split: %s", parsed.name, suggested)

    _log_unknown(parsed.name)
    logger.info("Ingredient '%s' marked unknown (no KB match).", parsed.name)
    return MatchResult(
        parsed_name=parsed.name,
        kb_entry=None,
        match_method="unknown",
        match_score=0.0,
        confidence=_CONFIDENCE["unknown"],
        possible_merged=parsed.possible_merged,
        suggested_split=parsed.suggested_split,
    )


def is_additive_class_item(ingredient: ParsedIngredient) -> bool:
    """Check if an ingredient is an expanded class-name additive item (e.g. 'Stabilizer (INS 412)')."""
    from backend.parser import _RE_CLASS_NAME, _extract_paren_group
    name_part, inner = _extract_paren_group(ingredient.name)
    return bool(inner and _RE_CLASS_NAME.match(name_part))


def match_ingredients(
    parsed: list[ParsedIngredient],
    kb: Optional[KnowledgeBase] = None,
    config: Settings | None = None,
) -> list[MatchResult]:
    """Match a list of parsed ingredients.

    When ``kb`` is None, resolves without curated KB lookup: recognizes additive
    codes by pattern and marks remaining ingredients as unknown.

    Parameters
    ----------
    parsed:
        Parsed ingredient list (may include nested children).
    kb:
        Optional :class:`KnowledgeBase`. If None, runs without KB lookup.
    config:
        Application settings.  Falls back to :func:`get_settings` when
        ``None``.

    Returns
    -------
    list[MatchResult]
        One :class:`MatchResult` per ingredient.
    """
    if config is None:
        config = get_settings()

    results: list[MatchResult] = []
    for ingredient in parsed:
        # Match the ingredient itself
        results.append(_match_single(ingredient, kb, config))

        # Recursively match children (sub-ingredients in parenthesised groups),
        # but avoid redundant matching if parent is an expanded class additive item
        if ingredient.children:
            if is_additive_class_item(ingredient):
                continue
            results.extend(
                match_ingredients(ingredient.children, kb, config)
            )

    return results
