"""Deterministic ingredient text parser.

Splits raw ingredient label text into structured ParsedIngredient trees,
extracts E-numbers, percentages, and allergen "Contains:" / "May contain:"
statements.  Every decision is rule-based — no AI model is involved.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from backend.models import ParsedIngredient, ParsedResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable patterns (all compiled once at import time)
# ---------------------------------------------------------------------------

# Flexible marker: "Ingredients:", "Ingredients.", "INGREDIENTS -", or just "Ingredients"
_RE_INGREDIENTS_MARKER = re.compile(r"\bingredients\b[:.\s-]*", re.IGNORECASE)

# Legacy alias
_RE_HEADER = _RE_INGREDIENTS_MARKER

# "Contains: X, Y and Z" — typically at the end of the text
_RE_CONTAINS = re.compile(
    r"[.;,]?\s*contains\s*:\s*(.+?)\.?\s*$",
    re.IGNORECASE,
)

# "May contain traces of X, Y and Z" or "May contain X, Y"
_RE_MAY_CONTAIN = re.compile(
    r"[.;,]?\s*may\s+contain(?:\s+traces\s+of)?\s*:?\s*(.+?)\.?\s*$",
    re.IGNORECASE,
)

# E-number variants: E330, E-322, E 330, INS 330, INS330, (330), bare 330, 160a(i), 150(iv)
_RE_E_NUMBER = re.compile(
    r"""
    (?:E\s*[-]?\s*(\d{3,4}[a-z]?(?:\([ivxIVX0-9a-zA-Z]+\)|i{0,3}|iv|v)?))
    |(?:INS\s*[-]?\s*(\d{3,4}[a-z]?(?:\([ivxIVX0-9a-zA-Z]+\)|i{0,3}|iv|v)?))
    |(?:^\(?(\d{3,4}[a-z]?(?:\([ivxIVX0-9a-zA-Z]+\)|i{0,3}|iv|v)?)\)?$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Common OCR misreads in digit positions for additive codes (O/0, l/1, S/5, Z/2)
_MISREAD_MAP = {
    "O": "0", "o": "0",
    "l": "1", "I": "1", "|": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
}

# Regex matching candidate additive codes that may have OCR misreads in digit positions
_RE_CODE_CANDIDATE = re.compile(
    r"""(?xi)
    \b
    (?P<prefix>INS\s*[-]?\s*|E\s*[-]?\s*)?
    (?P<digits>[0-9OlISZolisz]{3,4})
    (?P<suffix>[a-zA-Z]?(?:\([ivxIVX0-9a-zA-Z]+\))?)
    \b
    """
)

# Canonical additive code matcher (digits + optional suffix)
_RE_ADDITIVE_CODE = re.compile(
    r"""(?xi)
    ^\s*
    (?:INS\s*[-]?\s*|E\s*[-]?\s*)?
    (?P<digits>\d{3,4})
    (?P<suffix>[a-zA-Z]?(?:\([ivxIVX0-9a-zA-Z]+\))?)
    \s*$
    """
)

# Recognized additive class names:
# stabilizer/stabiliser, emulsifier, preservative, colour/color, acidity regulator,
# antioxidant, raising agent, anti-caking agent, thickener, sweetener, flavour enhancer, food acid
_CLASS_NAME_PATTERNS = [
    r"anti[-\s]?caking\s+agents?",
    r"acidity\s+regulators?",
    r"food\s+acids?",
    r"flavo?u?r\s+enhancers?",
    r"raising\s+agents?",
    r"anti[-\s]?oxidants?",
    r"thickening\s+agents?",
    r"thickeners?",
    r"stabili[zs]ers?",
    r"emulsifiers?",
    r"preservatives?",
    r"colou?ri?ngs?",
    r"colou?rs?",
    r"sweeteners?",
]
_RE_CLASS_NAME = re.compile(
    r"^(" + "|".join(_CLASS_NAME_PATTERNS) + r")$",
    re.IGNORECASE,
)

# Protective or descriptive clause prefixes inside brackets like "Protected with Antioxidant (319)"
_RE_PROTECTIVE_PREFIX = re.compile(
    r"^(?:protected|preserved|treated|fortified|prepared)?\s*with\s+",
    re.IGNORECASE,
)

# Percentage:  "60%", "(60%)", "10 %", "2.5%"
_RE_PERCENTAGE = re.compile(
    r"\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?",
)

# Check whether a parenthesised group is a pure percentage like "(60%)"
_RE_PURE_PERCENTAGE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*%\s*$",
)

# Stray trailing period / excessive whitespace
_RE_TRAILING_PERIOD = re.compile(r"\.\s*$")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip."""
    return _RE_MULTI_SPACE.sub(" ", text).strip()


def repair_additive_ocr(text: str) -> tuple[str, list[dict]]:
    """Fix common OCR misreads (O/0, l/1, S/5, Z/2) in additive code candidates.

    Returns the repaired text and the list of recorded corrections:
    [{"original": str, "corrected": str, "score": 0.95}].
    """
    corrections: list[dict] = []

    def _repl(m: re.Match) -> str:
        prefix = m.group("prefix") or ""
        digits_raw = m.group("digits")
        suffix = m.group("suffix") or ""
        has_prefix = bool(prefix.strip())
        has_misreads = any(ch in _MISREAD_MAP for ch in digits_raw)
        has_digits = any(ch.isdigit() for ch in digits_raw)
        if (has_prefix or has_digits) and has_misreads:
            fixed_digits = "".join(_MISREAD_MAP.get(ch, ch) for ch in digits_raw)
            if fixed_digits.isdigit() and len(fixed_digits) in (3, 4):
                original_token = m.group(0)
                corrected_token = f"{prefix}{fixed_digits}{suffix}"
                if original_token != corrected_token:
                    corrections.append({
                        "original": original_token,
                        "corrected": corrected_token,
                        "score": 0.95,
                    })
                    logger.info("OCR repaired additive code: '%s' -> '%s'", original_token, corrected_token)
                    return corrected_token
        return m.group(0)

    repaired = _RE_CODE_CANDIDATE.sub(_repl, text)
    return repaired, corrections


def normalize_additive_code(raw: str) -> Optional[str]:
    """Map E322, E-322, E 322, INS 322, INS322, and a bare 322 to canonical 'INS 322'.

    Preserves letter suffixes (150d, 160a) and nested Roman numerals like 150(iv) and 160a(i).
    Logs the original text.
    """
    if not raw or not raw.strip():
        return None
    m = _RE_ADDITIVE_CODE.match(raw.strip())
    if not m:
        return None
    digits = m.group("digits")
    suffix = m.group("suffix") or ""
    canonical = f"INS {digits}{suffix}"
    logger.info("Normalized additive code '%s' -> '%s'", raw.strip(), canonical)
    return canonical


def is_pure_additive_code(text: str) -> bool:
    """Return True if the entire string represents an additive code."""
    if not text or not text.strip():
        return False
    return bool(_RE_ADDITIVE_CODE.match(text.strip()))


def parse_additive_codes_from_inner(inner: str) -> Optional[list[str]]:
    """Parse comma-separated additive codes from the inside of a class bracket.

    Returns a list of canonical codes (e.g. ['INS 412', 'INS 415']) if ALL
    non-empty chunks are additive codes. Returns None if any chunk is not a code.
    """
    if not inner or not inner.strip():
        return None
    parts = _split_top_level(inner, separators=",")
    if not parts:
        return None
    codes: list[str] = []
    for p in parts:
        clean = p.strip()
        if not clean:
            continue
        c = normalize_additive_code(clean)
        if c is None:
            return None
        codes.append(c)
    return codes if codes else None


def _extract_e_number(text: str) -> Optional[str]:
    """Return normalised E-number (e.g. 'E330') if text contains one, else None."""
    m = _RE_E_NUMBER.search(text)
    if m:
        # One of the three groups will have matched
        digits = m.group(1) or m.group(2) or m.group(3)
        if digits:
            return f"E{digits.upper()}" if not digits[0].isalpha() else f"E{digits.upper()}"
    return None


def _extract_percentage(text: str) -> tuple[Optional[float], str]:
    """Extract percentage from text, returning (value, cleaned_text).

    The cleaned text has the percentage token removed so it doesn't
    pollute the ingredient name.
    """
    m = _RE_PERCENTAGE.search(text)
    if m:
        value = float(m.group(1))
        cleaned = text[: m.start()] + text[m.end() :]
        cleaned = _normalise_whitespace(cleaned)
        return value, cleaned
    return None, text


def _split_allergen_list(text: str) -> list[str]:
    """Split a string like 'Wheat, Soy and Milk' into lowercase allergens."""
    # Replace " and " with comma to unify
    unified = re.sub(r"\s+and\s+", ",", text, flags=re.IGNORECASE)
    parts = [p.strip().lower() for p in unified.split(",")]
    return [p for p in parts if p]


def _find_matching_close(text: str, start: int) -> int:
    """Find the index of the closing bracket/paren that matches the opener at *start*.

    Handles arbitrary nesting depth and mixed bracket types.
    Returns -1 if no matching close is found.
    """
    openers = "(["
    closers = ")]"
    open_char = text[start]
    close_char = closers[openers.index(open_char)]
    depth = 1
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch in openers:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
        elif ch in closers:
            # Mismatched closer — treat as closing to be lenient
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_level(text: str, separators: str = ",;\n\r") -> list[str]:
    """Split *text* on *separators* (including newlines), but only at depth-0 (outside brackets/parens).

    Returns the raw chunks with leading/trailing whitespace stripped.
    Handles arbitrary nesting of ( ) and [ ]. Does NOT split on decimal points.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch in separators and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    # Flush remaining
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return [p for p in parts if p]


def _extract_paren_group(text: str) -> tuple[str, Optional[str]]:
    """Separate a name from its parenthesised group at the TOP level.

    E.g.  "Emulsifier (Soy Lecithin, E322)"  ->  ("Emulsifier", "Soy Lecithin, E322")
          "Wheat Flour (60%)"                 ->  ("Wheat Flour", "60%")
          "Salt"                              ->  ("Salt", None)
    """
    text_clean = text.strip()
    # Standalone additive codes like E160a(i) or 150(iv) must not have nested parens stripped
    if is_pure_additive_code(text_clean):
        return text_clean, None

    # Find the first top-level opening paren/bracket
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            if depth == 0:
                # This is the first top-level opener
                close_idx = _find_matching_close(text, i)
                if close_idx == -1:
                    # Unmatched — give up, return whole text as name
                    break
                name_part = text[:i].strip()
                inner = text[i + 1 : close_idx].strip()
                # Check for anything after the closing bracket (rare but handle)
                after = text[close_idx + 1 :].strip()
                if after:
                    name_part = name_part or text[:close_idx + 1].strip()
                return name_part, inner if inner else None
            else:
                depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
    return text.strip(), None


def _handle_trailing_and(parts: list[str]) -> list[str]:
    """Handle the last element containing ' and ' which often joins two items.

    E.g. ['Sugar', 'Salt and Pepper'] -> ['Sugar', 'Salt', 'Pepper']
    But we only split on ' and ' for the LAST element of the top-level list,
    as 'and' in middle items can be part of names.
    """
    if not parts:
        return parts

    last = parts[-1]
    # Only split if ' and ' appears and it's not inside brackets
    depth = 0
    and_positions: list[int] = []
    i = 0
    lower_last = last.lower()
    while i < len(last):
        ch = last[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        elif depth == 0:
            # Check for ' and ' at this position
            if lower_last[i:i+5] == " and " and i > 0:
                and_positions.append(i)
        i += 1

    if and_positions:
        # Split on the LAST ' and ' (most common: "X and Y")
        pos = and_positions[-1]
        before = last[:pos].strip()
        after = last[pos + 5:].strip()
        if before and after:
            return parts[:-1] + [before, after]

    return parts


def _parse_children(inner_text: str) -> list[ParsedIngredient]:
    """Recursively parse the contents of a parenthesised group into children."""
    child_parts = _split_top_level(inner_text, separators=",;")
    child_parts = _handle_trailing_and(child_parts)

    children: list[ParsedIngredient] = []
    for idx, raw_child in enumerate(child_parts):
        child = _parse_single_ingredient(raw_child, position=idx)
        children.append(child)
    return children


def _clean_name(name: str) -> str:
    """Clean up an ingredient name: strip trailing period, normalise whitespace."""
    name = _RE_TRAILING_PERIOD.sub("", name)
    name = _normalise_whitespace(name)
    # Remove empty trailing/leading parentheses
    name = re.sub(r"\(\s*\)", "", name).strip()
    name = re.sub(r"\[\s*\]", "", name).strip()
    return name


def _parse_single_ingredient(raw: str, position: int = 0) -> ParsedIngredient:
    """Parse a single ingredient chunk into a ParsedIngredient.

    Handles percentage extraction, E-number detection, and child
    sub-ingredients inside parentheses.
    """
    raw = _normalise_whitespace(raw)
    raw_text = raw  # preserve original for raw_text field

    # Clean trailing period
    raw = _RE_TRAILING_PERIOD.sub("", raw).strip()

    # Early check: if the entire text is a standalone E-number like "(330)",
    # don't treat the parens as a sub-ingredient group.
    e_early = _extract_e_number(raw)
    if e_early is not None and re.match(r"^\(\d{3,4}[a-z]?\)$", raw, re.IGNORECASE):
        return ParsedIngredient(
            name=raw,
            raw_text=raw_text,
            e_number=e_early,
            percentage=None,
            children=[],
            position=position,
        )

    # Separate name from parenthesised group
    name_part, inner = _extract_paren_group(raw)

    percentage: Optional[float] = None
    children: list[ParsedIngredient] = []

    if inner is not None:
        # Decide: is the inner group a pure percentage, or sub-ingredients?
        if _RE_PURE_PERCENTAGE.match(inner):
            # Pure percentage — assign to parent, no children
            pct_match = re.search(r"(\d+(?:\.\d+)?)", inner)
            if pct_match:
                percentage = float(pct_match.group(1))
        else:
            # Sub-ingredients — parse as children
            children = _parse_children(inner)

    # If no percentage from parens, check the name for inline percentages
    if percentage is None:
        percentage, name_part = _extract_percentage(name_part)

    # Clean final name
    name_part = _clean_name(name_part)

    # Extract E-number from the cleaned name or inner code
    e_number = _extract_e_number(name_part)
    if not e_number and inner:
        e_number = _extract_e_number(inner) or normalize_additive_code(inner)
    if not e_number:
        e_number = normalize_additive_code(raw)

    return ParsedIngredient(
        name=name_part,
        raw_text=raw_text,
        e_number=e_number,
        percentage=percentage,
        children=children,
        position=position,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_ingredients(text: str) -> ParsedResult:
    """Parse a raw ingredient label string into a structured ParsedResult.

    This is a purely deterministic parser — no AI model involvement.
    It handles nested parenthesised sub-ingredient lists, E-numbers,
    percentages, and allergen "Contains:" / "May contain:" statements.

    Parameters
    ----------
    text:
        Raw ingredient text, possibly with a leading "Ingredients:" header
        and trailing allergen statements.

    Returns
    -------
    ParsedResult with structured ingredients, contains_allergens, and
    may_contain_allergens.
    """
    if not text or not text.strip():
        return ParsedResult(raw_text=text or "", has_ingredients_marker=False)

    original_text = text
    # Normalize horizontal whitespace while preserving newlines
    text = re.sub(r"[ \t]+", " ", text).strip()

    # 1. Discard everything before an 'Ingredients' marker (case-insensitive,
    #    with or without colon or period). If no marker exists, parse everything
    #    but flag has_ingredients_marker = False.
    m_marker = _RE_INGREDIENTS_MARKER.search(text)
    if m_marker:
        has_ingredients_marker = True
        text = text[m_marker.end() :].strip()
    else:
        has_ingredients_marker = False
        text = text.strip()

    # 2. Extract and remove "May contain" statement FIRST (it usually comes
    #    after "Contains:"), then extract "Contains:" statement.
    #    We work from the end of the string backwards.
    may_contain_allergens: list[str] = []
    contains_allergens: list[str] = []

    m_may = _RE_MAY_CONTAIN.search(text)
    if m_may:
        may_contain_allergens = _split_allergen_list(m_may.group(1))
        text = text[: m_may.start()].strip()

    m_contains = _RE_CONTAINS.search(text)
    if m_contains:
        contains_allergens = _split_allergen_list(m_contains.group(1))
        text = text[: m_contains.start()].strip()

    # 3. Strip trailing period from the remaining ingredient text
    text = _RE_TRAILING_PERIOD.sub("", text).strip()

    # 4. Split top-level ingredients (respecting brackets/parens, commas, semicolons, and newlines)
    top_parts = _split_top_level(text, separators=",;\n\r")

    # 5. Handle "and" joining in the last element
    top_parts = _handle_trailing_and(top_parts)

    # 6. Parse each top-level ingredient
    ingredients: list[ParsedIngredient] = []
    corrections: list[dict] = []
    for raw_part in top_parts:
        # Step 6a: OCR repair on candidate codes
        repaired_part, part_corrections = repair_additive_ocr(raw_part)
        corrections.extend(part_corrections)

        # Step 6b: Check if this chunk has parenthesised / bracketed additive code(s) or clauses
        name_part, inner = _extract_paren_group(repaired_part)
        if inner:
            # Case 1: Pure additive code(s) inside brackets
            # Per spec: Keep a trailing code on ANY "<name> (<code>)" item, not only known class names.
            codes = parse_additive_codes_from_inner(inner)
            if codes:
                for code in codes:
                    child_e = _extract_e_number(code) or code
                    child = ParsedIngredient(
                        name=code,
                        raw_text=code,
                        e_number=child_e,
                        position=0,
                    )
                    item_name = f"{name_part} ({code})" if name_part else code
                    ing = ParsedIngredient(
                        name=item_name,
                        raw_text=raw_part,
                        e_number=code,
                        percentage=None,
                        children=[child],
                        position=len(ingredients),
                    )
                    ingredients.append(ing)
                continue

            # Case 2: Protective / descriptive clause like [Protected with Antioxidant (319)]
            m_prot = _RE_PROTECTIVE_PREFIX.match(inner)
            if m_prot:
                stripped_inner = inner[m_prot.end():].strip()
                if name_part:
                    ingredients.append(_parse_single_ingredient(name_part, position=len(ingredients)))

                sub_parts = _split_top_level(stripped_inner, separators=",;")
                for sp in sub_parts:
                    sp_repaired, sp_corrs = repair_additive_ocr(sp)
                    corrections.extend(sp_corrs)
                    sp_name, sp_sub = _extract_paren_group(sp_repaired)
                    if sp_sub:
                        sp_codes = parse_additive_codes_from_inner(sp_sub)
                        if sp_codes:
                            for c in sp_codes:
                                c_e = _extract_e_number(c) or c
                                c_child = ParsedIngredient(name=c, raw_text=c, e_number=c_e, position=0)
                                c_name = f"{sp_name} ({c})" if sp_name else c
                                ingredients.append(ParsedIngredient(
                                    name=c_name,
                                    raw_text=sp,
                                    e_number=c,
                                    children=[c_child],
                                    position=len(ingredients),
                                ))
                            continue
                    ingredients.append(_parse_single_ingredient(sp_repaired, position=len(ingredients)))
                continue

            # Case 3: Inner has its own sub-bracketed additive code like [Antioxidant (319)]
            inner_name, inner_sub = _extract_paren_group(inner)
            if inner_sub:
                inner_codes = parse_additive_codes_from_inner(inner_sub)
                if inner_codes:
                    if name_part:
                        ingredients.append(_parse_single_ingredient(name_part, position=len(ingredients)))
                    for c in inner_codes:
                        c_e = _extract_e_number(c) or c
                        c_child = ParsedIngredient(name=c, raw_text=c, e_number=c_e, position=0)
                        c_name = f"{inner_name} ({c})" if inner_name else c
                        ingredients.append(ParsedIngredient(
                            name=c_name,
                            raw_text=inner,
                            e_number=c,
                            children=[c_child],
                            position=len(ingredients),
                        ))
                    continue

        # Step 6c: Regular ingredient parsing
        ingredient = _parse_single_ingredient(repaired_part, position=len(ingredients))
        # Per spec: Never drop bracket content silently; if it can't be parsed into children, keep it as its own chip
        if inner and not ingredient.children and not _RE_PURE_PERCENTAGE.match(inner):
            if name_part:
                ingredient.name = name_part
                ingredients.append(ingredient)
                ingredients.append(_parse_single_ingredient(inner, position=len(ingredients)))
                continue
        ingredients.append(ingredient)

    return ParsedResult(
        ingredients=ingredients,
        contains_allergens=contains_allergens,
        may_contain_allergens=may_contain_allergens,
        raw_text=original_text,
        has_ingredients_marker=has_ingredients_marker,
        corrections=corrections,
    )
