"""Deterministic ingredient text parser.

Splits raw ingredient label text into structured ParsedIngredient trees,
extracts E-numbers, percentages, and allergen "Contains:" / "May contain:"
statements.  Every decision is rule-based — no AI model is involved.
"""
from __future__ import annotations

import re
from typing import Optional

from backend.models import ParsedIngredient, ParsedResult


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

# E-number variants:  E330, E 330, INS 330, INS330, (330)
_RE_E_NUMBER = re.compile(
    r"""
    (?:E\s*(\d{3,4}[a-z]?(?:i{0,3}|iv|v)?))  # E330, E 330, E330i, E330ii
    |(?:INS\s*(\d{3,4}[a-z]?(?:i{0,3}|iv|v)?))  # INS330, INS 330
    |(?:^\((\d{3,4}[a-z]?(?:i{0,3}|iv|v)?)\)$)   # (330) standalone
    """,
    re.IGNORECASE | re.VERBOSE,
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

    # Extract E-number from the cleaned name
    e_number = _extract_e_number(name_part)

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
    for idx, raw_part in enumerate(top_parts):
        ingredient = _parse_single_ingredient(raw_part, position=idx)
        ingredients.append(ingredient)

    return ParsedResult(
        ingredients=ingredients,
        contains_allergens=contains_allergens,
        may_contain_allergens=may_contain_allergens,
        raw_text=original_text,
        has_ingredients_marker=has_ingredients_marker,
    )
