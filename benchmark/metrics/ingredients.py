"""Ingredient-level metrics: Precision, Recall, F1, exact match, hallucination, and allergen recall."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from rapidfuzz import fuzz

from backend.parser import ParsedResult, parse_ingredients

# Load allergen vocabulary from backend
ALLERGENS_JSON = Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "allergens.json"
try:
    with open(ALLERGENS_JSON, "r", encoding="utf-8") as f:
        _allergen_data = json.load(f)
        ALLERGEN_VOCAB = set(t.lower() for t in _allergen_data.get("vocabulary", []))
        for terms in _allergen_data.get("hierarchy", {}).values():
            for t in terms:
                ALLERGEN_VOCAB.add(t.lower())
except Exception:
    ALLERGEN_VOCAB = {"milk", "soy", "peanut", "tree_nuts", "wheat", "gluten", "egg", "fish", "shellfish", "sesame", "mustard", "sulphites"}


def extract_normalized_ingredient_tokens(parsed: ParsedResult) -> list[str]:
    """Extract all normalized ingredient and sub-ingredient names."""
    tokens: list[str] = []
    for ing in parsed.ingredients:
        name_clean = ing.name.strip().lower()
        if name_clean:
            tokens.append(name_clean)
        for child in ing.children:
            child_clean = child.name.strip().lower()
            if child_clean and child_clean != name_clean:
                tokens.append(child_clean)
    return tokens


def _build_full_text(
    ingredients_text: str,
    contains_text: Optional[str] = None,
    may_contain_text: Optional[str] = None,
) -> str:
    """Combine ingredients, contains, and may contain statements into single parseable text."""
    full = (ingredients_text or "").strip()
    if contains_text and "contains:" not in full.lower():
        full = f"{full.rstrip('.')}. Contains: {contains_text.strip()}."
    if may_contain_text and "may contain" not in full.lower():
        full = f"{full.rstrip('.')}. May contain: {may_contain_text.strip()}."
    return full


def evaluate_ingredient_extraction(
    reference_text: str,
    hypothesis_text: str,
    reference_contains: Optional[str] = None,
    hypothesis_contains: Optional[str] = None,
    reference_may_contain: Optional[str] = None,
    hypothesis_may_contain: Optional[str] = None,
    fuzzy_match_threshold: float = 90.0,
) -> dict[str, Any]:
    """Compute ingredient-level precision, recall, F1, exact match, and hallucinations."""
    full_ref = _build_full_text(reference_text, reference_contains, reference_may_contain)
    full_hyp = _build_full_text(hypothesis_text, hypothesis_contains, hypothesis_may_contain)

    parsed_ref = parse_ingredients(full_ref)
    parsed_hyp = parse_ingredients(full_hyp)

    ref_tokens = extract_normalized_ingredient_tokens(parsed_ref)
    hyp_tokens = extract_normalized_ingredient_tokens(parsed_hyp)

    if not ref_tokens and not hyp_tokens:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_list_match": True,
            "hallucinated_count": 0,
            "allergen_term_recall": 1.0,
            "parsed_ref_count": 0,
            "parsed_hyp_count": 0,
        }

    # Match tokens greedily with RapidFuzz
    matched_ref_indices: set[int] = set()
    matched_hyp_indices: set[int] = set()

    # Exact pass first
    for h_idx, h_tok in enumerate(hyp_tokens):
        for r_idx, r_tok in enumerate(ref_tokens):
            if r_idx not in matched_ref_indices and h_tok == r_tok:
                matched_ref_indices.add(r_idx)
                matched_hyp_indices.add(h_idx)
                break

    # Fuzzy pass for remaining
    for h_idx, h_tok in enumerate(hyp_tokens):
        if h_idx in matched_hyp_indices:
            continue
        best_r_idx = None
        best_score = 0.0
        for r_idx, r_tok in enumerate(ref_tokens):
            if r_idx in matched_ref_indices:
                continue
            score = fuzz.ratio(h_tok, r_tok)
            if score >= fuzzy_match_threshold and score > best_score:
                best_score = score
                best_r_idx = r_idx

        if best_r_idx is not None:
            matched_ref_indices.add(best_r_idx)
            matched_hyp_indices.add(h_idx)

    tp = len(matched_hyp_indices)
    fp = len(hyp_tokens) - tp
    fn = len(ref_tokens) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    exact_list_match = sorted(ref_tokens) == sorted(hyp_tokens)

    # Allergen-term recall
    # Find all allergen terms present anywhere in the ground truth text
    full_ref_text = f"{reference_text or ''} {reference_contains or ''} {reference_may_contain or ''}".lower()
    full_hyp_text = f"{hypothesis_text or ''} {hypothesis_contains or ''} {hypothesis_may_contain or ''}".lower()

    gt_allergens_present = [term for term in ALLERGEN_VOCAB if term in full_ref_text and len(term) >= 3]
    if gt_allergens_present:
        detected_allergens = [term for term in gt_allergens_present if term in full_hyp_text]
        allergen_recall = len(detected_allergens) / len(gt_allergens_present)
    else:
        allergen_recall = 1.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_list_match": exact_list_match,
        "hallucinated_count": fp,
        "allergen_term_recall": round(allergen_recall, 4),
        "parsed_ref_count": len(ref_tokens),
        "parsed_hyp_count": len(hyp_tokens),
    }
