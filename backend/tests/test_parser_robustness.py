"""Tests for Phase 2.5: Parser robustness and OCR spell-correction integration."""
import pytest
from backend.allergen_engine import check_allergens
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, TranscriptionResult, UserProfile
from backend.ocr import spell_correct_tokens
from backend.parser import parse_ingredients


def test_discard_everything_before_ingredients_marker():
    """Verify everything before the Ingredients marker is discarded, and has_ingredients_marker=True."""
    text1 = "Delicious Crunchy Biscuit Net Wt 150g. INGREDIENTS: Wheat Flour, Sugar, Salt."
    res1 = parse_ingredients(text1)
    assert res1.has_ingredients_marker is True
    assert [i.name for i in res1.ingredients] == ["Wheat Flour", "Sugar", "Salt"]

    # With period: "Ingredients."
    text2 = "Brand Name. Best Quality. Ingredients. Palm Oil, Sugar, Salt."
    res2 = parse_ingredients(text2)
    assert res2.has_ingredients_marker is True
    assert [i.name for i in res2.ingredients] == ["Palm Oil", "Sugar", "Salt"]

    # Without colon or period: "Ingredients Sugar, Salt"
    text3 = "Organic Bakery Ingredients Sugar, Salt"
    res3 = parse_ingredients(text3)
    assert res3.has_ingredients_marker is True
    assert [i.name for i in res3.ingredients] == ["Sugar", "Salt"]


def test_no_ingredients_marker_flags_false():
    """If no marker exists, parse everything but flag has_ingredients_marker=False."""
    text = "Wheat Flour, Sugar, Palm Oil, Salt"
    res = parse_ingredients(text)
    assert res.has_ingredients_marker is False
    assert len(res.ingredients) == 4
    assert [i.name for i in res.ingredients] == ["Wheat Flour", "Sugar", "Palm Oil", "Salt"]


def test_handle_newline_separated_text():
    """Handle text separated by newlines instead of commas."""
    text = "Ingredients:\nWheat Flour\nSugar\nPalm Oil\nSalt"
    res = parse_ingredients(text)
    assert len(res.ingredients) == 4
    assert [i.name for i in res.ingredients] == ["Wheat Flour", "Sugar", "Palm Oil", "Salt"]


def test_decimal_points_not_treated_as_separators():
    """Do not treat decimal points in percentages (e.g. 62.5%) as separators."""
    text = "Ingredients: Wheat Flour (62.5%), Sugar (20.0%), Palm Oil (12.5%), Salt (0.5%)."
    res = parse_ingredients(text)
    assert len(res.ingredients) == 4
    assert res.ingredients[0].name == "Wheat Flour"
    assert res.ingredients[0].percentage == 62.5
    assert res.ingredients[1].percentage == 20.0
    assert res.ingredients[2].percentage == 12.5
    assert res.ingredients[3].percentage == 0.5


def test_possible_merged_ingredients_detected(kb: KnowledgeBase):
    """If an unmatched token contains two or more known aliases, mark possible_merged and include suggested split."""
    text = "Ingredients: Wheat Flour, Palm Oil Cocoa Powder, Salt Raising Agent"
    parsed = parse_ingredients(text)
    matches = match_ingredients(parsed.ingredients, kb)

    # 1. Wheat Flour -> normal match
    assert matches[0].match_method == "exact"
    assert parsed.ingredients[0].possible_merged is False

    # 2. "Palm Oil Cocoa Powder" -> unmatched composite containing "Palm Oil" and "Cocoa Powder"
    ing2 = parsed.ingredients[1]
    assert ing2.possible_merged is True
    assert "Palm Oil" in ing2.suggested_split
    assert "Cocoa Powder" in ing2.suggested_split

    # 3. "Salt Raising Agent" -> unmatched composite containing "Salt" and "Raising Agent" (Baking powder / agent)
    # Status stays UNCERTAIN until confirmed!
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.UNCERTAIN
    assert any("merged ingredients" in a.reason for a in res.alerts)


def test_ocr_spell_correction_returns_corrections_list(kb: KnowledgeBase):
    """spell_correct_tokens must return (corrected_text, corrections_list) with score."""
    raw = "Water, Sodiun Benzoate, Citric Acid"
    corrected_text, corrections = spell_correct_tokens(raw, kb, min_score=92.0)
    assert "Sodium Benzoate" in corrected_text
    assert len(corrections) == 1
    assert corrections[0]["original"] == "Sodiun Benzoate"
    assert corrections[0]["corrected"] == "Sodium Benzoate"
    assert corrections[0]["score"] >= 92.0


def test_allergen_engine_scans_both_original_and_corrected(kb: KnowledgeBase):
    """The allergen engine scans BOTH original and corrected tokens and uses the worse result."""
    # Suppose OCR mis-spelled an allergen as "Soy Lecitin"
    # Even if corrected text had something else, scanning both original and corrected catches the allergen!
    parsed = parse_ingredients("Sugar, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    tx = TranscriptionResult(
        ingredients_text="Sugar, Salt",
        corrections=[{"original": "Whey Powder", "corrected": "Sweetener", "score": 93.0}],
    )
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed, transcription=tx)

    # Original had "WheyPowder" -> milk CONTAINS
    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "milk" for a in res.alerts)
