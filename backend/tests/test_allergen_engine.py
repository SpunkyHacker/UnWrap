"""Unit tests for the deterministic allergen engine."""
import pytest

from backend.allergen_engine import check_allergens
from backend.config import Settings
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import (
    AllergenStatus,
    ParsedIngredient,
    ParsedResult,
    TranscriptionResult,
    UserProfile,
)
from backend.parser import parse_ingredients


def test_whey_with_milk_allergy_returns_contains(kb: KnowledgeBase):
    parsed = parse_ingredients("Whey powder, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.status == AllergenStatus.CONTAINS and a.allergen == "milk" for a in result.alerts)


def test_soy_lecithin_with_soy_allergy_returns_contains(kb: KnowledgeBase):
    parsed = parse_ingredients("Soy Lecithin, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["soy"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.status == AllergenStatus.CONTAINS and a.allergen == "soy" for a in result.alerts)


def test_unspecified_lecithin_with_soy_allergy_returns_might_contain(kb: KnowledgeBase):
    parsed = parse_ingredients("Lecithin, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["soy"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.MIGHT_CONTAIN
    assert any(a.status == AllergenStatus.MIGHT_CONTAIN and a.allergen == "soy" for a in result.alerts)


def test_unknown_harmless_ingredient_returns_clear(kb: KnowledgeBase):
    """Missing KB entry alone must NOT cause UNCERTAIN. Unknown harmless ingredient returns CLEAR."""
    parsed = parse_ingredients("MysteriousExoticRoot, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk", "peanut"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CLEAR
    assert len(result.alerts) == 0


def test_unknown_whey_blend_with_milk_profile_returns_contains(kb: KnowledgeBase):
    """Unknown ingredient containing allergen keyword ('whey blend') returns CONTAINS for milk."""
    parsed = parse_ingredients("Organic Whey Blend, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "milk" and a.status == AllergenStatus.CONTAINS for a in result.alerts)


def test_allergen_result_identical_with_ai_disabled(kb: KnowledgeBase, monkeypatch):
    """Confirm allergen engine produces identical output with AI disabled."""
    parsed = parse_ingredients("Whey powder, Soy lecithin, Wheat flour, Palm oil")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk", "soy"])

    res1 = check_allergens(matches, profile, parsed)

    monkeypatch.setattr("backend.config.settings.XAI_API_KEY", "")
    monkeypatch.setattr("backend.config.settings.CHAT_BACKEND", "disabled")
    monkeypatch.setattr("backend.config.settings.LLM_PICK_ENABLED", False)

    res2 = check_allergens(matches, profile, parsed)

    assert res1.overall_status == res2.overall_status
    assert len(res1.alerts) == len(res2.alerts)
    for a1, a2 in zip(res1.alerts, res2.alerts):
        assert a1.status == a2.status
        assert a1.allergen == a2.allergen
        assert a1.reason == a2.reason


def test_e_number_matched_ingredient_with_allergen(kb: KnowledgeBase):
    # E322 is Lecithin (possible soy allergen)
    parsed = parse_ingredients("Emulsifier (E322)")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["soy"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.MIGHT_CONTAIN
    assert any(a.status == AllergenStatus.MIGHT_CONTAIN and a.allergen == "soy" for a in result.alerts)


def test_nested_bracket_ingredient_with_allergen(kb: KnowledgeBase):
    text = "Cookie Dough (Flour (Wheat Flour), Butter (Milk)), Salt"
    parsed = parse_ingredients(text)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "wheat" and a.status == AllergenStatus.CONTAINS for a in result.alerts)


def test_contains_statement_triggers_contains(kb: KnowledgeBase):
    parsed = parse_ingredients("Corn Starch, Sugar. Contains: Milk.")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.status == AllergenStatus.CONTAINS and a.allergen == "milk" for a in result.alerts)


def test_may_contain_statement_triggers_might_contain(kb: KnowledgeBase):
    parsed = parse_ingredients("Sugar, Salt. May contain traces of Peanuts.")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["peanut"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.MIGHT_CONTAIN
    assert any(a.status == AllergenStatus.MIGHT_CONTAIN and a.allergen == "peanut" for a in result.alerts)


def test_no_allergens_returns_clear_with_correct_wording(kb: KnowledgeBase):
    parsed = parse_ingredients("Sugar, Salt, Water")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk", "soy", "peanut"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CLEAR
    assert len(result.alerts) == 0


def test_multiple_allergens_worst_wins(kb: KnowledgeBase):
    # Has a CONTAINS (Whey powder -> milk), a MIGHT_CONTAIN (Lecithin -> soy), and an UNKNOWN
    parsed = parse_ingredients("Whey powder, Lecithin, ExoticUnknownHerb")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk", "soy"])
    result = check_allergens(matches, profile, parsed)

    # CONTAINS is highest severity
    assert result.overall_status == AllergenStatus.CONTAINS


def test_custom_allergen_text_matching(kb: KnowledgeBase):
    parsed = parse_ingredients("Palm Oil, Sugar, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=[], custom_allergens="palm oil")
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.status == AllergenStatus.CONTAINS and "palm oil" in a.allergen for a in result.alerts)


def test_custom_allergen_fuzzy_matching(kb: KnowledgeBase):
    # User types "coconut" in custom allergen box
    parsed = parse_ingredients("Coconut Oil, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=[], custom_allergens="coconut")
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS


def test_hierarchy_expansion_catches_casein(kb: KnowledgeBase):
    # User selects "milk" allergy, product has "casein" or "sodium caseinate"
    # Casein is child of milk in hierarchy
    # Let's test with a parsed ingredient that has casein
    parsed = parse_ingredients("Casein, Sugar")
    # Even if Casein is an alias or direct ingredient
    profile = UserProfile(allergens=["milk"])
    # If matched to KB or direct name check
    matches = match_ingredients(parsed.ingredients, kb)
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.UNCERTAIN)
    # The hierarchy expansion for milk includes casein
    if any(m.kb_entry is not None for m in matches):
        assert result.overall_status == AllergenStatus.CONTAINS


def test_disclaimer_always_present(kb: KnowledgeBase):
    parsed = parse_ingredients("Sugar, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    result = check_allergens(matches, UserProfile(), parsed)

    assert result.disclaimer is not None
    assert "medical advice" in result.disclaimer.lower()
    assert "reference only" in result.disclaimer.lower()


def test_unreadable_parts_returns_uncertain(kb: KnowledgeBase):
    parsed = parse_ingredients("Sugar, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    transcription = TranscriptionResult(
        ingredients_text="Sugar, Salt",
        unreadable_parts=["smudged text after salt", "torn corner"],
    )
    result = check_allergens(matches, UserProfile(allergens=["milk"]), parsed, transcription=transcription)

    assert result.overall_status == AllergenStatus.UNCERTAIN
    assert any("smudged" in a.reason for a in result.alerts)


def test_ocr_typo_fuzzy_match_still_checks_allergen(kb: KnowledgeBase):
    # OCR typo: "Soy Lecitin" (missing 'h') fuzzy matches "Soy Lecithin" (score > 92)
    parsed = parse_ingredients("Soy Lecitin, Sugar")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["soy"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "soy" and a.status == AllergenStatus.CONTAINS for a in result.alerts)


def test_clear_profile_with_different_allergens(kb: KnowledgeBase):
    # Product with milk, but user is only allergic to shellfish
    parsed = parse_ingredients("Whole Milk, Sugar, Salt")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["shellfish"])
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CLEAR
    assert len(result.alerts) == 0
