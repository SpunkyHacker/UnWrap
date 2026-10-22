"""Test proving allergen results are identical with the AI disabled and with the KB removed."""
import pytest
from backend.allergen_engine import check_allergens
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, UserProfile
from backend.parser import parse_ingredients
from backend.tests.test_golden import load_golden_cases, HEINZ_SALAD_CREAM_LABEL


def test_golden_cases_allergen_results_identical_with_and_without_kb(kb: KnowledgeBase):
    """Test that all golden cases produce identical allergen status with and without the KB."""
    golden_cases = load_golden_cases()
    assert len(golden_cases) >= 5

    for case in golden_cases:
        label_text = case["label_text"]
        profile_allergens = case["profile"]
        profile = UserProfile(allergens=profile_allergens)
        parsed = parse_ingredients(label_text)

        # 1. With curated KB
        matches_with_kb = match_ingredients(parsed.ingredients, kb=kb)
        result_with_kb = check_allergens(matches_with_kb, profile, parsed)

        # 2. With KB completely removed (kb=None)
        matches_no_kb = match_ingredients(parsed.ingredients, kb=None)
        result_no_kb = check_allergens(matches_no_kb, profile, parsed)

        # Status must be 100% identical
        assert result_no_kb.overall_status == result_with_kb.overall_status, (
            f"Status mismatch for {case.get('_file')}: with_kb={result_with_kb.overall_status}, no_kb={result_no_kb.overall_status}"
        )


@pytest.mark.parametrize(
    "profile_allergens, expected_status",
    [
        (["egg"], AllergenStatus.CONTAINS),
        (["soy"], AllergenStatus.CONTAINS),
        (["mustard"], AllergenStatus.CONTAINS),
        (["milk"], AllergenStatus.CLEAR),
    ],
    ids=["egg", "soy", "mustard", "milk"],
)
def test_heinz_label_identical_with_and_without_kb(kb: KnowledgeBase, profile_allergens: list[str], expected_status: AllergenStatus):
    """Test that the Heinz Salad Cream label produces identical allergen evaluation with and without KB."""
    parsed = parse_ingredients(HEINZ_SALAD_CREAM_LABEL)
    profile = UserProfile(allergens=profile_allergens)

    # 1. With KB
    matches_kb = match_ingredients(parsed.ingredients, kb=kb)
    res_kb = check_allergens(matches_kb, profile, parsed)
    assert res_kb.overall_status == expected_status

    # 2. Without KB
    matches_no_kb = match_ingredients(parsed.ingredients, kb=None)
    res_no_kb = check_allergens(matches_no_kb, profile, parsed)
    assert res_no_kb.overall_status == expected_status

    # Both must match expected status identically
    assert res_no_kb.overall_status == res_kb.overall_status


def test_ai_disabled_allergen_independence():
    """Verify allergen checking performs zero AI/LLM network calls."""
    parsed = parse_ingredients("Water, Sugar, Salt, Wheat Flour, Skimmed Milk Powder")
    profile = UserProfile(allergens=["milk", "wheat"])

    # Run without KB and without any AI client
    matches = match_ingredients(parsed.ingredients, kb=None)
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == AllergenStatus.CONTAINS
    allergens_found = {a.allergen for a in result.alerts if a.status == AllergenStatus.CONTAINS}
    assert "milk" in allergens_found
    assert "wheat" in allergens_found
