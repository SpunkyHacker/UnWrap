"""Golden dataset runner to verify ZERO false CLEAR allergen evaluations."""
import json
from pathlib import Path
import pytest

from backend.allergen_engine import check_allergens
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, UserProfile
from backend.parser import parse_ingredients

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def load_golden_cases():
    cases = []
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file"] = p.name
            cases.append(data)
    return cases


@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda c: c.get("_file", c.get("name", "case")))
def test_golden_case_no_false_clears(kb: KnowledgeBase, case: dict):
    label_text = case["label_text"]
    profile_allergens = case["profile"]
    expected_status_str = case["expected_status"].lower()

    parsed = parse_ingredients(label_text)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=profile_allergens)
    result = check_allergens(matches, profile, parsed)

    actual_status = result.overall_status.value

    # ASSERT ZERO FALSE CLEARS:
    # If the expected status is "contains" or "might_contain", the engine
    # MUST NEVER return "clear".
    if expected_status_str in ("contains", "might_contain"):
        assert actual_status != AllergenStatus.CLEAR.value, (
            f"CRITICAL SAFETY VIOLATION in {case.get('_file')}: "
            f"Expected '{expected_status_str}' but engine returned CLEAR! "
            f"Label: '{label_text}' | Profile: {profile_allergens}"
        )

    # Furthermore, verify actual status meets or matches expected
    if expected_status_str == "contains":
        assert actual_status == AllergenStatus.CONTAINS.value, (
            f"Expected CONTAINS in {case.get('_file')}, got {actual_status}. "
            f"Alerts: {[a.reason for a in result.alerts]}"
        )
    elif expected_status_str == "might_contain":
        assert actual_status in (AllergenStatus.MIGHT_CONTAIN.value, AllergenStatus.CONTAINS.value), (
            f"Expected at least MIGHT_CONTAIN in {case.get('_file')}, got {actual_status}. "
            f"Alerts: {[a.reason for a in result.alerts]}"
        )


HEINZ_SALAD_CREAM_LABEL = (
    "White Vinegar, Water, Soya Bean Oil [Protected with Antioxidant (319)], "
    "Sugar, Egg Yolk, Salt, Mustard, Maize Thickener (1422), Food Acid (270), "
    "Stabiliser (412), Colours (171, 101), Herb Extract."
)


@pytest.mark.parametrize(
    "profile_allergens, expected_status",
    [
        (["egg"], AllergenStatus.CONTAINS),
        (["soy"], AllergenStatus.CONTAINS),
        (["mustard"], AllergenStatus.CONTAINS),
        (["milk"], AllergenStatus.CLEAR),
    ],
    ids=["egg_contains", "soy_contains", "mustard_contains", "milk_clear"],
)
def test_golden_heinz_salad_cream_label(kb: KnowledgeBase, profile_allergens: list[str], expected_status: AllergenStatus):
    """Golden test with the real Heinz Salad Cream label photo transcription.

    - egg -> CONTAINS ('Egg Yolk')
    - soy -> CONTAINS ('Soya Bean Oil')
    - mustard -> CONTAINS ('Mustard')
    - milk -> CLEAR (no dairy ingredients present)
    """
    parsed = parse_ingredients(HEINZ_SALAD_CREAM_LABEL)

    # Verify parser extracts the expected items without dropping codes or bracket content
    parsed_names = [ing.name for ing in parsed.ingredients]
    assert "Soya Bean Oil" in parsed_names
    assert "Antioxidant (INS 319)" in parsed_names
    assert "Maize Thickener (INS 1422)" in parsed_names
    assert "Food Acid (INS 270)" in parsed_names
    assert "Stabiliser (INS 412)" in parsed_names
    assert "Colours (INS 171)" in parsed_names
    assert "Colours (INS 101)" in parsed_names

    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=profile_allergens)
    result = check_allergens(matches, profile, parsed)

    assert result.overall_status == expected_status


def test_golden_confidence_scoring_and_reason(kb: KnowledgeBase):
    """Test confidence scoring:
    - KB-unknown ingredients do NOT lower confidence to 'low'.
    - Photo/transcription quality triggers 'low' with explicit confidence_reason.
    """
    from backend.models import TranscriptionResult

    parsed = parse_ingredients(HEINZ_SALAD_CREAM_LABEL)
    matches = match_ingredients(parsed.ingredients, kb)

    # Clean transcription with multiple unknown additives -> confidence is high, not low
    clean_tx = TranscriptionResult(
        ingredients_text=HEINZ_SALAD_CREAM_LABEL,
        engine_used="qwen2.5-vl-7b",
        fallback_occurred=False,
    )
    res_clean = check_allergens(matches, UserProfile(allergens=["milk"]), parsed, transcription=clean_tx)
    assert res_clean.confidence in ("high", "medium")
    assert res_clean.overall_status == AllergenStatus.CLEAR

    # Unreadable parts -> low confidence with reason
    unreadable_tx = TranscriptionResult(
        ingredients_text=HEINZ_SALAD_CREAM_LABEL,
        engine_used="qwen2.5-vl-7b",
        unreadable_parts=["crease over ingredient list"],
    )
    res_unreadable = check_allergens(matches, UserProfile(allergens=["milk"]), parsed, transcription=unreadable_tx)
    assert res_unreadable.confidence == "low"
    assert "crease over ingredient list" in res_unreadable.confidence_reason

    # Fallback engine -> low confidence with reason
    fallback_tx = TranscriptionResult(
        ingredients_text=HEINZ_SALAD_CREAM_LABEL,
        engine_used="apple_vision",
        fallback_occurred=True,
    )
    res_fallback = check_allergens(matches, UserProfile(allergens=["milk"]), parsed, transcription=fallback_tx)
    assert res_fallback.confidence == "low"
    assert "fallback" in res_fallback.confidence_reason.lower()

    # >= 3 corrections -> low confidence with reason
    parsed_many_corrs = parse_ingredients(HEINZ_SALAD_CREAM_LABEL)
    parsed_many_corrs.corrections = [
        {"original": "a", "corrected": "b"},
        {"original": "c", "corrected": "d"},
        {"original": "e", "corrected": "f"},
    ]
    res_corrs = check_allergens(matches, UserProfile(allergens=["milk"]), parsed_many_corrs, transcription=clean_tx)
    assert res_corrs.confidence == "low"
    assert "repaired" in res_corrs.confidence_reason.lower()
