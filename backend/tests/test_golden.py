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
