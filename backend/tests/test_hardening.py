"""Hardening pass tests (Phase 1.5): independent raw-scan, fuzzy safety, regional terms, and traps."""
import pytest

from backend.allergen_engine import check_allergens
from backend.config import Settings
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, ParsedIngredient, UserProfile
from backend.parser import parse_ingredients


# ---------------------------------------------------------------------------
# Traps from prompt requirement 6
# ---------------------------------------------------------------------------


def test_trap_gluten_free_wheat_starch_wheat_allergy(kb: KnowledgeBase):
    """'gluten-free wheat starch' must trigger CONTAINS for wheat allergy."""
    label = "gluten-free wheat starch, Sugar, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "wheat" for a in res.alerts)


def test_trap_gluten_free_wheat_starch_gluten_allergy(kb: KnowledgeBase):
    """'gluten-free wheat starch' must trigger MIGHT_CONTAIN for gluten allergy (cannot downgrade to clear)."""
    label = "gluten-free wheat starch, Sugar, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["gluten"])
    res = check_allergens(matches, profile, parsed)

    # Must NOT be clear!
    assert res.overall_status in (AllergenStatus.MIGHT_CONTAIN, AllergenStatus.CONTAINS)
    assert res.overall_status != AllergenStatus.CLEAR


def test_trap_lactose_free_milk(kb: KnowledgeBase):
    """'lactose-free milk' must trigger CONTAINS/MIGHT_CONTAIN for milk allergy (milk protein remains)."""
    label = "lactose-free milk, Sugar, Cocoa"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)
    assert res.overall_status != AllergenStatus.CLEAR


def test_trap_non_dairy_creamer_sodium_caseinate(kb: KnowledgeBase):
    """'non-dairy creamer (sodium caseinate)' contains sodium caseinate (milk derivative)."""
    label = "non-dairy creamer (sodium caseinate), Sugar"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)
    assert any(a.allergen == "milk" for a in res.alerts)


def test_trap_groundnut_oil_peanut_allergy(kb: KnowledgeBase):
    """'groundnut oil' must trigger CONTAINS for peanut allergy."""
    label = "Groundnut Oil, Salt, Spices"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["peanut"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "peanut" for a in res.alerts)


def test_trap_maida_uppercase_wheat(kb: KnowledgeBase):
    """'MAIDA' must trigger CONTAINS for wheat allergy."""
    label = "MAIDA, Water, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "wheat" for a in res.alerts)


def test_trap_maida_uppercase_gluten(kb: KnowledgeBase):
    """'MAIDA' must trigger CONTAINS for gluten allergy."""
    label = "MAIDA, Water, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["gluten"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "gluten" for a in res.alerts)


def test_trap_fuzzy_typo_lands_on_allergen_entry(kb: KnowledgeBase):
    """A fuzzy typo resolving to an allergen entry must escalate to at least MIGHT_CONTAIN with warning reason."""
    # "Sodiun Caseinate" is >= 5 chars, typos 'n' for 'm'
    label = "Sodiun Caseinate, Sugar"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)
    # Reason must note approximate match
    assert any("matched approximately" in a.reason for a in res.alerts)


def test_trap_under_5_chars_almost_match_no_fuzzy(kb: KnowledgeBase):
    """Strings under 5 chars ('mils') must NOT fuzzy match 'milk'. Must become unknown and evaluate to CLEAR, not CONTAINS."""
    label = "mils, Sugar, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    # The ingredient 'mils' has 4 letters -> exact match required, no fuzzy match!
    assert matches[0].match_method == "unknown"

    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    # Under Phase 5 rule: missing KB entry alone does not cause UNCERTAIN.
    # Since 'mils' did not match milk, status is CLEAR
    assert res.overall_status == AllergenStatus.CLEAR


def test_trap_kb_clear_but_raw_scan_finds_allergen_not_clear(kb: KnowledgeBase):
    """When KB path fails to match (e.g. unlisted compound ingredient), raw scan catches allergen."""
    label = "Special Herbal Extract with Ghee, Sugar, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)

    assert res.overall_status != AllergenStatus.CLEAR
    assert res.overall_status == AllergenStatus.CONTAINS
    assert any(a.allergen == "milk" for a in res.alerts)
    assert any("Raw text safety scan" in a.reason for a in res.alerts)


# ---------------------------------------------------------------------------
# Regional & Alternate names tests
# ---------------------------------------------------------------------------


def test_regional_atta_wheat(kb: KnowledgeBase):
    label = "Atta, Water, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_regional_sooji_wheat(kb: KnowledgeBase):
    label = "Sooji, Salt, Ghee"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_regional_paneer_milk(kb: KnowledgeBase):
    label = "Paneer, Spinach, Spices"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_regional_dahi_curd_milk(kb: KnowledgeBase):
    label = "Dahi (Curd), Rice, Mustard Seeds"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_regional_til_sesame(kb: KnowledgeBase):
    label = "Til, Jaggery"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["sesame"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_regional_gingelly_sesame(kb: KnowledgeBase):
    label = "Gingelly Oil, Red Chillies"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["sesame"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_ambiguous_hvp_soy(kb: KnowledgeBase):
    label = "Hydrolysed Vegetable Protein, Salt, Yeast Extract"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["soy"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)


def test_ambiguous_hvp_wheat(kb: KnowledgeBase):
    label = "Hydrolyzed Vegetable Protein, Salt, Spices"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)


def test_sulphites_e220_to_e228(kb: KnowledgeBase):
    label = "Dried Apricots, Preservative (E220)"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["sulphites"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)


def test_sulphites_e224(kb: KnowledgeBase):
    label = "Wine Vinegar, Preservative (E224)"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["sulphites"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)


def test_tree_nuts_pistachio(kb: KnowledgeBase):
    label = "Pistachio Kernels, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["tree_nuts"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


def test_tree_nuts_macadamia(kb: KnowledgeBase):
    label = "Roasted Macadamia Nuts, Sea Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["tree_nuts"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CONTAINS


# ---------------------------------------------------------------------------
# Cross-contact phrasing tests
# ---------------------------------------------------------------------------


def test_cross_contact_facility_handles_peanuts(kb: KnowledgeBase):
    label = "Corn Chips, Sunflower Oil, Salt. Manufactured in a facility that also handles peanuts."
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["peanut"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.MIGHT_CONTAIN


def test_cross_contact_shared_equipment_tree_nuts(kb: KnowledgeBase):
    label = "Granola (Oats, Honey). Processed on shared equipment with Tree Nuts."
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["tree_nuts"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.MIGHT_CONTAIN


def test_cross_contact_same_line_milk(kb: KnowledgeBase):
    label = "Dark Chocolate (Cocoa Mass, Sugar). Made on the same line as milk products."
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.MIGHT_CONTAIN


# ---------------------------------------------------------------------------
# False-positive protection tests
# ---------------------------------------------------------------------------


def test_false_positive_cocoa_butter_not_milk(kb: KnowledgeBase):
    """'Cocoa butter' is pure plant fat from cocoa beans, MUST NOT trigger milk allergy."""
    label = "Sugar, Cocoa Butter, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.CLEAR
    assert len(res.alerts) == 0


def test_false_positive_buckwheat_not_wheat(kb: KnowledgeBase):
    """'Buckwheat flour' is not wheat, MUST NOT trigger wheat allergy."""
    label = "Buckwheat Flour, Water, Sea Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["wheat"])
    res = check_allergens(matches, profile, parsed)
    # Buckwheat flour is not wheat
    assert not any(a.allergen == "wheat" and a.status == AllergenStatus.CONTAINS for a in res.alerts)


# ---------------------------------------------------------------------------
# Phase 1 Additions (Part A & Part B)
# ---------------------------------------------------------------------------


def test_empty_and_short_ingredients_text_returns_uncertain(kb: KnowledgeBase):
    """Empty string, whitespace-only, and short (<10 chars) must return UNCERTAIN with specific reason."""
    profile = UserProfile(allergens=["milk", "peanut"])
    for text in ("", "   ", "Salt", "Sugar"):
        parsed = parse_ingredients(text)
        matches = match_ingredients(parsed.ingredients, kb)
        res = check_allergens(matches, profile, parsed)
        assert res.overall_status == AllergenStatus.UNCERTAIN
        assert any("No ingredients could be read" in a.reason for a in res.alerts)


def test_unreadable_parts_never_returns_clear(kb: KnowledgeBase):
    """Transcription with unreadable_parts must return at least UNCERTAIN."""
    from backend.models import TranscriptionResult
    parsed = parse_ingredients("Sugar, Salt, Water")
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    tx = TranscriptionResult(ingredients_text="Sugar, Salt, Water", unreadable_parts=["torn label corner"])
    res = check_allergens(matches, profile, parsed, transcription=tx)
    assert res.overall_status == AllergenStatus.UNCERTAIN
    assert res.overall_status != AllergenStatus.CLEAR


def test_cream_of_tartar_not_milk(kb: KnowledgeBase):
    """'cream of tartar' must NOT trigger milk."""
    label = "Cream of Tartar, Sodium Bicarbonate, Corn Starch"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert not any(a.allergen == "milk" and a.status == AllergenStatus.CONTAINS for a in res.alerts)


def test_plant_milks_rules(kb: KnowledgeBase):
    """Plant milks must NOT trigger dairy CONTAINS, but trigger own allergens, and at most MIGHT_CONTAIN for milk."""
    # 1. Almond milk
    parsed1 = parse_ingredients("Organic Almond Milk, Cane Sugar, Sea Salt")
    matches1 = match_ingredients(parsed1.ingredients, kb)
    res_milk1 = check_allergens(matches1, UserProfile(allergens=["milk"]), parsed1)
    # Milk profile -> at most MIGHT_CONTAIN with plant-based note
    assert res_milk1.overall_status in (AllergenStatus.MIGHT_CONTAIN, AllergenStatus.CLEAR)
    assert not any(a.allergen == "milk" and a.status == AllergenStatus.CONTAINS for a in res_milk1.alerts)
    # Tree nut profile -> CONTAINS
    res_nut1 = check_allergens(matches1, UserProfile(allergens=["tree_nuts"]), parsed1)
    assert res_nut1.overall_status == AllergenStatus.CONTAINS

    # 2. Soy milk
    parsed2 = parse_ingredients("Soy Milk, Water, Salt")
    matches2 = match_ingredients(parsed2.ingredients, kb)
    res_milk2 = check_allergens(matches2, UserProfile(allergens=["milk"]), parsed2)
    assert res_milk2.overall_status in (AllergenStatus.MIGHT_CONTAIN, AllergenStatus.CLEAR)
    assert not any(a.allergen == "milk" and a.status == AllergenStatus.CONTAINS for a in res_milk2.alerts)
    # Soy profile -> CONTAINS
    res_soy2 = check_allergens(matches2, UserProfile(allergens=["soy"]), parsed2)
    assert res_soy2.overall_status == AllergenStatus.CONTAINS


def test_lactic_acid_and_lactate_not_milk_contains(kb: KnowledgeBase):
    """Lactic acid and sodium lactate must NOT trigger milk CONTAINS."""
    label = "Lactic Acid, Sodium Lactate, Water, Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert not any(a.allergen == "milk" and a.status == AllergenStatus.CONTAINS for a in res.alerts)


def test_peanut_butter_token(kb: KnowledgeBase):
    """Peanut butter with peanut profile returns CONTAINS; with milk profile returns CLEAR for that token."""
    label = "Peanut Butter, Sea Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)

    res_peanut = check_allergens(matches, UserProfile(allergens=["peanut"]), parsed)
    assert res_peanut.overall_status == AllergenStatus.CONTAINS

    res_milk = check_allergens(matches, UserProfile(allergens=["milk"]), parsed)
    assert not any(a.allergen == "milk" for a in res_milk.alerts)


def test_false_nuts_not_tree_nuts(kb: KnowledgeBase):
    """'butternut squash', 'nutmeg', 'water chestnut', 'coconut' must NOT trigger tree nut CONTAINS."""
    label = "Butternut Squash, Nutmeg, Water Chestnut, Sea Salt"
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["tree_nuts"])
    res = check_allergens(matches, profile, parsed)
    assert not any(a.allergen == "tree_nuts" and a.status == AllergenStatus.CONTAINS for a in res.alerts)

    # Coconut produces at most MIGHT_CONTAIN with note
    label_coconut = "Coconut, Sugar, Salt"
    parsed_coco = parse_ingredients(label_coconut)
    matches_coco = match_ingredients(parsed_coco.ingredients, kb)
    res_coco = check_allergens(matches_coco, profile, parsed_coco)
    assert res_coco.overall_status in (AllergenStatus.MIGHT_CONTAIN, AllergenStatus.UNCERTAIN)
    assert not any(a.allergen == "tree_nuts" and a.status == AllergenStatus.CONTAINS for a in res_coco.alerts)


def test_oats_gluten_profile_might_contain(kb: KnowledgeBase):
    """'oats', 'oat flour' with a gluten profile must return MIGHT_CONTAIN (cross-contact risk)."""
    for text in ("Oats, Honey, Salt", "Oat Flour, Water, Yeast"):
        parsed = parse_ingredients(text)
        matches = match_ingredients(parsed.ingredients, kb)
        profile = UserProfile(allergens=["gluten"])
        res = check_allergens(matches, profile, parsed)
        assert res.overall_status == AllergenStatus.MIGHT_CONTAIN
        assert any("cross-contact risk" in a.reason for a in res.alerts)


def test_allergens_json_has_mandatory_classes():
    """Confirm lupin, mustard, celery, molluscs, crustaceans, fish, sesame exist with derivatives."""
    from backend.allergen_engine import _load_allergen_hierarchy
    hierarchy = _load_allergen_hierarchy()
    mandatory = ["lupin", "mustard", "celery", "molluscs", "crustaceans", "fish", "sesame"]
    for m in mandatory:
        assert m in hierarchy, f"Missing allergen class '{m}' in allergens.json hierarchy"
        assert len(hierarchy[m]) >= 3, f"Class '{m}' should have at least 3 derivatives"


def test_may_contain_milk_never_contains(kb: KnowledgeBase):
    """'May contain traces of milk' must give MIGHT_CONTAIN for milk, never CONTAINS."""
    label = "Sugar, Cocoa Mass, Cocoa Butter. May contain traces of milk."
    parsed = parse_ingredients(label)
    matches = match_ingredients(parsed.ingredients, kb)
    profile = UserProfile(allergens=["milk"])
    res = check_allergens(matches, profile, parsed)
    assert res.overall_status == AllergenStatus.MIGHT_CONTAIN
    assert not any(a.status == AllergenStatus.CONTAINS for a in res.alerts)


def test_allergen_result_confidence(kb: KnowledgeBase):
    """Test confidence scoring: high for clean match, low for unreadable, medium for fuzzy."""
    from backend.allergen_engine import allergen_result_confidence
    from backend.models import TranscriptionResult

    # Clean exact match -> high
    parsed = parse_ingredients("Sugar, Salt, Water")
    matches = match_ingredients(parsed.ingredients, kb)
    res = check_allergens(matches, UserProfile(allergens=["milk"]), parsed)
    assert res.confidence == "high"

    # Unreadable parts -> low
    tx = TranscriptionResult(ingredients_text="Sugar, Salt", unreadable_parts=["tear on label"])
    res_low = check_allergens(matches, UserProfile(allergens=["milk"]), parsed, transcription=tx)
    assert res_low.confidence == "low"

