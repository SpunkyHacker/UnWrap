"""Unit tests for additive-code parsing, normalization, OCR repair, and matching."""
import pytest

from backend.config import Settings
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, UserProfile
from backend.parser import normalize_additive_code, parse_ingredients, repair_additive_ocr
from backend.allergen_engine import check_allergens


def test_additive_stabilizer_expansion(kb: KnowledgeBase, default_config: Settings):
    """Input 'Stabilizer (INS 412, 415)' expands into two top-level items keeping class name."""
    text = "Stabilizer (INS 412, 415)"
    parsed = parse_ingredients(text)

    # 1. Parser verification: exactly two top-level items
    assert len(parsed.ingredients) == 2
    assert parsed.ingredients[0].name == "Stabilizer (INS 412)"
    assert parsed.ingredients[0].e_number == "INS 412"
    assert parsed.ingredients[1].name == "Stabilizer (INS 415)"
    assert parsed.ingredients[1].e_number == "INS 415"

    # 2. Matcher verification: matched by e_number without duplicate child matches
    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 2
    assert matches[0].parsed_name == "Stabilizer (INS 412)"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is not None
    assert matches[0].kb_entry.id == "guar_gum"

    assert matches[1].parsed_name == "Stabilizer (INS 415)"
    assert matches[1].match_method == "e_number"
    assert matches[1].kb_entry is not None
    assert matches[1].kb_entry.id == "xanthan_gum"


def test_additive_colour_ins_150d(kb: KnowledgeBase, default_config: Settings):
    """Input 'Colour (INS 150d)' maps to recognized additive code for AI card path."""
    text = "Colour (INS 150d)"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "Colour (INS 150d)"
    assert parsed.ingredients[0].e_number == "INS 150d"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].parsed_name == "Colour (INS 150d)"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is None  # Not in seed KB -> handled by AI card path
    assert matches[0].confidence == "medium"


def test_additive_acidity_regulator_330(kb: KnowledgeBase, default_config: Settings):
    """Input 'Acidity regulator (330)' maps bare 330 inside class bracket to citric acid."""
    text = "Acidity regulator (330)"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "Acidity regulator (INS 330)"
    assert parsed.ingredients[0].e_number == "INS 330"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].parsed_name == "Acidity regulator (INS 330)"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is not None
    assert matches[0].kb_entry.id == "citric_acid"


def test_additive_emulsifier_e_322(kb: KnowledgeBase, default_config: Settings):
    """Input 'Emulsifier (E-322)' normalizes hyphenated code to INS 322 and matches lecithin."""
    text = "Emulsifier (E-322)"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "Emulsifier (INS 322)"
    assert parsed.ingredients[0].e_number == "INS 322"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].parsed_name == "Emulsifier (INS 322)"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is not None
    assert matches[0].kb_entry.id == "lecithin"

    # Allergen test: soy allergy -> MIGHT_CONTAIN (lecithin source ambiguous)
    res = check_allergens(matches, UserProfile(allergens=["soy"]), parsed)
    assert res.overall_status == AllergenStatus.MIGHT_CONTAIN


def test_additive_anticaking_ins551_unknown_code(kb: KnowledgeBase, default_config: Settings):
    """Input 'anti-caking agent(INS551)' recognized as valid additive code, not UNCERTAIN."""
    text = "anti-caking agent(INS551)"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "anti-caking agent (INS 551)"
    assert parsed.ingredients[0].e_number == "INS 551"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].parsed_name == "anti-caking agent (INS 551)"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is None
    assert matches[0].confidence == "medium"

    # Must NOT cause UNCERTAIN by itself
    res = check_allergens(matches, UserProfile(allergens=["milk", "soy", "gluten"]), parsed)
    assert res.overall_status == AllergenStatus.CLEAR
    assert len(res.alerts) == 0


def test_additive_colour_nested_bracket_e160a_i(kb: KnowledgeBase, default_config: Settings):
    """Input 'Colour (E160a(i))' preserves nested brackets in suffix and routes unseeded code to AI card path."""
    text = "Colour (E160a(i))"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "Colour (INS 160a(i))"
    assert parsed.ingredients[0].e_number == "INS 160a(i)"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].parsed_name == "Colour (INS 160a(i))"
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is None  # Unseeded subvariant -> AI card path
    assert matches[0].confidence == "medium"


def test_additive_nested_bracket_150_iv(kb: KnowledgeBase, default_config: Settings):
    """Input 'Colour (150(iv))' handles roman numeral in nested brackets."""
    text = "Colour (150(iv))"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 1
    assert parsed.ingredients[0].name == "Colour (INS 150(iv))"
    assert parsed.ingredients[0].e_number == "INS 150(iv)"

    matches = match_ingredients(parsed.ingredients, kb, default_config)
    assert len(matches) == 1
    assert matches[0].match_method == "e_number"
    assert matches[0].kb_entry is None  # Unseeded variant -> AI card path
    assert matches[0].confidence == "medium"


def test_additive_preservative_sulphites_ins223_and_e220(kb: KnowledgeBase, default_config: Settings):
    """Preservative (INS 223) and Preservative (E220) with sulphite profile -> CONTAINS or MIGHT_CONTAIN, even though not in KB."""
    # Test INS 223
    p1 = parse_ingredients("Preservative (INS 223)")
    m1 = match_ingredients(p1.ingredients, kb, default_config)
    assert m1[0].kb_entry is None  # Not in seed KB
    res1 = check_allergens(m1, UserProfile(allergens=["sulphites"]), p1)
    assert res1.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)

    # Test E220
    p2 = parse_ingredients("Preservative (E220)")
    m2 = match_ingredients(p2.ingredients, kb, default_config)
    assert m2[0].kb_entry is None  # Not in seed KB
    res2 = check_allergens(m2, UserProfile(allergens=["sulphites"]), p2)
    assert res2.overall_status in (AllergenStatus.CONTAINS, AllergenStatus.MIGHT_CONTAIN)


def test_additive_emulsifier_ins471_soy_lecithin_not_clear(kb: KnowledgeBase, default_config: Settings):
    """Emulsifier (INS 471, Soy Lecithin) with soy profile -> not CLEAR."""
    parsed = parse_ingredients("Emulsifier (INS 471, Soy Lecithin)")
    matches = match_ingredients(parsed.ingredients, kb, default_config)
    res = check_allergens(matches, UserProfile(allergens=["soy"]), parsed)
    assert res.overall_status != AllergenStatus.CLEAR
    assert res.overall_status == AllergenStatus.CONTAINS


def test_additive_decimal_protection():
    """Decimal points like 62.5% must not be treated as code or split point."""
    text = "Wheat Flour (62.5%), Sugar"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 2
    assert parsed.ingredients[0].name == "Wheat Flour"
    assert parsed.ingredients[0].percentage == 62.5
    assert parsed.ingredients[0].children == []
    assert parsed.ingredients[1].name == "Sugar"


def test_additive_ocr_repair():
    """OCR misreads (O/0, l/1, S/5, Z/2) in code tokens are repaired and logged in corrections."""
    text = "Stabilizer (INS 4l2), Preservative (E-32Z), Acidity regulator (INS 33O), anti-caking agent(INS 5S1)"
    parsed = parse_ingredients(text)

    # 4 corrections recorded
    assert len(parsed.corrections) == 4
    corrs_map = {c["original"]: c["corrected"] for c in parsed.corrections}
    assert corrs_map["INS 4l2"] == "INS 412"
    assert corrs_map["E-32Z"] == "E-322"
    assert corrs_map["INS 33O"] == "INS 330"
    assert corrs_map["INS 5S1"] == "INS 551"

    # Parsed ingredients have normalized repaired codes
    e_numbers = [ing.e_number for ing in parsed.ingredients]
    assert e_numbers == ["INS 412", "INS 322", "INS 330", "INS 551"]


def test_additive_compound_subingredients_not_expanded():
    """Compound groups with non-code names must NOT be expanded into top-level items."""
    text = "Emulsifier (Soy Lecithin, E322), Salt"
    parsed = parse_ingredients(text)
    assert len(parsed.ingredients) == 2
    emuls = parsed.ingredients[0]
    assert emuls.name == "Emulsifier"
    assert len(emuls.children) == 2
    assert emuls.children[0].name == "Soy Lecithin"
    assert emuls.children[1].name == "E322"


def test_additive_normalizer_canonical_format():
    """Normalizer maps all code variations to canonical 'INS <digits><suffix>'."""
    assert normalize_additive_code("E322") == "INS 322"
    assert normalize_additive_code("E-322") == "INS 322"
    assert normalize_additive_code("E 322") == "INS 322"
    assert normalize_additive_code("INS 322") == "INS 322"
    assert normalize_additive_code("INS322") == "INS 322"
    assert normalize_additive_code("322") == "INS 322"
    assert normalize_additive_code("INS 150d") == "INS 150d"
    assert normalize_additive_code("150d") == "INS 150d"
    assert normalize_additive_code("E160a(i)") == "INS 160a(i)"
    assert normalize_additive_code("150(iv)") == "INS 150(iv)"
    assert normalize_additive_code("INS551") == "INS 551"
    assert normalize_additive_code("62.5%") is None
