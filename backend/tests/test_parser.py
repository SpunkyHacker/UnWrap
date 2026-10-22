"""Unit tests for the deterministic ingredient text parser."""
import pytest
from backend.parser import parse_ingredients


def test_simple_comma_separated():
    text = "Sugar, Salt, Water"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    assert [i.name for i in result.ingredients] == ["Sugar", "Salt", "Water"]
    assert result.ingredients[0].position == 0
    assert result.ingredients[1].position == 1
    assert result.ingredients[2].position == 2


def test_list_with_brackets_groups():
    text = "Emulsifier (Soy Lecithin, E322), Salt"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 2
    emulsifier = result.ingredients[0]
    assert emulsifier.name == "Emulsifier"
    assert len(emulsifier.children) == 2
    assert emulsifier.children[0].name == "Soy Lecithin"
    assert emulsifier.children[1].name == "E322"
    assert emulsifier.children[1].e_number == "E322"
    assert result.ingredients[1].name == "Salt"


def test_nested_brackets():
    text = "Chocolate (Cocoa Mass, Sugar, Cocoa Butter, Emulsifier (Soy Lecithin)), Vanilla Extract"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 2
    choc = result.ingredients[0]
    assert choc.name == "Chocolate"
    assert len(choc.children) == 4
    assert choc.children[0].name == "Cocoa Mass"
    assert choc.children[1].name == "Sugar"
    assert choc.children[2].name == "Cocoa Butter"
    emuls = choc.children[3]
    assert emuls.name == "Emulsifier"
    assert len(emuls.children) == 1
    assert emuls.children[0].name == "Soy Lecithin"


def test_e_number_extraction():
    text = "Citric Acid (E330), Preservative (INS 211), Colour (150a)"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    # Citric acid group with E330 child
    assert result.ingredients[0].children[0].e_number == "E330"
    # Preservative with INS 211 child
    assert result.ingredients[1].children[0].e_number == "E211"


def test_percentage_extraction():
    text = "Wheat Flour (60%), Sugar (25.5%), Palm Oil"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    assert result.ingredients[0].name == "Wheat Flour"
    assert result.ingredients[0].percentage == 60.0
    assert result.ingredients[0].children == []  # Pure percentage is not parsed as children
    assert result.ingredients[1].name == "Sugar"
    assert result.ingredients[1].percentage == 25.5
    assert result.ingredients[2].name == "Palm Oil"
    assert result.ingredients[2].percentage is None


def test_contains_statement_parsing():
    text = "Sugar, Wheat Flour, Milk. Contains: Wheat, Milk and Soy."
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    assert set(result.contains_allergens) == {"wheat", "milk", "soy"}
    assert result.may_contain_allergens == []


def test_may_contain_statement_parsing():
    text = "Sugar, Cocoa Butter. May contain traces of Peanuts and Tree Nuts."
    result = parse_ingredients(text)
    assert len(result.ingredients) == 2
    assert "peanuts" in result.may_contain_allergens
    assert "tree nuts" in result.may_contain_allergens


def test_both_contains_and_may_contain():
    text = "Wheat Flour, Sugar, Butter. Contains: Wheat, Milk. May contain: Eggs, Peanuts."
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    assert set(result.contains_allergens) == {"wheat", "milk"}
    assert set(result.may_contain_allergens) == {"eggs", "peanuts"}


def test_leading_ingredients_stripping():
    text1 = "Ingredients: Sugar, Salt"
    text2 = "INGREDIENTS: Sugar, Salt"
    res1 = parse_ingredients(text1)
    res2 = parse_ingredients(text2)
    assert [i.name for i in res1.ingredients] == ["Sugar", "Salt"]
    assert [i.name for i in res2.ingredients] == ["Sugar", "Salt"]


def test_empty_input():
    assert parse_ingredients("").ingredients == []
    assert parse_ingredients("   ").ingredients == []
    assert parse_ingredients(None).ingredients == []


def test_complex_real_world():
    text = (
        "INGREDIENTS: Wheat Flour (62%), Sugar, Palm Oil, Emulsifier (Soy Lecithin, E322), "
        "Raising Agents (Sodium Bicarbonate, E500, Ammonium Bicarbonate), Salt, Natural Vanilla Flavour. "
        "Contains: Wheat, Soy. May contain traces of Milk, Sesame."
    )
    result = parse_ingredients(text)
    assert len(result.ingredients) == 7
    assert result.ingredients[0].name == "Wheat Flour"
    assert result.ingredients[0].percentage == 62.0
    assert result.ingredients[3].name == "Emulsifier"
    assert len(result.ingredients[3].children) == 2
    assert result.ingredients[4].name == "Raising Agents"
    assert len(result.ingredients[4].children) == 3
    assert set(result.contains_allergens) == {"wheat", "soy"}
    assert set(result.may_contain_allergens) == {"milk", "sesame"}


def test_semicolon_handling():
    text = "Enriched Flour (Wheat Flour; Niacin; Reduced Iron); Water; Sugar"
    result = parse_ingredients(text)
    assert len(result.ingredients) == 3
    assert result.ingredients[0].name == "Enriched Flour"
    assert len(result.ingredients[0].children) == 3
