"""Unit tests for the ingredient matching ladder."""
import json
from pathlib import Path

import pytest
from backend.config import Settings
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import ParsedIngredient


def test_exact_match_canonical_name(kb: KnowledgeBase, default_config: Settings):
    parsed = [ParsedIngredient(name="Sugar")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    m = matches[0]
    assert m.match_method == "exact"
    assert m.match_score == 100.0
    assert m.confidence == "high"
    assert m.kb_entry is not None
    assert m.kb_entry.canonical_name == "Sugar"


def test_exact_match_by_alias(kb: KnowledgeBase, default_config: Settings):
    parsed = [ParsedIngredient(name="cane sugar")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    m = matches[0]
    assert m.match_method == "exact"
    assert m.kb_entry is not None
    assert m.kb_entry.canonical_name == "Sugar"


def test_case_insensitive_matching(kb: KnowledgeBase, default_config: Settings):
    parsed = [ParsedIngredient(name="WHEAT FLOUR")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    assert matches[0].match_method == "exact"
    assert matches[0].kb_entry.canonical_name == "Wheat Flour"


def test_e_number_match(kb: KnowledgeBase, default_config: Settings):
    # E330 without explicit name match
    parsed = [ParsedIngredient(name="Acidity Regulator (330)", e_number="E330")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    m = matches[0]
    # Acidity Regulator (330) matches E330
    assert m.match_method == "e_number"
    assert m.kb_entry is not None
    assert m.kb_entry.id in ("citric_acid", "citric-acid")
    assert m.confidence == "high"


def test_fuzzy_match_with_typo(kb: KnowledgeBase, default_config: Settings):
    # Minor OCR-like typo: "Sodiun Benzoate" instead of "Sodium Benzoate"
    parsed = [ParsedIngredient(name="Sodiun Benzoate")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    m = matches[0]
    assert m.match_method == "fuzzy"
    assert m.match_score >= default_config.FUZZY_MATCH_THRESHOLD
    assert m.kb_entry is not None
    assert m.kb_entry.canonical_name == "Sodium Benzoate"
    assert m.confidence == "medium"


def test_no_match_returns_unknown(kb: KnowledgeBase, default_config: Settings):
    parsed = [ParsedIngredient(name="Kryptonite Extract 999")]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 1
    m = matches[0]
    assert m.match_method == "unknown"
    assert m.kb_entry is None
    assert m.confidence == "low"
    assert m.match_score == 0.0


def test_group_ingredient_matching_parent_and_children(kb: KnowledgeBase, default_config: Settings):
    parsed = [
        ParsedIngredient(
            name="Emulsifier",
            children=[
                ParsedIngredient(name="Soy Lecithin"),
                ParsedIngredient(name="E322", e_number="E322"),
            ],
        )
    ]
    matches = match_ingredients(parsed, kb, default_config)
    # 1 parent + 2 children = 3 matches
    assert len(matches) == 3
    # First is parent (Emulsifier is unknown or matched)
    names = [m.parsed_name for m in matches]
    assert names == ["Emulsifier", "Soy Lecithin", "E322"]
    # Child 1: Soy Lecithin
    assert matches[1].kb_entry is not None
    assert matches[1].kb_entry.canonical_name == "Soy Lecithin"
    # Child 2: E322 matched by e_number (Lecithin / Soy Lecithin)
    assert matches[2].kb_entry is not None
    assert "Lecithin" in matches[2].kb_entry.canonical_name


def test_multiple_ingredients_matching(kb: KnowledgeBase, default_config: Settings):
    parsed = [
        ParsedIngredient(name="Salt"),
        ParsedIngredient(name="Palm Oil"),
        ParsedIngredient(name="Citric Acid"),
    ]
    matches = match_ingredients(parsed, kb, default_config)
    assert len(matches) == 3
    assert all(m.kb_entry is not None for m in matches)
    assert [m.kb_entry.canonical_name for m in matches] == ["Salt", "Palm Oil", "Citric Acid"]


def test_unknown_logged_to_file(kb: KnowledgeBase, default_config: Settings):
    log_file = Path(__file__).resolve().parent.parent / "data" / "unknown_ingredients.jsonl"
    initial_count = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            initial_count = len(f.readlines())

    fake_ingredient = "SuperRareHerbExtractX123"
    parsed = [ParsedIngredient(name=fake_ingredient)]
    match_ingredients(parsed, kb, default_config)

    assert log_file.exists()
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) >= initial_count + 1
    last_record = json.loads(lines[-1])
    assert last_record["name"] == fake_ingredient
    assert "timestamp" in last_record


def test_match_method_and_confidence_recorded(kb: KnowledgeBase, default_config: Settings):
    parsed = [
        ParsedIngredient(name="Sugar"),             # exact
        ParsedIngredient(name="E330", e_number="E330"), # e_number
        ParsedIngredient(name="Sodiun Benzoate"),   # typo -> fuzzy
        ParsedIngredient(name="Unobtainium"),       # unknown
    ]
    matches = match_ingredients(parsed, kb, default_config)
    assert matches[0].match_method == "exact"
    assert matches[0].confidence == "high"

    assert matches[1].match_method in ("exact", "e_number")
    assert matches[1].confidence == "high"

    assert matches[2].match_method == "fuzzy"
    assert matches[2].confidence == "medium"

    assert matches[3].match_method == "unknown"
    assert matches[3].confidence == "low"
