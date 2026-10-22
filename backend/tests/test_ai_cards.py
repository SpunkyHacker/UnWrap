"""Tests for AI-generated ingredient cards, rate limiting, xAI Grok fallback, and schema validation."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.ai_cards import (
    GrokRateLimiter,
    _extract_json_from_llm_response,
    get_cached_card,
    get_or_generate_ingredient_card,
    get_rate_limiter,
    make_unavailable_card,
    sanitize_text,
    validate_and_normalize_ai_card,
)
from backend.config import Settings
from backend.models import IngredientKBEntry, HowMadeStep, DietFlags


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Ensure clean rate limiter state before each test."""
    limiter = get_rate_limiter()
    limiter.reset()
    yield
    limiter.reset()


def test_rate_limiter_per_ip_limit():
    """Verify that requests exceeding GROK_RPM_LIMIT are blocked for that IP."""
    limiter = GrokRateLimiter()
    rpm = 3
    daily_cap = 100

    # 3 allowed
    for _ in range(3):
        allowed, _ = limiter.check_and_record("1.2.3.4", rpm_limit=rpm, daily_cap=daily_cap)
        assert allowed is True

    # 4th blocked
    allowed, reason = limiter.check_and_record("1.2.3.4", rpm_limit=rpm, daily_cap=daily_cap)
    assert allowed is False
    assert "rate limit exceeded" in reason.lower()

    # Different IP is still allowed
    allowed_other, _ = limiter.check_and_record("5.6.7.8", rpm_limit=rpm, daily_cap=daily_cap)
    assert allowed_other is True


def test_rate_limiter_daily_cap():
    """Verify that reaching GROK_DAILY_CAP blocks all further requests."""
    limiter = GrokRateLimiter()
    rpm = 50
    daily_cap = 5

    for _ in range(5):
        allowed, _ = limiter.check_and_record("1.2.3.4", rpm_limit=rpm, daily_cap=daily_cap)
        assert allowed is True

    # 6th request blocked due to daily cap
    allowed, reason = limiter.check_and_record("1.2.3.4", rpm_limit=rpm, daily_cap=daily_cap)
    assert allowed is False
    assert "daily grok call cap reached" in reason.lower()


def test_validate_and_normalize_card_valid():
    """Valid raw LLM JSON produces a complete, normalized card."""
    sample = {
        "name": "Organic Rosemary Extract",
        "one_liner": "Natural plant-derived antioxidant.",
        "why_used": "Retards oxidation in oils",
        "origin": "plant",
        "how_made": [
            {"icon": "leaf", "text": "Harvest rosemary leaves"},
            {"icon": "extract", "text": "Extract active polyphenols"},
            {"icon": "filter", "text": "Purify and standardize extract"},
        ],
        "flags": {
            "vegan": "likely_yes",
            "vegetarian": "likely_yes",
            "halal": "likely_yes",
            "jain": "likely_yes",
        },
    }
    norm = validate_and_normalize_ai_card(sample, "Organic Rosemary Extract")
    assert norm is not None
    assert norm["name"] == "Organic Rosemary Extract"
    assert norm["why_used"] == "Retards oxidation in oils"
    assert norm["origin"] == "plant"
    assert len(norm["how_made"]) == 3
    assert norm["flags"]["vegan"] == "likely_yes"
    assert norm["verified"] is False
    assert norm["source"] == "ai"


def test_validate_rejects_placeholder_and_schema_example():
    """Card containing placeholder text or schema definitions is rejected."""
    bad_sample = {
        "name": "string",
        "one_liner": "TODO: insert description here",
        "why_used": "example",
        "origin": "plant",
        "how_made": [],
        "flags": {"vegan": "unclear"},
    }
    assert validate_and_normalize_ai_card(bad_sample, "Test") is None

    lorem_sample = {
        "name": "Real Name",
        "one_liner": "Lorem ipsum dolor sit amet",
        "why_used": "Flavor",
        "origin": "plant",
    }
    assert validate_and_normalize_ai_card(lorem_sample, "Real Name") is None


def test_sanitize_removes_safety_claims_and_never_says_safe():
    """Text sanitizer neutralizes 'safe to eat' and 'safe' claims."""
    raw = "This ingredient is 100% safe to eat and medically approved."
    cleaned = sanitize_text(raw)
    assert "safe to eat" not in cleaned.lower()
    assert "safe" not in cleaned.lower()


@pytest.mark.asyncio
async def test_missing_xai_key_falls_back_to_local_model(tmp_path):
    """When cloud LLM keys are empty, pipeline falls back to local model."""
    config = Settings(
        XAI_API_KEY="",
        GROQ_API_KEY="",
        AI_CARD_CACHE_DIR=str(tmp_path / "cache"),
        GROK_DAILY_CAP=100,
    )

    mock_local_card = {
        "name": "Local Beetroot Extract",
        "one_liner": "Natural red food coloring.",
        "why_used": "Coloring agent in confectionary",
        "origin": "plant",
        "how_made": [{"icon": "root", "text": "Crush fresh beets"}],
        "flags": {"vegan": "likely_yes", "vegetarian": "likely_yes", "halal": "likely_yes", "jain": "likely_no"},
    }

    with patch("backend.ai_cards._call_local_model", new=AsyncMock(return_value=mock_local_card)) as mock_local:
        with patch("backend.ai_cards._call_xai_grok", new=AsyncMock(return_value=None)) as mock_grok:
            card = await get_or_generate_ingredient_card("Local Beetroot Extract", config=config)

            # Grok should not have been called because XAI_API_KEY is empty
            assert mock_grok.called is False
            # Local model should have been called
            assert mock_local.called is True
            assert card.name == "Local Beetroot Extract"
            assert card.source == "ai"
            assert card.status == "available"


@pytest.mark.asyncio
async def test_both_providers_fail_returns_unavailable_card(tmp_path):
    """When both cloud and local models fail, returns 'unavailable right now' card, never invented data."""
    config = Settings(
        XAI_API_KEY="test_key",
        GROQ_API_KEY="",
        AI_CARD_CACHE_DIR=str(tmp_path / "cache"),
    )

    with patch("backend.ai_cards._call_xai_grok", new=AsyncMock(return_value=None)):
        with patch("backend.ai_cards._call_local_model", new=AsyncMock(return_value=None)):
            card = await get_or_generate_ingredient_card("Exotic Nonexistent Herb", config=config)
            assert card.status == "unavailable"
            assert "unavailable right now" in card.one_liner.lower()
            assert card.verified is False
            assert card.source == "ai"


@pytest.mark.asyncio
async def test_kb_entry_cached_and_never_calls_ai(kb):
    """If an ingredient is already in the KB (the 69 seed entries), it returns immediately with source='kb'."""
    entry = kb.get("whey_powder")
    assert entry is not None

    with patch("backend.ai_cards._call_xai_grok", new=AsyncMock()) as mock_grok:
        with patch("backend.ai_cards._call_local_model", new=AsyncMock()) as mock_local:
            card = await get_or_generate_ingredient_card("whey_powder", kb_entry=entry)
            assert mock_grok.called is False
            assert mock_local.called is False
            assert card.id == "whey_powder"
            assert card.source == "kb"
            assert card.status == "available"
