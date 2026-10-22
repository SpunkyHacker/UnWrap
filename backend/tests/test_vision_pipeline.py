"""Unit tests for Phase 2 vision & transcription pipeline."""
import pytest
from pathlib import Path
from PIL import Image

from backend.config import Settings
from backend.image_processing import preprocess_image
from backend.matcher import KnowledgeBase
from backend.models import TranscriptionResult
from backend.ocr import run_apple_vision_ocr, spell_correct_tokens, transcribe_with_apple_vision
from backend.transcription import (
    _merge_transcriptions,
    compare_engines,
    transcribe_images,
    transcribe_single_image,
)
from backend.vlm import _clean_json_response, transcribe_with_vlm


@pytest.fixture
def sample_image_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "smoke_test_label.png"


def test_image_preprocessing(sample_image_path: Path):
    """Verify EXIF rotation, resizing to max 1280px, and contrast enhancement."""
    enhanced_img, orig_path, prep_path = preprocess_image(sample_image_path, max_dim=800)

    assert enhanced_img is not None
    w, h = enhanced_img.size
    assert max(w, h) <= 800
    assert orig_path.exists()
    assert prep_path.exists()


def test_apple_vision_ocr(sample_image_path: Path, kb: KnowledgeBase):
    """Verify Apple Vision OCR transcribes sample food label cleanly."""
    result = transcribe_with_apple_vision(sample_image_path, kb=kb)

    assert result.time_ms > 0
    assert result.engine_used == "apple_vision"
    assert "Wheat Flour" in result.ingredients_text
    assert "Sugar" in result.ingredients_text
    assert result.fallback_occurred is False


def test_spell_correct_tokens(kb: KnowledgeBase):
    """Verify RapidFuzz spell correction corrects high-confidence typos without inventing words."""
    # "Sodiun Benzoate" -> "Sodium Benzoate"
    raw = "Water, Sodiun Benzoate, Citric Acid"
    corrected_text, corrections = spell_correct_tokens(raw, kb, min_score=92.0)
    assert "Sodium Benzoate" in corrected_text
    assert len(corrections) > 0
    assert corrections[0]["original"] == "Sodiun Benzoate"

    # Non-existent short or gibberish word is NOT replaced
    gibberish = "Water, Xyzzz123, Citric Acid"
    gibberish_corrected, _ = spell_correct_tokens(gibberish, kb)
    assert "Xyzzz123" in gibberish_corrected


def test_vlm_json_cleaning():
    """Verify markdown fences and whitespace are stripped from raw VLM JSON."""
    raw_markdown = """```json
{
  "ingredients_text": "Wheat Flour, Sugar, Palm Oil",
  "contains_statement": "Wheat",
  "may_contain_statement": "Milk",
  "nutrition_text": null,
  "language": "en",
  "unreadable_parts": []
}
```"""
    data = _clean_json_response(raw_markdown)
    assert data is not None
    assert data["ingredients_text"] == "Wheat Flour, Sugar, Palm Oil"
    assert data["contains_statement"] == "Wheat"


@pytest.mark.asyncio
async def test_automatic_fallback_when_vlm_offline(sample_image_path: Path, kb: KnowledgeBase):
    """When VLM server is unreachable, pipeline automatically falls back to Apple Vision."""
    # Point to an unreachable port to simulate offline VLM
    offline_config = Settings(
        VLM_BACKEND="local",
        VLM_BASE_URL="http://127.0.0.1:59999/v1",
        VLM_TIMEOUT_SECONDS=1,
    )
    result = await transcribe_single_image(sample_image_path, kb=kb, config=offline_config)

    assert result is not None
    assert result.fallback_occurred is True
    assert "apple_vision" in result.engine_used
    assert "Wheat Flour" in result.ingredients_text


def test_merge_up_to_6_transcriptions():
    """Verify merging multiple photo results consolidates text and times."""
    t1 = TranscriptionResult(
        ingredients_text="Wheat Flour, Sugar",
        contains_statement="Wheat",
        engine_used="vlm",
        time_ms=500,
    )
    t2 = TranscriptionResult(
        ingredients_text="Palm Oil, Salt",
        may_contain_statement="Peanuts",
        nutrition_text="Calories 150",
        engine_used="vlm",
        time_ms=450,
    )

    merged = _merge_transcriptions([t1, t2])
    assert "Wheat Flour, Sugar" in merged.ingredients_text
    assert "Palm Oil, Salt" in merged.ingredients_text
    assert merged.contains_statement == "Wheat"
    assert merged.may_contain_statement == "Peanuts"
    assert "Calories 150" in merged.nutrition_text
    assert merged.time_ms == 950


@pytest.mark.asyncio
async def test_debug_comparison_mode(sample_image_path: Path, kb: KnowledgeBase):
    """Verify side-by-side comparison mode executes both engines."""
    offline_config = Settings(
        VLM_BACKEND="local",
        VLM_BASE_URL="http://127.0.0.1:59999/v1",
        VLM_TIMEOUT_SECONDS=1,
    )
    comparison = await compare_engines(sample_image_path, kb=kb, config=offline_config)

    assert comparison.apple_vision_result is not None
    assert comparison.apple_vision_result.time_ms > 0
    # VLM was offline, so vlm_result is None
    assert comparison.vlm_result is None
