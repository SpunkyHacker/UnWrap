"""Unified transcription pipeline (VLM primary, Apple Vision fallback).

Supports:
1. Up to 6 photos per product (merged into one consolidated result).
2. Image preprocessing: EXIF rotation, resize to ~1280px, contrast enhancement.
3. Automatic fallback from VLM to Apple Vision OCR on failure / timeout / empty result.
4. Full fallback tracking (recorded in response).
5. Debug comparison mode: runs both VLM and Apple Vision side-by-side with individual timings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from backend.config import Settings, get_settings
from backend.image_processing import preprocess_image
from backend.matcher import KnowledgeBase
from backend.models import DebugComparisonResult, TranscriptionResult
from backend.ocr import transcribe_with_apple_vision
from backend.vlm import transcribe_with_vlm

logger = logging.getLogger(__name__)


def _merge_transcriptions(results: list[TranscriptionResult]) -> TranscriptionResult:
    """Merge transcription results from up to 6 product photos into one."""
    if not results:
        return TranscriptionResult()
    if len(results) == 1:
        return results[0]

    ingredients_parts: list[str] = []
    contains_parts: list[str] = []
    may_contain_parts: list[str] = []
    nutrition_parts: list[str] = []
    unreadable: list[str] = []
    engines: list[str] = []
    total_time = 0
    fallback = any(r.fallback_occurred for r in results)

    for r in results:
        total_time += r.time_ms
        if r.engine_used and r.engine_used not in engines:
            engines.append(r.engine_used)
        if r.ingredients_text:
            ingredients_parts.append(r.ingredients_text)
        if r.contains_statement:
            contains_parts.append(r.contains_statement)
        if r.may_contain_statement:
            may_contain_parts.append(r.may_contain_statement)
        if r.nutrition_text:
            nutrition_parts.append(r.nutrition_text)
        unreadable.extend(r.unreadable_parts)

    return TranscriptionResult(
        ingredients_text=" , ".join(ingredients_parts),
        contains_statement=" ; ".join(contains_parts),
        may_contain_statement=" ; ".join(may_contain_parts),
        nutrition_text="\n\n".join(nutrition_parts),
        language=results[0].language if results else "en",
        unreadable_parts=list(dict.fromkeys(unreadable)),  # deduplicate preserving order
        engine_used="+".join(engines),
        time_ms=total_time,
        fallback_occurred=fallback,
    )


async def transcribe_single_image(
    image_input: Union[bytes, str, Path, Image.Image],
    kb: Optional[KnowledgeBase] = None,
    config: Optional[Settings] = None,
    force_engine: Optional[str] = None,  # "vlm" | "apple_vision" | None
    save_debug: bool = False,
    temp_dir: Optional[Path] = None,
) -> TranscriptionResult:
    """Transcribe a single product image with VLM first and Apple Vision fallback.

    Parameters
    ----------
    image_input:
        Image file path, bytes, or PIL Image.
    kb:
        KnowledgeBase instance for OCR spell correction.
    config:
        Application settings.
    force_engine:
        Optional override to force a specific engine (used in debug comparison).
    save_debug:
        Persist debug copies if True.
    temp_dir:
        Directory for transient processing files.
    """
    if config is None:
        config = get_settings()

    # Step 1: Preprocess image
    enhanced_img, orig_path, prep_path = preprocess_image(
        image_input,
        save_debug=save_debug,
        temp_dir=temp_dir,
    )

    # If Apple Vision explicitly forced
    if force_engine == "apple_vision":
        return transcribe_with_apple_vision(prep_path, kb=kb)

    # Step 2: Attempt VLM transcription
    vlm_result = None
    if force_engine != "apple_vision":
        try:
            vlm_result = await transcribe_with_vlm(prep_path, config=config)
        except Exception as exc:
            logger.warning("VLM call exception: %s", exc)
            vlm_result = None

    if vlm_result is not None and vlm_result.ingredients_text:
        return vlm_result

    # Step 3: Automatic fallback to Apple Vision OCR
    logger.info("VLM transcription unavailable or invalid. Falling back to Apple Vision OCR.")
    ocr_result = transcribe_with_apple_vision(prep_path, kb=kb)
    ocr_result.fallback_occurred = True
    ocr_result.engine_used = f"apple_vision (fallback from {config.VLM_BACKEND})"
    return ocr_result


async def transcribe_images(
    images: list[Union[bytes, str, Path, Image.Image]],
    kb: Optional[KnowledgeBase] = None,
    config: Optional[Settings] = None,
    save_debug: bool = False,
    temp_dir: Optional[Path] = None,
) -> TranscriptionResult:
    """Transcribe up to 6 product images and merge into a single result."""
    if not images:
        return TranscriptionResult(
            engine_used="none",
            time_ms=0,
            unreadable_parts=["No image provided"],
        )

    # Limit to maximum 6 images
    selected_images = images[:6]
    results: list[TranscriptionResult] = []

    for img in selected_images:
        res = await transcribe_single_image(
            img,
            kb=kb,
            config=config,
            save_debug=save_debug,
            temp_dir=temp_dir,
        )
        results.append(res)

    return _merge_transcriptions(results)


async def compare_engines(
    image_input: Union[bytes, str, Path, Image.Image],
    kb: Optional[KnowledgeBase] = None,
    config: Optional[Settings] = None,
    save_debug: bool = False,
    temp_dir: Optional[Path] = None,
) -> DebugComparisonResult:
    """Run BOTH VLM and Apple Vision on the exact same photo and return side-by-side results.

    Used by the developer debug screen and smoke test to evaluate accuracy and timings.
    """
    if config is None:
        config = get_settings()

    enhanced_img, orig_path, prep_path = preprocess_image(
        image_input,
        save_debug=save_debug,
        temp_dir=temp_dir,
    )

    # 1. Run Apple Vision OCR
    ocr_result = transcribe_with_apple_vision(prep_path, kb=kb)

    # 2. Run VLM
    vlm_result = await transcribe_with_vlm(prep_path, config=config)

    time_diff = (vlm_result.time_ms if vlm_result else 0) - ocr_result.time_ms

    return DebugComparisonResult(
        vlm_result=vlm_result,
        apple_vision_result=ocr_result,
        time_difference_ms=time_diff,
    )
