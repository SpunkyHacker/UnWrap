"""Apple Vision OCR wrapper with RapidFuzz spell correction.

Uses macOS pyobjc Vision framework (VNRecognizeTextRequest) for fast,
on-device OCR on Apple Silicon, followed by RapidFuzz spell correction
against known ingredient aliases. Spell-corrects ONLY toward high-confidence matches.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from backend.matcher import KnowledgeBase
from backend.models import TranscriptionResult

logger = logging.getLogger(__name__)

# Try importing macOS Vision framework
try:
    import Vision
    from Cocoa import NSURL
    APPLE_VISION_AVAILABLE = True
except ImportError:
    APPLE_VISION_AVAILABLE = False
    logger.warning("pyobjc Vision framework not available; Apple Vision OCR disabled.")


# Regular expressions for OCR section routing
_RE_CONTAINS = re.compile(r"^\s*(?:contains|allergy\s+advice|allergens?)\s*:\s*(.*)$", re.IGNORECASE)
_RE_MAY_CONTAIN = re.compile(r"^\s*(?:may\s+contain(?:\s+traces\s+of)?|traces\s+of)\s*:?\s*(.*)$", re.IGNORECASE)
_RE_NUTRITION_HEADER = re.compile(r"\b(?:nutrition\s+facts|nutritional\s+information|per\s+100g|serving\s+size)\b", re.IGNORECASE)


def run_apple_vision_ocr(image_path: Union[str, Path]) -> tuple[list[str], int]:
    """Run native Apple Vision text recognition on an image file.

    Parameters
    ----------
    image_path:
        Path to image on disk.

    Returns
    -------
    tuple[list[str], int]
        List of recognized text lines and elapsed time in milliseconds.
    """
    if not APPLE_VISION_AVAILABLE:
        raise RuntimeError("Apple Vision framework is not available on this platform/environment.")

    path_str = str(Path(image_path).resolve())
    ns_url = NSURL.fileURLWithPath_(path_str)

    start_time = time.perf_counter()

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(ns_url, None)
    success, error = handler.performRequests_error_([request], None)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    if not success:
        logger.error("Apple Vision OCR failed: %s", error)
        return [], elapsed_ms

    results = request.results()
    lines: list[str] = []
    if results:
        for obs in results:
            top_candidates = obs.topCandidates_(1)
            if top_candidates:
                line_str = top_candidates[0].string().strip()
                if line_str:
                    lines.append(line_str)

    return lines, elapsed_ms


def spell_correct_tokens(
    text: str,
    kb: KnowledgeBase,
    min_score: float = 93.0,
    min_length: int = 5,
) -> tuple[str, list[dict]]:
    """Spell-correct OCR tokens toward known ingredient aliases using RapidFuzz.

    Only applies replacement when confidence is >= min_score and length >= min_length,
    guaranteeing the spell-corrector never invents non-existent words.

    Returns
    -------
    tuple[str, list[dict]]
        (corrected_text, list_of_corrections) where each correction is
        {"original": str, "corrected": str, "score": float}.
    """
    if not text or not kb._all_names:
        return text, []

    # Split text by commas/semicolons to correct ingredient segments
    segments = re.split(r"([,;])", text)
    corrected_segments: list[str] = []
    corrections: list[dict] = []

    for seg in segments:
        if seg in (",", ";"):
            corrected_segments.append(seg)
            continue

        stripped = seg.strip()
        # Keep parentheses structure intact if present
        if "(" in stripped or ")" in stripped or len(stripped) < min_length:
            corrected_segments.append(seg)
            continue

        # Check exact hit first
        if kb.lookup_name(stripped):
            corrected_segments.append(seg)
            continue

        # Try fuzzy match against known ingredient names
        best = process.extractOne(
            stripped.lower(),
            kb._all_names,
            scorer=fuzz.WRatio,
            score_cutoff=min_score,
        )
        if best is not None:
            matched_name, score, _idx = best
            entry = kb._name_to_entry[matched_name]
            leading_spaces = len(seg) - len(seg.lstrip())
            trailing_spaces = len(seg) - len(seg.rstrip())
            corrected_canonical = entry.canonical_name
            corrected = (" " * leading_spaces) + corrected_canonical + (" " * trailing_spaces)
            corrected_segments.append(corrected)
            corrections.append({
                "original": stripped,
                "corrected": corrected_canonical,
                "score": round(float(score), 1),
            })
        else:
            corrected_segments.append(seg)

    return "".join(corrected_segments), corrections


def process_ocr_to_transcription(
    lines: list[str],
    elapsed_ms: int,
    kb: Optional[KnowledgeBase] = None,
) -> TranscriptionResult:
    """Convert raw OCR lines into a structured TranscriptionResult."""
    ingredients_lines: list[str] = []
    contains_lines: list[str] = []
    may_contain_lines: list[str] = []
    nutrition_lines: list[str] = []
    in_nutrition = False

    for line in lines:
        # Check nutrition section
        if _RE_NUTRITION_HEADER.search(line):
            in_nutrition = True
            nutrition_lines.append(line)
            continue

        if in_nutrition:
            nutrition_lines.append(line)
            continue

        # Check Contains statement
        m_contains = _RE_CONTAINS.match(line)
        if m_contains:
            contains_lines.append(m_contains.group(1).strip() or line)
            continue

        # Check May contain statement
        m_may = _RE_MAY_CONTAIN.match(line)
        if m_may:
            may_contain_lines.append(m_may.group(1).strip() or line)
            continue

        ingredients_lines.append(line)

    raw_ingredients = " ".join(ingredients_lines).strip()

    # Apply spell correction against known ingredient aliases
    corrections: list[dict] = []
    if kb is not None and raw_ingredients:
        corrected_ingredients, corrections = spell_correct_tokens(raw_ingredients, kb)
    else:
        corrected_ingredients = raw_ingredients

    return TranscriptionResult(
        ingredients_text=corrected_ingredients,
        contains_statement="; ".join(contains_lines) if contains_lines else "",
        may_contain_statement="; ".join(may_contain_lines) if may_contain_lines else "",
        nutrition_text="\n".join(nutrition_lines) if nutrition_lines else "",
        language="en",
        unreadable_parts=[],
        engine_used="apple_vision",
        time_ms=elapsed_ms,
        fallback_occurred=False,
        corrections=corrections,
    )


def transcribe_with_apple_vision(
    image_path: Union[str, Path],
    kb: Optional[KnowledgeBase] = None,
) -> TranscriptionResult:
    """High-level entrypoint: runs Apple Vision OCR and returns TranscriptionResult."""
    lines, elapsed_ms = run_apple_vision_ocr(image_path)
    return process_ocr_to_transcription(lines, elapsed_ms, kb=kb)
