"""Apple Vision OCR engines for benchmarking."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import psutil

try:
    import mlx.core as mx
    def _get_peak_gpu_mb() -> float:
        try:
            fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
            return round(fn() / (1024 * 1024), 2) if fn else 0.0
        except Exception:
            return 0.0
except ImportError:
    def _get_peak_gpu_mb() -> float:
        return 0.0

from backend.image_processing import preprocess_image
from backend.matcher import KnowledgeBase
from backend.ocr import process_ocr_to_transcription, run_apple_vision_ocr
from benchmark.engines.base import BaseBenchmarkEngine


class AppleVisionBaseEngine(BaseBenchmarkEngine):
    """Base class for Apple Vision OCR engines."""

    def __init__(self, with_correction: bool = False):
        self.with_correction = with_correction
        self.engine_name = "apple_vision_corrected" if with_correction else "apple_vision_raw"
        self._kb: Optional[KnowledgeBase] = KnowledgeBase() if with_correction else None

    def transcribe(
        self,
        image_path: Path,
        max_dim: int = 1280,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        target_path = image_path

        # Handle preprocessing ablation
        if preprocess:
            _, _, preprocessed_path = preprocess_image(image_path, max_dim=max_dim, save_debug=False)
            target_path = preprocessed_path

        ocr_lines, ocr_elapsed_ms = run_apple_vision_ocr(target_path)
        transcription = process_ocr_to_transcription(ocr_lines, ocr_elapsed_ms, kb=self._kb)
        total_elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        gpu_mb = _get_peak_gpu_mb()

        return {
            "ingredients_text": transcription.ingredients_text,
            "contains_statement": transcription.contains_statement or None,
            "may_contain_statement": transcription.may_contain_statement or None,
            "raw_output": "\n".join(ocr_lines),
            "timings": {
                "elapsed_ms": total_elapsed_ms,
                "ocr_ms": ocr_elapsed_ms,
            },
            "unreadable_parts": transcription.unreadable_parts or [],
            "json_valid": False,  # Native OCR does not output JSON
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_sec": 0.0,
            "peak_process_rss_mb": rss_mb,
            "peak_gpu_mem_mb": gpu_mb,
            "corrections": transcription.corrections or [],
        }


class AppleVisionRawEngine(AppleVisionBaseEngine):
    def __init__(self):
        super().__init__(with_correction=False)


class AppleVisionCorrectedEngine(AppleVisionBaseEngine):
    def __init__(self):
        super().__init__(with_correction=True)
