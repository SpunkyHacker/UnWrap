"""VLM transcription engines for MLX-based models."""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil
from PIL import Image

try:
    import mlx.core as mx
    def _get_mlx_peak_gpu_mb() -> float:
        try:
            fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
            return round(fn() / (1024 * 1024), 2) if fn else 0.0
        except Exception:
            return 0.0
except ImportError:
    def _get_mlx_peak_gpu_mb() -> float:
        return 0.0

from backend.image_processing import preprocess_image
from backend.vlm import SYSTEM_PROMPT, USER_PROMPT, _clean_json_response, _encode_image_to_base64
from benchmark.engines.base import BaseBenchmarkEngine
from benchmark.engines.server_manager import MLXVLMServerManager

logger = logging.getLogger(__name__)


class VlmBenchmarkEngine(BaseBenchmarkEngine):
    """Benchmark engine for local MLX-VLM models."""

    def __init__(
        self,
        engine_name: str,
        target_model: str,
        fallback_model: Optional[str] = None,
        server_manager: Optional[MLXVLMServerManager] = None,
    ):
        self.engine_name = engine_name
        self.target_model = target_model
        self.fallback_model = fallback_model
        self.server_manager = server_manager or MLXVLMServerManager()
        self.active_model_name = target_model
        self.cold_start_ms: int = 0

    def prepare_server(self, force_restart_for_cold_start: bool = False) -> int:
        """Ensure server is running with the model, measuring cold start if requested."""
        model_to_use = self.target_model if force_restart_for_cold_start else self.active_model_name
        actual_model, cold_ms = self.server_manager.ensure_server(
            model_name=model_to_use,
            fallback_model=self.fallback_model,
            force_restart_for_cold_start=force_restart_for_cold_start,
        )
        self.active_model_name = actual_model
        self.cold_start_ms = cold_ms
        return cold_ms

    def transcribe(
        self,
        image_path: Path,
        max_dim: int = 1280,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        """Perform deterministic transcription via local VLM."""
        # Ensure server is running
        self.prepare_server(force_restart_for_cold_start=False)

        # Preprocessing ablation
        if preprocess:
            _, _, final_path = preprocess_image(image_path, max_dim=max_dim, save_debug=False)
            image_b64 = _encode_image_to_base64(final_path)
        else:
            with Image.open(image_path) as im:
                rgb_im = im.convert("RGB")
                buf = io.BytesIO()
                rgb_im.save(buf, format="JPEG", quality=90)
                image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        payload = {
            "model": self.active_model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        url = f"{self.server_manager.base_url}/v1/chat/completions"
        start_time = time.perf_counter()

        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract tokens and timings from response
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        server_timings = data.get("timings", {})
        prompt_ms = server_timings.get("prompt_ms", 0.0)
        predicted_ms = server_timings.get("predicted_ms", 0.0)
        tokens_per_sec = server_timings.get("predicted_per_second", 0.0)
        if not tokens_per_sec and predicted_ms > 0 and completion_tokens > 0:
            tokens_per_sec = round(completion_tokens / (predicted_ms / 1000.0), 2)
        elif not tokens_per_sec and elapsed_ms > 0 and completion_tokens > 0:
            tokens_per_sec = round(completion_tokens / (elapsed_ms / 1000.0), 2)

        # Memory tracking
        server_rss = self.server_manager.get_server_rss_mb()
        runner_rss = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        total_rss = round(server_rss + runner_rss, 2)

        # Peak GPU memory: check server timings peak_memory or local mx
        server_peak_gpu_gb = server_timings.get("peak_memory")
        if server_peak_gpu_gb is not None:
            gpu_mb = round(float(server_peak_gpu_gb) * 1024, 2)
        else:
            gpu_mb = _get_mlx_peak_gpu_mb()

        # Parse text output
        raw_content = ""
        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass

        parsed_json = _clean_json_response(raw_content)
        json_valid = parsed_json is not None

        ingredients_text = ""
        contains_stmt = None
        may_contain_stmt = None
        unreadable_parts: list[str] = []

        if parsed_json:
            ingredients_text = (parsed_json.get("ingredients_text") or "").strip()
            contains_stmt = parsed_json.get("contains_statement") or None
            may_contain_stmt = parsed_json.get("may_contain_statement") or None
            unreadable_parts = parsed_json.get("unreadable_parts") or []
        else:
            # Fallback if raw content is not clean JSON
            ingredients_text = raw_content.strip()

        return {
            "ingredients_text": ingredients_text,
            "contains_statement": contains_stmt,
            "may_contain_statement": may_contain_stmt,
            "raw_output": raw_content,
            "timings": {
                "elapsed_ms": elapsed_ms,
                "cold_start_ms": self.cold_start_ms,
                "prompt_ms": round(prompt_ms, 2),
                "decode_ms": round(predicted_ms, 2),
            },
            "unreadable_parts": unreadable_parts,
            "json_valid": json_valid,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_process_rss_mb": total_rss,
            "peak_gpu_mem_mb": gpu_mb,
            "model_used": self.active_model_name,
        }


def create_qwen25vl_7b_engine(server_manager: Optional[MLXVLMServerManager] = None) -> VlmBenchmarkEngine:
    return VlmBenchmarkEngine(
        engine_name="vlm_qwen25vl_7b",
        target_model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        fallback_model=None,
        server_manager=server_manager,
    )


def create_qwen3vl_8b_engine(server_manager: Optional[MLXVLMServerManager] = None) -> VlmBenchmarkEngine:
    return VlmBenchmarkEngine(
        engine_name="vlm_qwen3vl_8b",
        target_model="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        fallback_model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        server_manager=server_manager,
    )
