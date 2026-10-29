"""Efficiency metrics: FLOPs estimation, model parameters, disk footprint, and throughput."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


def get_directory_size_mb(path: Path) -> float:
    """Calculate recursive size of a directory in MB following symlinks."""
    total_bytes = 0
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)

    for root, _, files in os.walk(path, followlinks=True):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                pass
    return round(total_bytes / (1024 * 1024), 2)


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "apple_vision_raw": {
        "display_name": "Apple Vision (Raw)",
        "total_params": 50_000_000,  # ~50M estimated on-device OCR model
        "text_params": 0,
        "vision_params": 50_000_000,
        "disk_size_mb": 0.0,  # Built into macOS system framework
        "is_system_framework": True,
    },
    "apple_vision_corrected": {
        "display_name": "Apple Vision (Corrected)",
        "total_params": 50_000_000,
        "text_params": 0,
        "vision_params": 50_000_000,
        "disk_size_mb": 0.0,
        "is_system_framework": True,
    },
    "vlm_qwen25vl_7b": {
        "display_name": "Qwen2.5-VL-7B-Instruct (4-bit)",
        "total_params": 7_610_000_000,
        "text_params": 7_070_000_000,
        "vision_params": 540_000_000,
        "hf_repo": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "vision_config": {
            "depth": 32,
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "patch_size": 14,
        },
    },
    "vlm_qwen3vl_8b": {
        "display_name": "Qwen3-VL-8B / 3B Fallback (4-bit)",
        "total_params": 8_200_000_000,
        "text_params": 7_500_000_000,
        "vision_params": 700_000_000,
        "hf_repo": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "vision_config": {
            "depth": 32,
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "patch_size": 14,
        },
    },
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit": {
        "display_name": "Qwen2.5-VL-3B-Instruct (4-bit Fallback)",
        "total_params": 3_100_000_000,
        "text_params": 2_700_000_000,
        "vision_params": 400_000_000,
        "hf_repo": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        "vision_config": {
            "depth": 32,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "patch_size": 14,
        },
    },
}


def estimate_vision_encoder_flops(
    image_width: int = 1280,
    image_height: int = 1280,
    vision_config: Optional[dict[str, Any]] = None,
) -> float:
    """Estimate Vision Transformer encoder FLOPs.

    Calculates analytical FLOPs for ViT layers:
    FLOPs = L * (8 * N * D^2 + 4 * N^2 * D + 4 * N * D * D_mlp)
    where N = (W/patch_size) * (H/patch_size).
    """
    cfg = vision_config or {
        "depth": 32,
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "patch_size": 14,
    }

    depth = cfg.get("depth", 32)
    hidden = cfg.get("hidden_size", 1280)
    mlp_dim = cfg.get("intermediate_size", 3420)
    patch_size = cfg.get("patch_size", 14)

    patches_w = max(1, image_width // patch_size)
    patches_h = max(1, image_height // patch_size)
    n_patches = patches_w * patches_h

    # Layer FLOPs: QKV + Out projection (8 N D^2), Attention Matrix (4 N^2 D), MLP (4 N D D_mlp)
    layer_flops = (8 * n_patches * (hidden ** 2)) + (4 * (n_patches ** 2) * hidden) + (4 * n_patches * hidden * mlp_dim)
    total_flops = depth * layer_flops
    return float(total_flops)


def estimate_flops(
    engine_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    image_width: int = 1280,
    image_height: int = 1280,
    model_override: Optional[str] = None,
) -> dict[str, Any]:
    """Estimate total inference FLOPs for a run.

    Returns dict with decoder FLOPs, vision encoder FLOPs, and total estimated FLOPs.
    """
    key = model_override if model_override in MODEL_METADATA else engine_name
    meta = MODEL_METADATA.get(key, MODEL_METADATA.get("vlm_qwen25vl_7b"))

    if meta.get("is_system_framework", False):
        # Proprietary OCR model estimate: ~50M params * ~500 visual features
        est_ocr_flops = 50_000_000 * 500 * 2
        return {
            "decoder_flops_est": 0.0,
            "vision_encoder_flops_est": float(est_ocr_flops),
            "total_flops_est": float(est_ocr_flops),
            "is_estimate": True,
            "label": "Estimated Apple Vision OCR FLOPs (proprietary on-device model)",
        }

    text_params = meta.get("text_params", 7_000_000_000)
    vision_cfg = meta.get("vision_config", {})

    total_tokens = prompt_tokens + completion_tokens
    decoder_flops = 2.0 * text_params * total_tokens
    vision_flops = estimate_vision_encoder_flops(image_width, image_height, vision_cfg)
    total_flops = decoder_flops + vision_flops

    return {
        "decoder_flops_est": float(decoder_flops),
        "vision_encoder_flops_est": float(vision_flops),
        "total_flops_est": float(total_flops),
        "is_estimate": True,
        "label": "Estimated FLOPs (decoder ~ 2*params*tokens + analytical ViT patch encoder estimate)",
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Compute mean, P50 (median), and P95 latency in milliseconds."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "std": 0.0}

    arr = np.array(latencies_ms)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "std": round(float(np.std(arr)), 2),
    }


def compute_throughput_images_per_min(mean_latency_ms: float) -> float:
    """Compute throughput in images per minute."""
    if mean_latency_ms <= 0:
        return 0.0
    return round(60000.0 / mean_latency_ms, 2)
