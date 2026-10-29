"""Benchmark evaluation metrics package."""
from benchmark.metrics.text import calculate_cer, calculate_wer
from benchmark.metrics.ingredients import evaluate_ingredient_extraction
from benchmark.metrics.allergen_eval import AllergenEvaluator
from benchmark.metrics.efficiency import (
    estimate_flops,
    compute_latency_stats,
    compute_throughput_images_per_min,
    get_directory_size_mb,
    MODEL_METADATA,
)

__all__ = [
    "calculate_cer",
    "calculate_wer",
    "evaluate_ingredient_extraction",
    "AllergenEvaluator",
    "estimate_flops",
    "compute_latency_stats",
    "compute_throughput_images_per_min",
    "get_directory_size_mb",
    "MODEL_METADATA",
]
