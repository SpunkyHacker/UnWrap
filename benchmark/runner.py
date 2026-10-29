"""Benchmark execution orchestrator.

Implements the protocol:
- 2 warm-up runs, then 3 timed runs per image per engine.
- Measures cold start by restarting the VLM server once per VLM engine.
- Ensures only one VLM engine is loaded at a time (16GB RAM constraint).
- Logs thermal state and AC/battery power.
- Evaluates text metrics (CER, WER), ingredient metrics (Precision, Recall, F1, exact match, allergen recall, hallucination),
  end-to-end allergen checks (FALSE_CLEAR tracking), and efficiency.
- Supports ablations (preprocessing on/off, max_dim 1024/1280/1536).
- Resumable: skips already-completed runs in results_raw.csv.
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import psutil
from PIL import Image

from benchmark.engines.apple_vision import AppleVisionCorrectedEngine, AppleVisionRawEngine
from benchmark.engines.base import BaseBenchmarkEngine
from benchmark.engines.server_manager import MLXVLMServerManager, stop_server_on_port
from benchmark.engines.vlm import create_qwen25vl_7b_engine, create_qwen3vl_8b_engine
from benchmark.metrics.allergen_eval import AllergenEvaluator
from benchmark.metrics.efficiency import estimate_flops
from benchmark.metrics.ingredients import evaluate_ingredient_extraction
from benchmark.metrics.text import calculate_cer, calculate_wer

logger = logging.getLogger(__name__)

RAW_CSV_COLUMNS = [
    "image_name",
    "engine_name",
    "run_type",
    "ablation_key",
    "preprocess",
    "max_dim",
    "model_used",
    "cold_start_ms",
    "elapsed_ms",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_sec",
    "peak_process_rss_mb",
    "peak_gpu_mem_mb",
    "cer",
    "wer",
    "precision",
    "recall",
    "f1",
    "exact_list_match",
    "allergen_term_recall",
    "hallucinated_count",
    "json_valid",
    "unreadable_parts_nonempty",
    "total_allergen_checks",
    "allergen_matches",
    "allergen_accuracy",
    "false_clears",
    "decoder_flops_est",
    "vision_encoder_flops_est",
    "total_flops_est",
    "on_power",
    "thermal_state",
    "tags",
    "hypothesis_text",
    "reference_text",
]


def get_power_telemetry() -> tuple[bool, str]:
    """Check whether laptop is on power and battery state."""
    on_power = False
    batt_summary = "unknown"
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True).strip()
        lines = out.splitlines()
        first_line = lines[0] if lines else ""
        on_power = "AC Power" in first_line
        batt_summary = first_line
    except Exception:
        try:
            batt = psutil.sensors_battery()
            if batt:
                on_power = batt.power_plugged
                batt_summary = f"{batt.percent}% (plugged={on_power})"
        except Exception:
            pass
    return on_power, batt_summary


def get_thermal_telemetry() -> str:
    """Check macOS thermal pressure state."""
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if not lines:
            return "nominal"
        # Check if warning level recorded
        warning_lines = [l for l in lines if "level has been recorded" not in l and "status has been recorded" not in l]
        return " | ".join(warning_lines) if warning_lines else "nominal"
    except Exception:
        return "nominal"


class BenchmarkRunner:
    """Runs the benchmark across images, engines, and ablations."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        raw_csv_path: Optional[Path] = None,
        photos_dir: Optional[Path] = None,
        ground_truth_file: Optional[Path] = None,
    ):
        base_benchmark = Path("benchmark").resolve()
        self.data_dir = Path(data_dir).resolve() if data_dir else (base_benchmark / "data")
        self.results_dir = Path(results_dir).resolve() if results_dir else (base_benchmark / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_csv_path = raw_csv_path or (self.results_dir / "results_raw.csv")

        # Check for user-provided photos directory (benchmark/photos or benchmark/data/images)
        if photos_dir and Path(photos_dir).exists():
            self.photos_dir = Path(photos_dir).resolve()
        elif (base_benchmark / "photos").exists():
            self.photos_dir = (base_benchmark / "photos").resolve()
        elif (self.data_dir / "images").exists():
            self.photos_dir = (self.data_dir / "images").resolve()
        else:
            self.photos_dir = self.data_dir / "images"

        # Check for unified ground_truth.json
        self.unified_gt_file: Optional[Path] = None
        if ground_truth_file and Path(ground_truth_file).exists():
            self.unified_gt_file = Path(ground_truth_file).resolve()
        elif (base_benchmark / "ground_truth.json").exists():
            self.unified_gt_file = (base_benchmark / "ground_truth.json").resolve()
        elif (self.data_dir / "ground_truth.json").exists():
            self.unified_gt_file = (self.data_dir / "ground_truth.json").resolve()

        self.gt_dir = self.data_dir / "ground_truth"
        self._unified_gt_map: dict[str, dict[str, Any]] = {}
        self.server_manager = MLXVLMServerManager()
        self.allergen_evaluator = AllergenEvaluator()
        self._load_unified_ground_truth()
        self._completed_runs: set[tuple[str, str, str, str]] = set()
        self._load_completed_runs()

    def _load_unified_ground_truth(self):
        """Load unified ground_truth.json if present."""
        if not self.unified_gt_file or not self.unified_gt_file.exists():
            return

        with open(self.unified_gt_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            std_profiles = ["milk", "soy", "wheat", "peanut", "tree_nuts", "egg", "fish", "shellfish", "sesame", "mustard"]
            from backend.allergen_engine import check_allergens
            from backend.matcher import match_ingredients
            from backend.models import TranscriptionResult, UserProfile
            from backend.parser import parse_ingredients

            for item in raw_data:
                photo_id = str(item.get("photo_id", ""))
                # Derive expected allergens if not already defined
                if "expected_allergen_results" not in item:
                    full_text = item.get("true_text", "")
                    if item.get("contains_statement"):
                        full_text += f". Contains: {item['contains_statement']}"
                    if item.get("may_contain_statement"):
                        full_text += f". May contain: {item['may_contain_statement']}"
                    parsed = parse_ingredients(full_text)
                    matches = match_ingredients(parsed.ingredients, self.allergen_evaluator.kb)
                    tr = TranscriptionResult(
                        ingredients_text=item.get("true_text", ""),
                        contains_statement=item.get("contains_statement") or "",
                        may_contain_statement=item.get("may_contain_statement") or "",
                        unreadable_parts=item.get("unreadable_parts", []),
                        engine_used="gt",
                        time_ms=0,
                        fallback_occurred=False,
                    )
                    expected = {}
                    for p in std_profiles:
                        res = check_allergens(matches, UserProfile(allergens=[p]), parsed, transcription=tr)
                        expected[p] = res.overall_status.value.upper()
                    item["expected_allergen_results"] = expected

                # Index by photo_id, zero-padded id, and full stem
                self._unified_gt_map[photo_id] = item
                self._unified_gt_map[photo_id.zfill(3)] = item
                self._unified_gt_map[photo_id.lstrip("0")] = item

            logger.info("Loaded %d ground truth definitions from %s", len(raw_data), self.unified_gt_file.name)

    def _load_completed_runs(self):
        """Read results_raw.csv to support resumption without duplicating runs."""
        if not self.raw_csv_path.exists():
            with open(self.raw_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RAW_CSV_COLUMNS)
                writer.writeheader()
            return

        with open(self.raw_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get("image_name", ""),
                    row.get("engine_name", ""),
                    row.get("run_type", ""),
                    row.get("ablation_key", "default"),
                )
                self._completed_runs.add(key)
        logger.info("Loaded %d previously completed runs from %s", len(self._completed_runs), self.raw_csv_path.name)

    def _append_result(self, row: dict[str, Any]):
        """Append a single run result to results_raw.csv immediately."""
        with open(self.raw_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_CSV_COLUMNS)
            writer.writerow(row)
        key = (row["image_name"], row["engine_name"], row["run_type"], row["ablation_key"])
        self._completed_runs.add(key)

    def load_ground_truth(self, image_path: Path) -> Optional[dict[str, Any]]:
        """Load corresponding JSON ground truth for an image."""
        stem = image_path.stem
        item = (
            self._unified_gt_map.get(stem)
            or self._unified_gt_map.get(stem.zfill(3))
            or self._unified_gt_map.get(stem.lstrip("0"))
        )
        if item:
            tags = []
            if item.get("condition"):
                tags.append(str(item["condition"]))
            if "is_real_photo" in item:
                tags.append("real_photo" if item["is_real_photo"] else "synthetic")

            return {
                "ingredients_text": item.get("true_text") or item.get("ingredients_text", ""),
                "contains_statement": item.get("contains_statement"),
                "may_contain_statement": item.get("may_contain_statement"),
                "tags": tags,
                "expected_allergen_results": item.get("expected_allergen_results", {}),
                "unreadable_parts": item.get("unreadable_parts", []),
            }

        # Fallback to per-file JSON
        gt_path = self.gt_dir / f"{image_path.stem}.json"
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.warning("No ground truth found for %s", image_path.name)
        return None

    def execute_single_run(
        self,
        engine: BaseBenchmarkEngine,
        image_path: Path,
        gt_data: dict[str, Any],
        run_type: str,
        ablation_key: str = "default",
        preprocess: bool = True,
        max_dim: int = 1280,
    ) -> dict[str, Any]:
        """Execute one run (warmup or timed), evaluate all metrics, and append to CSV."""
        key = (image_path.name, engine.engine_name, run_type, ablation_key)
        if key in self._completed_runs:
            logger.info("Skipping already completed run: %s", key)
            return {}

        on_power, batt_summary = get_power_telemetry()
        thermal = get_thermal_telemetry()

        # Get image dimensions for FLOPs estimation
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 1280, 1280

        # Perform transcription
        transcription_res = engine.transcribe(
            image_path=image_path,
            max_dim=max_dim,
            preprocess=preprocess,
        )

        hyp_ingredients = transcription_res.get("ingredients_text", "")
        hyp_contains = transcription_res.get("contains_statement")
        hyp_may_contain = transcription_res.get("may_contain_statement")

        ref_ingredients = gt_data.get("ingredients_text", "")
        ref_contains = gt_data.get("contains_statement")
        ref_may_contain = gt_data.get("may_contain_statement")
        tags = gt_data.get("tags", [])
        expected_allergens = gt_data.get("expected_allergen_results", {})

        # Compute text metrics
        cer = calculate_cer(ref_ingredients, hyp_ingredients)
        wer = calculate_wer(ref_ingredients, hyp_ingredients)

        # Compute ingredient metrics
        ing_metrics = evaluate_ingredient_extraction(
            reference_text=ref_ingredients,
            hypothesis_text=hyp_ingredients,
            reference_contains=ref_contains,
            hypothesis_contains=hyp_contains,
            reference_may_contain=ref_may_contain,
            hypothesis_may_contain=hyp_may_contain,
        )

        # Compute end-to-end allergen evaluation
        allergen_eval = self.allergen_evaluator.evaluate(
            transcription_data=transcription_res,
            expected_allergen_results=expected_allergens,
            engine_name=engine.engine_name,
        )

        # Efficiency & FLOPs
        timings = transcription_res.get("timings", {})
        prompt_tokens = transcription_res.get("prompt_tokens", 0)
        completion_tokens = transcription_res.get("completion_tokens", 0)
        flops_info = estimate_flops(
            engine_name=engine.engine_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            image_width=min(img_w, max_dim) if preprocess else img_w,
            image_height=min(img_h, max_dim) if preprocess else img_h,
            model_override=transcription_res.get("model_used"),
        )

        unreadable = transcription_res.get("unreadable_parts") or []

        row = {
            "image_name": image_path.name,
            "engine_name": engine.engine_name,
            "run_type": run_type,
            "ablation_key": ablation_key,
            "preprocess": preprocess,
            "max_dim": max_dim,
            "model_used": transcription_res.get("model_used", engine.engine_name),
            "cold_start_ms": timings.get("cold_start_ms", 0),
            "elapsed_ms": timings.get("elapsed_ms", 0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": transcription_res.get("tokens_per_sec", 0.0),
            "peak_process_rss_mb": transcription_res.get("peak_process_rss_mb", 0.0),
            "peak_gpu_mem_mb": transcription_res.get("peak_gpu_mem_mb", 0.0),
            "cer": cer,
            "wer": wer,
            "precision": ing_metrics["precision"],
            "recall": ing_metrics["recall"],
            "f1": ing_metrics["f1"],
            "exact_list_match": ing_metrics["exact_list_match"],
            "allergen_term_recall": ing_metrics["allergen_term_recall"],
            "hallucinated_count": ing_metrics["hallucinated_count"],
            "json_valid": transcription_res.get("json_valid", False),
            "unreadable_parts_nonempty": len(unreadable) > 0,
            "total_allergen_checks": allergen_eval["total_checks"],
            "allergen_matches": allergen_eval["exact_matches"],
            "allergen_accuracy": allergen_eval["allergen_accuracy"],
            "false_clears": allergen_eval["false_clears"],
            "decoder_flops_est": flops_info["decoder_flops_est"],
            "vision_encoder_flops_est": flops_info["vision_encoder_flops_est"],
            "total_flops_est": flops_info["total_flops_est"],
            "on_power": on_power,
            "thermal_state": thermal,
            "tags": ",".join(tags),
            "hypothesis_text": hyp_ingredients,
            "reference_text": ref_ingredients,
        }

        self._append_result(row)
        return row

    def run_engine_protocol(
        self,
        engine: BaseBenchmarkEngine,
        image_paths: list[Path],
        ablation_key: str = "default",
        preprocess: bool = True,
        max_dim: int = 1280,
    ):
        """Run the standard protocol: 2 warmups + 3 timed runs per image."""
        logger.info("=== Running engine: %s (ablation=%s, max_dim=%d, preprocess=%s) ===", engine.engine_name, ablation_key, max_dim, preprocess)

        # For VLM engines, record cold-start by restarting the server once
        if hasattr(engine, "prepare_server"):
            logger.info("Restarting VLM server to measure cold-start for %s...", engine.engine_name)
            cold_ms = getattr(engine, "prepare_server")(force_restart_for_cold_start=True)
            logger.info("Cold start recorded for %s: %d ms", engine.engine_name, cold_ms)

        for img_path in image_paths:
            gt_data = self.load_ground_truth(img_path)
            if not gt_data:
                continue

            logger.info("Processing %s on %s...", img_path.name, engine.engine_name)

            # 2 Warm-up runs
            for w in (1, 2):
                self.execute_single_run(
                    engine=engine,
                    image_path=img_path,
                    gt_data=gt_data,
                    run_type=f"warmup_{w}",
                    ablation_key=ablation_key,
                    preprocess=preprocess,
                    max_dim=max_dim,
                )

            # 3 Timed runs
            for t in (1, 2, 3):
                self.execute_single_run(
                    engine=engine,
                    image_path=img_path,
                    gt_data=gt_data,
                    run_type=f"run_{t}",
                    ablation_key=ablation_key,
                    preprocess=preprocess,
                    max_dim=max_dim,
                )

    def run_all(
        self,
        include_ablations: bool = True,
        selected_engines: Optional[list[str]] = None,
    ):
        """Execute the full benchmark sequence across all engines one at a time."""
        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".heic")
        image_paths = sorted([p for p in self.photos_dir.iterdir() if p.suffix.lower() in valid_exts])
        if not image_paths:
            logger.error("No images found in %s! Please place images there.", self.photos_dir)
            return

        logger.info("Found %d images to benchmark in %s: %s", len(image_paths), self.photos_dir.name, [p.name for p in image_paths[:10]])

        engine_keys = selected_engines or [
            "apple_vision_raw",
            "apple_vision_corrected",
            "vlm_qwen25vl_7b",
            "vlm_qwen3vl_8b",
        ]

        # Engine A: apple_vision_raw
        if "apple_vision_raw" in engine_keys:
            engine_a = AppleVisionRawEngine()
            self.run_engine_protocol(engine_a, image_paths, ablation_key="default", preprocess=True, max_dim=1280)
            if include_ablations:
                # Preprocessing off ablation
                self.run_engine_protocol(engine_a, image_paths, ablation_key="no_preproc", preprocess=False, max_dim=1280)

        # Engine B: apple_vision_corrected
        if "apple_vision_corrected" in engine_keys:
            engine_b = AppleVisionCorrectedEngine()
            self.run_engine_protocol(engine_b, image_paths, ablation_key="default", preprocess=True, max_dim=1280)
            if include_ablations:
                self.run_engine_protocol(engine_b, image_paths, ablation_key="no_preproc", preprocess=False, max_dim=1280)

        # Engine C: vlm_qwen25vl_7b
        if "vlm_qwen25vl_7b" in engine_keys:
            engine_c = create_qwen25vl_7b_engine(self.server_manager)
            self.run_engine_protocol(engine_c, image_paths, ablation_key="default", preprocess=True, max_dim=1280)
            if include_ablations:
                # Longest side 1024
                self.run_engine_protocol(engine_c, image_paths, ablation_key="max_dim_1024", preprocess=True, max_dim=1024)
                # Longest side 1536
                self.run_engine_protocol(engine_c, image_paths, ablation_key="max_dim_1536", preprocess=True, max_dim=1536)
                # Preprocessing off
                self.run_engine_protocol(engine_c, image_paths, ablation_key="no_preproc", preprocess=False, max_dim=1280)

        # Engine D: vlm_qwen3vl_8b (with fallback)
        if "vlm_qwen3vl_8b" in engine_keys:
            engine_d = create_qwen3vl_8b_engine(self.server_manager)
            self.run_engine_protocol(engine_d, image_paths, ablation_key="default", preprocess=True, max_dim=1280)
            if include_ablations:
                self.run_engine_protocol(engine_d, image_paths, ablation_key="max_dim_1024", preprocess=True, max_dim=1024)
                self.run_engine_protocol(engine_d, image_paths, ablation_key="max_dim_1536", preprocess=True, max_dim=1536)
                self.run_engine_protocol(engine_d, image_paths, ablation_key="no_preproc", preprocess=False, max_dim=1280)

        logger.info("Benchmark execution complete. All runs saved to %s", self.raw_csv_path)
