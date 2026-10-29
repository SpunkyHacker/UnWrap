"""CLI runner for the benchmark suite.

Usage:
    python -m benchmark.run
    python benchmark/run.py --no-ablations
    python benchmark/run.py --generate-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

from benchmark.runner import BenchmarkRunner
from benchmark.visualize import BenchmarkReporter


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic OCR & VLM Benchmarking Harness")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/data"),
        help="Path to data directory containing images/ and ground_truth/ (default: benchmark/data)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Path to store outputs (default: benchmark/results)",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Engines to evaluate (default: apple_vision_raw apple_vision_corrected vlm_qwen25vl_7b vlm_qwen3vl_8b)",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip ablation configurations (preprocessing on/off, max_dim 1024/1536)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Skip runs and only regenerate summary.csv, breakdown_by_tag.csv, plots, diffs, and report.md",
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=None,
        help="Path to photos directory (default: auto-detects benchmark/photos or benchmark/data/images)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to unified ground_truth.json (default: auto-detects benchmark/ground_truth.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing results_raw.csv before running",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = results_dir / "results_raw.csv"

    if args.force and raw_csv.exists():
        logger.warning("Forcing fresh benchmark: removing %s", raw_csv)
        raw_csv.unlink()

    if not args.generate_only:
        runner = BenchmarkRunner(
            data_dir=data_dir,
            results_dir=results_dir,
            raw_csv_path=raw_csv,
            photos_dir=args.photos_dir,
            ground_truth_file=args.ground_truth,
        )
        runner.run_all(
            include_ablations=not args.no_ablations,
            selected_engines=args.engines,
        )

    # Visualization and reporting phase
    logger.info("Generating summary tables, charts, failures diffs, and report.md...")
    reporter = BenchmarkReporter(results_dir=results_dir, raw_csv_path=raw_csv)
    df = reporter.load_df()

    summary_df = reporter.generate_summary_csv(df)
    breakdown_df = reporter.generate_tag_breakdown_csv(df)

    reporter.plot_accuracy_vs_latency(summary_df)
    reporter.plot_latency_boxplots(df)
    reporter.plot_peak_memory_barchart(summary_df)
    reporter.generate_per_engine_error_tables(df)
    diff_reports = reporter.generate_failures_diffs(df)
    reporter.write_report_markdown(summary_df, breakdown_df, diff_reports)

    print("\n" + "=" * 70)
    print(" Benchmark Completed Successfully!")
    print(f" Summary:       {results_dir / 'summary.csv'}")
    print(f" Tag Breakdown: {results_dir / 'breakdown_by_tag.csv'}")
    print(f" Plots:         {results_dir / '*.png'}")
    print(f" Failures:      {results_dir / 'failures/'}")
    print(f" Full Report:   {results_dir / 'report.md'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
