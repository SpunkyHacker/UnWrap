"""Visualization, summary aggregation, diff generation, and report compilation."""
from __future__ import annotations

import csv
import difflib
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.metrics.efficiency import (
    MODEL_METADATA,
    compute_latency_stats,
    compute_throughput_images_per_min,
)

logger = logging.getLogger(__name__)


def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calculate empirical bootstrap confidence interval."""
    clean_vals = [float(v) for v in values if pd.notnull(v)]
    if not clean_vals:
        return 0.0, 0.0
    if len(clean_vals) < 2 or len(set(clean_vals)) == 1:
        m = float(np.mean(clean_vals))
        return round(m, 4), round(m, 4)

    alpha_low = ((1.0 - ci) / 2.0) * 100
    alpha_high = (ci + (1.0 - ci) / 2.0) * 100

    boot_means = []
    n = len(clean_vals)
    for _ in range(n_resamples):
        sample = np.random.choice(clean_vals, size=n, replace=True)
        boot_means.append(np.mean(sample))

    low = float(np.percentile(boot_means, alpha_low))
    high = float(np.percentile(boot_means, alpha_high))
    return round(low, 4), round(high, 4)


class BenchmarkReporter:
    """Generates all summary CSVs, breakdown CSVs, plots, diffs, and report.md."""

    def __init__(self, results_dir: Path, raw_csv_path: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self.raw_csv_path = raw_csv_path or (self.results_dir / "results_raw.csv")
        self.failures_dir = self.results_dir / "failures"
        self.failures_dir.mkdir(parents=True, exist_ok=True)

    def load_df(self) -> pd.DataFrame:
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Missing {self.raw_csv_path}")
        df = pd.read_csv(self.raw_csv_path)
        # Filter for timed runs only for summary metrics
        return df

    def generate_summary_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary.csv with mean, std, and bootstrap 95% CI."""
        timed_df = df[df["run_type"].str.startswith("run_") | (df["run_type"] == "timed")].copy()
        if timed_df.empty:
            timed_df = df.copy()

        summary_rows = []
        for (engine, ablation), group in timed_df.groupby(["engine_name", "ablation_key"]):
            n_runs = len(group)
            latencies = group["elapsed_ms"].tolist()
            lat_stats = compute_latency_stats(latencies)

            cer_ci = bootstrap_ci(group["cer"].tolist())
            wer_ci = bootstrap_ci(group["wer"].tolist())
            f1_ci = bootstrap_ci(group["f1"].tolist())
            lat_ci = bootstrap_ci(latencies)

            throughput = compute_throughput_images_per_min(lat_stats["mean"])
            cold_start = group["cold_start_ms"].max() if "cold_start_ms" in group else 0

            row = {
                "engine_name": engine,
                "ablation_key": ablation,
                "num_runs": n_runs,
                "cer_mean": round(group["cer"].mean(), 4),
                "cer_std": round(group["cer"].std(), 4),
                "cer_ci95_low": cer_ci[0],
                "cer_ci95_high": cer_ci[1],
                "wer_mean": round(group["wer"].mean(), 4),
                "wer_std": round(group["wer"].std(), 4),
                "wer_ci95_low": wer_ci[0],
                "wer_ci95_high": wer_ci[1],
                "precision_mean": round(group["precision"].mean(), 4),
                "recall_mean": round(group["recall"].mean(), 4),
                "f1_mean": round(group["f1"].mean(), 4),
                "f1_std": round(group["f1"].std(), 4),
                "f1_ci95_low": f1_ci[0],
                "f1_ci95_high": f1_ci[1],
                "exact_match_rate": round(group["exact_list_match"].astype(bool).mean(), 4),
                "allergen_term_recall_mean": round(group["allergen_term_recall"].mean(), 4),
                "hallucinated_count_mean": round(group["hallucinated_count"].mean(), 2),
                "allergen_accuracy_mean": round(group["allergen_accuracy"].mean(), 4),
                "false_clears_total": int(group["false_clears"].sum()),
                "latency_mean_ms": lat_stats["mean"],
                "latency_p50_ms": lat_stats["p50"],
                "latency_p95_ms": lat_stats["p95"],
                "latency_std_ms": lat_stats["std"],
                "latency_ci95_low": lat_ci[0],
                "latency_ci95_high": lat_ci[1],
                "cold_start_ms": int(cold_start),
                "tokens_per_sec_mean": round(group["tokens_per_sec"].mean(), 2) if "tokens_per_sec" in group else 0.0,
                "peak_rss_mb": round(group["peak_process_rss_mb"].max(), 2),
                "peak_gpu_mb": round(group["peak_gpu_mem_mb"].max(), 2),
                "throughput_img_per_min": throughput,
                "total_flops_est": group["total_flops_est"].mean() if "total_flops_est" in group else 0.0,
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.results_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary to %s", summary_path)
        return summary_df

    def generate_tag_breakdown_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Break down metrics by image tag."""
        timed_df = df[df["run_type"].str.startswith("run_") | (df["run_type"] == "timed")].copy()
        if timed_df.empty:
            timed_df = df.copy()

        rows = []
        for (engine, ablation), group in timed_df.groupby(["engine_name", "ablation_key"]):
            tag_records: list[tuple[str, Any]] = []
            for _, row in group.iterrows():
                tags = [t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()]
                if not tags:
                    tags = ["untagged"]
                for t in tags:
                    tag_records.append((t, row))

            # Group by tag
            tag_dict: dict[str, list[Any]] = {}
            for t, r in tag_records:
                tag_dict.setdefault(t, []).append(r)

            for tag, records in tag_dict.items():
                tag_df = pd.DataFrame(records)
                rows.append({
                    "engine_name": engine,
                    "ablation_key": ablation,
                    "tag": tag,
                    "sample_count": len(tag_df),
                    "cer_mean": round(tag_df["cer"].mean(), 4),
                    "wer_mean": round(tag_df["wer"].mean(), 4),
                    "f1_mean": round(tag_df["f1"].mean(), 4),
                    "allergen_accuracy_mean": round(tag_df["allergen_accuracy"].mean(), 4),
                    "false_clears_total": int(tag_df["false_clears"].sum()),
                    "latency_mean_ms": round(tag_df["elapsed_ms"].mean(), 2),
                })

        breakdown_df = pd.DataFrame(rows)
        breakdown_path = self.results_dir / "breakdown_by_tag.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        logger.info("Saved tag breakdown to %s", breakdown_path)
        return breakdown_df

    def plot_accuracy_vs_latency(self, summary_df: pd.DataFrame):
        """Generate accuracy-vs-latency scatter plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        # Filter for default ablation
        defaults = summary_df[summary_df["ablation_key"] == "default"]
        if defaults.empty:
            defaults = summary_df

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (_, row) in enumerate(defaults.iterrows()):
            eng = row["engine_name"]
            lat = row["latency_mean_ms"]
            f1 = row["f1_mean"]
            c = colors[i % len(colors)]
            ax.scatter(lat, f1, s=140, color=c, label=eng, zorder=5)
            ax.annotate(
                f"{eng}\n({lat:.0f}ms, F1:{f1:.2f})",
                (lat, f1),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title("Ingredient Extraction Accuracy (F1) vs Latency", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Latency (ms)", fontsize=11)
        ax.set_ylabel("Mean Ingredient F1 Score", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        out_path = self.results_dir / "accuracy_vs_latency.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved accuracy vs latency plot to %s", out_path)

    def plot_latency_boxplots(self, df: pd.DataFrame):
        """Generate latency distribution boxplots across timed runs."""
        timed_df = df[df["run_type"].str.startswith("run_") | (df["run_type"] == "timed")]
        if timed_df.empty:
            timed_df = df

        fig, ax = plt.subplots(figsize=(9, 5))
        engines = sorted(timed_df["engine_name"].unique())
        data_by_eng = [timed_df[timed_df["engine_name"] == eng]["elapsed_ms"].values for eng in engines]

        try:
            ax.boxplot(data_by_eng, tick_labels=engines, patch_artist=True, boxprops=dict(facecolor="#cce5ff", color="#1f77b4"))
        except TypeError:
            ax.boxplot(data_by_eng, labels=engines, patch_artist=True, boxprops=dict(facecolor="#cce5ff", color="#1f77b4"))
        ax.set_title("Latency Distribution by Engine (ms)", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Elapsed Time (ms)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        out_path = self.results_dir / "latency_boxplots.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved latency boxplots to %s", out_path)

    def plot_peak_memory_barchart(self, summary_df: pd.DataFrame):
        """Generate grouped bar chart for peak RSS and peak GPU memory."""
        defaults = summary_df[summary_df["ablation_key"] == "default"]
        if defaults.empty:
            defaults = summary_df

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(defaults))
        width = 0.35

        rss_vals = defaults["peak_rss_mb"].values
        gpu_vals = defaults["peak_gpu_mb"].values
        labels = defaults["engine_name"].values

        ax.bar(x - width / 2, rss_vals, width, label="Peak Process RSS (MB)", color="#4a90e2")
        ax.bar(x + width / 2, gpu_vals, width, label="Peak MLX GPU Memory (MB)", color="#50e3c2")

        ax.set_title("Peak Memory Footprint by Engine (MB)", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Memory (MB)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend()

        out_path = self.results_dir / "peak_memory_barchart.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved peak memory chart to %s", out_path)

    def generate_per_engine_error_tables(self, df: pd.DataFrame):
        """Render an image table per engine showing image-level metrics."""
        timed_df = df[df["run_type"].str.startswith("run_") | (df["run_type"] == "timed")]
        if timed_df.empty:
            timed_df = df

        for engine, group in timed_df.groupby("engine_name"):
            # Aggregate per image
            img_agg = group.groupby("image_name").agg({
                "cer": "mean",
                "wer": "mean",
                "f1": "mean",
                "allergen_accuracy": "mean",
                "false_clears": "sum",
                "elapsed_ms": "mean",
            }).reset_index()

            cell_text = []
            for _, row in img_agg.iterrows():
                cell_text.append([
                    row["image_name"],
                    f"{row['cer']:.3f}",
                    f"{row['wer']:.3f}",
                    f"{row['f1']:.3f}",
                    f"{row['allergen_accuracy']:.2f}",
                    str(int(row["false_clears"])),
                    f"{row['elapsed_ms']:.0f} ms",
                ])

            fig, ax = plt.subplots(figsize=(10, max(2, len(cell_text) * 0.6 + 1.2)))
            ax.axis("tight")
            ax.axis("off")

            col_labels = ["Image", "CER", "WER", "F1", "Allergen Acc", "False Clears", "Latency"]
            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.4)

            # Style header
            for c in range(len(col_labels)):
                cell = table[0, c]
                cell.set_facecolor("#2c3e50")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")

            ax.set_title(f"Image Error & Metric Table: {engine}", fontsize=13, fontweight="bold", pad=15)
            out_path = self.results_dir / f"table_{engine}_errors.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved error table for %s to %s", engine, out_path)

    def generate_failures_diffs(self, df: pd.DataFrame, max_failures: int = 6):
        """Generate word-level diffs for the worst images per engine."""
        timed_df = df[df["run_type"].str.startswith("run_") | (df["run_type"] == "timed")]
        if timed_df.empty:
            timed_df = df

        diff_reports: dict[str, list[dict]] = {}

        for engine, group in timed_df.groupby("engine_name"):
            # Sort by lowest F1 score (worst first) then highest CER
            worst_cases = group.sort_values(by=["f1", "cer"], ascending=[True, False]).drop_duplicates(subset=["image_name"]).head(max_failures)

            engine_diffs = []
            for _, row in worst_cases.iterrows():
                img_name = row["image_name"]
                ref_text = str(row.get("reference_text", ""))
                hyp_text = str(row.get("hypothesis_text", ""))

                # Word-level diff
                ref_words = ref_text.split()
                hyp_words = hyp_text.split()
                diff = list(difflib.ndiff(ref_words, hyp_words))
                diff_str = " ".join(diff)

                out_file = self.failures_dir / f"{engine}_{Path(img_name).stem}_diff.md"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# Word Diff for {engine} on {img_name}\n\n")
                    f.write(f"- **F1**: {row.get('f1', 0.0):.4f}\n")
                    f.write(f"- **CER**: {row.get('cer', 0.0):.4f}\n")
                    f.write(f"- **WER**: {row.get('wer', 0.0):.4f}\n")
                    f.write(f"- **False Clears**: {row.get('false_clears', 0)}\n\n")
                    f.write("### Reference (Ground Truth)\n")
                    f.write(f"```\n{ref_text}\n```\n\n")
                    f.write("### Hypothesis (Transcribed)\n")
                    f.write(f"```\n{hyp_text}\n```\n\n")
                    f.write("### Word-level Diff\n")
                    f.write("Legend: `- word` (missing), `+ word` (hallucinated/extra), `  word` (correct)\n\n")
                    f.write(f"```\n{diff_str}\n```\n")

                engine_diffs.append({
                    "image": img_name,
                    "f1": row.get("f1", 0.0),
                    "cer": row.get("cer", 0.0),
                    "diff_file": out_file.name,
                })
            diff_reports[engine] = engine_diffs

        return diff_reports

    def write_report_markdown(self, summary_df: pd.DataFrame, breakdown_df: pd.DataFrame, diff_reports: dict):
        """Compile comprehensive report.md."""
        report_path = self.results_dir / "report.md"

        defaults = summary_df[summary_df["ablation_key"] == "default"]
        if defaults.empty:
            defaults = summary_df

        # Check for false clears across all runs
        total_fc = int(summary_df["false_clears_total"].sum())

        lines = [
            "# Label OCR & VLM Transcription Benchmark Report",
            "",
            "## Executive Summary",
            f"This benchmark evaluates on-device text transcription across **4 engines** on local Apple Silicon hardware without any cloud LLMs.",
            "",
            f"- **Safety Status**: Total False Clears across all evaluated engines = **{total_fc}**.",
        ]

        if total_fc > 0:
            lines.append("> [!CAUTION]")
            lines.append(f"> **SAFETY VIOLATION**: Detected {total_fc} instance(s) of `FALSE_CLEAR` where allergens were expected but the engine output resulted in a CLEAR safety status!")
        else:
            lines.append("> [!NOTE]")
            lines.append("> **Zero False Clears**: No critical `FALSE_CLEAR` safety violations were observed on tested profiles.")

        lines.extend([
            "",
            "## Overall Engine Performance (Default Config)",
            "",
            "| Engine | CER (95% CI) | WER (95% CI) | Ingredient F1 (95% CI) | Allergen Acc | False Clears | Latency Mean (P95) | Peak RSS | Peak GPU | Throughput |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for _, r in defaults.iterrows():
            eng = r["engine_name"]
            cer_str = f"{r['cer_mean']:.3f} [{r['cer_ci95_low']:.3f}, {r['cer_ci95_high']:.3f}]"
            wer_str = f"{r['wer_mean']:.3f} [{r['wer_ci95_low']:.3f}, {r['wer_ci95_high']:.3f}]"
            f1_str = f"{r['f1_mean']:.3f} [{r['f1_ci95_low']:.3f}, {r['f1_ci95_high']:.3f}]"
            acc_str = f"{r['allergen_accuracy_mean']:.2f}"
            fc_str = f"**{r['false_clears_total']}**" if r["false_clears_total"] > 0 else "0"
            lat_str = f"{r['latency_mean_ms']:.0f} ms ({r['latency_p95_ms']:.0f} ms)"
            rss_str = f"{r['peak_rss_mb']:.0f} MB"
            gpu_str = f"{r['peak_gpu_mb']:.0f} MB"
            tp_str = f"{r['throughput_img_per_min']:.1f} img/min"
            lines.append(f"| `{eng}` | {cer_str} | {wer_str} | {f1_str} | {acc_str} | {fc_str} | {lat_str} | {rss_str} | {gpu_str} | {tp_str} |")

        lines.extend([
            "",
            "## Efficiency & Resource Footprint",
            "",
            "| Engine | Parameters | Model Size on Disk | Est. Total FLOPs / img | Cold-Start Latency |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        for _, r in defaults.iterrows():
            eng = r["engine_name"]
            meta = MODEL_METADATA.get(eng, {})
            params = f"{meta.get('total_params', 0) / 1e9:.2f}B" if meta.get("total_params", 0) > 1e8 else f"{meta.get('total_params', 0) / 1e6:.0f}M"
            disk = f"{r.get('disk_size_mb', 0):.1f} MB" if r.get('disk_size_mb', 0) > 0 else ("Built-in System" if "apple_vision" in eng else "Cached")
            flops = f"{r['total_flops_est'] / 1e12:.2f} TFLOPs" if r.get('total_flops_est', 0) > 1e11 else f"{r.get('total_flops_est', 0) / 1e9:.2f} GFLOPs"
            cold = f"{r.get('cold_start_ms', 0):.0f} ms"
            lines.append(f"| `{eng}` | {params} | {disk} | {flops}* | {cold} |")

        lines.append("")
        lines.append("> \\* *FLOPs are analytical estimates based on decoder (2 * params * tokens) + ViT patch encoder calculations.*")

        # Ablations
        ablations = summary_df[summary_df["ablation_key"] != "default"]
        if not ablations.empty:
            lines.extend([
                "",
                "## Ablation Study",
                "",
                "| Engine | Ablation | Latency Mean | Ingredient F1 | CER | False Clears |",
                "| :--- | :--- | :---: | :---: | :---: | :---: |",
            ])
            for _, r in ablations.iterrows():
                lines.append(f"| `{r['engine_name']}` | `{r['ablation_key']}` | {r['latency_mean_ms']:.0f} ms | {r['f1_mean']:.3f} | {r['cer_mean']:.3f} | {r['false_clears_total']} |")

        # Visualizations references
        lines.extend([
            "",
            "## Visualizations",
            "- Accuracy vs Latency: `accuracy_vs_latency.png`",
            "- Latency Boxplots: `latency_boxplots.png`",
            "- Peak Memory Footprint: `peak_memory_barchart.png`",
            "- Per-Engine Error Tables: `table_<engine>_errors.png`",
            "- Detailed Word-Level Diffs: `failures/*.md`",
            "",
            "## Commands to Reproduce",
            "```bash",
            "# Run the complete deterministic benchmark:",
            "python -m benchmark.run",
            "",
            "# Run without ablations (quick mode):",
            "python -m benchmark.run --no-ablations",
            "",
            "# Run specific engines only:",
            "python -m benchmark.run --engines apple_vision_raw apple_vision_corrected vlm_qwen25vl_7b",
            "```",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Compiled report to %s", report_path)
