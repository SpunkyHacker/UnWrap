"""Compare Apple Vision OCR vs VLM on a folder of real label photos.

Usage:
    python3 scripts/compare_engines.py [path/to/folder]

Default folder: backend/data/real_photos/
Writes: <folder>/comparison_report.md
NOTE: This script does NOT run any allergen logic.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import re
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.image_processing import preprocess_image
from backend.matcher import KnowledgeBase
from backend.ocr import transcribe_with_apple_vision
from backend.vlm import transcribe_with_vlm


def word_level_diff(text_a: str, text_b: str) -> str:
    """Compute a compact word-level diff between two text strings."""
    words_a = re.findall(r"\S+", text_a or "")
    words_b = re.findall(r"\S+", text_b or "")

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    diff_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            del_words = " ".join(words_a[i1:i2])
            add_words = " ".join(words_b[j1:j2])
            diff_parts.append(f"[-{del_words}-] [+{add_words}+]")
        elif tag == "delete":
            del_words = " ".join(words_a[i1:i2])
            diff_parts.append(f"[-{del_words}-]")
        elif tag == "insert":
            add_words = " ".join(words_b[j1:j2])
            diff_parts.append(f"[+{add_words}+]")

    if not diff_parts:
        return "Identical"
    return " ; ".join(diff_parts[:6])  # limit length for table


async def process_folder(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)
    config = get_settings()
    kb = KnowledgeBase()

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.heic", "*.JPG", "*.PNG", "*.JPEG")
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(ext))
    image_files = sorted(set(image_files))

    print(f"Scanning folder: {folder_path} (Found {len(image_files)} image(s))")

    if not image_files:
        print(f"ℹ No images found in {folder_path}. Please place test images there and re-run.")
        sample_img = PROJECT_ROOT / "backend" / "data" / "smoke_test_label.png"
        if sample_img.exists():
            print(f"Running comparison on smoke test label: {sample_img.name}")
            image_files = [sample_img]
        else:
            return

    table_rows = []

    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        enhanced_img, _, prep_path = preprocess_image(img_path)

        # 1. Apple Vision OCR
        start_ocr = time.perf_counter()
        ocr_res = transcribe_with_apple_vision(prep_path, kb=kb)
        ocr_time = ocr_res.time_ms

        # 2. VLM
        start_vlm = time.perf_counter()
        vlm_res = None
        try:
            vlm_res = await transcribe_with_vlm(prep_path, config=config)
        except Exception as exc:
            print(f"  VLM failed: {exc}")

        vlm_text = vlm_res.ingredients_text if vlm_res else "[VLM Offline/Failed]"
        vlm_time = vlm_res.time_ms if vlm_res else 0

        diff_summary = word_level_diff(ocr_res.ingredients_text, vlm_text)

        # Sanitize for markdown table
        clean_ocr_text = ocr_res.ingredients_text.replace("\n", " ").replace("|", "\\|")
        clean_vlm_text = vlm_text.replace("\n", " ").replace("|", "\\|")
        clean_diff = diff_summary.replace("\n", " ").replace("|", "\\|")

        table_rows.append(
            f"| `{img_path.name}` | {clean_ocr_text[:80]}... | {ocr_time}ms | {clean_vlm_text[:80]}... | {vlm_time}ms | {clean_diff[:60]} |"
        )

    # Build report
    report_lines = [
        "# OCR vs VLM Transcription Comparison Report",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**VLM Config**: `{config.VLM_BACKEND}` @ `{config.VLM_BASE_URL}` (Model: `{config.VLM_MODEL_NAME}`)",
        "",
        "| Image | Apple Vision Text | OCR Time | VLM Text | VLM Time | Word Diff (OCR vs VLM) |",
        "| :--- | :--- | :---: | :--- | :---: | :--- |",
    ]
    report_lines.extend(table_rows)
    report_content = "\n".join(report_lines) + "\n"

    report_path = folder_path / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"✓ Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Apple Vision vs VLM transcription on a folder")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(PROJECT_ROOT / "backend" / "data" / "real_photos"),
        help="Folder containing product packaging photos",
    )
    args = parser.parse_args()
    asyncio.run(process_folder(Path(args.folder)))


if __name__ == "__main__":
    main()
