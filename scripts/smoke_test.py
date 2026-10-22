"""Smoke test for Unwrap vision pipeline: Apple Vision OCR + Local VLM."""
import asyncio
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.matcher import KnowledgeBase
from backend.ocr import transcribe_with_apple_vision
from backend.transcription import compare_engines, transcribe_single_image
from backend.vlm import transcribe_with_vlm


def create_sample_label_image(output_path: Path) -> Path:
    """Generate a clean synthetic food packaging label for testing."""
    img = Image.new("RGB", (900, 420), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)

    lines = [
        "CRISPY CHOCOLATE COOKIES",
        "----------------------------------------------------------------",
        "INGREDIENTS: Wheat Flour (62%), Sugar, Palm Oil, Cocoa Powder,",
        "Emulsifier (Soy Lecithin, E322), Salt, Raising Agent (Sodium Bicarbonate),",
        "Natural Vanilla Flavour.",
        "",
        "Contains: Wheat, Soy.",
        "May contain traces of Milk, Peanuts.",
        "",
        "Nutrition Facts: Calories 140, Total Fat 7g, Carbohydrate 19g",
    ]

    y = 30
    for line in lines:
        draw.text((35, y), line, fill=(30, 30, 30))
        y += 35

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


async def main():
    print("==================================================================")
    print("Unwrap Vision Pipeline Smoke Test (Apple Vision vs Local VLM)")
    print("==================================================================")

    config = get_settings()
    kb = KnowledgeBase()

    sample_img_path = PROJECT_ROOT / "backend" / "data" / "smoke_test_label.png"
    create_sample_label_image(sample_img_path)
    print(f"✓ Generated sample test image at: {sample_img_path}")

    # -----------------------------------------------------------------------
    # 1. Test Apple Vision OCR
    # -----------------------------------------------------------------------
    print("\n--- [Engine 1: Native Apple Vision OCR] ---")
    start_ocr = time.perf_counter()
    ocr_result = transcribe_with_apple_vision(sample_img_path, kb=kb)
    total_ocr_ms = int((time.perf_counter() - start_ocr) * 1000)

    print(f"Timing: {ocr_result.time_ms} ms (Total pipeline: {total_ocr_ms} ms)")
    print(f"Engine Used: {ocr_result.engine_used}")
    print(f"Ingredients Text:\n  '{ocr_result.ingredients_text}'")
    print(f"Contains Statement: '{ocr_result.contains_statement}'")
    print(f"May Contain Statement: '{ocr_result.may_contain_statement}'")

    # -----------------------------------------------------------------------
    # 2. Test Local / Configured VLM
    # -----------------------------------------------------------------------
    print(f"\n--- [Engine 2: VLM ({config.VLM_BACKEND} @ {config.VLM_BASE_URL})] ---")
    start_vlm = time.perf_counter()
    vlm_result = None
    try:
        vlm_result = await transcribe_with_vlm(sample_img_path, config=config)
    except Exception as exc:
        print(f"VLM call raised: {exc}")

    if vlm_result is not None:
        total_vlm_ms = int((time.perf_counter() - start_vlm) * 1000)
        print(f"Timing: {vlm_result.time_ms} ms (Total pipeline: {total_vlm_ms} ms)")
        print(f"Engine Used: {vlm_result.engine_used}")
        print(f"Ingredients Text:\n  '{vlm_result.ingredients_text}'")
        print(f"Contains Statement: '{vlm_result.contains_statement}'")
        print(f"May Contain Statement: '{vlm_result.may_contain_statement}'")
        print(f"Unreadable Parts: {vlm_result.unreadable_parts}")
    else:
        print(f"ℹ VLM endpoint at {config.VLM_BASE_URL} is currently offline or unreachable.")
        print("  -> Pipeline will seamlessly fall back to Apple Vision OCR.")

    # -----------------------------------------------------------------------
    # 3. Test Full Pipeline (Fallback Verification)
    # -----------------------------------------------------------------------
    print("\n--- [Full Pipeline with Automatic Fallback & Tracking] ---")
    pipeline_result = await transcribe_single_image(sample_img_path, kb=kb, config=config)
    print(f"Engine Used: {pipeline_result.engine_used}")
    print(f"Fallback Occurred: {pipeline_result.fallback_occurred}")
    print(f"Total Time: {pipeline_result.time_ms} ms")

    # -----------------------------------------------------------------------
    # 4. Side-by-Side Debug Comparison Mode
    # -----------------------------------------------------------------------
    print("\n--- [Side-by-Side Debug Comparison Mode] ---")
    comparison = await compare_engines(sample_img_path, kb=kb, config=config)
    print(f"Apple Vision Time: {comparison.apple_vision_result.time_ms} ms")
    if comparison.vlm_result:
        print(f"VLM Time: {comparison.vlm_result.time_ms} ms")
        print(f"Time Delta: {comparison.time_difference_ms} ms")
    else:
        print("VLM: [Offline / Fallback]")

    print("\n✓ Smoke test finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())
