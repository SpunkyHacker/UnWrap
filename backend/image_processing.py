"""Image preprocessing pipeline for the Unwrap vision modules.

1. Fixes EXIF rotation.
2. Resizes so the longest dimension is ~1280px.
3. Applies light contrast enhancement.
4. Preserves the original image for diagnostics / debug side-by-side view.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

import tempfile

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"


def preprocess_image(
    image_input: Union[bytes, str, Path, Image.Image],
    max_dim: int = 1280,
    save_debug: bool = False,
    temp_dir: Optional[Path] = None,
) -> tuple[Image.Image, Path, Path]:
    """Preprocess an input image for OCR and VLM processing.

    Parameters
    ----------
    image_input:
        Image as bytes, file path, or PIL Image.
    max_dim:
        Longest edge pixel size (default: 1280px).
    save_debug:
        If True, persists original and preprocessed images to CACHE_DIR for debug UI.
    temp_dir:
        Optional directory to store temporary files in. If None and save_debug is False,
        uses system temporary directory.

    Returns
    -------
    tuple[Image.Image, Path, Path]
        (preprocessed_pil_image, original_file_path, preprocessed_file_path)
    """
    import uuid

    file_id = str(uuid.uuid4())[:8]

    # Determine storage directory
    if save_debug:
        target_dir = CACHE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
    elif temp_dir is not None:
        target_dir = Path(temp_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = Path(tempfile.gettempdir()) / "unwrap_vision"
        target_dir.mkdir(parents=True, exist_ok=True)

    # Load image (with native macOS HEIC conversion support via sips)
    if isinstance(image_input, Image.Image):
        original_img = image_input
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.suffix.lower() in (".heic", ".heif"):
            import subprocess
            out_jpg = target_dir / f"{file_id}_heic_converted.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(out_jpg)], check=True, capture_output=True)
            original_img = Image.open(out_jpg)
        else:
            original_img = Image.open(p)
    elif isinstance(image_input, bytes):
        if len(image_input) > 12 and (b"ftypheic" in image_input[:32] or b"ftypmif1" in image_input[:32]):
            import subprocess
            temp_heic = target_dir / f"{file_id}_temp.heic"
            temp_heic.write_bytes(image_input)
            out_jpg = target_dir / f"{file_id}_heic_converted.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(temp_heic), "--out", str(out_jpg)], check=True, capture_output=True)
            original_img = Image.open(out_jpg)
        else:
            original_img = Image.open(io.BytesIO(image_input))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 1. Correct EXIF orientation
    try:
        transposed_img = ImageOps.exif_transpose(original_img)
    except Exception as exc:
        logger.warning("EXIF orientation correction skipped: %s", exc)
        transposed_img = original_img

    # Convert to RGB (handles RGBA, P, etc.)
    if transposed_img.mode != "RGB":
        transposed_img = transposed_img.convert("RGB")

    # Save original copy
    original_path = target_dir / f"{file_id}_original.jpg"
    transposed_img.save(original_path, "JPEG", quality=95)

    # 2. Resize so long side is about max_dim (1280px)
    w, h = transposed_img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized_img = transposed_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        resized_img = transposed_img.copy()

    # 3. Light contrast enhancement (1.15x for crisper label text)
    enhancer = ImageEnhance.Contrast(resized_img)
    enhanced_img = enhancer.enhance(1.15)

    # Save preprocessed copy
    preprocessed_path = target_dir / f"{file_id}_preprocessed.jpg"
    enhanced_img.save(preprocessed_path, "JPEG", quality=90)

    return enhanced_img, original_path, preprocessed_path
