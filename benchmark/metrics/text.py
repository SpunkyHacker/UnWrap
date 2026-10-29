"""Text distance metrics: CER (Character Error Rate) and WER (Word Error Rate)."""
from __future__ import annotations

import jiwer


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) using jiwer.

    Returns float in range [0.0, ...] (0.0 = perfect match).
    """
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()

    if not ref and not hyp:
        return 0.0
    if not ref and hyp:
        return 1.0  # Or length of hyp, bounded to 1.0 for rate

    try:
        score = float(jiwer.cer(ref, hyp))
        return round(score, 4)
    except Exception:
        return 1.0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using jiwer.

    Returns float in range [0.0, ...] (0.0 = perfect match).
    """
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()

    if not ref and not hyp:
        return 0.0
    if not ref and hyp:
        return 1.0

    try:
        score = float(jiwer.wer(ref, hyp))
        return round(score, 4)
    except Exception:
        return 1.0
