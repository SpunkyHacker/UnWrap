"""Base interface for all benchmark transcription engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseBenchmarkEngine(ABC):
    """Abstract base class for benchmark transcription engines.

    Every engine must implement transcribe(image_path) -> dict.
    """

    engine_name: str

    @abstractmethod
    def transcribe(
        self,
        image_path: Path,
        max_dim: int = 1280,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        """Transcribe an image file.

        Parameters
        ----------
        image_path:
            Path to the target image.
        max_dim:
            Target max dimension for image preprocessing (default 1280).
        preprocess:
            Whether image preprocessing (resizing/contrast/orientation) is enabled.

        Returns
        -------
        dict[str, Any]
            Must include:
            - ingredients_text: str
            - contains_statement: Optional[str]
            - may_contain_statement: Optional[str]
            - raw_output: Any
            - timings: dict (elapsed_ms, etc.)
            - unreadable_parts: list[str]
            - json_valid: bool
            - prompt_tokens: int
            - completion_tokens: int
            - tokens_per_sec: float
            - peak_process_rss_mb: float
            - peak_gpu_mem_mb: float
        """
        pass
