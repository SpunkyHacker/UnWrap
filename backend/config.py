"""Application settings loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised, typed configuration for the Unwrap backend.

    Values are read from environment variables first, then from a `.env`
    file in the project root.  Every field has a sensible default so the
    app can start without any configuration.
    """

    # -- Matching ----------------------------------------------------------
    FUZZY_MATCH_THRESHOLD: int = 92
    """Minimum score (0–100) for a fuzzy match to be accepted."""

    LLM_PICK_ENABLED: bool = False
    """Allow an LLM to disambiguate when fuzzy matching is uncertain."""

    # -- VLM (Vision-Language Model) ---------------------------------------
    VLM_BACKEND: str = "local"
    """Which VLM backend to use ('local' or 'remote')."""

    VLM_BASE_URL: str = "http://localhost:8080/v1"
    """Base URL of the VLM inference server."""

    VLM_MODEL_NAME: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    """Model identifier sent to the VLM server."""

    VLM_TIMEOUT_SECONDS: int = 60
    """HTTP timeout (seconds) for VLM requests."""

    # -- Chat / LLM / xAI Grok --------------------------------------------
    CHAT_BACKEND: str = "xai"
    """Chat & Card generation backend ('xai', 'local', 'gemini', or 'groq')."""

    XAI_API_KEY: str = ""
    """xAI API key for Grok models."""

    XAI_BASE_URL: str = "https://api.x.ai/v1"
    """Base URL for xAI OpenAI-compatible API."""

    XAI_MODEL: str = "grok-2-latest"
    """xAI model name (e.g. grok-2-latest)."""

    GROK_RPM_LIMIT: int = 30
    """Per-IP rate limit for Grok requests per minute."""

    GROK_DAILY_CAP: int = 100
    """Daily cap on total Grok requests."""

    AI_CARD_CACHE_DIR: str = "backend/data/ai_cards_cache"
    """Directory where AI-generated cards are cached on disk."""

    GEMINI_API_KEY: str = ""
    """Google Gemini API key (optional fallback)."""

    GROQ_API_KEY: str = ""
    """Groq API key."""

    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    """Base URL for Groq API."""

    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    """Groq model identifier (e.g. qwen/qwen3.8-27b)."""

    # -- Developer ---------------------------------------------------------
    DEBUG_MODE: bool = False
    """Enable verbose logging and extra diagnostics."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Module-level singleton — import this directly for convenience.
settings = Settings()


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Use this as a FastAPI dependency so settings can be overridden in
    tests via ``app.dependency_overrides``.
    """
    return Settings()
