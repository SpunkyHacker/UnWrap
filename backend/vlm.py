"""Vision-Language Model (VLM) transcription client.

Supports:
1. Local VLM server (mlx-vlm / Ollama OpenAI-compatible endpoint).
2. Google Gemini API (fallback).
3. Groq API (fallback).

Requires strict JSON output matching the product specification.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Union

import httpx
from PIL import Image

from backend.config import Settings, get_settings
from backend.models import TranscriptionResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an OCR and label transcription engine.
Your sole job is to transcribe text from food product packaging photos.

STRICT INSTRUCTIONS:
1. Copy the text EXACTLY as printed on the packaging.
2. Do NOT interpret, correct, translate, summarise, or add anything.
3. NEVER guess unreadable words. If any word or area is smudged, blurry, cut off, or illegible, list it in "unreadable_parts".
4. You MUST reply ONLY with a valid JSON object matching the requested schema. No conversational filler, no explanations.
"""

USER_PROMPT = """Transcribe the food packaging text in this photo into the following JSON format:
{
  "ingredients_text": "exact text of the ingredients list, as printed",
  "contains_statement": "text after 'Contains:' or 'Allergy advice' if present, else null",
  "may_contain_statement": "text after 'May contain traces of' or 'May contain' if present, else null",
  "nutrition_text": "raw nutrition table text if visible, else null",
  "language": "detected language code (e.g. en)",
  "unreadable_parts": ["any parts you could not read"]
}
"""


def _encode_image_to_base64(image_input: Union[str, Path, Image.Image, bytes]) -> str:
    """Encode an image to a base64 JPEG data URL."""
    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            data = f.read()
    elif isinstance(image_input, bytes):
        data = image_input
    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    b64_str = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _clean_json_response(raw_text: str) -> Optional[dict]:
    """Extract and parse a JSON object from raw model output."""
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Attempt to locate first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = cleaned[first_brace : last_brace + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


async def transcribe_with_local_vlm(
    image_b64: str,
    config: Settings,
) -> tuple[Optional[dict], int]:
    """Call an OpenAI-compatible local VLM server (mlx-vlm or Ollama)."""
    base_url = config.VLM_BASE_URL.rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": config.VLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.VLM_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    raw_content = data["choices"][0]["message"]["content"]
    parsed_json = _clean_json_response(raw_content)
    return parsed_json, elapsed_ms


async def transcribe_with_gemini(
    image_b64: str,
    config: Settings,
) -> tuple[Optional[dict], int]:
    """Call Google Gemini API as remote VLM fallback."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    # Extract raw base64 data
    header, b64_data = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
    mime_type = "image/jpeg"
    if "image/png" in header:
        mime_type = "image/png"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.VLM_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed_json = _clean_json_response(raw_text)
        return parsed_json, elapsed_ms
    except (KeyError, IndexError):
        return None, elapsed_ms


async def transcribe_with_groq(
    image_b64: str,
    config: Settings,
) -> tuple[Optional[dict], int]:
    """Call Groq vision endpoint as remote VLM fallback."""
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            },
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.VLM_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    raw_content = data["choices"][0]["message"]["content"]
    parsed_json = _clean_json_response(raw_content)
    return parsed_json, elapsed_ms


async def transcribe_with_vlm(
    image_input: Union[str, Path, Image.Image, bytes],
    config: Optional[Settings] = None,
) -> Optional[TranscriptionResult]:
    """Attempt VLM transcription using configured backend.

    Returns None if the VLM call fails, times out, returns invalid JSON,
    or returns empty/suspiciously short ingredients text.
    """
    if config is None:
        config = get_settings()

    try:
        image_b64 = _encode_image_to_base64(image_input)
    except Exception as exc:
        logger.warning("Failed to encode image to base64: %s", exc)
        return None

    backend = (config.VLM_BACKEND or "local").lower()
    parsed_json: Optional[dict] = None
    elapsed_ms: int = 0

    try:
        if backend == "gemini":
            parsed_json, elapsed_ms = await transcribe_with_gemini(image_b64, config)
        elif backend == "groq":
            parsed_json, elapsed_ms = await transcribe_with_groq(image_b64, config)
        else:
            parsed_json, elapsed_ms = await transcribe_with_local_vlm(image_b64, config)
    except Exception as exc:
        logger.warning("VLM call (%s) failed or timed out: %s", backend, exc)
        return None

    if not parsed_json:
        logger.warning("VLM returned invalid or empty JSON.")
        return None

    ingredients_text = parsed_json.get("ingredients_text") or ""
    # Suspiciously short check (<10 chars)
    if not isinstance(ingredients_text, str) or len(ingredients_text.strip()) < 10:
        logger.warning("VLM ingredients_text was empty or suspiciously short: '%s'", ingredients_text)
        return None

    unreadable = parsed_json.get("unreadable_parts")
    if not isinstance(unreadable, list):
        unreadable = [str(unreadable)] if unreadable else []

    return TranscriptionResult(
        ingredients_text=ingredients_text.strip(),
        contains_statement=str(parsed_json.get("contains_statement") or "").strip(),
        may_contain_statement=str(parsed_json.get("may_contain_statement") or "").strip(),
        nutrition_text=str(parsed_json.get("nutrition_text") or "").strip(),
        language=str(parsed_json.get("language") or "en"),
        unreadable_parts=[str(p) for p in unreadable if p],
        engine_used=f"vlm_{backend}",
        time_ms=elapsed_ms,
        fallback_occurred=False,
    )
