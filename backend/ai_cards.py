"""AI-generated ingredient card generator with xAI Grok, caching, and fallback ladder.

Features:
- Primary: xAI Grok (grok-2-latest) via OpenAI-compatible endpoint.
- Secondary: Local LLM/VLM at http://localhost:8080/v1.
- Tertiary: Return 'unavailable right now' (never invent cards).
- Per-IP rate limiting and daily cap.
- Disk caching in backend/data/ai_cards_cache/.
- Strict schema validation, rejection of placeholder text, sanitization.
- Privacy-first: sends ONLY the ingredient name to LLMs.
"""
from __future__ import annotations

import collections
import datetime
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.config import Settings, get_settings
from backend.models import (
    AICardFlags,
    AIIngredientCard,
    DietFlags,
    HowMadeStep,
    IngredientKBEntry,
)

logger = logging.getLogger("unwrap.ai_cards")

VALID_ORIGINS = {"plant", "animal", "fermentation", "lab", "mineral", "unclear"}
VALID_FLAG_VALUES = {"likely_yes", "likely_no", "unclear"}
PLACEHOLDER_SUBSTRINGS = {
    "placeholder",
    "todo",
    "lorem ipsum",
    "insert here",
    "example.com",
    "sample text",
    "schema-example",
}

# ---------------------------------------------------------------------------
# Rate Limiting & Daily Cap State
# ---------------------------------------------------------------------------
class GrokRateLimiter:
    """In-memory rate limiter tracking requests per minute per IP and daily total."""

    def __init__(self):
        self.ip_timestamps: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self.daily_count: int = 0
        self.current_date: datetime.date = datetime.date.today()

    def _rollover_day_if_needed(self):
        today = datetime.date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_count = 0
            self.ip_timestamps.clear()

    def check_and_record(self, client_ip: str, rpm_limit: int, daily_cap: int) -> tuple[bool, str]:
        """Check if request is permitted. If yes, record it and return (True, "").

        If no, return (False, reason).
        """
        self._rollover_day_if_needed()

        # Check daily cap
        if self.daily_count >= daily_cap:
            return False, f"Daily Grok call cap reached ({self.daily_count}/{daily_cap})"

        # Check IP rate limit (rolling 60 seconds)
        now = time.time()
        ip_queue = self.ip_timestamps[client_ip]
        while ip_queue and ip_queue[0] < now - 60.0:
            ip_queue.popleft()

        if len(ip_queue) >= rpm_limit:
            return False, f"Per-IP rate limit exceeded ({len(ip_queue)}/{rpm_limit} req/min)"

        # Record
        ip_queue.append(now)
        self.daily_count += 1
        return True, ""

    def reset(self):
        """Reset all counters (for testing)."""
        self.ip_timestamps.clear()
        self.daily_count = 0
        self.current_date = datetime.date.today()


_rate_limiter = GrokRateLimiter()


def get_rate_limiter() -> GrokRateLimiter:
    """Return the global rate limiter instance."""
    return _rate_limiter


def _slugify(name: str) -> str:
    """Convert ingredient name to filesystem-safe slug."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_") or "unknown_ingredient"


# ---------------------------------------------------------------------------
# Validation & Sanitization
# ---------------------------------------------------------------------------
def sanitize_text(text: str) -> str:
    """Remove HTML, escape harmful markup, and enforce no-safety-claims policy."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", "", text)
    clean = html.unescape(clean).strip()
    # Enforce NO "safe" claims
    clean = re.sub(r"\b(safe to eat|safe for consumption|100% safe|safe)\b", "common", clean, flags=re.IGNORECASE)
    return clean


def validate_and_normalize_ai_card(
    raw: dict[str, Any],
    ingredient_name: str,
) -> Optional[dict[str, Any]]:
    """Validate JSON response against AI card schema and reject placeholders."""
    if not isinstance(raw, dict):
        return None

    # Check for placeholder strings in any field
    raw_str = json.dumps(raw).lower()
    for sub in PLACEHOLDER_SUBSTRINGS:
        if sub in raw_str:
            logger.warning(f"Rejected AI card containing placeholder '{sub}'")
            return None

    # Exact schema placeholder match check
    if raw.get("name") in ("string", "example", "<string>", ""):
        return None

    name = sanitize_text(str(raw.get("name") or ingredient_name))
    one_liner = sanitize_text(str(raw.get("one_liner") or f"{name} is an ingredient used in food products."))

    # why_used: max 5 words
    why_raw = sanitize_text(str(raw.get("why_used") or "Commonly used in food preparation."))
    why_words = why_raw.split()
    why_used = " ".join(why_words[:5])

    # origin
    origin_raw = str(raw.get("origin") or "unclear").strip().lower()
    origin = origin_raw if origin_raw in VALID_ORIGINS else "unclear"

    # how_made: max 3 steps
    raw_steps = raw.get("how_made") or []
    if not isinstance(raw_steps, list):
        raw_steps = []
    how_made: list[dict[str, str]] = []
    for step in raw_steps[:3]:
        if isinstance(step, dict):
            icon = str(step.get("icon") or "box").strip().lower()
            text = sanitize_text(str(step.get("text") or ""))
            if text:
                how_made.append({"icon": icon, "text": " ".join(text.split()[:7])})
        elif isinstance(step, str) and step.strip():
            how_made.append({"icon": "box", "text": " ".join(sanitize_text(step).split()[:7])})

    # flags
    flags_raw = raw.get("flags") or {}
    if not isinstance(flags_raw, dict):
        flags_raw = {}
    flags = {
        "vegan": flags_raw.get("vegan") if flags_raw.get("vegan") in VALID_FLAG_VALUES else "unclear",
        "vegetarian": flags_raw.get("vegetarian") if flags_raw.get("vegetarian") in VALID_FLAG_VALUES else "unclear",
        "halal": flags_raw.get("halal") if flags_raw.get("halal") in VALID_FLAG_VALUES else "unclear",
        "jain": flags_raw.get("jain") if flags_raw.get("jain") in VALID_FLAG_VALUES else "unclear",
    }

    return {
        "id": _slugify(name),
        "name": name,
        "canonical_name": name,
        "role": raw.get("role") or "other",
        "one_liner": one_liner,
        "why_used": why_used,
        "origin": origin,
        "origin_type": origin,
        "how_made": how_made,
        "flags": flags,
        "verified": False,
        "source": "ai",
        "status": "available",
        "confidence": "low",
    }


def make_unavailable_card(ingredient_id_or_name: str) -> AIIngredientCard:
    """Return an explicitly unavailable card when generation fails completely."""
    clean_name = ingredient_id_or_name.replace("_", " ").title()
    slug_id = _slugify(ingredient_id_or_name)
    return AIIngredientCard(
        id=slug_id,
        name=clean_name,
        canonical_name=clean_name,
        role="other",
        one_liner="Ingredient card unavailable right now.",
        why_used="Information unavailable",
        origin="unclear",
        origin_type="unclear",
        how_made=[],
        flags=AICardFlags(vegan="unclear", vegetarian="unclear", halal="unclear", jain="unclear"),
        diet_flags=DietFlags(),
        verified=False,
        source="ai",
        status="unavailable",
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Disk Cache Helpers
# ---------------------------------------------------------------------------
def _get_cache_dir(config: Settings) -> Path:
    cache_path = Path(config.AI_CARD_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_cached_card(ingredient_id: str, config: Settings) -> Optional[AIIngredientCard]:
    """Retrieve card from disk cache if it exists."""
    cache_dir = _get_cache_dir(config)
    slug = _slugify(ingredient_id)
    cache_file = cache_dir / f"{slug}.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "available":
                return AIIngredientCard(**data)
        except Exception as exc:
            logger.warning(f"Failed to read cached card for {slug}: {exc}")
    return None


def save_cached_card(card: AIIngredientCard, config: Settings) -> None:
    """Persist generated card to disk cache."""
    if card.status != "available":
        return
    cache_dir = _get_cache_dir(config)
    slug = _slugify(card.id or card.name)
    cache_file = cache_dir / f"{slug}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(card.model_dump(), f, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to write card cache for {slug}: {exc}")


# ---------------------------------------------------------------------------
# LLM Generation Prompts
# ---------------------------------------------------------------------------
CARD_SYSTEM_PROMPT = """You are an objective ingredient information explainer.
Explain the specified food ingredient factually, concisely, and neutrally.

MANDATORY RULES:
1. Absolutely NO health claims, medical claims, or nutritional benefits.
2. Absolutely NO safety claims. NEVER use the word 'safe' anywhere in any field.
3. If uncertain about any origin, step, or suitability flag, strictly specify 'unclear'.
4. why_used must be 5 words or fewer.
5. how_made must have at most 3 concise steps (each with an icon name and short text).
6. Output MUST be valid JSON only matching the schema below. No markdown code blocks, no preamble.
7. Do NOT include placeholder text, 'string', 'example', 'TODO', or schema definitions.

JSON Schema:
{
  "name": "Ingredient Name",
  "one_liner": "Concise factual one-sentence description.",
  "why_used": "Max 5 words explanation",
  "origin": "plant" | "animal" | "fermentation" | "lab" | "mineral" | "unclear",
  "how_made": [
    {"icon": "source_icon", "text": "Step 1 text"},
    {"icon": "process_icon", "text": "Step 2 text"},
    {"icon": "finish_icon", "text": "Step 3 text"}
  ],
  "flags": {
    "vegan": "likely_yes" | "likely_no" | "unclear",
    "vegetarian": "likely_yes" | "likely_no" | "unclear",
    "halal": "likely_yes" | "likely_no" | "unclear",
    "jain": "likely_yes" | "likely_no" | "unclear"
  }
}"""


def _extract_json_from_llm_response(text: str) -> Optional[dict]:
    """Parse JSON from model response, handling optional markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Try regex extract
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Call Providers: xAI Grok & Local Model
# ---------------------------------------------------------------------------
async def _call_groq_api(
    ingredient_name: str,
    config: Settings,
    client_ip: str,
) -> Optional[dict]:
    """Call Groq API (e.g. qwen/qwen3.8-27b) with rate limit check and JSON format."""
    if not config.GROQ_API_KEY:
        return None

    allowed, reason = _rate_limiter.check_and_record(
        client_ip=client_ip,
        rpm_limit=config.GROK_RPM_LIMIT,
        daily_cap=config.GROK_DAILY_CAP,
    )
    if not allowed:
        logger.warning(f"Groq call blocked by rate limiter: {reason}")
        return None

    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": CARD_SYSTEM_PROMPT},
            {"role": "user", "content": f"Ingredient: {ingredient_name}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    url = f"{config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _extract_json_from_llm_response(content)
            else:
                logger.warning(f"Groq API returned HTTP {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.warning(f"Groq API call error: {exc}")
    return None


async def _call_xai_grok(
    ingredient_name: str,
    config: Settings,
    client_ip: str,
) -> Optional[dict]:
    """Call xAI Grok API with rate limit check and strict timeout."""
    if not config.XAI_API_KEY:
        logger.info("xAI API key not configured; skipping Grok call.")
        return None

    # Check rate limit and daily cap
    allowed, reason = _rate_limiter.check_and_record(
        client_ip=client_ip,
        rpm_limit=config.GROK_RPM_LIMIT,
        daily_cap=config.GROK_DAILY_CAP,
    )
    if not allowed:
        logger.warning(f"Grok call blocked by rate limiter: {reason}")
        return None

    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.XAI_MODEL,
        "messages": [
            {"role": "system", "content": CARD_SYSTEM_PROMPT},
            {"role": "user", "content": f"Ingredient: {ingredient_name}"},
        ],
        "temperature": 0.1,
    }

    url = f"{config.XAI_BASE_URL.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _extract_json_from_llm_response(content)
            else:
                logger.warning(f"xAI Grok returned HTTP {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.warning(f"xAI Grok API call error: {exc}")
    return None


async def _call_local_model(
    ingredient_name: str,
    config: Settings,
) -> Optional[dict]:
    """Fall back to local OpenAI-compatible inference server."""
    url = f"{config.VLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": config.VLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": CARD_SYSTEM_PROMPT},
            {"role": "user", "content": f"Ingredient: {ingredient_name}"},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _extract_json_from_llm_response(content)
            else:
                logger.warning(f"Local model endpoint returned HTTP {resp.status_code}")
    except Exception as exc:
        logger.info(f"Local model call failed or unreachable: {exc}")
    return None


# ---------------------------------------------------------------------------
# Main Entry Point: get_or_generate_ingredient_card
# ---------------------------------------------------------------------------
async def get_or_generate_ingredient_card(
    ingredient_name_or_id: str,
    kb_entry: Optional[IngredientKBEntry] = None,
    client_ip: str = "127.0.0.1",
    config: Optional[Settings] = None,
) -> AIIngredientCard:
    """Retrieve card from KB, disk cache, or generate via Grok / Local fallback."""
    cfg = config or get_settings()

    # 1. If existing in curated KnowledgeBase (the 69 seed entries act as cache)
    if kb_entry is not None:
        return AIIngredientCard(
            id=kb_entry.id,
            name=kb_entry.canonical_name,
            canonical_name=kb_entry.canonical_name,
            role=kb_entry.role,
            one_liner=kb_entry.one_liner,
            why_used=kb_entry.why_used,
            origin=kb_entry.origin_type,
            origin_type=kb_entry.origin_type,
            how_made=kb_entry.how_made,
            flags=AICardFlags(
                vegan="likely_yes" if kb_entry.diet_flags.vegan is True else ("likely_no" if kb_entry.diet_flags.vegan is False else "unclear"),
                vegetarian="likely_yes" if kb_entry.diet_flags.vegetarian is True else ("likely_no" if kb_entry.diet_flags.vegetarian is False else "unclear"),
                halal="likely_yes" if kb_entry.diet_flags.halal is True else ("likely_no" if kb_entry.diet_flags.halal is False else "unclear"),
                jain="likely_yes" if kb_entry.diet_flags.jain is True else ("likely_no" if kb_entry.diet_flags.jain is False else "unclear"),
            ),
            diet_flags=kb_entry.diet_flags,
            verified=kb_entry.verified,
            source="kb",
            status="available",
            confidence=kb_entry.confidence,
        )

    clean_name = ingredient_name_or_id.replace("_", " ").strip()

    # 2. Check disk cache
    cached = get_cached_card(clean_name, cfg)
    if cached is not None:
        return cached

    # 3. Call Cloud LLMs (Groq / xAI) (Level 1)
    if cfg.GROQ_API_KEY:
        raw_data = await _call_groq_api(clean_name, cfg, client_ip)
        if raw_data:
            norm = validate_and_normalize_ai_card(raw_data, clean_name)
            if norm:
                card = AIIngredientCard(**norm)
                save_cached_card(card, cfg)
                return card

    if cfg.XAI_API_KEY:
        raw_data = await _call_xai_grok(clean_name, cfg, client_ip)
        if raw_data:
            norm = validate_and_normalize_ai_card(raw_data, clean_name)
            if norm:
                card = AIIngredientCard(**norm)
                save_cached_card(card, cfg)
                return card

    # 4. Fall back to local model (Level 2)
    raw_data = await _call_local_model(clean_name, cfg)
    if raw_data:
        norm = validate_and_normalize_ai_card(raw_data, clean_name)
        if norm:
            card = AIIngredientCard(**norm)
            save_cached_card(card, cfg)
            return card

    # 5. Both failed -> return unavailable card (Level 3, Never invent or mock)
    return make_unavailable_card(clean_name)
