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
VALID_ROLES = {"base", "sweetener", "preservative", "texture", "color_flavor", "other"}
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

ROLE_MAPPING = {
    "base": "base",
    "solvent": "base",
    "fat_oil": "base",
    "fat": "base",
    "oil": "base",
    "lipid": "base",
    "protein": "base",
    "grain": "base",
    "flour": "base",
    "carrier": "base",
    "sweetener": "sweetener",
    "sugar": "sweetener",
    "sweetening": "sweetener",
    "preservative": "preservative",
    "antioxidant": "preservative",
    "acidity_regulator": "preservative",
    "acid": "preservative",
    "food_acid": "preservative",
    "texture": "texture",
    "emulsifier": "texture",
    "thickener": "texture",
    "stabilizer": "texture",
    "stabiliser": "texture",
    "gelling_agent": "texture",
    "binder": "texture",
    "color_flavor": "color_flavor",
    "flavour": "color_flavor",
    "flavor": "color_flavor",
    "colour": "color_flavor",
    "color": "color_flavor",
    "flavour_enhancer": "color_flavor",
    "flavor_enhancer": "color_flavor",
    "spice": "color_flavor",
    "seasoning": "color_flavor",
    "extract": "color_flavor",
    "other": "other",
}


def normalize_card_lookup_name(name_or_code: str) -> str:
    """Normalize ingredient name or additive code to canonical lookup form.

    E.g. 'E-322' -> 'INS 322', 'e322' -> 'INS 322', '322' inside code context -> 'INS 322'.
    Normalizes whitespace and canonicalizes additive code format.
    """
    from backend.parser import normalize_additive_code, _normalise_whitespace

    clean = _normalise_whitespace(name_or_code)
    # Check if the whole string is an additive code
    code = normalize_additive_code(clean)
    if code:
        return code
    # Check if string has trailing code in parens like "Stabiliser (412)"
    m = re.match(r"^(.*?)\s*[\(\[]([^()\[\]]+)[\)\]]$", clean)
    if m:
        prefix, inner = m.group(1).strip(), m.group(2).strip()
        inner_code = normalize_additive_code(inner)
        if inner_code:
            return f"{prefix} ({inner_code})" if prefix else inner_code
    return clean

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
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
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

    # role: map and validate to one of 6 roles (base, sweetener, preservative, texture, color_flavor, other)
    role_raw = str(raw.get("role") or "other").strip().lower()
    role = ROLE_MAPPING.get(role_raw, "other")
    if role not in VALID_ROLES:
        role = "other"

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

    return {
        "id": _slugify(name),
        "name": name,
        "canonical_name": name,
        "role": role,
        "one_liner": one_liner,
        "why_used": why_used,
        "origin": origin,
        "origin_type": origin,
        "how_made": how_made,
        "verified": False,
        "source": "ai",
        "status": "available",
        "confidence": "low",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
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
    canonical = normalize_card_lookup_name(ingredient_id)
    slug = _slugify(canonical)
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
    canonical = normalize_card_lookup_name(card.id or card.name)
    slug = _slugify(canonical)
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
5. role must be one of:
   - "base": bulk structure, solvent, carrier, flour, fat or oil (e.g. "water", "soya bean oil", "wheat flour")
   - "sweetener": caloric or non-caloric sweetening agent (e.g. "sugar", "stevia", "glucose syrup")
   - "preservative": prevents spoilage, inhibits oxidation, or regulates acidity (e.g. "antioxidant 319", "food acid 270", "sodium benzoate", "white vinegar")
   - "texture": thickener, emulsifier, stabilizer, gelling agent (e.g. "maize thickener 1422", "stabiliser 412", "soy lecithin", "xanthan gum")
   - "color_flavor": adds color, flavor, seasoning, or aroma (e.g. "colours 171", "mustard", "salt", "vanilla extract")
   - "other": specialized function not covered above (e.g. "calcium silicate")
   DO NOT lazily classify functional ingredients as "other". Assign the best functional category.
6. how_made must have at most 3 concise steps (each with an icon name and short text).
7. Output MUST be valid JSON only matching the schema below. No markdown code blocks, no preamble.
8. Do NOT include placeholder text, 'string', 'example', 'TODO', or schema definitions.

JSON Schema:
{
  "name": "Ingredient Name",
  "one_liner": "Concise factual one-sentence description.",
  "why_used": "Max 5 words explanation",
  "origin": "plant" | "animal" | "fermentation" | "lab" | "mineral" | "unclear",
  "role": "base" | "sweetener" | "preservative" | "texture" | "color_flavor" | "other",
  "how_made": [
    {"icon": "source_icon", "text": "Step 1 text"},
    {"icon": "process_icon", "text": "Step 2 text"},
    {"icon": "finish_icon", "text": "Step 3 text"}
  ]
}"""

BATCH_CARD_SYSTEM_PROMPT = """You are an objective food ingredient information explainer.
Provide factual, concise explanations for each ingredient specified in the user request.

MANDATORY RULES FOR EVERY CARD:
1. Absolutely NO health claims, medical claims, or nutritional benefits.
2. Absolutely NO safety claims. NEVER use the word 'safe' anywhere in any field.
3. If uncertain about any origin, step, or role, strictly specify 'unclear'.
4. why_used must be 5 words or fewer.
5. role must be one of:
   - "base": bulk structure, solvent, carrier, flour, fat or oil (e.g. "water", "soya bean oil", "wheat flour")
   - "sweetener": caloric or non-caloric sweetening agent (e.g. "sugar", "stevia", "glucose syrup")
   - "preservative": prevents spoilage, inhibits oxidation, or regulates acidity (e.g. "antioxidant 319", "food acid 270", "sodium benzoate", "white vinegar")
   - "texture": thickener, emulsifier, stabilizer, gelling agent (e.g. "maize thickener 1422", "stabiliser 412", "soy lecithin", "xanthan gum")
   - "color_flavor": adds color, flavor, seasoning, or aroma (e.g. "colours 171", "mustard", "salt", "vanilla extract")
   - "other": specialized function not covered above (e.g. "calcium silicate")
   DO NOT lazily classify functional ingredients as "other". Assign the best functional category.
6. how_made must have at most 3 concise steps (each with an icon name and short text).
7. Output MUST be valid JSON only matching the schema below. No markdown fences.

Schema:
{
  "cards": {
    "<Ingredient Name>": {
      "name": "Ingredient Name",
      "one_liner": "Concise factual one-sentence description.",
      "why_used": "Max 5 words explanation",
      "origin": "plant" | "animal" | "fermentation" | "lab" | "mineral" | "unclear",
      "role": "base" | "sweetener" | "preservative" | "texture" | "color_flavor" | "other",
      "how_made": [
        {"icon": "source", "text": "Step 1"},
        {"icon": "process", "text": "Step 2"},
        {"icon": "finish", "text": "Step 3"}
      ]
    }
  }
}
"""


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


async def _call_batch_llm(
    ingredients: list[str],
    config: Settings,
    client_ip: str,
) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """Call Cloud LLM (xAI Grok or Groq) in a single batch request."""
    if not ingredients:
        return None, None, None

    # Check rate limit
    allowed, reason = _rate_limiter.check_and_record(
        client_ip=client_ip,
        rpm_limit=config.GROK_RPM_LIMIT,
        daily_cap=config.GROK_DAILY_CAP,
    )
    if not allowed:
        logger.warning(f"Batch LLM call blocked by rate limiter: {reason}")
        return None, None, None

    user_content = f"Ingredients: {json.dumps(ingredients)}"

    # Primary: xAI Grok
    if config.XAI_API_KEY:
        headers = {
            "Authorization": f"Bearer {config.XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.XAI_MODEL,
            "messages": [
                {"role": "system", "content": BATCH_CARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }
        url = f"{config.XAI_BASE_URL.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return _extract_json_from_llm_response(content), "xai", config.XAI_MODEL
                else:
                    logger.warning(f"xAI Grok batch returned HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            logger.warning(f"xAI batch call failed: {exc}")

    # Secondary: Groq
    if config.GROQ_API_KEY:
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": BATCH_CARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        url = f"{config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return _extract_json_from_llm_response(content), "groq", config.GROQ_MODEL
                else:
                    logger.warning(f"Groq batch returned HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            logger.warning(f"Groq batch call failed: {exc}")

    # Tertiary: Local model fallback
    url = f"{config.VLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": config.VLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": BATCH_CARD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _extract_json_from_llm_response(content), "local", config.VLM_MODEL_NAME
    except Exception as exc:
        logger.info(f"Local model batch call failed: {exc}")

    return None, None, None


# ---------------------------------------------------------------------------
# Main Entry Point: get_or_generate_ingredient_card
# ---------------------------------------------------------------------------
async def get_or_generate_ingredient_card(
    ingredient_name_or_id: str,
    kb_entry: Optional[IngredientKBEntry] = None,
    client_ip: str = "127.0.0.1",
    config: Optional[Settings] = None,
) -> AIIngredientCard:
    """Retrieve card from disk cache, or generate via Grok / Local fallback and cache on disk.

    Curated KnowledgeBase lookup is completely removed. Every card is AI-generated and cached.
    """
    cfg = config or get_settings()
    canonical_name = normalize_card_lookup_name(ingredient_name_or_id)

    # 1. Check disk cache (key = normalized name, additive codes in canonical form)
    cached = get_cached_card(canonical_name, cfg)
    if cached is not None:
        return cached

    # 2. Call Cloud LLMs: xAI Grok first (Level 1)
    if cfg.XAI_API_KEY:
        raw_data = await _call_xai_grok(canonical_name, cfg, client_ip)
        if raw_data:
            norm = validate_and_normalize_ai_card(raw_data, canonical_name, llm_provider="xai", llm_model=cfg.XAI_MODEL)
            if norm:
                card = AIIngredientCard(**norm)
                save_cached_card(card, cfg)
                return card

    if cfg.GROQ_API_KEY:
        raw_data = await _call_groq_api(canonical_name, cfg, client_ip)
        if raw_data:
            norm = validate_and_normalize_ai_card(raw_data, canonical_name, llm_provider="groq", llm_model=cfg.GROQ_MODEL)
            if norm:
                card = AIIngredientCard(**norm)
                save_cached_card(card, cfg)
                return card

    # 3. Fall back to local model (Level 2)
    raw_data = await _call_local_model(canonical_name, cfg)
    if raw_data:
        norm = validate_and_normalize_ai_card(raw_data, canonical_name, llm_provider="local", llm_model=cfg.VLM_MODEL_NAME)
        if norm:
            card = AIIngredientCard(**norm)
            save_cached_card(card, cfg)
            return card

    # 4. Failed -> return unavailable card (Level 3, Never invent or mock)
    return make_unavailable_card(canonical_name)


async def get_or_generate_cards_batch(
    ingredients: list[str],
    client_ip: str = "127.0.0.1",
    config: Optional[Settings] = None,
) -> dict[str, AIIngredientCard]:
    """Retrieve cards for all chips in as few Grok calls as possible, with per-chip fallback.

    1. Checks disk cache for each ingredient.
    2. Collects all missing ingredients and requests them in a single batch LLM call.
    3. Validates and caches returned cards on disk.
    4. Falls back to individual per-chip calls for any items that failed in batch.
    5. Never invents cards; returns unavailable card if failed.
    """
    cfg = config or get_settings()
    results: dict[str, AIIngredientCard] = {}
    missing_items: list[tuple[str, str]] = []  # (orig_name, canonical_name)

    # 1. Disk cache check
    for raw in ingredients:
        clean = raw.strip()
        if not clean:
            continue
        canonical = normalize_card_lookup_name(clean)
        cached = get_cached_card(canonical, cfg)
        if cached is not None and cached.status == "available":
            results[clean] = cached
        else:
            missing_items.append((clean, canonical))

    if not missing_items:
        return results

    # 2. Batch LLM call for all missing items
    missing_canons = list(dict.fromkeys(c for _, c in missing_items))
    try:
        batch_raw = await _call_batch_llm(missing_canons, cfg, client_ip)
        if isinstance(batch_raw, tuple) and len(batch_raw) == 3:
            batch_res, batch_provider, batch_model = batch_raw
        else:
            batch_res, batch_provider, batch_model = batch_raw, "ai", "model"

        if batch_res and isinstance(batch_res, dict):
            cards_map = batch_res.get("cards", batch_res)
            if isinstance(cards_map, dict):
                for orig, canon in missing_items:
                    card_raw = cards_map.get(canon) or cards_map.get(orig)
                    if not card_raw:
                        # Case-insensitive check
                        for k, v in cards_map.items():
                            if k.lower() in (canon.lower(), orig.lower()):
                                card_raw = v
                                break
                    if card_raw and isinstance(card_raw, dict):
                        norm = validate_and_normalize_ai_card(card_raw, canon, llm_provider=batch_provider, llm_model=batch_model)
                        if norm:
                            card = AIIngredientCard(**norm)
                            save_cached_card(card, cfg)
                            results[orig] = card
    except Exception as exc:
        logger.warning(f"Batch generation error: {exc}")

    # 3. Per-chip fallback for anything still missing
    for orig, canon in missing_items:
        if orig not in results or results[orig].status != "available":
            try:
                card = await get_or_generate_ingredient_card(
                    ingredient_name_or_id=canon,
                    client_ip=client_ip,
                    config=cfg,
                )
                results[orig] = card
            except Exception as exc:
                logger.warning(f"Per-chip fallback failed for '{orig}': {exc}")
                results[orig] = make_unavailable_card(canon)

    return results
