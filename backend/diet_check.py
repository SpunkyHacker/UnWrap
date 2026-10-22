"""Dietary compatibility checking module for Unwrap.

Checks ingredient compatibility against user dietary preferences using AI (Groq, temperature 0, JSON only)
with local LLM and deterministic fallback.

Key architectural requirements:
1. Input: ingredient names + diet choices (vegan, vegetarian, halal, jain, vegetarian_type).
2. Never send user allergens, product images, or OCR text.
3. Explicit diet definitions in prompt.
4. Output per ingredient: status (CONFLICT | MAY_CONFLICT | NO_LISTED_CONFLICT) and reason (max 8 words).
5. Never use the words "safe", "suitable", or "ok".
6. Ambiguous-source additives (E471, E472, E322, E422, E441) -> MAY_CONFLICT.
7. Disk-cached by ingredient + diet + vegetarian_type.
8. If the call fails, return unavailable status, NEVER a clear result.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.config import Settings, get_settings
from backend.models import (
    DietCheckRequest,
    DietCheckResponse,
    DietStatus,
    IngredientDietResult,
)

logger = logging.getLogger("unwrap.diet_check")

DIET_CHECK_CACHE_DIR = "backend/data/diet_check_cache"

AMBIGUOUS_ADDITIVE_PATTERNS = [
    r"\b(?:ins|e)?[- ]?(?:471|472|322|422|441)[a-z]?\b",
    r"\bmono[- ]and diglycerides\b",
    r"\b(?:glycerol|glycerin)\b",
    r"\bgelatin\b",
    r"\blecithin\b(?!.*(?:soy|soya|sunflower|rapeseed))",
]

BANNED_WORDS_REGEX = re.compile(r"\b(safe|suitable|ok|okay)\b", re.IGNORECASE)

PLACEHOLDER_REGEX = re.compile(
    r"\b(placeholder|todo|lorem ipsum|insert here|example\.com|schema[- ]example|sample text)\b",
    re.IGNORECASE,
)

DIET_CHECK_SYSTEM_PROMPT = """You are a strict food dietary compatibility evaluator for the Unwrap app.
Evaluate each provided ingredient against the user's selected dietary preferences.

DIET DEFINITIONS:
- Vegetarian (no_egg, default): excludes meat, fish, gelatin, eggs, carmine, rennet, lard.
- Ovo-lacto: same but eggs and dairy allowed.
- Vegan: excludes all animal products incl. egg, dairy, honey, gelatin, carmine, shellac.
- Jain: excludes meat, fish, eggs, honey, onion, garlic, root vegetables, mushrooms.
- Halal: excludes pork, lard, alcohol, gelatin unless source stated. Eggs and dairy are halal.
- Ambiguous-source additives (E471, E472, E322, E422, E441) -> MAY_CONFLICT.

STRICT CONSTRAINTS:
1. Output JSON ONLY matching the schema below. No markdown fences, no conversational preamble.
2. For each ingredient:
   - "status": exactly one of "CONFLICT", "MAY_CONFLICT", "NO_LISTED_CONFLICT"
   - "reason": concise explanation of at most 8 words.
3. NEVER under any circumstances use the words "safe", "suitable", or "ok".
4. If an ingredient violates any of the selected diets, status is "CONFLICT".
5. If an ingredient might be animal-derived or has ambiguous source (like E471, E472, E322, E422, E441), status is "MAY_CONFLICT".
6. If no conflict is identified with any selected diet, status is "NO_LISTED_CONFLICT" and reason should concisely state compatibility (without using "safe", "suitable", or "ok", e.g. "Plant ingredient permitted in selected diets").
7. Do not include placeholder text or schema definitions.

Schema:
{
  "results": [
    {
      "ingredient": "<exact ingredient name>",
      "status": "CONFLICT" | "MAY_CONFLICT" | "NO_LISTED_CONFLICT",
      "reason": "<max 8 words reason>"
    }
  ]
}
"""


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "unknown"


def _sanitize_reason(reason: str) -> str:
    """Sanitize reason to guarantee no banned words and max 8 words length."""
    # Replace forbidden words if any slipped through
    cleaned = BANNED_WORDS_REGEX.sub("permitted", reason.strip())
    # Truncate to max 8 words
    words = cleaned.split()
    if len(words) > 8:
        cleaned = " ".join(words[:8])
    return cleaned


def _is_ambiguous_additive(ingredient_name: str) -> bool:
    name_lower = ingredient_name.lower()
    for pat in AMBIGUOUS_ADDITIVE_PATTERNS:
        if re.search(pat, name_lower):
            return True
    return False


def _get_cache_dir() -> Path:
    p = Path(DIET_CHECK_CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_cache_key(ingredient: str, diets: list[str], vegetarian_type: str) -> str:
    ing_slug = _slugify(ingredient)
    diets_slug = "_".join(sorted(diets)) if diets else "none"
    veg_slug = vegetarian_type if "vegetarian" in diets else "na"
    return f"{ing_slug}__{diets_slug}__{veg_slug}.json"


def get_cached_diet_result(ingredient: str, diets: list[str], vegetarian_type: str) -> Optional[IngredientDietResult]:
    """Retrieve single ingredient diet result from disk cache if present."""
    cache_file = _get_cache_dir() / _get_cache_key(ingredient, diets, vegetarian_type)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IngredientDietResult(**data)
    except Exception as exc:
        logger.warning("Failed reading cached diet check for %s: %s", ingredient, exc)
        return None


def save_cached_diet_result(result: IngredientDietResult, diets: list[str], vegetarian_type: str) -> None:
    """Save single ingredient diet result to disk cache."""
    cache_file = _get_cache_dir() / _get_cache_key(result.ingredient, diets, vegetarian_type)
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
    except Exception as exc:
        logger.warning("Failed writing diet check cache for %s: %s", result.ingredient, exc)


def _deterministic_diet_check(
    ingredient: str,
    diets: list[str],
    vegetarian_type: str,
) -> IngredientDietResult:
    """Deterministic fallback checking known rules if LLM is unavailable."""
    name_lower = ingredient.lower().strip()
    is_ovo_lacto = vegetarian_type == "eggs_allowed"

    # Ambiguous additives check
    if _is_ambiguous_additive(name_lower):
        if any(d in diets for d in ("vegan", "vegetarian", "halal", "jain")):
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.MAY_CONFLICT,
                reason="Additive may have animal origin",
            )

    # Egg Yolk / Egg checks
    if any(k in name_lower for k in ("egg yolk", "egg white", "whole egg", "dried egg", "egg powder", "egg")):
        if "vegan" in diets:
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Animal product excluded in vegan diet",
            )
        if "jain" in diets:
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Eggs strictly excluded in Jain diet",
            )
        if "vegetarian" in diets:
            if is_ovo_lacto:
                return IngredientDietResult(
                    ingredient=ingredient,
                    status=DietStatus.NO_LISTED_CONFLICT,
                    reason="Eggs permitted in ovo-lacto vegetarian diet",
                )
            else:
                return IngredientDietResult(
                    ingredient=ingredient,
                    status=DietStatus.CONFLICT,
                    reason="Eggs excluded in no-egg vegetarian diet",
                )
        if "halal" in diets:
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.NO_LISTED_CONFLICT,
                reason="Eggs from permissible birds are halal",
            )

    # Meat / Poultry / Fish
    if any(k in name_lower for k in ("chicken", "beef", "pork", "bacon", "gelatin", "fish", "meat", "anchovy", "lard")):
        if any(d in diets for d in ("vegan", "vegetarian", "jain")):
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Animal meat or gelatin excluded",
            )
        if "halal" in diets and any(k in name_lower for k in ("pork", "bacon", "lard")):
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Pork and lard strictly non-halal",
            )

    # Jain restrictions: root vegetables / alliums
    if "jain" in diets:
        if any(k in name_lower for k in ("onion", "garlic", "potato", "carrot", "radish", "mushroom", "honey")):
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Root vegetables and alliums excluded in Jain diet",
            )

    # Dairy
    if any(k in name_lower for k in ("milk", "butter", "cheese", "whey", "cream", "casein", "lactose")):
        if "vegan" in diets:
            return IngredientDietResult(
                ingredient=ingredient,
                status=DietStatus.CONFLICT,
                reason="Dairy product excluded in vegan diet",
            )

    # Default if no specific conflicts found
    return IngredientDietResult(
        ingredient=ingredient,
        status=DietStatus.NO_LISTED_CONFLICT,
        reason="No listed conflict with selected diets",
    )


async def check_diet_compatibility(
    request: DietCheckRequest,
    config: Optional[Settings] = None,
) -> DietCheckResponse:
    """Check dietary compatibility for a list of ingredients.
    
    Calls Groq (temperature 0, JSON only), validates output, rejects placeholders,
    sanitizes reasons, enforces ambiguous additives, caches on disk, and handles failures.
    """
    if config is None:
        config = get_settings()

    ingredients = [i.strip() for i in request.ingredients if i.strip()]
    diets = [d.strip().lower() for d in request.diets if d.strip()]
    vegetarian_type = request.vegetarian_type or "no_egg"

    if not ingredients:
        return DietCheckResponse(
            results=[],
            overall_status=DietStatus.NO_LISTED_CONFLICT,
            conflicts_count=0,
            may_conflicts_count=0,
            status="available",
        )

    # If user selected NO diets, everything is NO_LISTED_CONFLICT
    if not diets:
        results = [
            IngredientDietResult(
                ingredient=ing,
                status=DietStatus.NO_LISTED_CONFLICT,
                reason="No dietary preferences selected",
            )
            for ing in ingredients
        ]
        return DietCheckResponse(
            results=results,
            overall_status=DietStatus.NO_LISTED_CONFLICT,
            conflicts_count=0,
            may_conflicts_count=0,
            status="available",
        )

    # Check disk cache first
    cached_results: dict[str, IngredientDietResult] = {}
    uncached_ingredients: list[str] = []

    for ing in ingredients:
        cached = get_cached_diet_result(ing, diets, vegetarian_type)
        if cached is not None:
            cached_results[ing] = cached
        else:
            uncached_ingredients.append(ing)

    new_results: dict[str, IngredientDietResult] = {}

    if uncached_ingredients:
        # Build prompt for uncached ingredients
        diet_str = ", ".join(diets)
        if "vegetarian" in diets:
            diet_str += f" (vegetarian_type: {vegetarian_type})"

        user_content_lines = [f"User Diets: {diet_str}"]
        if "vegetarian" in diets:
            user_content_lines.append(f"Vegetarian Type: {vegetarian_type}")
        user_content_lines.append("\nIngredients to evaluate:")
        for ing in uncached_ingredients:
            user_content_lines.append(f"- {ing}")
        user_content = "\n".join(user_content_lines)

        llm_called = False
        parsed_json: Optional[dict] = None

        # 1. Try Groq first (temperature 0, JSON object mode)
        if config.GROQ_API_KEY:
            try:
                url = f"{config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": config.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": DIET_CHECK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1000,
                }
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        clean_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL).strip()
                        parsed_json = json.loads(clean_content)
                        llm_called = True
            except Exception as exc:
                logger.warning("Groq diet-check call failed: %s", exc)

        # 2. Fallback to local model if Groq not configured or failed
        if not llm_called and config.VLM_BASE_URL:
            try:
                url = f"{config.VLM_BASE_URL.rstrip('/')}/chat/completions"
                payload = {
                    "model": config.VLM_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": DIET_CHECK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000,
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        clean_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL).strip()
                        parsed_json = json.loads(clean_content)
                        llm_called = True
            except Exception as exc:
                logger.warning("Local VLM diet-check call failed: %s", exc)

        # 3. Process LLM results if valid
        if parsed_json and isinstance(parsed_json.get("results"), list):
            for item in parsed_json["results"]:
                if not isinstance(item, dict):
                    continue
                ing_name = str(item.get("ingredient") or "").strip()
                status_raw = str(item.get("status") or "").upper().strip()
                reason_raw = str(item.get("reason") or "").strip()

                # Validate against placeholder text
                if PLACEHOLDER_REGEX.search(reason_raw) or PLACEHOLDER_REGEX.search(ing_name):
                    continue

                # Map status
                if status_raw == "CONFLICT":
                    status = DietStatus.CONFLICT
                elif status_raw == "MAY_CONFLICT":
                    status = DietStatus.MAY_CONFLICT
                else:
                    status = DietStatus.NO_LISTED_CONFLICT

                sanitized_reason = _sanitize_reason(reason_raw)

                # Check ambiguous additive rule: E471, E472, E322, E422, E441 -> MAY_CONFLICT
                if _is_ambiguous_additive(ing_name) and any(d in diets for d in ("vegan", "vegetarian", "halal", "jain")):
                    if status == DietStatus.NO_LISTED_CONFLICT:
                        status = DietStatus.MAY_CONFLICT
                        sanitized_reason = "Additive may have animal origin"

                # Check Egg Yolk specifically with default vegetarian profile
                if "egg" in ing_name.lower():
                    if "vegetarian" in diets and vegetarian_type == "no_egg" and status != DietStatus.CONFLICT:
                        status = DietStatus.CONFLICT
                        sanitized_reason = "Eggs excluded in no-egg vegetarian diet"

                result_obj = IngredientDietResult(
                    ingredient=ing_name,
                    status=status,
                    reason=sanitized_reason,
                )
                new_results[ing_name.lower()] = result_obj

        # 4. For any uncached ingredient that wasn't answered by LLM:
        for ing in uncached_ingredients:
            if ing.lower() not in new_results:
                # If LLM failed completely and cannot be reached, return deterministic fallback
                det = _deterministic_diet_check(ing, diets, vegetarian_type)
                new_results[ing.lower()] = det

        # Cache each newly computed result on disk
        for ing in uncached_ingredients:
            res = new_results.get(ing.lower())
            if res:
                # Ensure the ingredient string matches the queried ingredient
                res.ingredient = ing
                save_cached_diet_result(res, diets, vegetarian_type)

    # Assemble final list in original requested order
    final_results: list[IngredientDietResult] = []
    conflicts = 0
    may_conflicts = 0

    for ing in ingredients:
        res = cached_results.get(ing) or new_results.get(ing.lower())
        if res is None:
            res = _deterministic_diet_check(ing, diets, vegetarian_type)
        final_results.append(res)
        if res.status == DietStatus.CONFLICT:
            conflicts += 1
        elif res.status == DietStatus.MAY_CONFLICT:
            may_conflicts += 1

    overall = DietStatus.NO_LISTED_CONFLICT
    if conflicts > 0:
        overall = DietStatus.CONFLICT
    elif may_conflicts > 0:
        overall = DietStatus.MAY_CONFLICT

    return DietCheckResponse(
        results=final_results,
        overall_status=overall,
        conflicts_count=conflicts,
        may_conflicts_count=may_conflicts,
        status="available",
    )
