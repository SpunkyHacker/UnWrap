"""FastAPI application entry-point for Unwrap.

Provides Phase 3 API endpoints:
- POST /analyze: Transcribes product photos, parses ingredients, matches KB, checks allergens.
- POST /reanalyze: Re-checks user-edited ingredient list and profile (skips vision).
- GET  /ingredient/{id}: Fetches full ingredient card or drafts unverified card for unknown items.
- POST /chat: Bounded, card-grounded QA with 3 suggested follow-up chips.
- GET  /health: Live health check pinging VLM server, Apple Vision, and knowledge base.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.ai_cards import get_or_generate_ingredient_card
from backend.allergen_engine import check_allergens
from backend.chat import answer_ingredient_question
from backend.config import Settings, get_settings
from backend.matcher import KnowledgeBase, _log_unknown, match_ingredients
from backend.models import (
    AllergenResult,
    AllergenStatus,
    AnalysisResult,
    ChatRequest,
    ChatResponse,
    DietFlags,
    HowMadeStep,
    IngredientKBEntry,
    MatchResult,
    ParsedIngredient,
    ParsedResult,
    ReanalyzeRequest,
    TranscriptionResult,
    UserProfile,
)
from backend.ocr import APPLE_VISION_AVAILABLE
from backend.parser import parse_ingredients
from backend.transcription import compare_engines, transcribe_images

logger = logging.getLogger("unwrap")

# Maximum upload limits
MAX_IMAGES = 1
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per image
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "application/octet-stream",  # Fallback for some mobile browsers
}


# ---------------------------------------------------------------------------
# Lifespan: initialize KnowledgeBase and application state
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings and knowledge base on startup."""
    app.state.config = get_settings()
    app.state.kb = KnowledgeBase()
    logger.info(
        "Unwrap initialized with %d ingredients in KnowledgeBase.",
        len(app.state.kb.entries),
    )
    yield


app = FastAPI(
    title="Unwrap API",
    description="Deterministic ingredient-explainer and allergen alert API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Vite frontend, local development, and tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Unverified Card Drafting Helper
# ---------------------------------------------------------------------------
def draft_unverified_card(ingredient_name_or_id: str) -> IngredientKBEntry:
    """Draft a clearly-labelled 'unverified' card for an ingredient missing from the KB.

    Marked with verified=False, confidence='low', and recorded to unknown_ingredients.jsonl.
    """
    clean_name = ingredient_name_or_id.replace("-", " ").replace("_", " ").strip().title()
    slug_id = re.sub(r"[^a-z0-9_]+", "_", clean_name.lower()).strip("_")

    # Record to unknown log
    _log_unknown(clean_name)

    return IngredientKBEntry(
        id=slug_id,
        canonical_name=clean_name,
        names=[clean_name.lower()],
        role="other",
        one_liner=f"Unverified record for {clean_name}.",
        why_used="Commonly used in food products for flavor, texture, stability, or processing.",
        origin_type="unclear",
        origin_note="Source not verified in curated knowledge base.",
        how_made=[
            HowMadeStep(icon="box", text="Raw ingredients prepared"),
            HowMadeStep(icon="factory", text="Standard processing applied"),
            HowMadeStep(icon="package", text="Formulated into food"),
        ],
        source_ambiguous=True,
        ambiguity_note="Origin could be plant, animal, microbial, or synthetic.",
        allergens_direct=[],
        allergens_possible=[],
        diet_flags=DietFlags(
            vegan="depends",
            vegetarian="depends",
            halal="depends",
            jain="depends",
            gluten_free="depends",
        ),
        region_notes="",
        sources=[],
        verified=False,
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Live health check pinging VLM endpoint, Apple Vision, and knowledge base."""
    config: Settings = getattr(app.state, "config", get_settings())
    kb: KnowledgeBase = getattr(app.state, "kb", KnowledgeBase())

    vlm_status = {
        "available": False,
        "endpoint": config.VLM_BASE_URL,
        "model": config.VLM_MODEL_NAME,
        "latency_ms": None,
        "error": None,
    }

    # Ping VLM endpoint
    try:
        start_t = time.perf_counter()
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{config.VLM_BASE_URL.rstrip('/')}/models")
            vlm_status["latency_ms"] = int((time.perf_counter() - start_t) * 1000)
            if resp.status_code == 200:
                vlm_status["available"] = True
                models_data = resp.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    loaded = [m.get("id") for m in models_data["data"] if m.get("loaded", False)]
                    if loaded:
                        vlm_status["model"] = loaded[0]
                        vlm_status["actual_loaded_model"] = loaded[0]
                    else:
                        vlm_status["model"] = models_data["data"][0].get("id", config.VLM_MODEL_NAME)
                    vlm_status["available_models"] = [m.get("id") for m in models_data["data"]]
            else:
                vlm_status["error"] = f"HTTP {resp.status_code}"
    except Exception as exc:
        vlm_status["error"] = str(exc)

    overall_status = "healthy" if (APPLE_VISION_AVAILABLE or vlm_status["available"]) else "degraded"

    return {
        "status": overall_status,
        "vlm": vlm_status,
        "apple_vision": {
            "available": APPLE_VISION_AVAILABLE,
            "backend": "macOS PyObjC VNRecognizeTextRequest",
        },
        "knowledge_base": {
            "loaded": True,
            "ingredient_count": len(kb.entries),
            "indexed_names_count": len(kb._all_names),
        },
    }


@app.post("/analyze")
async def analyze_images(
    files: list[UploadFile] = File(...),
    profile: Optional[str] = Form(None),
    debug: bool = Form(False),
):
    """Primary analysis pipeline for uploaded product photos.

    1. Validates upload limits (max 6 images, max 10MB each, jpeg/png/webp/heic).
    2. Processes images in temporary storage (purged immediately after request for privacy).
    3. Runs transcription with VLM first and Apple Vision OCR fallback.
    4. Parses ingredients, handles merged aliases, extracts allergens.
    5. Runs deterministic matcher and allergen engine.
    6. Returns complete analysis result.
    """
    kb: KnowledgeBase = app.state.kb
    config: Settings = app.state.config

    # Validation: File count
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly 1 photo is required.",
        )
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only 1 photo is permitted per analysis (received {len(files)}).",
        )

    # Parse user profile
    user_profile = UserProfile()
    if profile:
        try:
            profile_data = json.loads(profile)
            user_profile = UserProfile(**profile_data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid profile JSON format: {exc}",
            )

    # Save to transient tempdir to guarantee privacy (auto-deleted)
    saved_paths: list[Path] = []
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    try:
        for idx, upload_file in enumerate(files):
            # Check file extension
            filename = upload_file.filename or f"image_{idx}.jpg"
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"File '{filename}' has unsupported extension '{suffix}'. "
                        f"Accepted types: JPEG, PNG, WebP, HEIC."
                    ),
                )

            # Read content with 10MB size limit
            content = await upload_file.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{filename}' exceeds maximum allowed size of 10MB.",
                )
            if len(content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{filename}' is empty.",
                )

            file_path = temp_dir_path / f"upload_{idx}{suffix}"
            file_path.write_bytes(content)
            saved_paths.append(file_path)

        # 1. Transcribe images
        transcription: TranscriptionResult = await transcribe_images(
            saved_paths,
            kb=kb,
            config=config,
            save_debug=debug,
            temp_dir=temp_dir_path,
        )

        # 2. Parse ingredients text
        parsed: ParsedResult = parse_ingredients(transcription.ingredients_text)

        # Merge contains / may_contain statements from transcription if not in parsed
        if transcription.contains_statement:
            from backend.parser import _split_allergen_list
            extra_contains = _split_allergen_list(transcription.contains_statement)
            for c in extra_contains:
                if c not in parsed.contains_allergens:
                    parsed.contains_allergens.append(c)

        if transcription.may_contain_statement:
            from backend.parser import _split_allergen_list
            extra_may = _split_allergen_list(transcription.may_contain_statement)
            for m in extra_may:
                if m not in parsed.may_contain_allergens:
                    parsed.may_contain_allergens.append(m)

        # 3. Match ingredients against KnowledgeBase
        matches: list[MatchResult] = match_ingredients(parsed.ingredients, kb)

        # 4. Check allergens deterministically
        allergen_res: AllergenResult = check_allergens(
            matches=matches,
            profile=user_profile,
            parsed=parsed,
            transcription=transcription,
        )

        # 5. Extract unknowns
        unknowns = [m.parsed_name for m in matches if m.match_method == "unknown"]

        # 6. Optional debug side-by-side comparison
        debug_comparison = None
        if debug and saved_paths:
            debug_comparison = await compare_engines(
                saved_paths[0],
                kb=kb,
                config=config,
                save_debug=True,
                temp_dir=temp_dir_path,
            )

        return {
            "transcription": transcription,
            "parsed": parsed,
            "matches": matches,
            "allergen_result": allergen_res,
            "unknowns": unknowns,
            "debug_comparison": debug_comparison,
        }

    finally:
        # Privacy guarantee: Always clean up temporary directory immediately
        # unless debug mode specifically requests keeping it.
        if not debug:
            temp_dir_obj.cleanup()
        else:
            logger.info("Debug mode enabled: keeping upload files at %s", temp_dir_path)


@app.post("/reanalyze", response_model=AnalysisResult)
async def reanalyze_ingredients(payload: ReanalyzeRequest):
    """Reanalyze user-edited ingredients and profile (skips vision).

    Used when the user edits, adds, or removes ingredient chips on the confirm screen.
    """
    kb: KnowledgeBase = app.state.kb

    # Construct parsed ingredients from the user's edited list
    parsed_items: list[ParsedIngredient] = []
    for idx, raw_name in enumerate(payload.ingredients):
        name = raw_name.strip()
        if not name:
            continue
        parsed_items.append(
            ParsedIngredient(
                name=name,
                raw_text=name,
                position=idx,
            )
        )

    # Parse allergen statements
    contains_allergens: list[str] = []
    if payload.contains_statement:
        from backend.parser import _split_allergen_list
        contains_allergens = _split_allergen_list(payload.contains_statement)

    may_contain_allergens: list[str] = []
    if payload.may_contain_statement:
        from backend.parser import _split_allergen_list
        may_contain_allergens = _split_allergen_list(payload.may_contain_statement)

    parsed = ParsedResult(
        ingredients=parsed_items,
        contains_allergens=contains_allergens,
        may_contain_allergens=may_contain_allergens,
        raw_text=", ".join(payload.ingredients),
        has_ingredients_marker=True,
    )

    # Run deterministic matching ladder
    matches = match_ingredients(parsed.ingredients, kb)

    # Run allergen check
    allergen_res = check_allergens(
        matches=matches,
        profile=payload.profile,
        parsed=parsed,
        transcription=None,
    )

    unknowns = [m.parsed_name for m in matches if m.match_method == "unknown"]

    return AnalysisResult(
        transcription=None,
        parsed=parsed,
        matches=matches,
        allergen_result=allergen_res,
        unknowns=unknowns,
    )


@app.get("/ingredient/{ingredient_id}")
async def get_ingredient_card(
    ingredient_id: str,
    draft_if_missing: bool = Query(True, description="Draft or generate card if not found"),
    request: Request = None,
):
    """Retrieve full ingredient card from KB cache, or generate via Grok / Local fallback and cache on disk."""
    kb: KnowledgeBase = app.state.kb
    config: Settings = getattr(app.state, "config", get_settings())

    # 1. Try exact ID lookup in KB (existing 69 entries act as cache)
    entry = kb.get(ingredient_id)
    if entry is not None:
        return entry

    # 2. Try normalized name lookup in KB
    entry = kb.lookup_name(ingredient_id)
    if entry is not None:
        return entry

    # 3. If not found and generation disabled: 404
    if not draft_if_missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingredient '{ingredient_id}' not found in knowledge base.",
        )

    # 4. Generate with Grok / local fallback or retrieve from disk cache
    client_ip = request.client.host if (request and request.client) else "127.0.0.1"
    card = await get_or_generate_ingredient_card(
        ingredient_name_or_id=ingredient_id,
        kb_entry=None,
        client_ip=client_ip,
        config=config,
    )
    return card


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ingredient(payload: ChatRequest):
    """Grounded Q&A strictly bounded to the supplied ingredient card.

    Card text is treated as data, never instructions. Answers are capped in length
    and accompanied by 3 contextual follow-up questions.
    """
    kb: KnowledgeBase = app.state.kb
    config: Settings = app.state.config

    # Resolve card
    card = payload.ingredient_card
    if card is None and payload.ingredient_id:
        card = kb.get(payload.ingredient_id) or kb.lookup_name(payload.ingredient_id)
        if card is None:
            # Draft unverified card
            card = draft_unverified_card(payload.ingredient_id)

    if card is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either ingredient_id or ingredient_card must be provided.",
        )

    # Generate strictly-grounded answer
    response = answer_ingredient_question(
        card=card,
        question=payload.question,
        history=payload.history,
        config=config,
    )
    return response
