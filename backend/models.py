"""Pydantic models for the Unwrap ingredient-explainer app."""
from __future__ import annotations

import enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
class TranscriptionResult(BaseModel):
    """Result from OCR / VLM transcription of a product label."""
    ingredients_text: str = ""
    contains_statement: str = ""
    may_contain_statement: str = ""
    nutrition_text: str = ""
    language: str = "en"
    unreadable_parts: list[str] = Field(default_factory=list)
    engine_used: str = ""
    time_ms: int = 0
    fallback_occurred: bool = False
    corrections: list[dict] = Field(default_factory=list)  # [{original, corrected, score}]


class DebugComparisonResult(BaseModel):
    """Side-by-side comparison of VLM and Apple Vision transcriptions."""
    vlm_result: Optional[TranscriptionResult] = None
    apple_vision_result: Optional[TranscriptionResult] = None
    time_difference_ms: int = 0


# ---------------------------------------------------------------------------
# Parsed ingredients
# ---------------------------------------------------------------------------
class ParsedIngredient(BaseModel):
    """A single parsed ingredient, possibly with nested children."""
    name: str
    raw_text: str = ""
    e_number: Optional[str] = None
    percentage: Optional[float] = None
    children: list[ParsedIngredient] = Field(default_factory=list)
    position: int = 0
    possible_merged: bool = False
    suggested_split: list[str] = Field(default_factory=list)


class ParsedResult(BaseModel):
    """Full result of parsing an ingredient text block."""
    ingredients: list[ParsedIngredient] = Field(default_factory=list)
    contains_allergens: list[str] = Field(default_factory=list)
    may_contain_allergens: list[str] = Field(default_factory=list)
    raw_text: str = ""
    has_ingredients_marker: bool = True
    corrections: list[dict] = Field(default_factory=list)  # [{original, corrected, score}]


# ---------------------------------------------------------------------------
# Knowledge-base entry
# ---------------------------------------------------------------------------
class HowMadeStep(BaseModel):
    """One step in the 'how it's made' explainer."""
    icon: str
    text: str


class DietFlags(BaseModel):
    """Dietary-compatibility flags."""
    vegan: bool | str = False          # True / False / "depends"
    vegetarian: bool | str = False
    halal: bool | str = True
    jain: bool | str = False
    gluten_free: bool | str = True


class AICardFlags(BaseModel):
    """Dietary suitability flags for AI-generated ingredient cards."""
    vegan: str = "unclear"        # "likely_yes" | "likely_no" | "unclear"
    vegetarian: str = "unclear"   # "likely_yes" | "likely_no" | "unclear"
    halal: str = "unclear"        # "likely_yes" | "likely_no" | "unclear"
    jain: str = "unclear"         # "likely_yes" | "likely_no" | "unclear"


class AIIngredientCard(BaseModel):
    """Schema for on-the-fly AI-generated or cached ingredient cards.
    
    Dietary suitability flags have been removed; diet conflicts are checked
    dynamically via the /diet-check endpoint.
    """
    id: str = ""
    name: str = ""
    canonical_name: str = ""
    role: str = "other"
    one_liner: str = ""
    why_used: str = ""  # max 5 words
    origin: str = "unclear"  # plant | animal | fermentation | lab | mineral | unclear
    origin_type: str = "unclear"
    how_made: list[HowMadeStep] = Field(default_factory=list)  # max 3 steps
    verified: bool = False
    source: str = "ai"  # "ai" | "kb"
    status: str = "available"  # "available" | "unavailable"
    confidence: str = "low"
    llm_provider: Optional[str] = None  # e.g. "xai", "groq", "local"
    llm_model: Optional[str] = None     # e.g. "grok-2-latest", "qwen/qwen3.8-27b"


class IngredientKBEntry(BaseModel):
    """Full knowledge-base record for one ingredient."""
    id: str
    canonical_name: str
    name: Optional[str] = None
    names: list[str] = Field(default_factory=list)
    e_number: Optional[str] = None
    role: str = "other"
    one_liner: str = ""
    why_used: str = ""
    origin: Optional[str] = None
    origin_type: str = "unclear"
    origin_note: str = ""
    how_made: list[HowMadeStep] = Field(default_factory=list)
    source_ambiguous: bool = False
    ambiguity_note: str = ""
    allergens_direct: list[str] = Field(default_factory=list)
    allergens_possible: list[str] = Field(default_factory=list)
    diet_flags: DietFlags = Field(default_factory=DietFlags)
    flags: Optional[dict[str, str]] = None
    region_notes: str = ""
    sources: list[str] = Field(default_factory=list)
    verified: bool = False
    confidence: str = "medium"
    source: str = "kb"
    status: str = "available"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
class MatchResult(BaseModel):
    """Result of matching a parsed ingredient to the knowledge base."""
    parsed_name: str
    kb_entry: Optional[IngredientKBEntry] = None
    match_method: str = "unknown"   # exact | e_number | fuzzy | llm_pick | unknown
    match_score: float = 0.0
    confidence: str = "low"
    possible_merged: bool = False
    suggested_split: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Allergen checking
# ---------------------------------------------------------------------------
class AllergenStatus(str, enum.Enum):
    """Severity levels for allergen alerts."""
    CONTAINS = "contains"
    MIGHT_CONTAIN = "might_contain"
    UNCERTAIN = "uncertain"
    CLEAR = "clear"


class AllergenAlert(BaseModel):
    """A single allergen alert raised by the engine."""
    status: AllergenStatus
    allergen: str
    ingredient_name: str
    reason: str


class AllergenResult(BaseModel):
    """Full allergen-check result."""
    overall_status: AllergenStatus = AllergenStatus.CLEAR
    alerts: list[AllergenAlert] = Field(default_factory=list)
    confidence: str = "high"  # "high" | "medium" | "low"
    confidence_reason: str = ""
    disclaimer: str = (
        "This information is for reference only and does not constitute "
        "medical advice. Always consult product packaging and your "
        "healthcare provider for allergy-related decisions."
    )


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------
class AnalysisResult(BaseModel):
    """Complete analysis result combining all stages."""
    transcription: Optional[TranscriptionResult] = None
    parsed: Optional[ParsedResult] = None
    matches: list[MatchResult] = Field(default_factory=list)
    allergen_result: Optional[AllergenResult] = None
    unknowns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------
class UserProfile(BaseModel):
    """User's allergy and dietary profile."""
    allergens: list[str] = Field(default_factory=list)
    custom_allergens: str = ""
    diet_preferences: list[str] = Field(default_factory=list)
    vegetarian_type: str = "no_egg"  # "no_egg" (default) or "eggs_allowed"


# ---------------------------------------------------------------------------
# Dietary Check Models
# ---------------------------------------------------------------------------
class DietStatus(str, enum.Enum):
    """Dietary conflict status."""
    CONFLICT = "CONFLICT"
    MAY_CONFLICT = "MAY_CONFLICT"
    NO_LISTED_CONFLICT = "NO_LISTED_CONFLICT"


class IngredientDietResult(BaseModel):
    """Dietary check result for a single ingredient."""
    ingredient: str
    status: DietStatus
    reason: str = ""  # max 8 words


class DietCheckRequest(BaseModel):
    """Payload for checking ingredient compatibility against dietary preferences.
    
    Contains strictly ingredient names and diet choices. NEVER includes allergens,
    images, or OCR transcriptions.
    """
    ingredients: list[str] = Field(default_factory=list)
    diets: list[str] = Field(default_factory=list)
    vegetarian_type: str = "no_egg"  # "no_egg" (default) or "eggs_allowed"


class DietCheckResponse(BaseModel):
    """Aggregate response for dietary compatibility check."""
    results: list[IngredientDietResult] = Field(default_factory=list)
    overall_status: DietStatus = DietStatus.NO_LISTED_CONFLICT
    conflicts_count: int = 0
    may_conflicts_count: int = 0
    status: str = "available"  # "available" | "unavailable"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------
class BatchIngredientsRequest(BaseModel):
    """Request payload to retrieve or generate ingredient cards in batch."""
    ingredients: list[str] = Field(default_factory=list)


class ReanalyzeRequest(BaseModel):
    """Request payload to re-run matching and allergen checking on user-edited ingredients."""
    ingredients: list[str] = Field(default_factory=list)
    contains_statement: str = ""
    may_contain_statement: str = ""
    profile: UserProfile = Field(default_factory=UserProfile)


class ChatRequest(BaseModel):
    """Request for grounded QA on a single ingredient card."""
    ingredient_id: Optional[str] = None
    ingredient_card: Optional[Union[AIIngredientCard, IngredientKBEntry, dict[str, Any]]] = None
    question: str = Field(..., max_length=300)
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response containing bounded answer and follow-up suggestion chips."""
    answer: str
    suggested_questions: list[str] = Field(default_factory=list)
    ingredient_id: Optional[str] = None
    source: str = "card"
