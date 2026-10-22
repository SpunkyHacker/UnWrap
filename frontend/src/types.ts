export type AllergenStatus = 'contains' | 'might_contain' | 'uncertain' | 'clear';

export interface AllergenAlert {
  status: AllergenStatus;
  allergen: string;
  ingredient_name: string;
  reason: string;
}

export interface AllergenResult {
  overall_status: AllergenStatus;
  alerts: AllergenAlert[];
  confidence: 'high' | 'medium' | 'low';
  confidence_reason?: string;
  disclaimer: string;
}

export interface HowMadeStep {
  icon: string;
  text: string;
}

export interface DietFlags {
  vegan: boolean | string;
  vegetarian: boolean | string;
  halal: boolean | string;
  jain: boolean | string;
  gluten_free: boolean | string;
}

export interface IngredientKBEntry {
  id: string;
  canonical_name: string;
  names: string[];
  e_number?: string | null;
  role: string;
  one_liner: string;
  why_used: string;
  origin_type: string;
  origin_note: string;
  how_made: HowMadeStep[];
  source_ambiguous: boolean;
  ambiguity_note: string;
  allergens_direct: string[];
  allergens_possible: string[];
  diet_flags: DietFlags;
  region_notes: string;
  sources: string[];
  verified: boolean;
  confidence: string;
}

export interface ParsedIngredient {
  name: string;
  raw_text: string;
  e_number?: string | null;
  percentage?: number | null;
  children: ParsedIngredient[];
  position: number;
  possible_merged: boolean;
  suggested_split: string[];
}

export interface MatchResult {
  parsed_name: string;
  kb_entry?: IngredientKBEntry | null;
  match_method: string;
  match_score: number;
  confidence: string;
  possible_merged?: boolean;
  suggested_split?: string[];
}

export interface TranscriptionResult {
  ingredients_text: string;
  contains_statement: string;
  may_contain_statement: string;
  nutrition_text: string;
  language: string;
  unreadable_parts: string[];
  engine_used: string;
  time_ms: number;
  fallback_occurred: boolean;
  corrections: Array<{ original: string; corrected: string; score: number }>;
}

export interface AnalysisResponse {
  transcription: TranscriptionResult;
  parsed: {
    ingredients: ParsedIngredient[];
    contains_allergens: string[];
    may_contain_allergens: string[];
    raw_text: string;
    has_ingredients_marker: boolean;
    corrections?: Array<{ original: string; corrected: string; score: number }>;
  };
  matches: MatchResult[];
  allergen_result: AllergenResult;
  unknowns: string[];
  debug_comparison?: any;
}

export interface UserProfile {
  allergens: string[];
  custom_allergens: string;
  diet_preferences: string[];
}

export interface CapturedPhoto {
  id: string;
  file: File;
  previewUrl: string;
}

export interface AICardFlags {
  vegan: 'likely_yes' | 'likely_no' | 'unclear';
  vegetarian: 'likely_yes' | 'likely_no' | 'unclear';
  halal: 'likely_yes' | 'likely_no' | 'unclear';
  jain: 'likely_yes' | 'likely_no' | 'unclear';
}

export interface AIIngredientCard {
  id: string;
  name: string;
  canonical_name: string;
  role: string;
  one_liner: string;
  why_used: string;
  origin: 'plant' | 'animal' | 'fermentation' | 'lab' | 'mineral' | 'unclear' | string;
  origin_type: string;
  how_made: HowMadeStep[];
  flags: AICardFlags;
  diet_flags?: DietFlags;
  verified: boolean;
  source: 'ai' | 'kb' | string;
  status: 'available' | 'unavailable' | string;
  confidence: string;
}

export interface ChatResponse {
  answer: string;
  suggested_questions: string[];
  ingredient_id?: string;
  source?: string;
}
