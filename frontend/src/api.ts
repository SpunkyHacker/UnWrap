import { AIIngredientCard, AnalysisResponse, ChatResponse, DietCheckResponse, UserProfile } from './types';

const API_BASE_URL = (() => {
  // If running in browser and on a tunnel host, mobile device, or custom domain,
  // ALWAYS use relative '/api' to avoid Mixed Content (HTTPS -> HTTP) and localhost binding issues.
  if (
    typeof window !== 'undefined' &&
    window.location.hostname !== 'localhost' &&
    window.location.hostname !== '127.0.0.1'
  ) {
    return '/api';
  }

  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (
    envUrl &&
    envUrl.trim() !== '' &&
    envUrl !== 'http://localhost:8000' &&
    envUrl !== 'http://127.0.0.1:8000'
  ) {
    return envUrl.replace(/\/+$/, '');
  }

  return '/api';
})();

export async function analyzePhotos(
  files: File[],
  profile: UserProfile,
  debug: boolean = false
): Promise<AnalysisResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  formData.append('profile', JSON.stringify(profile));
  formData.append('debug', debug ? 'true' : 'false');

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = `Server error (${response.status})`;
    try {
      const errData = await response.json();
      if (errData && errData.detail) {
        errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function reanalyzeIngredients(
  ingredients: string[],
  profile: UserProfile,
  containsStatement: string = '',
  mayContainStatement: string = ''
): Promise<AnalysisResponse> {
  const payload = {
    ingredients,
    contains_statement: containsStatement,
    may_contain_statement: mayContainStatement,
    profile,
  };

  const response = await fetch(`${API_BASE_URL}/reanalyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMsg = `Server error (${response.status})`;
    try {
      const errData = await response.json();
      if (errData && errData.detail) {
        errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function fetchIngredientCard(ingredientIdOrName: string): Promise<AIIngredientCard> {
  const cleanId = encodeURIComponent(ingredientIdOrName.trim());
  const response = await fetch(`${API_BASE_URL}/ingredient/${cleanId}`);
  if (!response.ok) {
    throw new Error(`Failed to load card for ${ingredientIdOrName}`);
  }
  return response.json();
}

export async function fetchIngredientCardsBatch(
  ingredients: string[]
): Promise<Record<string, AIIngredientCard>> {
  if (!ingredients || ingredients.length === 0) {
    return {};
  }
  const response = await fetch(`${API_BASE_URL}/ingredient/batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ingredients }),
  });
  if (!response.ok) {
    throw new Error(`Failed to load batch ingredient cards (${response.status})`);
  }
  const data = await response.json();
  return data.cards || {};
}

export async function sendChatQuestion(
  ingredientId: string,
  question: string,
  cardData?: AIIngredientCard
): Promise<ChatResponse> {
  const payload: any = {
    ingredient_id: ingredientId,
    question,
  };
  if (cardData) {
    payload.ingredient_card = cardData;
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Chat request failed');
  }

  return response.json();
}

export async function performDietCheck(
  ingredients: string[],
  diets: string[],
  vegetarianType: string = 'no_egg'
): Promise<DietCheckResponse> {
  const response = await fetch(`${API_BASE_URL}/diet-check`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ingredients,
      diets,
      vegetarian_type: vegetarianType,
    }),
  });

  if (!response.ok) {
    throw new Error(`Diet check failed with status ${response.status}`);
  }

  return response.json();
}

export async function checkBackendHealth(): Promise<{
  status: string;
  vlm: any;
  apple_vision: any;
  knowledge_base: any;
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      return res.json();
    }
  } catch {
    // offline or unreachable
  }
  return {
    status: 'unreachable',
    vlm: { available: false },
    apple_vision: { available: false },
    knowledge_base: { loaded: false },
  };
}

/** Explicit demo mode data strictly behind flag */
export function getDemoAnalysis(): AnalysisResponse {
  return {
    transcription: {
      ingredients_text: 'Wheat Flour (62%), Sugar, Palm Oil, Cocoa Powder, Emulsifier (Soy Lecithin, E322), Salt, Raising Agent (Sodium Bicarbonate), Natural Vanilla Flavour.',
      contains_statement: 'Wheat, Soy',
      may_contain_statement: 'Milk, Peanuts',
      nutrition_text: 'Calories 140, Total Fat 7g',
      language: 'en',
      unreadable_parts: [],
      engine_used: 'mlx-community/Qwen2.5-VL-7B-Instruct-4bit',
      time_ms: 8740,
      fallback_occurred: false,
      corrections: [],
    },
    parsed: {
      ingredients: [
        { name: 'Wheat Flour', raw_text: 'Wheat Flour (62%)', percentage: 62, children: [], position: 0, possible_merged: false, suggested_split: [] },
        { name: 'Sugar', raw_text: 'Sugar', children: [], position: 1, possible_merged: false, suggested_split: [] },
        { name: 'Palm Oil Cocoa Powder', raw_text: 'Palm Oil Cocoa Powder', children: [], position: 2, possible_merged: true, suggested_split: ['Palm Oil', 'Cocoa Powder'] },
        { name: 'Emulsifier', raw_text: 'Emulsifier (Soy Lecithin, E322)', children: [
          { name: 'Soy Lecithin', raw_text: 'Soy Lecithin', e_number: 'E322', children: [], position: 0, possible_merged: false, suggested_split: [] }
        ], position: 3, possible_merged: false, suggested_split: [] },
        { name: 'Salt Raising Agent', raw_text: 'Salt Raising Agent', children: [], position: 4, possible_merged: true, suggested_split: ['Salt', 'Raising Agent'] },
        { name: 'Natural Vanilla Flavour', raw_text: 'Natural Vanilla Flavour', children: [], position: 5, possible_merged: false, suggested_split: [] },
      ],
      contains_allergens: ['wheat', 'soy'],
      may_contain_allergens: ['milk', 'peanuts'],
      raw_text: 'Wheat Flour, Sugar, Palm Oil Cocoa Powder...',
      has_ingredients_marker: true,
    },
    matches: [],
    allergen_result: {
      overall_status: 'might_contain',
      alerts: [
        { status: 'might_contain', allergen: 'milk', ingredient_name: 'May contain statement', reason: 'Cross-contact advisory lists Milk' },
      ],
      confidence: 'medium',
      disclaimer: 'This information is for reference only and does not constitute medical advice. Always consult product packaging.',
    },
    unknowns: ['Palm Oil Cocoa Powder', 'Salt Raising Agent'],
  };
}
