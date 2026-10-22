import { UserProfile } from './types';

const STORAGE_KEY_PROFILE = 'unwrap_user_profile';
const STORAGE_KEY_FIRST_TIME = 'unwrap_onboarding_completed';

export const DEFAULT_ALLERGEN_OPTIONS = [
  { id: 'milk', label: 'Milk / Dairy', icon: '🥛' },
  { id: 'egg', label: 'Egg', icon: '🥚' },
  { id: 'peanut', label: 'Peanuts', icon: '🥜' },
  { id: 'tree_nuts', label: 'Tree Nuts', icon: '🌰' },
  { id: 'wheat', label: 'Wheat', icon: '🌾' },
  { id: 'gluten', label: 'Gluten (all grains)', icon: '🍞' },
  { id: 'soy', label: 'Soy', icon: '🌱' },
  { id: 'sesame', label: 'Sesame', icon: '🥯' },
  { id: 'fish', label: 'Fish', icon: '🐟' },
  { id: 'shellfish', label: 'Shellfish', icon: '🦐' },
  { id: 'mustard', label: 'Mustard', icon: '🌭' },
  { id: 'celery', label: 'Celery', icon: '🥬' },
  { id: 'sulphites', label: 'Sulphites', icon: '🍷' },
  { id: 'lupin', label: 'Lupin', icon: '🌿' },
  { id: 'molluscs', label: 'Molluscs', icon: '🦪' },
];

export const DEFAULT_DIET_OPTIONS = [
  { id: 'vegan', label: 'Vegan' },
  { id: 'vegetarian', label: 'Vegetarian' },
  { id: 'gluten_free', label: 'Gluten-Free' },
  { id: 'halal', label: 'Halal' },
  { id: 'jain', label: 'Jain' },
];

export function getStoredProfile(): UserProfile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_PROFILE);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch (err) {
    console.error('Failed reading user profile from storage:', err);
  }
  return {
    allergens: [],
    custom_allergens: '',
    diet_preferences: [],
    vegetarian_type: 'no_egg',
  };
}

export function saveStoredProfile(profile: UserProfile): void {
  try {
    localStorage.setItem(STORAGE_KEY_PROFILE, JSON.stringify(profile));
    localStorage.setItem(STORAGE_KEY_FIRST_TIME, 'true');
  } catch (err) {
    console.error('Failed saving user profile to storage:', err);
  }
}

export function isOnboardingCompleted(): boolean {
  return localStorage.getItem(STORAGE_KEY_FIRST_TIME) === 'true';
}
