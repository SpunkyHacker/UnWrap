import React, { useState } from 'react';
import { X, Check, ShieldAlert } from 'lucide-react';
import { UserProfile } from '../types';
import { DEFAULT_ALLERGEN_OPTIONS, DEFAULT_DIET_OPTIONS } from '../storage';

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialProfile: UserProfile;
  onSaveProfile: (profile: UserProfile) => void;
  isFirstTime?: boolean;
}

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  isOpen,
  onClose,
  initialProfile,
  onSaveProfile,
  isFirstTime = false,
}) => {
  const [selectedAllergens, setSelectedAllergens] = useState<string[]>(initialProfile.allergens || []);
  const [customText, setCustomText] = useState(initialProfile.custom_allergens || '');
  const [selectedDiets, setSelectedDiets] = useState<string[]>(initialProfile.diet_preferences || []);
  const [vegetarianType, setVegetarianType] = useState<'no_egg' | 'eggs_allowed'>(
    initialProfile.vegetarian_type || 'no_egg'
  );

  if (!isOpen) return null;

  const toggleAllergen = (id: string) => {
    setSelectedAllergens((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const toggleDiet = (id: string) => {
    setSelectedDiets((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSave = () => {
    onSaveProfile({
      allergens: selectedAllergens,
      custom_allergens: customText.trim(),
      diet_preferences: selectedDiets,
      vegetarian_type: vegetarianType,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-zinc-900 border border-zinc-800 rounded-3xl max-w-md w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-zinc-800 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-emerald-400" />
              {isFirstTime ? 'Your Allergy Profile' : 'Edit Allergen Profile'}
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Select allergens you need to avoid. Stored securely on your device.
            </p>
          </div>
          {!isFirstTime && (
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Scrollable Body */}
        <div className="p-5 overflow-y-auto space-y-6">
          {/* Common Allergens Grid */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400 block mb-3">
              Common Allergens
            </label>
            <div className="grid grid-cols-2 gap-2">
              {DEFAULT_ALLERGEN_OPTIONS.map((item) => {
                const isChecked = selectedAllergens.includes(item.id);
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => toggleAllergen(item.id)}
                    className={`flex items-center gap-2 p-2.5 rounded-xl border text-sm font-medium transition-all text-left ${
                      isChecked
                        ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-200 shadow-sm'
                        : 'bg-zinc-950/40 border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    }`}
                  >
                    <span className="text-base">{item.icon}</span>
                    <span className="flex-1 truncate">{item.label}</span>
                    <div
                      className={`w-4 h-4 rounded-md flex items-center justify-center border transition-colors ${
                        isChecked
                          ? 'bg-emerald-500 border-emerald-500 text-zinc-950'
                          : 'border-zinc-700 bg-zinc-900'
                      }`}
                    >
                      {isChecked && <Check className="w-3 h-3 stroke-[3]" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Custom Allergen Input */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400 block mb-2">
              Custom Allergies or Sensitivities
            </label>
            <input
              type="text"
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="e.g. strawberries, artificial red dye, kiwi"
              className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-500 transition-colors"
            />
            <p className="text-[11px] text-zinc-500 mt-1.5">
              Enter specific ingredients separated by commas.
            </p>
          </div>

          {/* Diet Preferences */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400 block mb-2.5">
              Dietary Preferences
            </label>
            <div className="flex flex-wrap gap-2">
              {DEFAULT_DIET_OPTIONS.map((diet) => {
                const isSelected = selectedDiets.includes(diet.id);
                return (
                  <button
                    key={diet.id}
                    type="button"
                    onClick={() => toggleDiet(diet.id)}
                    className={`px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${
                      isSelected
                        ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300'
                        : 'bg-zinc-950/40 border-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {diet.label}
                  </button>
                );
              })}
            </div>

            {/* Vegetarian Type Sub-Choice */}
            {selectedDiets.includes('vegetarian') && (
              <div className="mt-3 p-3 rounded-2xl bg-zinc-950/60 border border-zinc-800 space-y-2 animate-in fade-in duration-200">
                <label className="text-[11px] font-semibold text-zinc-300 block">
                  Vegetarian Type
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setVegetarianType('no_egg')}
                    className={`p-2.5 rounded-xl border text-xs font-medium text-left transition-all ${
                      vegetarianType === 'no_egg'
                        ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-200'
                        : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <span className="block font-semibold">No eggs</span>
                    <span className="text-[10px] text-zinc-400">Default vegetarian</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setVegetarianType('eggs_allowed')}
                    className={`p-2.5 rounded-xl border text-xs font-medium text-left transition-all ${
                      vegetarianType === 'eggs_allowed'
                        ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-200'
                        : 'bg-zinc-900/60 border-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <span className="block font-semibold">Eggs allowed</span>
                    <span className="text-[10px] text-zinc-400">Ovo-lacto</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-zinc-800 bg-zinc-950/40 flex items-center justify-between">
          <span className="text-xs text-zinc-500">
            {selectedAllergens.length} allergen{selectedAllergens.length === 1 ? '' : 's'} chosen
          </span>
          <button
            onClick={handleSave}
            className="px-5 py-2.5 rounded-xl bg-emerald-500 text-zinc-950 font-semibold text-sm hover:bg-emerald-400 transition-colors shadow-lg shadow-emerald-500/20"
          >
            {isFirstTime ? 'Get Started' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};
