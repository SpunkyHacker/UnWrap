import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  X,
  Edit2,
  Plus,
  ArrowRight,
  ShieldAlert,
  ShieldCheck,
  HelpCircle,
  Split,
  RefreshCw,
  Camera,
} from 'lucide-react';
import { AnalysisResponse, AllergenStatus, UserProfile } from '../types';
import { EngineBadge } from './EngineBadge';

interface ConfirmScreenProps {
  initialAnalysis: AnalysisResponse;
  profile: UserProfile;
  onConfirmReanalyze: (editedIngredients: string[]) => Promise<void>;
  onRetakePhotos: () => void;
  onExplore: () => void;
  isReanalyzing: boolean;
}

interface EditableChip {
  id: string;
  name: string;
  possible_merged: boolean;
  suggested_split: string[];
}

export const ConfirmScreen: React.FC<ConfirmScreenProps> = ({
  initialAnalysis,
  profile,
  onConfirmReanalyze,
  onRetakePhotos,
  onExplore,
  isReanalyzing,
}) => {
  // Convert parsed ingredients to editable local state
  const [chips, setChips] = useState<EditableChip[]>(() => {
    return initialAnalysis.parsed.ingredients.map((ing, idx) => ({
      id: `chip-${idx}-${ing.name}`,
      name: ing.name,
      possible_merged: ing.possible_merged,
      suggested_split: ing.suggested_split || [],
    }));
  });

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [newIngredientText, setNewIngredientText] = useState('');
  const [isAddingNew, setIsAddingNew] = useState(false);

  // Check if unreadable parts or low confidence
  const unreadable = initialAnalysis.transcription?.unreadable_parts || [];
  const confidence = initialAnalysis.allergen_result?.confidence || 'medium';
  const showRecheckWarning = unreadable.length > 0 || confidence === 'low';

  // Allergen result helper
  const allergenStatus = initialAnalysis.allergen_result?.overall_status || 'uncertain';

  const getStatusBadge = (status: AllergenStatus) => {
    switch (status) {
      case 'contains':
        return {
          label: 'CONTAINS YOUR ALLERGEN',
          className: 'bg-rose-500/15 border-rose-500/50 text-rose-300',
          icon: ShieldAlert,
        };
      case 'might_contain':
        return {
          label: 'MIGHT CONTAIN YOUR ALLERGEN',
          className: 'bg-amber-500/15 border-amber-500/50 text-amber-300',
          icon: AlertTriangle,
        };
      case 'uncertain':
        return {
          label: 'UNCERTAIN — UNIDENTIFIED ITEMS',
          className: 'bg-yellow-500/15 border-yellow-500/50 text-yellow-300',
          icon: HelpCircle,
        };
      case 'clear':
      default:
        // NEVER USE THE WORD "SAFE"
        return {
          label: 'NO LISTED ALLERGENS FOUND',
          className: 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300',
          icon: ShieldCheck,
        };
    }
  };

  const statusBadge = getStatusBadge(allergenStatus);
  const StatusIcon = statusBadge.icon;

  // Actions on chips
  const handleDeleteChip = (id: string) => {
    setChips((prev) => prev.filter((c) => c.id !== id));
  };

  const handleStartEdit = (chip: EditableChip) => {
    setEditingId(chip.id);
    setEditText(chip.name);
  };

  const handleSaveEdit = (id: string) => {
    if (!editText.trim()) {
      handleDeleteChip(id);
    } else {
      setChips((prev) =>
        prev.map((c) => (c.id === id ? { ...c, name: editText.trim(), possible_merged: false } : c))
      );
    }
    setEditingId(null);
  };

  const handleAcceptSplit = (chip: EditableChip) => {
    const splitChips: EditableChip[] = chip.suggested_split.map((name, i) => ({
      id: `${chip.id}-split-${i}-${Date.now()}`,
      name,
      possible_merged: false,
      suggested_split: [],
    }));

    setChips((prev) => {
      const idx = prev.findIndex((c) => c.id === chip.id);
      if (idx === -1) return prev;
      const copy = [...prev];
      copy.splice(idx, 1, ...splitChips);
      return copy;
    });
  };

  const handleRejectSplit = (chipId: string) => {
    setChips((prev) =>
      prev.map((c) => (c.id === chipId ? { ...c, possible_merged: false } : c))
    );
  };

  const handleAddNewChip = () => {
    if (newIngredientText.trim()) {
      setChips((prev) => [
        ...prev,
        {
          id: `new-${Date.now()}`,
          name: newIngredientText.trim(),
          possible_merged: false,
          suggested_split: [],
        },
      ]);
      setNewIngredientText('');
      setIsAddingNew(false);
    }
  };

  const handleConfirm = async () => {
    const ingredientNames = chips.map((c) => c.name.trim()).filter(Boolean);
    await onConfirmReanalyze(ingredientNames);
    onExplore();
  };

  return (
    <div className="flex-1 max-w-md mx-auto w-full p-5 space-y-6 flex flex-col justify-between min-h-[calc(100vh-80px)]">
      <div className="space-y-5">
        {/* Top bar */}
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-white tracking-tight">
            Confirm Ingredients
          </h2>
          <button
            onClick={onRetakePhotos}
            className="text-xs text-zinc-400 hover:text-white flex items-center gap-1 p-1"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Retake Photo</span>
          </button>
        </div>

        {/* Engine and Fallback Info */}
        <EngineBadge transcription={initialAnalysis.transcription} />

        {/* Allergen Status Banner */}
        <div className={`p-4 rounded-2xl border ${statusBadge.className} flex items-start gap-3 shadow-sm`}>
          <StatusIcon className="w-5 h-5 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h3 className="font-bold text-sm tracking-wide">{statusBadge.label}</h3>
            {initialAnalysis.allergen_result.alerts.length > 0 ? (
              <ul className="text-xs space-y-1 mt-1 text-zinc-300">
                {initialAnalysis.allergen_result.alerts.slice(0, 3).map((a, i) => (
                  <li key={i} className="leading-snug">
                    • {a.reason}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-zinc-400">
                Deterministic lookup found no matches for your {profile.allergens.length} listed allergens.
              </p>
            )}
          </div>
        </div>

        {/* Warning if unreadable parts or low confidence */}
        {showRecheckWarning && (
          <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/40 text-amber-200 text-xs flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1.5">
              <strong className="block text-amber-300 text-sm">Re-check photo recommended</strong>
              <p className="leading-relaxed">
                {unreadable.length > 0
                  ? `Part of the label was hard to read (${unreadable.join(', ')}). Confidence is ${confidence}.`
                  : `Transcribed with ${confidence} confidence. Please verify all chips below before eating.`}
              </p>
              <button
                onClick={onRetakePhotos}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 font-semibold text-[11px] transition-colors mt-1"
              >
                <Camera className="w-3.5 h-3.5" />
                <span>Snap clearer photo</span>
              </button>
            </div>
          </div>
        )}

        {/* Merged Items Alert & Actions */}
        {chips.some((c) => c.possible_merged) && (
          <div className="space-y-2">
            {chips
              .filter((c) => c.possible_merged)
              .map((chip) => (
                <div
                  key={`merged-${chip.id}`}
                  className="p-3.5 rounded-2xl bg-zinc-900 border border-amber-500/50 text-xs space-y-2"
                >
                  <div className="flex items-center gap-1.5 text-amber-400 font-medium">
                    <Split className="w-4 h-4" />
                    <span>Possible merged ingredient: <strong>"{chip.name}"</strong></span>
                  </div>
                  <p className="text-zinc-400 text-[11px]">
                    Did you mean: {chip.suggested_split.join(' + ')}?
                  </p>
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={() => handleAcceptSplit(chip)}
                      className="px-3 py-1 rounded-lg bg-emerald-500 text-zinc-950 font-bold text-xs hover:bg-emerald-400 transition-colors"
                    >
                      Accept Split
                    </button>
                    <button
                      onClick={() => handleRejectSplit(chip.id)}
                      className="px-3 py-1 rounded-lg bg-zinc-800 text-zinc-300 hover:text-white transition-colors"
                    >
                      Keep As Is
                    </button>
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* Detected Chips Grid */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Ingredients ({chips.length})
            </label>
            <span className="text-[11px] text-zinc-500">Tap chip to edit</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {chips.map((chip) => {
              const isEditing = editingId === chip.id;
              const isMerged = chip.possible_merged;

              if (isEditing) {
                return (
                  <div
                    key={chip.id}
                    className="flex items-center gap-1 bg-zinc-900 border border-emerald-500 rounded-xl px-2 py-1 shadow-md"
                  >
                    <input
                      type="text"
                      value={editText}
                      autoFocus
                      onChange={(e) => setEditText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit(chip.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      className="bg-transparent text-sm text-white focus:outline-none w-36 sm:w-44"
                    />
                    <button
                      onClick={() => handleSaveEdit(chip.id)}
                      className="p-1 rounded-md text-emerald-400 hover:text-emerald-300 hover:bg-zinc-800"
                    >
                      <Check className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setEditingId(null)}
                      className="p-1 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              }

              return (
                <div
                  key={chip.id}
                  className={`group inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-sm font-medium transition-all ${
                    isMerged
                      ? 'bg-amber-500/15 border-amber-500/60 text-amber-200 shadow-sm'
                      : 'bg-zinc-900/90 border-zinc-800 text-zinc-200 hover:border-zinc-700 hover:bg-zinc-850'
                  }`}
                >
                  <span
                    onClick={() => handleStartEdit(chip)}
                    className="cursor-pointer hover:underline underline-offset-2"
                  >
                    {chip.name}
                  </span>
                  <button
                    onClick={() => handleStartEdit(chip)}
                    className="text-zinc-500 hover:text-zinc-300 p-0.5 -mr-0.5 opacity-60 group-hover:opacity-100"
                    title="Edit ingredient"
                  >
                    <Edit2 className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => handleDeleteChip(chip.id)}
                    className="text-zinc-500 hover:text-rose-400 p-0.5 rounded-md hover:bg-zinc-800 opacity-60 group-hover:opacity-100"
                    title="Remove ingredient"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              );
            })}

            {/* Add ingredient button / input */}
            {isAddingNew ? (
              <div className="flex items-center gap-1 bg-zinc-900 border border-emerald-500 rounded-xl px-2 py-1 shadow-md">
                <input
                  type="text"
                  value={newIngredientText}
                  placeholder="e.g. Canola Oil"
                  autoFocus
                  onChange={(e) => setNewIngredientText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleAddNewChip();
                    if (e.key === 'Escape') setIsAddingNew(false);
                  }}
                  className="bg-transparent text-sm text-white focus:outline-none w-36"
                />
                <button
                  onClick={handleAddNewChip}
                  className="p-1 rounded-md text-emerald-400 hover:text-emerald-300 hover:bg-zinc-800"
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setIsAddingNew(false)}
                  className="p-1 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAddingNew(true)}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-xl border border-dashed border-zinc-700/80 hover:border-emerald-500 text-xs font-medium text-zinc-400 hover:text-emerald-400 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Ingredient</span>
              </button>
            )}
          </div>
        </div>

        {/* Non-medical note */}
        <p className="text-[11px] text-zinc-600 leading-normal text-center pt-2">
          {initialAnalysis.allergen_result.disclaimer}
        </p>
      </div>

      {/* Confirm Bottom Action */}
      <div className="pt-6 pb-2 space-y-2">
        <button
          onClick={handleConfirm}
          disabled={isReanalyzing || chips.length === 0}
          className="w-full py-4 rounded-2xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-zinc-950 font-bold text-base flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/20 active:scale-[0.99] transition-all"
        >
          {isReanalyzing ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              <span>Verifying Allergens...</span>
            </>
          ) : (
            <>
              <span>Confirm & Explore Ingredients</span>
              <ArrowRight className="w-5 h-5" />
            </>
          )}
        </button>

        <button
          onClick={onExplore}
          disabled={isReanalyzing || chips.length === 0}
          className="w-full py-2.5 rounded-xl text-xs font-semibold text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
        >
          Skip edit & explore current ingredients &rarr;
        </button>
      </div>
    </div>
  );
};
