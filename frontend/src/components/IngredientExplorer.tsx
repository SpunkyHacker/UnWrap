import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Leaf,
  FlaskConical,
  Beaker,
  Gem,
  HelpCircle,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  RotateCcw,
  Check,
  X,
  Info,
  Layers,
} from 'lucide-react';
import {
  AnalysisResponse,
  AIIngredientCard,
  UserProfile,
  AllergenStatus,
} from '../types';
import { fetchIngredientCard } from '../api';
import { IngredientChatModal } from './IngredientChatModal';

interface IngredientExplorerProps {
  analysis: AnalysisResponse;
  profile?: UserProfile;
  onRetake: () => void;
  onBackToConfirm: () => void;
}

// Consistent role color mapping
export const ROLE_STYLES: Record<string, { bg: string; text: string; border: string; label: string }> = {
  sweetener: { bg: 'bg-pink-500/15', text: 'text-pink-300', border: 'border-pink-500/30', label: 'Sweetener' },
  emulsifier: { bg: 'bg-purple-500/15', text: 'text-purple-300', border: 'border-purple-500/30', label: 'Emulsifier' },
  fat_oil: { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30', label: 'Fat / Oil' },
  base: { bg: 'bg-yellow-500/15', text: 'text-yellow-300', border: 'border-yellow-500/30', label: 'Base' },
  preservative: { bg: 'bg-rose-500/15', text: 'text-rose-300', border: 'border-rose-500/30', label: 'Preservative' },
  flavour: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/30', label: 'Flavour' },
  colour: { bg: 'bg-cyan-500/15', text: 'text-cyan-300', border: 'border-cyan-500/30', label: 'Colour' },
  thickener: { bg: 'bg-blue-500/15', text: 'text-blue-300', border: 'border-blue-500/30', label: 'Thickener' },
  acidity_regulator: { bg: 'bg-lime-500/15', text: 'text-lime-300', border: 'border-lime-500/30', label: 'Acidity Regulator' },
  raising_agent: { bg: 'bg-indigo-500/15', text: 'text-indigo-300', border: 'border-indigo-500/30', label: 'Raising Agent' },
  vitamin_mineral: { bg: 'bg-teal-500/15', text: 'text-teal-300', border: 'border-teal-500/30', label: 'Nutrient' },
  other: { bg: 'bg-zinc-800', text: 'text-zinc-300', border: 'border-zinc-700', label: 'Other' },
};

export const getRoleStyle = (role: string) => {
  const key = role?.toLowerCase().trim() || 'other';
  return ROLE_STYLES[key] || ROLE_STYLES.other;
};

export const getOriginIcon = (origin: string) => {
  const o = origin?.toLowerCase().trim();
  switch (o) {
    case 'plant':
      return Leaf;
    case 'fermentation':
      return FlaskConical;
    case 'lab':
      return Beaker;
    case 'mineral':
      return Gem;
    default:
      return HelpCircle;
  }
};

export const IngredientExplorer: React.FC<IngredientExplorerProps> = ({
  analysis,
  profile: _profile,
  onRetake,
  onBackToConfirm,
}) => {
  const ingredients = analysis.parsed.ingredients;
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [cardsCache, setCardsCache] = useState<Record<string, AIIngredientCard>>({});
  const [loadingCards, setLoadingCards] = useState<Record<string, boolean>>({});
  const [selectedRoleFilter, setSelectedRoleFilter] = useState<string>('all');
  const [activeChatCard, setActiveChatCard] = useState<AIIngredientCard | null>(null);

  const currentIngredient = ingredients[selectedIndex] || ingredients[0];

  // Lazy-load card when selectedIndex changes
  useEffect(() => {
    if (!currentIngredient) return;
    const name = currentIngredient.name;
    if (cardsCache[name] || loadingCards[name]) return;

    // First check if match exists in analysis.matches with kb_entry
    const match = analysis.matches?.find((m) => m.parsed_name.toLowerCase() === name.toLowerCase());
    if (match && match.kb_entry) {
      const kb = match.kb_entry;
      const card: AIIngredientCard = {
        id: kb.id,
        name: kb.canonical_name,
        canonical_name: kb.canonical_name,
        role: kb.role,
        one_liner: kb.one_liner,
        why_used: kb.why_used,
        origin: kb.origin_type,
        origin_type: kb.origin_type,
        how_made: kb.how_made,
        flags: {
          vegan: kb.diet_flags.vegan === true ? 'likely_yes' : (kb.diet_flags.vegan === false ? 'likely_no' : 'unclear'),
          vegetarian: kb.diet_flags.vegetarian === true ? 'likely_yes' : (kb.diet_flags.vegetarian === false ? 'likely_no' : 'unclear'),
          halal: kb.diet_flags.halal === true ? 'likely_yes' : (kb.diet_flags.halal === false ? 'likely_no' : 'unclear'),
          jain: kb.diet_flags.jain === true ? 'likely_yes' : (kb.diet_flags.jain === false ? 'likely_no' : 'unclear'),
        },
        verified: kb.verified,
        source: 'kb',
        status: 'available',
        confidence: kb.confidence,
      };
      setCardsCache((prev) => ({ ...prev, [name]: card }));
      return;
    }

    // Otherwise fetch via API (will retrieve from Grok / disk cache / local model)
    setLoadingCards((prev) => ({ ...prev, [name]: true }));
    fetchIngredientCard(name)
      .then((card) => {
        setCardsCache((prev) => ({ ...prev, [name]: card }));
      })
      .catch((err) => {
        console.error(`Failed to fetch card for ${name}:`, err);
        // Fallback placeholder card
        setCardsCache((prev) => ({
          ...prev,
          [name]: {
            id: name.toLowerCase().replace(/\s+/g, '_'),
            name,
            canonical_name: name,
            role: 'other',
            one_liner: `${name} is an ingredient in this food item.`,
            why_used: 'Food formulation component',
            origin: 'unclear',
            origin_type: 'unclear',
            how_made: [],
            flags: { vegan: 'unclear', vegetarian: 'unclear', halal: 'unclear', jain: 'unclear' },
            verified: false,
            source: 'ai',
            status: 'unavailable',
            confidence: 'low',
          },
        }));
      })
      .finally(() => {
        setLoadingCards((prev) => ({ ...prev, [name]: false }));
      });
  }, [selectedIndex, currentIngredient]);

  // Compute functional roles summary
  const roleCounts: Record<string, number> = {};
  ingredients.forEach((ing) => {
    const match = analysis.matches?.find((m) => m.parsed_name.toLowerCase() === ing.name.toLowerCase());
    const role = match?.kb_entry?.role || 'other';
    roleCounts[role] = (roleCounts[role] || 0) + 1;
  });

  const distinctRoles = Object.keys(roleCounts);
  const summaryLine = `${ingredients.length} ingredients across ${distinctRoles.length} functional roles.`;

  const activeCard = currentIngredient ? cardsCache[currentIngredient.name] : null;
  const isCardLoading = currentIngredient ? loadingCards[currentIngredient.name] : false;

  const handleNext = () => {
    if (selectedIndex < ingredients.length - 1) {
      setSelectedIndex((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (selectedIndex > 0) {
      setSelectedIndex((prev) => prev - 1);
    }
  };

  const allergenStatus = analysis.allergen_result.overall_status;

  const getAllergenBanner = (status: AllergenStatus) => {
    switch (status) {
      case 'contains':
        return {
          label: 'Contains Your Allergen',
          cls: 'bg-rose-500/15 border-rose-500/40 text-rose-300',
          icon: ShieldAlert,
        };
      case 'might_contain':
        return {
          label: 'May Contain Your Allergen',
          cls: 'bg-amber-500/15 border-amber-500/40 text-amber-300',
          icon: AlertTriangle,
        };
      case 'uncertain':
        return {
          label: 'Uncertain Items on Label',
          cls: 'bg-yellow-500/15 border-yellow-500/40 text-yellow-300',
          icon: HelpCircle,
        };
      case 'clear':
      default:
        return {
          label: 'No Listed Allergens Found',
          cls: 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300',
          icon: ShieldCheck,
        };
    }
  };

  const banner = getAllergenBanner(allergenStatus);
  const BannerIcon = banner.icon;

  return (
    <div className="flex-1 max-w-md mx-auto w-full p-4 space-y-4 flex flex-col justify-between min-h-[calc(100vh-80px)]">
      {/* Top Header & Allergen Banner */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <button
            onClick={onBackToConfirm}
            className="text-xs font-medium text-zinc-400 hover:text-white flex items-center gap-1 p-1 -ml-1 rounded-lg hover:bg-zinc-900 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Chips</span>
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onRetake}
              className="text-xs font-medium text-zinc-400 hover:text-white flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>New Scan</span>
            </button>
          </div>
        </div>

        {/* Compact Allergen Summary Banner */}
        <div
          className={`px-3 py-2 rounded-2xl border flex items-center justify-between text-xs ${banner.cls}`}
        >
          <div className="flex items-center gap-2">
            <BannerIcon className="w-4 h-4 shrink-0" />
            <span className="font-semibold tracking-tight">{banner.label}</span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-zinc-950/60 border border-current">
            {analysis.allergen_result.alerts.length} alerts
          </span>
        </div>

        {/* ------------------------------------------------------------- */}
        {/* LAYER 1: COMPOSITION STRIP (Label Order + Role Colors) */}
        {/* ------------------------------------------------------------- */}
        <div className="space-y-2 pt-1">
          <div className="flex items-center justify-between text-[11px] px-0.5">
            <span className="text-zinc-400 font-medium flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-zinc-500" />
              <span>Label Composition</span>
            </span>
            <span className="text-zinc-500 font-mono text-[10px]">
              {selectedIndex + 1} of {ingredients.length}
            </span>
          </div>

          {/* Horizontal Scrolling Strip in original label order */}
          <div className="flex gap-1.5 overflow-x-auto pb-2 scrollbar-none pt-0.5 snap-x">
            {ingredients.map((ing, idx) => {
              const match = analysis.matches?.find((m) => m.parsed_name.toLowerCase() === ing.name.toLowerCase());
              const role = match?.kb_entry?.role || 'other';
              const roleStyle = getRoleStyle(role);
              const isSelected = idx === selectedIndex;

              return (
                <button
                  key={`${ing.name}-${idx}`}
                  onClick={() => setSelectedIndex(idx)}
                  className={`snap-start shrink-0 px-2.5 py-1.5 rounded-xl border text-xs font-medium transition-all flex items-center gap-1.5 ${
                    isSelected
                      ? `${roleStyle.bg} ${roleStyle.border} ${roleStyle.text} ring-2 ring-emerald-500/50 shadow-md scale-105`
                      : 'bg-zinc-900/80 border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700'
                  }`}
                >
                  <span className="text-[10px] font-mono opacity-60">#{idx + 1}</span>
                  <span className="truncate max-w-[110px]">{ing.name}</span>
                </button>
              );
            })}
          </div>

          {/* Summary Line & Role Filter Pills */}
          <div className="space-y-1.5 bg-zinc-900/40 p-2.5 rounded-2xl border border-zinc-800/60">
            <p className="text-[11px] text-zinc-400 leading-tight">
              {summaryLine}
            </p>
            <div className="flex gap-1 overflow-x-auto scrollbar-none pt-0.5">
              <button
                onClick={() => setSelectedRoleFilter('all')}
                className={`px-2 py-0.5 rounded-full text-[10px] font-medium border transition-colors ${
                  selectedRoleFilter === 'all'
                    ? 'bg-zinc-100 text-zinc-900 border-zinc-100'
                    : 'bg-zinc-800/80 text-zinc-400 border-zinc-700 hover:text-white'
                }`}
              >
                All ({ingredients.length})
              </button>
              {distinctRoles.map((roleKey) => {
                const style = getRoleStyle(roleKey);
                const count = roleCounts[roleKey] || 0;
                return (
                  <button
                    key={roleKey}
                    onClick={() => {
                      setSelectedRoleFilter(roleKey);
                      // Jump to first ingredient matching this role
                      const firstIdx = ingredients.findIndex((ing) => {
                        const m = analysis.matches?.find((match) => match.parsed_name.toLowerCase() === ing.name.toLowerCase());
                        return (m?.kb_entry?.role || 'other') === roleKey;
                      });
                      if (firstIdx !== -1) setSelectedIndex(firstIdx);
                    }}
                    className={`px-2 py-0.5 rounded-full text-[10px] font-medium border transition-colors shrink-0 ${
                      selectedRoleFilter === roleKey
                        ? `${style.bg} ${style.border} ${style.text} ring-1 ring-white/20`
                        : 'bg-zinc-900/60 text-zinc-400 border-zinc-800 hover:text-zinc-200'
                    }`}
                  >
                    {style.label} ({count})
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* LAYER 2: SWIPEABLE CARD DECK */}
      {/* ------------------------------------------------------------- */}
      <div className="my-auto py-2">
        <AnimatePresence mode="wait">
          {isCardLoading || !activeCard ? (
            /* Shimmer Skeleton Card while generating / fetching */
            <motion.div
              key="skeleton"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="w-full bg-zinc-900/90 border border-zinc-800 rounded-3xl p-5 space-y-4 shadow-xl relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-zinc-800/20 to-transparent -translate-x-full animate-[shimmer_1.5s_infinite]" />
              <div className="flex items-center justify-between">
                <div className="h-6 w-36 bg-zinc-800 rounded-lg animate-pulse" />
                <div className="h-5 w-20 bg-zinc-800 rounded-full animate-pulse" />
              </div>
              <div className="h-4 w-full bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-3/4 bg-zinc-800 rounded animate-pulse" />
              <div className="h-8 w-28 bg-zinc-800 rounded-xl animate-pulse" />
              <div className="grid grid-cols-3 gap-2 pt-2">
                <div className="h-16 bg-zinc-800/80 rounded-2xl animate-pulse" />
                <div className="h-16 bg-zinc-800/80 rounded-2xl animate-pulse" />
                <div className="h-16 bg-zinc-800/80 rounded-2xl animate-pulse" />
              </div>
              <div className="h-10 w-full bg-zinc-800/60 rounded-2xl animate-pulse" />
            </motion.div>
          ) : (
            /* Real Ingredient Card with Spring Animation */
            <motion.div
              key={activeCard.name}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ type: 'spring', stiffness: 300, damping: 28 }}
              className="w-full bg-gradient-to-b from-zinc-900 to-zinc-950 border border-zinc-800/90 rounded-3xl p-5 space-y-4 shadow-2xl relative"
            >
              {/* Card Header */}
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
                      #{selectedIndex + 1}
                    </span>
                    {activeCard.source === 'ai' ? (
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300">
                        AI-generated, not verified
                      </span>
                    ) : (
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                        Curated Record
                      </span>
                    )}
                  </div>
                  <h3 className="text-xl font-bold text-white tracking-tight">
                    {activeCard.name}
                  </h3>
                </div>

                {/* Role Badge */}
                {(() => {
                  const style = getRoleStyle(activeCard.role);
                  return (
                    <span
                      className={`text-[11px] font-semibold px-2.5 py-1 rounded-xl border shrink-0 ${style.bg} ${style.border} ${style.text}`}
                    >
                      {style.label}
                    </span>
                  );
                })()}
              </div>

              {/* One Liner Description */}
              <p className="text-xs text-zinc-300 leading-relaxed">
                {activeCard.one_liner}
              </p>

              {/* Why Used Callout */}
              <div className="p-3 rounded-2xl bg-zinc-900/90 border border-zinc-800 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span className="text-xs text-zinc-400 font-medium">Why it's used:</span>
                </div>
                <span className="text-xs font-bold text-white text-right">
                  {activeCard.why_used}
                </span>
              </div>

              {/* Origin Chip */}
              <div className="flex items-center justify-between text-xs py-0.5">
                <span className="text-zinc-400 font-medium">Origin:</span>
                {(() => {
                  const OriginIcon = getOriginIcon(activeCard.origin || activeCard.origin_type);
                  return (
                    <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-zinc-800/80 border border-zinc-700/60 text-zinc-200 font-medium">
                      <OriginIcon className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="capitalize">{activeCard.origin || activeCard.origin_type || 'Unclear'}</span>
                    </div>
                  );
                })()}
              </div>

              {/* 3-Step How It's Made */}
              {activeCard.how_made && activeCard.how_made.length > 0 && (
                <div className="space-y-1.5">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider block">
                    How it's made (3 stages)
                  </span>
                  <div className="grid grid-cols-3 gap-2">
                    {activeCard.how_made.slice(0, 3).map((step, sIdx) => (
                      <div
                        key={sIdx}
                        className="p-2.5 rounded-2xl bg-zinc-900/80 border border-zinc-800/80 flex flex-col justify-between min-h-[64px]"
                      >
                        <span className="text-[9px] font-mono text-zinc-500 font-bold">
                          Stage {sIdx + 1}
                        </span>
                        <p className="text-[10px] text-zinc-200 font-medium leading-tight line-clamp-2 mt-1">
                          {step.text}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Dietary Flags */}
              <div className="space-y-1.5 pt-1">
                <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider block">
                  Suitability Flags
                </span>
                <div className="grid grid-cols-4 gap-1.5">
                  {(['vegan', 'vegetarian', 'halal', 'jain'] as const).map((flagKey) => {
                    const val = activeCard.flags ? activeCard.flags[flagKey] : 'unclear';
                    const isYes = val === 'likely_yes';
                    const isNo = val === 'likely_no';

                    return (
                      <div
                        key={flagKey}
                        className={`px-2 py-1.5 rounded-xl border text-center flex flex-col items-center justify-center gap-0.5 ${
                          isYes
                            ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                            : isNo
                            ? 'bg-rose-500/10 border-rose-500/25 text-rose-300'
                            : 'bg-zinc-800/60 border-zinc-700/50 text-zinc-400'
                        }`}
                      >
                        <span className="text-[10px] uppercase font-bold tracking-tight capitalize">
                          {flagKey}
                        </span>
                        <span className="text-[9px] font-medium flex items-center gap-0.5">
                          {isYes && <Check className="w-2.5 h-2.5 text-emerald-400" />}
                          {isNo && <X className="w-2.5 h-2.5 text-rose-400" />}
                          {isYes ? 'Likely' : isNo ? 'Unlikely' : 'Unclear'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* ------------------------------------------------------------- */}
              {/* LAYER 3: "ASK ABOUT THIS" CHAT CTA */}
              {/* ------------------------------------------------------------- */}
              <div className="pt-2">
                <button
                  onClick={() => setActiveChatCard(activeCard)}
                  className="w-full py-3 rounded-2xl bg-gradient-to-r from-indigo-500/20 via-purple-500/20 to-teal-500/20 border border-indigo-500/40 hover:border-indigo-400 text-indigo-200 hover:text-white font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-lg active:scale-[0.99]"
                >
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <span>Ask about {activeCard.name}</span>
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation Controls: Prev / Next */}
      <div className="pt-2 pb-1 flex items-center justify-between gap-3">
        <button
          onClick={handlePrev}
          disabled={selectedIndex === 0}
          className="flex-1 py-3 px-4 rounded-2xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 disabled:opacity-40 disabled:pointer-events-none text-zinc-300 hover:text-white font-medium text-xs flex items-center justify-center gap-1.5 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Previous</span>
        </button>

        <span className="text-xs font-mono text-zinc-500 px-3 py-1 rounded-full bg-zinc-900/60 border border-zinc-800 shrink-0">
          {selectedIndex + 1} / {ingredients.length}
        </span>

        <button
          onClick={handleNext}
          disabled={selectedIndex === ingredients.length - 1}
          className="flex-1 py-3 px-4 rounded-2xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:pointer-events-none text-zinc-950 font-bold text-xs flex items-center justify-center gap-1.5 transition-colors shadow-lg shadow-emerald-500/10"
        >
          <span>Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* Grounded Chat Modal Drawer */}
      {activeChatCard && (
        <IngredientChatModal
          isOpen={!!activeChatCard}
          onClose={() => setActiveChatCard(null)}
          card={activeChatCard}
        />
      )}
    </div>
  );
};
