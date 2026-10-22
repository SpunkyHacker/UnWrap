import React, { useState, useEffect, useMemo, useCallback } from 'react';
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
  Info,
  Layers,
  RefreshCw,
} from 'lucide-react';
import {
  AnalysisResponse,
  AIIngredientCard,
  UserProfile,
  AllergenStatus,
  DietCheckResponse,
  IngredientDietResult,
} from '../types';
import { fetchIngredientCard, fetchIngredientCardsBatch, performDietCheck } from '../api';
import { IngredientChatModal } from './IngredientChatModal';

interface IngredientExplorerProps {
  analysis: AnalysisResponse;
  profile?: UserProfile;
  onRetake: () => void;
  onBackToConfirm: () => void;
}

// 6 canonical functional roles mapping
export const ROLE_STYLES: Record<string, { bg: string; text: string; border: string; label: string }> = {
  base: { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30', label: 'Base' },
  sweetener: { bg: 'bg-pink-500/15', text: 'text-pink-300', border: 'border-pink-500/30', label: 'Sweetener' },
  preservative: { bg: 'bg-rose-500/15', text: 'text-rose-300', border: 'border-rose-500/30', label: 'Preservative' },
  texture: { bg: 'bg-sky-500/15', text: 'text-sky-300', border: 'border-sky-500/30', label: 'Texture' },
  color_flavor: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/30', label: 'Color / Flavor' },
  other: { bg: 'bg-zinc-800', text: 'text-zinc-300', border: 'border-zinc-700', label: 'Other' },
};

export const getRoleStyle = (role?: string) => {
  const key = role?.toLowerCase().trim() || 'other';
  if (key === 'emulsifier' || key === 'thickener' || key === 'stabilizer' || key === 'stabiliser') {
    return ROLE_STYLES.texture;
  }
  if (
    key === 'flavour' ||
    key === 'flavor' ||
    key === 'colour' ||
    key === 'color' ||
    key === 'flavour_enhancer' ||
    key === 'flavor_enhancer'
  ) {
    return ROLE_STYLES.color_flavor;
  }
  if (key === 'fat_oil' || key === 'fat' || key === 'oil' || key === 'protein' || key === 'solvent') {
    return ROLE_STYLES.base;
  }
  if (key === 'antioxidant' || key === 'acidity_regulator') {
    return ROLE_STYLES.preservative;
  }
  return ROLE_STYLES[key] || ROLE_STYLES.other;
};

export const getOriginIcon = (origin?: string) => {
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
  profile,
  onRetake,
  onBackToConfirm,
}) => {
  const ingredients = analysis.parsed.ingredients;
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [cardsCache, setCardsCache] = useState<Record<string, AIIngredientCard>>({});
  const [loadingCards, setLoadingCards] = useState<Record<string, boolean>>({});
  const [selectedRoleFilter, setSelectedRoleFilter] = useState<string>('all');
  const [activeChatCard, setActiveChatCard] = useState<AIIngredientCard | null>(null);

  // Diet check state
  const [dietResponse, setDietResponse] = useState<DietCheckResponse | null>(null);
  const [dietLoading, setDietLoading] = useState<boolean>(false);
  const [dietError, setDietError] = useState<boolean>(false);

  const userDiets = profile?.diet_preferences || [];
  const vegType = profile?.vegetarian_type || 'no_egg';

  const runDietCheck = useCallback(() => {
    if (!ingredients || ingredients.length === 0 || userDiets.length === 0) {
      setDietResponse(null);
      setDietError(false);
      setDietLoading(false);
      return;
    }

    setDietLoading(true);
    setDietError(false);

    performDietCheck(
      ingredients.map((ing) => ing.name),
      userDiets,
      vegType
    )
      .then((res) => {
        if (res.status === 'unavailable') {
          setDietError(true);
          setDietResponse(null);
        } else {
          setDietResponse(res);
          setDietError(false);
        }
      })
      .catch((err) => {
        console.error('Diet check error:', err);
        setDietError(true);
        setDietResponse(null);
      })
      .finally(() => {
        setDietLoading(false);
      });
  }, [ingredients, userDiets, vegType]);

  useEffect(() => {
    runDietCheck();
  }, [runDietCheck]);

  const dietResultsMap = useMemo(() => {
    const map: Record<string, IngredientDietResult> = {};
    if (dietResponse && Array.isArray(dietResponse.results)) {
      dietResponse.results.forEach((item) => {
        map[item.ingredient.toLowerCase().trim()] = item;
      });
    }
    return map;
  }, [dietResponse]);

  const currentIngredient = ingredients[selectedIndex] || ingredients[0];

  // Batch load cards on mount and whenever ingredients change
  useEffect(() => {
    if (!ingredients || ingredients.length === 0) return;

    const namesToFetch = ingredients
      .map((ing) => ing.name)
      .filter((name) => !cardsCache[name]);

    if (namesToFetch.length === 0) return;

    const loadingMap: Record<string, boolean> = {};
    namesToFetch.forEach((name) => {
      loadingMap[name] = true;
    });
    setLoadingCards((prev) => ({ ...prev, ...loadingMap }));

    fetchIngredientCardsBatch(namesToFetch)
      .then((batchResult) => {
        const updates: Record<string, AIIngredientCard> = { ...batchResult };
        namesToFetch.forEach((name) => {
          if (!updates[name]) {
            updates[name] = {
              id: name.toLowerCase().replace(/\s+/g, '_'),
              name,
              canonical_name: name,
              role: 'other',
              one_liner: '',
              why_used: '',
              origin: 'unclear',
              origin_type: 'unclear',
              how_made: [],
              verified: false,
              source: 'ai',
              status: 'unavailable',
              confidence: 'low',
            };
          }
        });
        setCardsCache((prev) => ({ ...prev, ...updates }));
      })
      .catch((err) => {
        console.error('Batch card fetch error, falling back to individual calls:', err);
        namesToFetch.forEach((name) => {
          fetchIngredientCard(name)
            .then((card) => {
              setCardsCache((prev) => ({ ...prev, [name]: card }));
            })
            .catch(() => {
              setCardsCache((prev) => ({
                ...prev,
                [name]: {
                  id: name.toLowerCase().replace(/\s+/g, '_'),
                  name,
                  canonical_name: name,
                  role: 'other',
                  one_liner: '',
                  why_used: '',
                  origin: 'unclear',
                  origin_type: 'unclear',
                  how_made: [],
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
        });
      })
      .finally(() => {
        setLoadingCards((prev) => {
          const next = { ...prev };
          namesToFetch.forEach((name) => {
            next[name] = false;
          });
          return next;
        });
      });
  }, [ingredients]);

  // Compute functional roles summary from loaded cards
  const roleCounts: Record<string, number> = {};
  ingredients.forEach((ing) => {
    const card = cardsCache[ing.name];
    const roleKey = card?.role?.toLowerCase() || 'other';
    const canonicalKey = ROLE_STYLES[roleKey] ? roleKey : 'other';
    roleCounts[canonicalKey] = (roleCounts[canonicalKey] || 0) + 1;
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

        {/* Prominent Allergen Summary Banner */}
        <div
          className={`px-3.5 py-2.5 rounded-2xl border flex items-center justify-between text-xs ${banner.cls} shadow-sm`}
        >
          <div className="flex items-center gap-2">
            <BannerIcon className="w-4 h-4 shrink-0" />
            <span className="font-bold tracking-tight">{banner.label}</span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-zinc-950/60 border border-current">
            {analysis.allergen_result.alerts.length} alerts
          </span>
        </div>

        {/* Separate Diet Banner Directly Under Allergen Banner */}
        {userDiets.length > 0 && (
          <div className="space-y-1">
            {dietLoading ? (
              <div className="px-3.5 py-2 rounded-2xl bg-zinc-900/80 border border-zinc-800 text-xs text-zinc-400 flex items-center justify-between">
                <span>Checking dietary compatibility...</span>
                <span className="w-3.5 h-3.5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : dietError ? (
              <div className="px-3.5 py-2.5 rounded-2xl border border-amber-500/40 bg-amber-500/10 text-amber-300 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span className="font-medium">Diet check unavailable, retry</span>
                </div>
                <button
                  type="button"
                  onClick={runDietCheck}
                  className="px-2.5 py-1 rounded-xl bg-zinc-900 border border-zinc-700 text-[11px] font-medium hover:text-white transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : dietResponse ? (
              <div
                className={`p-3.5 rounded-2xl border text-xs space-y-1.5 ${
                  dietResponse.conflicts_count > 0
                    ? 'bg-rose-500/10 border-rose-500/30 text-rose-200'
                    : dietResponse.may_conflicts_count > 0
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
                    : 'bg-zinc-900/80 border-zinc-800 text-zinc-300'
                }`}
              >
                <div className="flex items-center justify-between font-semibold">
                  <span>
                    {dietResponse.conflicts_count > 0
                      ? `${dietResponse.conflicts_count} conflict${dietResponse.conflicts_count > 1 ? 's' : ''} with your diet`
                      : dietResponse.may_conflicts_count > 0
                      ? `${dietResponse.may_conflicts_count} ingredient${dietResponse.may_conflicts_count > 1 ? 's' : ''} may conflict with your diet`
                      : 'No listed dietary conflicts found'}
                  </span>
                  <span className="text-[10px] font-mono uppercase opacity-75">
                    {userDiets.join(', ')}
                  </span>
                </div>

                {/* List conflicting reasons if any */}
                {dietResponse.results
                  .filter((r) => r.status === 'CONFLICT' || r.status === 'MAY_CONFLICT')
                  .slice(0, 3)
                  .map((r, i) => (
                    <div key={i} className="text-[11px] opacity-90 flex items-start gap-1.5 pt-0.5">
                      <span className="text-zinc-500">•</span>
                      <span>
                        <strong>{r.ingredient}:</strong> {r.reason}
                      </span>
                    </div>
                  ))}

                {/* Single muted line inside diet banner */}
                <p className="text-[10px] text-zinc-500 pt-0.5">
                  Check the pack for certification.
                </p>
              </div>
            ) : null}
          </div>
        )}

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
              const card = cardsCache[ing.name];
              const isLoaded = !!card && card.status === 'available';
              const roleStyle = getRoleStyle(card?.role);
              const isSelected = idx === selectedIndex;

              // Neutral color before card loads, then animate to role color
              const colorClasses = isLoaded
                ? `${roleStyle.bg} ${roleStyle.border} ${roleStyle.text}`
                : 'bg-zinc-800/60 border-zinc-700/50 text-zinc-400';

              return (
                <button
                  key={`${ing.name}-${idx}`}
                  onClick={() => setSelectedIndex(idx)}
                  className={`snap-start shrink-0 px-2.5 py-1.5 rounded-xl border text-xs font-medium transition-all duration-700 ease-out flex items-center gap-1.5 ${colorClasses} ${
                    isSelected
                      ? 'ring-2 ring-emerald-500/50 shadow-md scale-105'
                      : 'hover:brightness-110'
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
                      const firstIdx = ingredients.findIndex((ing) => {
                        const c = cardsCache[ing.name];
                        const r = (c?.role || 'other').toLowerCase();
                        return (ROLE_STYLES[r] ? r : 'other') === roleKey;
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
          ) : activeCard.status === 'unavailable' ? (
            /* Unavailable Card State */
            <motion.div
              key={`unavailable-${activeCard.name}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="w-full bg-zinc-900/90 border border-zinc-800 rounded-3xl p-6 text-center space-y-3 shadow-xl"
            >
              <HelpCircle className="w-8 h-8 text-zinc-500 mx-auto" />
              <h3 className="text-lg font-bold text-white">{currentIngredient?.name}</h3>
              <p className="text-xs text-zinc-400">Card unavailable right now</p>
              <div className="pt-2">
                <button
                  onClick={() => {
                    if (!currentIngredient) return;
                    setLoadingCards((prev) => ({ ...prev, [currentIngredient.name]: true }));
                    fetchIngredientCard(currentIngredient.name)
                      .then((card) => setCardsCache((prev) => ({ ...prev, [currentIngredient.name]: card })))
                      .catch(() => {})
                      .finally(() => setLoadingCards((prev) => ({ ...prev, [currentIngredient.name]: false })));
                  }}
                  className="px-3.5 py-1.5 text-xs font-medium rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors inline-flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry</span>
                </button>
              </div>
              <div className="pt-3 text-center">
                <span className="text-[11px] text-zinc-500 select-none">
                  AI-generated, may contain errors
                </span>
              </div>
            </motion.div>
          ) : (
            /* Real Ingredient Card with Swipe Gestures */
            <motion.div
              key={activeCard.name}
              drag="x"
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={0.2}
              onDragEnd={(_, info) => {
                if (info.offset.x < -50 || info.velocity.x < -300) {
                  handleNext();
                } else if (info.offset.x > 50 || info.velocity.x > 300) {
                  handlePrev();
                }
              }}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ type: 'spring', stiffness: 300, damping: 28 }}
              className="w-full bg-gradient-to-b from-zinc-900 to-zinc-950 border border-zinc-800/90 rounded-3xl p-5 space-y-4 shadow-2xl relative cursor-grab active:cursor-grabbing touch-pan-y"
            >
              {/* Card Header (No badge or colored chip on top) */}
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
                      #{selectedIndex + 1}
                    </span>
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
              {activeCard.one_liner && (
                <p className="text-xs text-zinc-300 leading-relaxed">
                  {activeCard.one_liner}
                </p>
              )}

              {/* Why Used Callout */}
              {activeCard.why_used && (
                <div className="p-3 rounded-2xl bg-zinc-900/90 border border-zinc-800 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Info className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span className="text-xs text-zinc-400 font-medium">Why it's used:</span>
                  </div>
                  <span className="text-xs font-bold text-white text-right">
                    {activeCard.why_used}
                  </span>
                </div>
              )}

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
                    How it's made (max 3 stages)
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

              {/* Single Diet Conflict / May Conflict Line (replaces dietary suitability section) */}
              {(() => {
                const currentNameKey = (currentIngredient?.name || '').toLowerCase().trim();
                const dietRes = dietResultsMap[currentNameKey];
                if (!dietRes || (dietRes.status !== 'CONFLICT' && dietRes.status !== 'MAY_CONFLICT')) {
                  return null;
                }
                const isConflict = dietRes.status === 'CONFLICT';
                return (
                  <div
                    className={`p-2.5 rounded-xl border text-xs font-medium flex items-start gap-2 ${
                      isConflict
                        ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                        : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                    }`}
                  >
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>
                      {isConflict
                        ? `Conflicts with your diet: ${dietRes.reason}`
                        : `May conflict with your diet: ${dietRes.reason}`}
                    </span>
                  </div>
                );
              })()}

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

              {/* Subtle Card-Bottom AI Disclosure (Low contrast ~11px muted gray, no badge) */}
              <div className="pt-1 text-center">
                <span className="text-[11px] text-zinc-500 select-none">
                  AI-generated, may contain errors
                </span>
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

      {/* Single Quiet Footer Line for AI disclosure on Explorer screen */}
      <footer className="pt-1 pb-1 text-center">
        <p className="text-[11px] text-zinc-500">
          Ingredient explanations are AI-generated.
        </p>
      </footer>

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
