import React from 'react';
import { ShieldCheck, User, Sparkles } from 'lucide-react';
import { UserProfile } from '../types';

interface NavbarProps {
  profile: UserProfile;
  onOpenProfile: () => void;
  backendHealth: { status: string; vlm?: any; apple_vision?: any };
  isDemo: boolean;
  onToggleDemo: () => void;
  onHomeClick: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  profile,
  onOpenProfile,
  backendHealth,
  isDemo,
  onToggleDemo,
  onHomeClick,
}) => {
  const isHealthy = backendHealth.status === 'healthy';
  const vlmModel = backendHealth.vlm?.actual_loaded_model || backendHealth.vlm?.model || 'VLM';

  return (
    <header className="sticky top-0 z-30 backdrop-blur-md bg-zinc-950/80 border-b border-zinc-800/80 px-4 py-3">
      <div className="max-w-md mx-auto flex items-center justify-between">
        {/* Brand */}
        <button
          onClick={onHomeClick}
          className="flex items-center gap-2 text-left focus:outline-none group"
        >
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-emerald-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 group-hover:scale-105 transition-transform">
            <ShieldCheck className="w-5 h-5 text-zinc-950 font-bold" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight tracking-tight text-white flex items-center gap-1.5">
              Unwrap
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700/50">
                PWA
              </span>
            </h1>
          </div>
        </button>

        {/* Right side controls */}
        <div className="flex items-center gap-2">
          {/* VLM Status Pill */}
          <div
            title={`Backend: ${backendHealth.status} | Model: ${vlmModel}`}
            className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-zinc-900 border border-zinc-800 text-zinc-400"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="truncate max-w-[110px]">{vlmModel.replace('mlx-community/', '').replace('-Instruct-4bit', '')}</span>
          </div>

          {/* Demo toggle */}
          <button
            onClick={onToggleDemo}
            className={`px-2 py-1 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1 ${
              isDemo
                ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200'
            }`}
          >
            <Sparkles className="w-3 h-3" />
            {isDemo ? 'Demo On' : 'Demo'}
          </button>

          {/* Profile button */}
          <button
            onClick={onOpenProfile}
            className="p-2 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-white hover:border-zinc-700 transition-colors relative"
            aria-label="Edit Allergen Profile"
          >
            <User className="w-4 h-4" />
            {profile.allergens.length > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-500 text-zinc-950 font-bold text-[10px] rounded-full flex items-center justify-center shadow">
                {profile.allergens.length}
              </span>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
