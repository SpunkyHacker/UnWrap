import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Cpu, ShieldCheck, Sparkles, Clock } from 'lucide-react';

interface ProgressIndicatorProps {
  photoCount: number;
}

const STAGES = [
  { label: 'Formatting & checking photo orientation', icon: Eye, duration: 1.5 },
  { label: 'Local Apple Silicon Vision Model transcription', icon: Cpu, duration: 8.0 },
  { label: 'Parsing ingredient tree & checking fractions', icon: Sparkles, duration: 2.0 },
  { label: 'Deterministic allergen scan & raw text safety checks', icon: ShieldCheck, duration: 3.0 },
];

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ photoCount }) => {
  const [elapsed, setElapsed] = useState(0);
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (elapsed < 2) setStageIndex(0);
    else if (elapsed < 14) setStageIndex(1);
    else if (elapsed < 18) setStageIndex(2);
    else setStageIndex(3);
  }, [elapsed]);

  const currentStage = STAGES[stageIndex];
  const Icon = currentStage.icon;

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-zinc-900 border border-zinc-800 rounded-3xl shadow-2xl text-center space-y-6">
      {/* Animated Center Orb */}
      <div className="relative w-24 h-24 mx-auto flex items-center justify-center">
        <motion.div
          animate={{ scale: [1, 1.15, 1], rotate: [0, 180, 360] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute inset-0 rounded-full bg-gradient-to-tr from-emerald-500/20 via-emerald-400/10 to-transparent border border-emerald-500/30"
        />
        <motion.div
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          className="w-16 h-16 rounded-2xl bg-zinc-950 border border-emerald-500/50 flex items-center justify-center shadow-lg shadow-emerald-500/10 text-emerald-400"
        >
          <Icon className="w-8 h-8 animate-pulse" />
        </motion.div>
      </div>

      {/* Stage info */}
      <div className="space-y-2">
        <h3 className="text-lg font-bold text-white tracking-tight">
          Analyzing {photoCount} {photoCount === 1 ? 'Photo' : 'Photos'}...
        </h3>
        <p className="text-sm text-emerald-400 font-medium min-h-[40px] flex items-center justify-center px-4">
          {currentStage.label}
        </p>
      </div>

      {/* Animated Glowing Progress Bar */}
      <div className="w-full bg-zinc-950 rounded-full h-2.5 overflow-hidden border border-zinc-800/80 relative">
        <motion.div
          className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
          initial={{ width: '5%' }}
          animate={{ width: `${Math.min(95, Math.max(10, elapsed * 5))}%` }}
          transition={{ ease: 'linear', duration: 1 }}
        />
      </div>

      {/* Footer stats */}
      <div className="flex items-center justify-between text-xs text-zinc-500 pt-2 border-t border-zinc-800/60 font-mono">
        <span className="flex items-center gap-1">
          <Clock className="w-3.5 h-3.5 text-zinc-400" />
          Elapsed: {elapsed}s
        </span>
        <span className="text-zinc-400">Privacy: Zero cloud storage</span>
      </div>
    </div>
  );
};
