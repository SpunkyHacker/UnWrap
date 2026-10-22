import React from 'react';
import { Cpu, AlertTriangle } from 'lucide-react';
import { TranscriptionResult } from '../types';

interface EngineBadgeProps {
  transcription?: TranscriptionResult | null;
}

export const EngineBadge: React.FC<EngineBadgeProps> = ({ transcription }) => {
  if (!transcription || !transcription.engine_used) return null;

  const isFallback = transcription.fallback_occurred;
  const isVLM = transcription.engine_used.toLowerCase().includes('qwen') || transcription.engine_used.toLowerCase().includes('vlm');
  const isApple = transcription.engine_used.toLowerCase().includes('apple_vision');

  const engineName = isVLM
    ? 'Qwen2.5-VL (Local VLM)'
    : isApple
    ? 'Apple Vision OCR'
    : transcription.engine_used;

  return (
    <div className="space-y-2">
      {/* Fallback Alert Banner */}
      {isFallback && (
        <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Automatic Fallback Active</span>
            The vision model was offline or timed out; product text was transcribed using native on-device Apple Vision OCR.
          </div>
        </div>
      )}

      {/* Engine pill */}
      <div className="flex items-center justify-between text-xs text-zinc-400 bg-zinc-900/60 border border-zinc-800/80 px-3.5 py-2 rounded-xl">
        <div className="flex items-center gap-1.5 font-mono">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>Engine: <strong className="text-zinc-200">{engineName}</strong></span>
        </div>
        {transcription.time_ms > 0 && (
          <span className="font-mono text-zinc-500">
            {(transcription.time_ms / 1000).toFixed(2)}s
          </span>
        )}
      </div>
    </div>
  );
};
