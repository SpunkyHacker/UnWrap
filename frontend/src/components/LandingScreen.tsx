import React, { useRef } from 'react';
import { Camera, Upload, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

interface LandingScreenProps {
  onPhotosSelected: (files: File[]) => void;
}

export const LandingScreen: React.FC<LandingScreenProps> = ({ onPhotosSelected }) => {
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onPhotosSelected(Array.from(e.target.files));
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-md mx-auto w-full min-h-[calc(100vh-80px)]">
      {/* Hidden file inputs */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={handleFileChange}
      />
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/heic,.heic"
        className="hidden"
        onChange={handleFileChange}
      />

      <div className="flex-1 flex flex-col items-center justify-center space-y-8 w-full my-auto">
        {/* Subtle tagline */}
        <div className="space-y-2">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium"
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Deterministic Allergen Engine</span>
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl"
          >
            Scan Ingredients
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-sm text-zinc-400 max-w-xs mx-auto"
          >
            Photograph any food packet to reveal allergens, origins, and how it is made.
          </motion.p>
        </div>

        {/* ONE BIG CAMERA BUTTON (Nothing else competing) */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', damping: 15, delay: 0.25 }}
          className="relative py-4"
        >
          {/* Outer glowing pulsing halo */}
          <div className="absolute inset-0 rounded-full bg-emerald-500/20 blur-2xl animate-pulse" />

          <button
            onClick={() => cameraInputRef.current?.click()}
            className="relative w-44 h-44 sm:w-48 sm:h-48 rounded-full bg-gradient-to-b from-zinc-800 to-zinc-900 border-2 border-emerald-500/40 p-2 shadow-2xl shadow-emerald-500/20 flex flex-col items-center justify-center gap-3 group active:scale-95 transition-all hover:border-emerald-400 hover:shadow-emerald-500/30"
            aria-label="Scan Food Package"
          >
            {/* Inner Ring */}
            <div className="w-36 h-36 sm:w-40 sm:h-40 rounded-full bg-gradient-to-tr from-emerald-600 via-emerald-500 to-teal-400 flex flex-col items-center justify-center text-zinc-950 font-bold shadow-inner group-hover:scale-[1.02] transition-transform">
              <Camera className="w-14 h-14 sm:w-16 sm:h-16 stroke-[1.75] drop-shadow-md" />
              <span className="text-xs uppercase font-extrabold tracking-wider mt-1 text-zinc-950">
                Snap Label
              </span>
            </div>
          </button>
        </motion.div>

        {/* Secondary subtle option */}
        <div className="pt-2">
          <button
            onClick={() => galleryInputRef.current?.click()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium text-zinc-400 hover:text-white bg-zinc-900/60 border border-zinc-800/80 hover:border-zinc-700 transition-colors"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload from photo library</span>
          </button>
        </div>
      </div>

      {/* Non-medical disclaimer footer */}
      <footer className="w-full text-center py-4 border-t border-zinc-900 text-[11px] text-zinc-600">
        Reference information only. Not medical advice. Always check product packaging.
      </footer>
    </div>
  );
};
