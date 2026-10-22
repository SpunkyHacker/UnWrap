import React, { useRef } from 'react';
import { Camera, RefreshCw, ArrowRight, Upload, AlertCircle } from 'lucide-react';
import { CapturedPhoto } from '../types';
import { ProgressIndicator } from './ProgressIndicator';

interface CaptureScreenProps {
  photo: CapturedPhoto | null;
  onRetakePhoto: (file: File) => void;
  onStartAnalysis: () => void;
  onBack: () => void;
  isAnalyzing: boolean;
  errorMessage?: string | null;
}

export const CaptureScreen: React.FC<CaptureScreenProps> = ({
  photo,
  onRetakePhoto,
  onStartAnalysis,
  onBack,
  isAnalyzing,
  errorMessage,
}) => {
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onRetakePhoto(e.target.files[0]);
    }
  };

  if (isAnalyzing) {
    return (
      <div className="flex-1 flex items-center justify-center p-6 min-h-[calc(100vh-80px)]">
        <ProgressIndicator photoCount={1} />
      </div>
    );
  }

  return (
    <div className="flex-1 max-w-md mx-auto w-full p-5 space-y-6 flex flex-col justify-between min-h-[calc(100vh-80px)]">
      {/* Hidden file inputs for retake */}
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

      <div className="space-y-5">
        {/* Top Header */}
        <div className="flex items-center justify-between">
          <button
            onClick={onBack}
            className="p-2 -ml-2 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors flex items-center gap-1.5 text-xs font-medium"
          >
            <span>&larr; Back</span>
          </button>
          <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400">
            1 Photo
          </span>
        </div>

        {/* Prompt Header */}
        <div className="space-y-1.5">
          <h2 className="text-xl font-bold text-white tracking-tight">
            Photograph the ingredients list.
          </h2>
          <p className="text-xs text-zinc-400">
            Make sure the ingredient names and any allergen warnings are clear and well lit.
          </p>
        </div>

        {/* Error notification if any */}
        {errorMessage && (
          <div className="p-3.5 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Single Photo Preview Card */}
        {photo ? (
          <div className="relative aspect-[4/3] rounded-3xl overflow-hidden border border-zinc-800 bg-zinc-900 shadow-2xl group">
            <img
              src={photo.previewUrl}
              alt="Ingredients list photo"
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-black/20" />

            {/* Retake Floating Action Overlay */}
            <div className="absolute bottom-3 right-3 flex items-center gap-2">
              <button
                onClick={() => cameraInputRef.current?.click()}
                className="px-3 py-1.5 rounded-xl bg-zinc-950/80 hover:bg-zinc-900 border border-zinc-700/60 text-zinc-200 hover:text-white text-xs font-medium flex items-center gap-1.5 backdrop-blur-md transition-all shadow-lg active:scale-95"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retake</span>
              </button>
              <button
                onClick={() => galleryInputRef.current?.click()}
                className="p-1.5 rounded-xl bg-zinc-950/80 hover:bg-zinc-900 border border-zinc-700/60 text-zinc-200 hover:text-white backdrop-blur-md transition-all shadow-lg active:scale-95"
                title="Choose different photo from gallery"
              >
                <Upload className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        ) : (
          <div className="aspect-[4/3] rounded-3xl border-2 border-dashed border-zinc-800 bg-zinc-900/30 flex flex-col items-center justify-center gap-3 p-6 text-center">
            <div className="w-14 h-14 rounded-2xl bg-zinc-800/80 flex items-center justify-center text-zinc-400">
              <Camera className="w-7 h-7" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold text-zinc-300">No photo selected</p>
              <p className="text-xs text-zinc-500">Capture a clear photo of the ingredients list</p>
            </div>
            <button
              onClick={() => cameraInputRef.current?.click()}
              className="mt-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-bold text-xs flex items-center gap-2 shadow-lg"
            >
              <Camera className="w-4 h-4" />
              <span>Take Photo</span>
            </button>
          </div>
        )}
      </div>

      {/* Bottom Action Button */}
      <div className="pt-6 pb-2">
        <button
          onClick={onStartAnalysis}
          disabled={!photo}
          className="w-full py-4 rounded-2xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 disabled:pointer-events-none text-zinc-950 font-bold text-base flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/20 active:scale-[0.99] transition-all"
        >
          <span>Analyze Ingredients</span>
          <ArrowRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};
