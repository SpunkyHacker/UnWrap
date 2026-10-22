import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OnboardingModal } from './components/OnboardingModal';
import { LandingScreen } from './components/LandingScreen';
import { CaptureScreen } from './components/CaptureScreen';
import { ConfirmScreen } from './components/ConfirmScreen';
import { IngredientExplorer } from './components/IngredientExplorer';
import {
  UserProfile,
  CapturedPhoto,
  AnalysisResponse,
} from './types';
import {
  getStoredProfile,
  saveStoredProfile,
  isOnboardingCompleted,
} from './storage';
import {
  analyzePhotos,
  reanalyzeIngredients,
  checkBackendHealth,
  getDemoAnalysis,
} from './api';

type CurrentScreen = 'landing' | 'capture' | 'confirm' | 'explorer';

export const App: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<CurrentScreen>('landing');
  const [profile, setProfile] = useState<UserProfile>(getStoredProfile);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [isFirstTime, setIsFirstTime] = useState(false);

  // Photos
  const [photos, setPhotos] = useState<CapturedPhoto[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isReanalyzing, setIsReanalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Results
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);

  // Health & Demo
  const [backendHealth, setBackendHealth] = useState<any>({ status: 'checking' });
  const [isDemo, setIsDemo] = useState(() => {
    return new URLSearchParams(window.location.search).get('demo') === 'true';
  });

  // Check first time launch
  useEffect(() => {
    if (!isOnboardingCompleted()) {
      setIsFirstTime(true);
      setShowProfileModal(true);
    }
    // Poll backend health once
    checkBackendHealth().then(setBackendHealth);
  }, []);

  const handleSaveProfile = (newProfile: UserProfile) => {
    setProfile(newProfile);
    saveStoredProfile(newProfile);
    setIsFirstTime(false);
  };

  // Screen 1: Photo selected from Landing
  const handlePhotosFromLanding = (files: File[]) => {
    if (files.length === 0) return;
    const file = files[0];
    const newPhoto: CapturedPhoto = {
      id: `photo-${Date.now()}`,
      file,
      previewUrl: URL.createObjectURL(file),
    };
    setPhotos([newPhoto]);
    setCurrentScreen('capture');
  };

  // Screen 2: Retake photo
  const handleRetakePhoto = (file: File) => {
    photos.forEach((p) => URL.revokeObjectURL(p.previewUrl));
    const newPhoto: CapturedPhoto = {
      id: `photo-${Date.now()}`,
      file,
      previewUrl: URL.createObjectURL(file),
    };
    setPhotos([newPhoto]);
  };

  // Start analysis
  const handleStartAnalysis = async () => {
    setErrorMessage(null);
    setIsAnalyzing(true);

    try {
      if (isDemo) {
        // Simulated latency for demo
        await new Promise((r) => setTimeout(r, 2000));
        const demoData = getDemoAnalysis();
        setAnalysisResult(demoData);
        setCurrentScreen('confirm');
        return;
      }

      const files = photos.map((p) => p.file);
      const res = await analyzePhotos(files, profile);
      setAnalysisResult(res);
      setCurrentScreen('confirm');
    } catch (err: any) {
      console.error('Analysis error:', err);
      setErrorMessage(err.message || 'Failed to analyze photos. Please check the backend server.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Screen 3: Reanalyze when chips are edited
  const handleConfirmReanalyze = async (editedIngredients: string[]) => {
    setIsReanalyzing(true);
    try {
      if (isDemo && analysisResult) {
        await new Promise((r) => setTimeout(r, 800));
        // Mock update
        setAnalysisResult((prev) => prev ? {
          ...prev,
          parsed: {
            ...prev.parsed,
            ingredients: editedIngredients.map((name, i) => ({
              name,
              raw_text: name,
              children: [],
              position: i,
              possible_merged: false,
              suggested_split: [],
            })),
          },
        } : null);
        return;
      }

      const res = await reanalyzeIngredients(
        editedIngredients,
        profile,
        analysisResult?.transcription?.contains_statement || '',
        analysisResult?.transcription?.may_contain_statement || ''
      );

      // Preserve transcription metadata from first pass
      setAnalysisResult((prev) => ({
        ...res,
        transcription: prev?.transcription || res.transcription,
      }));
    } catch (err: any) {
      console.error('Reanalysis error:', err);
      alert(`Reanalysis failed: ${err.message}`);
    } finally {
      setIsReanalyzing(false);
    }
  };

  const handleRetake = () => {
    photos.forEach((p) => URL.revokeObjectURL(p.previewUrl));
    setPhotos([]);
    setAnalysisResult(null);
    setCurrentScreen('landing');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans antialiased">
      {/* Top Navigation */}
      <Navbar
        profile={profile}
        onOpenProfile={() => setShowProfileModal(true)}
        backendHealth={backendHealth}
        isDemo={isDemo}
        onToggleDemo={() => setIsDemo((prev) => !prev)}
        onHomeClick={handleRetake}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col">
        {currentScreen === 'landing' && (
          <LandingScreen onPhotosSelected={handlePhotosFromLanding} />
        )}

        {currentScreen === 'capture' && (
          <CaptureScreen
            photo={photos[0] || null}
            onRetakePhoto={handleRetakePhoto}
            onStartAnalysis={handleStartAnalysis}
            onBack={() => setCurrentScreen('landing')}
            isAnalyzing={isAnalyzing}
            errorMessage={errorMessage}
          />
        )}

        {currentScreen === 'confirm' && analysisResult && (
          <ConfirmScreen
            initialAnalysis={analysisResult}
            profile={profile}
            onConfirmReanalyze={handleConfirmReanalyze}
            onRetakePhotos={handleRetake}
            onExplore={() => setCurrentScreen('explorer')}
            isReanalyzing={isReanalyzing}
          />
        )}

        {currentScreen === 'explorer' && analysisResult && (
          <IngredientExplorer
            analysis={analysisResult}
            profile={profile}
            onRetake={handleRetake}
            onBackToConfirm={() => setCurrentScreen('confirm')}
          />
        )}
      </main>

      {/* Onboarding / Profile Settings Modal */}
      <OnboardingModal
        isOpen={showProfileModal}
        onClose={() => setShowProfileModal(false)}
        initialProfile={profile}
        onSaveProfile={handleSaveProfile}
        isFirstTime={isFirstTime}
      />
    </div>
  );
};

export default App;
