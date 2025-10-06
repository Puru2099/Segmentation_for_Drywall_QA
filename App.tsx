
import React, { useState, useCallback } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { PromptControls } from './components/PromptControls';
import { ResultDisplay } from './components/ResultDisplay';
import { generateSegmentationMask, fileToBase64 } from './services/geminiService';

const App: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string>('');
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = useCallback((file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImageUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
    setResultImageUrl(null);
    setError(null);
  }, []);
  
  const handleGenerateMask = useCallback(async () => {
    if (!imageFile || !prompt) {
      setError("Please upload an image and provide a prompt.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResultImageUrl(null);

    try {
      const { base64Data, mimeType } = await fileToBase64(imageFile);
      const maskBase64 = await generateSegmentationMask(base64Data, mimeType, prompt);
      
      if (maskBase64) {
        setResultImageUrl(`data:image/png;base64,${maskBase64}`);
      } else {
        setError("The model did not return an image. Please try a different prompt or image.");
      }

    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  }, [imageFile, prompt]);

  return (
    <div className="min-h-screen bg-slate-900 text-gray-200 font-sans">
      <header className="bg-slate-800/50 backdrop-blur-sm shadow-lg p-4 sticky top-0 z-10 border-b border-slate-700">
        <div className="container mx-auto">
          <h1 className="text-2xl md:text-3xl font-bold text-cyan-400">
            Prompted Segmentation for Drywall QA
          </h1>
        </div>
      </header>

      <main className="container mx-auto p-4 md:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-800 p-6 rounded-lg shadow-xl border border-slate-700">
              <h2 className="text-xl font-semibold mb-4 border-b border-slate-600 pb-2">1. Upload Image</h2>
              <ImageUploader onImageUpload={handleImageUpload} />
            </div>
            <div className="bg-slate-800 p-6 rounded-lg shadow-xl border border-slate-700">
              <h2 className="text-xl font-semibold mb-4 border-b border-slate-600 pb-2">2. Provide Prompt</h2>
              <PromptControls
                prompt={prompt}
                setPrompt={setPrompt}
                onSubmit={handleGenerateMask}
                isLoading={isLoading}
                isReady={!!imageFile && !!prompt}
              />
            </div>
          </div>

          <div className="lg:col-span-8">
            <ResultDisplay
              originalImageUrl={imageUrl}
              resultImageUrl={resultImageUrl}
              isLoading={isLoading}
              error={error}
            />
          </div>

        </div>
      </main>
      
      <footer className="text-center p-4 mt-8 text-slate-500 text-sm">
        <p>Built with React, Tailwind CSS, and the Google Gemini API.</p>
      </footer>
    </div>
  );
};

export default App;
