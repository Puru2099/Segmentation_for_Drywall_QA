
import React from 'react';
import { ImageIcon, WarningIcon } from './icons';

interface ResultDisplayProps {
  originalImageUrl: string | null;
  resultImageUrl: string | null;
  isLoading: boolean;
  error: string | null;
}

const ImagePanel: React.FC<{ title: string; imageUrl: string | null; isLoading?: boolean }> = ({ title, imageUrl, isLoading = false }) => {
  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold text-center mb-3 text-slate-300">{title}</h3>
      <div className="aspect-square bg-slate-800 rounded-lg overflow-hidden border-2 border-slate-700 flex items-center justify-center relative">
        {imageUrl ? (
          <img src={imageUrl} alt={title} className="w-full h-full object-contain" />
        ) : (
          <div className="text-slate-500 flex flex-col items-center">
            <ImageIcon className="w-16 h-16" />
            <span className="mt-2 text-sm">{title} will appear here</span>
          </div>
        )}
        {isLoading && (
            <div className="absolute inset-0 bg-slate-900/70 flex items-center justify-center">
              <div className="w-16 h-16 border-4 border-dashed border-cyan-400 rounded-full animate-spin"></div>
            </div>
        )}
      </div>
    </div>
  );
};

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ originalImageUrl, resultImageUrl, isLoading, error }) => {
  return (
    <div className="bg-slate-800/50 p-4 sm:p-6 rounded-lg shadow-xl border border-slate-700">
      {error && (
        <div className="mb-4 bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-md flex items-start" role="alert">
          <WarningIcon className="w-5 h-5 mr-3 mt-1 flex-shrink-0" />
          <div>
            <strong className="font-bold">Error:</strong>
            <span className="block sm:inline ml-1">{error}</span>
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
        <ImagePanel title="Original Image" imageUrl={originalImageUrl} />
        <ImagePanel title="Generated Mask" imageUrl={resultImageUrl} isLoading={isLoading && !resultImageUrl} />
      </div>
    </div>
  );
};
