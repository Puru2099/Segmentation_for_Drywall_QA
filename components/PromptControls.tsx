
import React from 'react';
import { SparklesIcon, SpinnerIcon } from './icons';

interface PromptControlsProps {
  prompt: string;
  setPrompt: (prompt: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  isReady: boolean;
}

const examplePrompts = [
  "segment crack",
  "segment taping area",
  "segment drywall seam",
  "segment wall crack",
];

export const PromptControls: React.FC<PromptControlsProps> = ({ prompt, setPrompt, onSubmit, isLoading, isReady }) => {
  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="prompt" className="block text-sm font-medium text-slate-300 mb-1">
          Enter a prompt
        </label>
        <textarea
          id="prompt"
          rows={3}
          className="w-full bg-slate-700 border border-slate-600 rounded-md shadow-sm p-2 text-gray-200 focus:ring-cyan-500 focus:border-cyan-500 transition"
          placeholder="e.g., 'segment the large crack on the wall'"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>
      
      <div>
        <p className="text-sm font-medium text-slate-400 mb-2">Or try an example:</p>
        <div className="flex flex-wrap gap-2">
          {examplePrompts.map((p) => (
            <button
              key={p}
              onClick={() => setPrompt(p)}
              className="px-3 py-1 text-sm bg-slate-700 hover:bg-slate-600 rounded-full text-slate-300 transition-colors"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="pt-2">
        <button
          onClick={onSubmit}
          disabled={!isReady || isLoading}
          className="w-full flex items-center justify-center px-4 py-3 bg-cyan-600 text-white font-bold rounded-lg shadow-md hover:bg-cyan-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-200"
        >
          {isLoading ? (
            <>
              <SpinnerIcon className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
              Generating...
            </>
          ) : (
            <>
              <SparklesIcon className="-ml-1 mr-2 h-5 w-5" />
              Generate Mask
            </>
          )}
        </button>
      </div>
    </div>
  );
};
