import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import ProcessingStatus from './components/ProcessingStatus';
import Results from './components/Results';
import ThemeToggle from './components/ThemeToggle';

function App() {
  const [stage, setStage] = useState('upload'); // 'upload', 'processing', 'results'
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState({
    stage: 'uploading',
    progress: 0,
    message: ''
  });

  const handleFileSelect = async (file) => {
    setError(null); // Clear any previous errors
    setStage('processing');
    setProcessingStatus({
      stage: 'uploading',
      progress: 10,
      message: 'Uploading your file...'
    });

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate processing stages
      setTimeout(() => {
        setProcessingStatus({
          stage: 'extracting',
          progress: 40,
          message: 'Extracting text from document...'
        });
      }, 1000);

      setTimeout(() => {
        setProcessingStatus({
          stage: 'classifying',
          progress: 60,
          message: 'Classifying sentences with PrivBERT (using mock data)...'
        });
      }, 2000);

      setTimeout(() => {
        setProcessingStatus({
          stage: 'summarizing',
          progress: 80,
          message: 'Generating AI summary with GPT...'
        });
      }, 3000);

      // Make the actual API call
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setProcessingStatus({
        stage: 'complete',
        progress: 100,
        message: 'Analysis complete!'
      });

      // Show results after a brief delay
      setTimeout(() => {
        setResults(response.data.data);
        setStage('results');
      }, 1000);

    } catch (error) {
      console.error('Analysis error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Analysis failed. Please try again.';
      setError(errorMessage);
      setStage('upload');
    }
  };

  const handleUrlSubmit = async (url) => {
    setError(null); // Clear any previous errors
    setStage('processing');
    setProcessingStatus({
      stage: 'uploading',
      progress: 10,
      message: 'Fetching privacy policy from URL...'
    });

    try {
      // Simulate processing stages
      setTimeout(() => {
        setProcessingStatus({
          stage: 'extracting',
          progress: 40,
          message: 'Extracting text from webpage...'
        });
      }, 1000);

      setTimeout(() => {
        setProcessingStatus({
          stage: 'classifying',
          progress: 60,
          message: 'Classifying sentences with PrivBERT (using mock data)...'
        });
      }, 2000);

      setTimeout(() => {
        setProcessingStatus({
          stage: 'summarizing',
          progress: 80,
          message: 'Generating AI summary with GPT...'
        });
      }, 3000);

      // Make the actual API call
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/api/analyze-url`, {
        url: url
      });

      setProcessingStatus({
        stage: 'complete',
        progress: 100,
        message: 'Analysis complete!'
      });

      // Show results after a brief delay
      setTimeout(() => {
        setResults(response.data.data);
        setStage('results');
      }, 1000);

    } catch (error) {
      console.error('Analysis error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Analysis failed. Please try again.';
      setError(errorMessage);
      setStage('upload');
    }
  };

  const handleReset = () => {
    setStage('upload');
    setResults(null);
    setError(null);
    setProcessingStatus({
      stage: 'uploading',
      progress: 0,
      message: ''
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">
                PPAnalyzer
              </h1>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {stage === 'upload' && (
          <div className="space-y-8 animate-fadeIn">
            <div className="text-center mb-12">
              <h2 className="text-5xl font-extrabold text-foreground">
                Analyze Your Privacy Policy
              </h2>
            </div>
            {error && (
              <div className="bg-red-50 dark:bg-red-950 border-2 border-red-500 rounded-lg p-4 mb-4">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-red-800 dark:text-red-200 mb-1">Error</h3>
                    <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            )}
            <FileUpload onFileSelect={handleFileSelect} onUrlSubmit={handleUrlSubmit} error={error} />
          </div>
        )}

        {stage === 'processing' && (
          <ProcessingStatus 
            status={processingStatus} 
            error={error}
            onRetry={() => {
              setError(null);
              // Retry logic would go here based on last action
            }}
            onCancel={() => {
              setError(null);
              setStage('upload');
            }}
          />
        )}

        {stage === 'results' && results && (
          <Results data={results} onReset={handleReset} />
        )}
      </main>

    </div>
  );
}

export default App;

