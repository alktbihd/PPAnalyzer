import React, { useCallback, useState } from 'react';
import { Upload, FileText, Link, AlertCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export default function FileUpload({ onFileSelect, onUrlSubmit, error }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputMethod, setInputMethod] = useState('file'); // 'file' or 'url'
  const [urlInput, setUrlInput] = useState('');
  const [fileError, setFileError] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'html', 'htm'].includes(extension)) {
      setFileError('Please upload a PDF or HTML file');
      setSelectedFile(null);
      return;
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB
      setFileError('File size exceeds 10MB limit. Please upload a smaller file.');
      setSelectedFile(null);
      return;
    }
    setFileError(null);
    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  const handleUrlSubmit = () => {
    if (urlInput.trim()) {
      onUrlSubmit(urlInput.trim());
    }
  };

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader className="space-y-4">
        <CardTitle className="text-2xl font-bold text-center">
          Choose Your Input Method
        </CardTitle>
        <div className="flex gap-2 p-1 bg-muted rounded-lg">
            <Button
              variant={inputMethod === 'file' ? 'default' : 'ghost'}
              onClick={() => {
                setInputMethod('file');
                setUrlInput('');
                setFileError(null);
              }}
              className="flex-1"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload File
            </Button>
            <Button
              variant={inputMethod === 'url' ? 'default' : 'ghost'}
              onClick={() => {
                setInputMethod('url');
                setSelectedFile(null);
                setFileError(null);
              }}
              className="flex-1"
            >
              <Link className="w-4 h-4 mr-2" />
              Enter URL
            </Button>
        </div>
      </CardHeader>
      <CardContent>
        {inputMethod === 'file' ? (
          <div
            className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
              dragActive
                ? 'border-primary bg-muted/50 scale-105'
                : 'border-border hover:border-primary/50 hover:bg-muted/30'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".pdf,.html,.htm"
              onChange={handleChange}
            />
            
            <div className="flex flex-col items-center gap-4">
              {selectedFile ? (
                <>
                  <div className="bg-primary/10 p-4 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-10 h-10 text-foreground" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold">{selectedFile.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                  <div className="flex gap-3 mt-2">
                    <Button 
                      onClick={handleUpload} 
                      size="lg"
                    >
                      Analyze Policy →
                    </Button>
                    <Button
                      onClick={() => setSelectedFile(null)}
                      variant="outline"
                      size="lg"
                    >
                      Change File
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="bg-muted p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-12 h-12 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-xl font-bold mb-2">
                      Upload Privacy Policy
                    </p>
                    <p className="text-sm text-muted-foreground mb-6">
                      Drag and drop your file here, or click to browse
                    </p>
                    <label htmlFor="file-upload">
                      <Button 
                        asChild 
                        size="lg"
                      >
                        <span>Browse Files</span>
                      </Button>
                    </label>
                  </div>
                  <div className="mt-6 flex items-center justify-center gap-2 text-xs text-muted-foreground">
                    <span className="px-3 py-1 bg-muted rounded-full">PDF</span>
                    <span className="px-3 py-1 bg-muted rounded-full">HTML</span>
                    <span>•</span>
                    <span>Max 10MB</span>
                  </div>
                </>
              )}
              {fileError && (
                <div className="mt-4 bg-red-50 dark:bg-red-950 border-2 border-red-500 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                    <p className="text-sm text-red-700 dark:text-red-300">{fileError}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4 p-8">
            <div className="flex flex-col items-center gap-6">
              <div className="bg-muted p-6 rounded-full w-24 h-24 flex items-center justify-center">
                <Link className="w-12 h-12 text-muted-foreground" />
              </div>
              <div className="w-full max-w-md">
                <label htmlFor="url-input" className="block text-sm font-semibold mb-3">
                  Privacy Policy URL
                </label>
                <input
                  id="url-input"
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/privacy-policy"
                  className="w-full px-4 py-3 bg-background border-2 border-input rounded-lg focus:ring-2 focus:ring-ring focus:border-ring transition-all outline-none"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && urlInput.trim()) {
                      handleUrlSubmit();
                    }
                  }}
                />
                <p className="text-xs text-muted-foreground mt-3">
                  Enter a complete URL starting with http:// or https://
                </p>
                {error && inputMethod === 'url' && (
                  <div className="mt-3 bg-red-50 dark:bg-red-950 border-2 border-red-500 rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                      <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                    </div>
                  </div>
                )}
              </div>
              <Button
                onClick={handleUrlSubmit}
                size="lg"
                disabled={!urlInput.trim()}
              >
                Analyze Policy →
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

