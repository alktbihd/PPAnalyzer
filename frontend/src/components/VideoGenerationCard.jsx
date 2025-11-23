import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Video, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { Progress } from './ui/progress';

export default function VideoGenerationCard({ summary, onVideoGenerated }) {
  const [status, setStatus] = useState('idle'); // 'idle', 'generating', 'success', 'error'
  const [videoUrl, setVideoUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');

  const generateVideo = async () => {
    setStatus('generating');
    setProgress(0);
    setErrorMessage('');

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev;
        return prev + 5;
      });
    }, 15000); // Update every 15 seconds (longer wait time)

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/generate-video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ summary }),
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Video generation failed');
      }

      const data = await response.json();
      setVideoUrl(`${API_URL}${data.data.video_url}`);
      setProgress(100);
      setStatus('success');
      
      if (onVideoGenerated) {
        onVideoGenerated(data.data.video_url);
      }
    } catch (error) {
      clearInterval(progressInterval);
      setStatus('error');
      setErrorMessage(error.message);
      console.error('Video generation error:', error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="bg-primary p-3 rounded-lg">
            <Video className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <CardTitle className="text-xl">
              Video Explanation
            </CardTitle>
            <CardDescription>
              Generate an AI-narrated video of the summary
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {status === 'idle' && (
          <div className="text-center py-8">
            <div className="bg-muted p-6 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
              <Video className="w-10 h-10 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground mb-6">
              Create a short 20-30 second narrated video of this privacy policy
            </p>
            <Button onClick={generateVideo} size="lg">
              Generate Video
            </Button>
            <p className="text-xs text-muted-foreground mt-4">
              Short video (20-30 seconds) • Takes 2-3 minutes to generate
            </p>
          </div>
        )}

        {status === 'generating' && (
          <div className="py-8">
            <div className="flex items-center justify-center gap-3 mb-6">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span className="font-medium">Generating video...</span>
            </div>
            <Progress value={progress} className="mb-4" />
            <div className="space-y-2 text-sm text-center">
              <p className="text-muted-foreground">
                {progress < 30 && "Creating video script..."}
                {progress >= 30 && progress < 60 && "Submitting to HeyGen..."}
                {progress >= 60 && progress < 90 && "Rendering video with AI avatar..."}
                {progress >= 90 && "Finalizing video..."}
              </p>
              <p className="text-xs text-muted-foreground">
                Generating a 20-30 second video. This may take 2-3 minutes. Please wait...
              </p>
            </div>
          </div>
        )}

        {status === 'success' && videoUrl && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-green-600 bg-green-50 dark:bg-green-950 p-3 rounded-lg">
              <CheckCircle className="w-5 h-5" />
              <span className="font-medium">Video generated successfully!</span>
            </div>
            <div className="aspect-video bg-black rounded-lg overflow-hidden border-2 border-border">
              <video controls className="w-full h-full" src={videoUrl}>
                Your browser does not support the video tag.
              </video>
            </div>
            <Button 
              onClick={generateVideo} 
              variant="outline" 
              className="w-full"
            >
              Regenerate Video
            </Button>
          </div>
        )}

        {status === 'error' && (
          <div className="py-8">
            <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-950 p-4 rounded-lg mb-4">
              <AlertCircle className="w-5 h-5" />
              <div className="flex-1">
                <p className="font-medium">Video generation failed</p>
                <p className="text-sm mt-1">{errorMessage}</p>
              </div>
            </div>
            <Button onClick={generateVideo} variant="outline" className="w-full">
              Try Again
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

