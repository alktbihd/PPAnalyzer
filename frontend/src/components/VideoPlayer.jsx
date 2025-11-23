import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Play } from 'lucide-react';

export default function VideoPlayer({ videoUrl }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="bg-primary p-3 rounded-lg">
            <Play className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <CardTitle className="text-2xl">
              Video Explanation
            </CardTitle>
            <CardDescription>
              AI-generated narration of the policy summary
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="aspect-video bg-black dark:bg-black rounded-lg overflow-hidden border-2 border-border">
          <video
            controls
            className="w-full h-full"
            src={videoUrl}
          >
            Your browser does not support the video tag.
          </video>
        </div>
      </CardContent>
    </Card>
  );
}

