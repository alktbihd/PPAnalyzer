import React from 'react';
import { Loader2, Check, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Button } from './ui/button';

export default function ProcessingStatus({ status, error, onRetry, onCancel }) {
  const steps = [
    { id: 'uploading', label: 'Uploading file', progress: 20 },
    { id: 'extracting', label: 'Extracting text', progress: 40 },
    { id: 'classifying', label: 'Classifying sentences with PrivBERT', progress: 60 },
    { id: 'summarizing', label: 'Generating AI summary with GPT', progress: 80 },
    { id: 'complete', label: 'Analysis complete', progress: 100 },
  ];

  const currentStepIndex = steps.findIndex(step => step.id === status.stage);

  return (
    <Card className="w-full max-w-3xl mx-auto animate-slideUp">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-2xl">
          <div className="bg-primary p-3 rounded-lg">
            <Loader2 className="w-7 h-7 animate-spin text-primary-foreground" />
          </div>
          <span>Processing Privacy Policy</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-8">
          <div className="flex justify-between mb-2 text-sm font-medium">
            <span>Progress</span>
            <span className="font-bold">{status.progress || 0}%</span>
          </div>
          <Progress value={status.progress || 0} className="h-2.5" />
        </div>
        
        <div className="space-y-3">
          {steps.map((step, index) => {
            const isComplete = index < currentStepIndex;
            const isCurrent = index === currentStepIndex;
            
            return (
              <div
                key={step.id}
                className={`flex items-center gap-4 p-4 rounded-lg border-2 transition-all duration-300 ${
                  isComplete
                    ? 'bg-secondary border-primary/20'
                    : isCurrent
                    ? 'bg-accent border-primary shadow-md'
                    : 'bg-muted/30 border-border'
                }`}
              >
                <div className="flex-shrink-0">
                  {isComplete ? (
                    <div className="bg-primary p-1 rounded-full">
                      <Check className="w-4 h-4 text-primary-foreground" />
                    </div>
                  ) : isCurrent ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <div className="w-6 h-6 rounded-full border-2 border-border" />
                  )}
                </div>
                <span className={`flex-1 font-medium ${
                  isComplete || isCurrent ? 'text-foreground' : 'text-muted-foreground'
                }`}>
                  {step.label}
                </span>
                {isComplete && (
                  <span className="text-xs font-semibold">✓</span>
                )}
              </div>
            );
          })}
        </div>

        {status.message && !error && (
          <div className="mt-8 p-4 bg-muted border-l-4 border-primary rounded-lg">
            <p className="text-sm font-medium">{status.message}</p>
          </div>
        )}

        {error && (
          <div className="mt-8 p-4 bg-red-50 dark:bg-red-950 border-2 border-red-500 rounded-lg">
            <div className="flex items-start gap-3 mb-4">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-semibold text-red-800 dark:text-red-200 mb-1">Error</h3>
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </div>
            {(onRetry || onCancel) && (
              <div className="flex gap-2">
                {onRetry && (
                  <Button onClick={onRetry} variant="outline" size="sm" className="border-red-500 text-red-700 hover:bg-red-100 dark:text-red-300 dark:hover:bg-red-900">
                    Try Again
                  </Button>
                )}
                {onCancel && (
                  <Button onClick={onCancel} variant="ghost" size="sm">
                    Cancel
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

