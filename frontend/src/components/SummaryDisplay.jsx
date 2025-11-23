import React from 'react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { AlertCircle, Shield, Lightbulb } from 'lucide-react';

export default function SummaryDisplay({ summary }) {
  // Parse the summary into sections
  const parseSummary = (text) => {
    const sections = {
      summary: '',
      observations: '',
      recommendations: ''
    };

    // Split by common headers
    const summaryMatch = text.match(/Summary of the Policy:?([\s\S]*?)(?=Key Observations:|Recommendations:|$)/i);
    const observationsMatch = text.match(/Key Observations:?([\s\S]*?)(?=Recommendations:|$)/i);
    const recommendationsMatch = text.match(/Recommendations:?([\s\S]*?)$/i);

    if (summaryMatch) sections.summary = summaryMatch[1].trim();
    if (observationsMatch) sections.observations = observationsMatch[1].trim();
    if (recommendationsMatch) sections.recommendations = recommendationsMatch[1].trim();

    return sections;
  };

  const sections = parseSummary(summary);

  return (
    <Accordion type="multiple" defaultValue={['summary', 'observations', 'recommendations']} className="w-full">
      {/* Summary Section */}
      {sections.summary && (
        <AccordionItem value="summary">
          <AccordionTrigger className="text-lg font-semibold hover:no-underline">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Summary of the Policy
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 text-sm leading-relaxed pl-7">
              {sections.summary.split('\n\n').map((paragraph, index) => (
                <div key={index}>
                  {paragraph.split('\n').map((line, i) => {
                    // Check if it's a bullet point
                    if (line.trim().startsWith('-')) {
                      return (
                        <div key={i} className="flex gap-2 mb-2">
                          <span className="text-muted-foreground">•</span>
                          <span className="flex-1">{line.trim().substring(1).trim()}</span>
                        </div>
                      );
                    }
                    return <p key={i} className="mb-2">{line}</p>;
                  })}
                </div>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Key Observations */}
      {sections.observations && (
        <AccordionItem value="observations">
          <AccordionTrigger className="text-lg font-semibold hover:no-underline">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              Key Observations
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-2 text-sm leading-relaxed pl-7">
              {sections.observations.split('\n').map((line, index) => {
                if (line.trim().startsWith('-')) {
                  return (
                    <div key={index} className="flex gap-2 p-2 rounded-lg bg-muted/50">
                      <span className="text-muted-foreground">•</span>
                      <span className="flex-1">{line.trim().substring(1).trim()}</span>
                    </div>
                  );
                }
                return line.trim() ? <p key={index} className="mb-2">{line}</p> : null;
              })}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Recommendations */}
      {sections.recommendations && (
        <AccordionItem value="recommendations" className="border-b-0">
          <AccordionTrigger className="text-lg font-semibold hover:no-underline">
            <div className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Recommendations
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-2 text-sm leading-relaxed pl-7">
              {sections.recommendations.split('\n').map((line, index) => {
                if (line.trim().startsWith('-')) {
                  return (
                    <div key={index} className="flex gap-2 p-3 rounded-lg bg-accent border border-border">
                      <span className="text-primary font-bold">→</span>
                      <span className="flex-1">{line.trim().substring(1).trim()}</span>
                    </div>
                  );
                }
                return line.trim() ? <p key={index} className="mb-2">{line}</p> : null;
              })}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}

