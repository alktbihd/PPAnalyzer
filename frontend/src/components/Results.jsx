import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { FileText, PieChart, Download, ArrowLeft } from 'lucide-react';
import VideoGenerationCard from './VideoGenerationCard';
import SummaryDisplay from './SummaryDisplay';
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const COLORS = {
  'Collection': '#3b82f6',
  'Usage': '#10b981',
  'Sharing': '#f59e0b',
  'User Control': '#8b5cf6',
  'Other': '#6b7280'
};

export default function Results({ data, onReset }) {
  const chartData = Object.entries(data.categories_percent).map(([name, value]) => ({
    name,
    value,
    count: data.categories[name]
  }));

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
        <div>
          <h1 className="text-4xl font-bold mb-2">
            Analysis Complete
          </h1>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">
              {data.total_sentences} sentences analyzed
            </Badge>
          </div>
        </div>
        <Button 
          onClick={onReset} 
          variant="outline"
          size="lg"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Analyze Another
        </Button>
      </div>
      
      <Separator className="my-6" />

      {/* Two Column Layout */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left Column - Summary */}
        <Card className="h-fit">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="bg-primary p-3 rounded-lg">
                <FileText className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">
                  Policy Summary
                </CardTitle>
                <CardDescription>
                  Few-Shot AI Analysis
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <SummaryDisplay summary={data.summary} />
          </CardContent>
        </Card>

        {/* Right Column - Category Breakdown */}
        <Card className="h-fit">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="bg-primary p-3 rounded-lg">
                <PieChart className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">
                  Category Breakdown
                </CardTitle>
                <CardDescription>
                  Content Distribution
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {/* Chart */}
            <div className="h-64 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.value}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </RechartsPie>
              </ResponsiveContainer>
            </div>

            {/* Stats */}
            <div className="space-y-2">
              {Object.entries(data.categories).map(([category, count]) => (
                <div 
                  key={category} 
                  className="flex items-center justify-between p-3 bg-muted hover:bg-accent rounded-lg border transition-all"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: COLORS[category] }}
                    />
                    <span className="text-sm font-medium">
                      {category}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold">{count}</span>
                    <span className="text-xs text-muted-foreground ml-1">
                      ({data.categories_percent[category]}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Video Generation Card - User can generate video after analysis */}
      <VideoGenerationCard 
        summary={data.summary}
        onVideoGenerated={(url) => console.log('Video generated:', url)}
      />

      {/* Sample Sentences - Full Width */}
      {data.classified_sentences && data.classified_sentences.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="bg-primary p-3 rounded-lg">
                <FileText className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">
                  Sample Classified Sentences
                </CardTitle>
                <CardDescription>
                  First 10 sentences from the analysis
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-3">
              {data.classified_sentences.slice(0, 10).map((item, index) => (
                <div 
                  key={index} 
                  className="p-4 border-2 border-border rounded-lg hover:border-primary/50 hover:bg-accent transition-all duration-200"
                >
                  <p className="text-sm leading-relaxed mb-3">
                    {item.sentence}
                  </p>
                  <Badge
                    style={{
                      backgroundColor: COLORS[['Collection', 'Usage', 'Sharing', 'User Control', 'Other'][item.label]],
                      color: 'white',
                      borderColor: 'transparent'
                    }}
                  >
                    {['Collection', 'Usage', 'Sharing', 'User Control', 'Other'][item.label]}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

