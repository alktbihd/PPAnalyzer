#!/bin/bash

echo "Starting PPAnalyzer Backend..."

# Download spaCy model if not already present
if [ ! -d "/home/.spacy/data/en_core_web_sm-3.7.0" ]; then
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
fi

# Create necessary directories
mkdir -p uploads outputs

echo "Starting Uvicorn server..."
# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000

