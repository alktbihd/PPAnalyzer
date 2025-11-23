# PPAnalyzer Backend

FastAPI backend for privacy policy analysis.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
HEYGEN_API_KEY=your_key_here
```

4. Run server:
```bash
python main.py
# or
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`

## API Endpoints

### POST /api/analyze
Upload and analyze a privacy policy

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF or HTML)

**Response:**
```json
{
  "status": "success",
  "data": {
    "file_id": "uuid",
    "summary": "AI-generated summary text",
    "video_url": "/api/video/filename.mp4",
    "total_sentences": 123,
    "categories": {
      "Collection": 30,
      "Usage": 25,
      ...
    },
    "categories_percent": {
      "Collection": 24.4,
      ...
    },
    "classified_sentences": [...]
  }
}
```

### GET /api/video/{video_filename}
Retrieve generated video file

### GET /api/health
Health check endpoint

## Services

### text_extractor.py
Extracts text from PDF and HTML files using:
- PyPDF2 for PDFs
- BeautifulSoup for HTML
- spaCy for sentence segmentation

### summarizer.py
Generates few-shot summaries using OpenAI GPT-4

### video_generator.py
Creates narrated videos using HeyGen API

## Mock Data

Currently uses `../privbert_summary_results.json` for classification.
To integrate real PrivBERT:
1. Create `services/classifier.py`
2. Replace `get_mock_classification()` call in `main.py`

# Docker Deployment
