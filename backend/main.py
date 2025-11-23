from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os
import logging

from services.text_extractor import extract_text_from_file
from services.summarizer import generate_fewshot_summary
from services.script_generator import generate_video_script
from services.video_generator import generate_video
from services.url_fetcher import fetch_policy_from_url, validate_url
from services.classifier import classify_sentences, load_model
from mock_data import get_mock_classification
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPAnalyzer API",
    description="Privacy Policy Analyzer for AR/VR Applications",
    version="1.0.0"
)

# CORS for React frontend
# Allow localhost for development and Azure domains for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://salmon-rock-0b1a8df03.3.azurestaticapps.net",  # Production frontend
        "https://ppanalyzer-backend.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global flag to determine whether to use real PrivBERT or mock data
USE_REAL_PRIVBERT = False

# Load PrivBERT model on startup if available
@app.on_event("startup")
async def startup_event():
    global USE_REAL_PRIVBERT
    try:
        load_model()
        USE_REAL_PRIVBERT = True
        logger.info("✓ PrivBERT model loaded - using real classification")
    except Exception as e:
        logger.warning(f"⚠ Could not load PrivBERT model: {str(e)}")
        logger.info("✓ Using mock classification data")
        USE_REAL_PRIVBERT = False

@app.get("/")
async def root():
    return {"message": "PPAnalyzer API is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Request model for URL analysis
class URLAnalysisRequest(BaseModel):
    url: str

@app.post("/api/analyze")
async def analyze_policy(file: UploadFile = File(...)):
    """
    Main endpoint: Upload privacy policy → Extract → Classify → Summarize → Generate Video
    """
    file_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{file_id}] Received file: {file.filename}")
        
        # 1. Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.pdf', '.html', '.htm']:
            raise HTTPException(status_code=400, detail="Only PDF and HTML files are supported")
        
        # 2. Save uploaded file
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"[{file_id}] File saved, extracting text...")
        
        # 3. Extract text and sentences
        sentences = extract_text_from_file(str(file_path))
        
        if not sentences or len(sentences) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        logger.info(f"[{file_id}] Extracted {len(sentences)} sentences")
        
        # 4. Classify sentences using PrivBERT
        if USE_REAL_PRIVBERT:
            # Use real PrivBERT model with async threading for concurrency
            from fastapi.concurrency import run_in_threadpool
            classified_sentences = await run_in_threadpool(classify_sentences, sentences)
            logger.info(f"[{file_id}] Classified {len(classified_sentences)} sentences using PrivBERT model")
        else:
            # Use mock data
            classified_sentences = get_mock_classification()
            logger.info(f"[{file_id}] Using mock classification data ({len(classified_sentences)} sentences)")
        
        # 5. Calculate category distribution
        categories = {
            "Collection": 0,
            "Usage": 0, 
            "Sharing": 0,
            "User Control": 0,
            "Other": 0
        }
        
        label_names = ["Collection", "Usage", "Sharing", "User Control", "Other"]
        
        for item in classified_sentences:
            label_name = label_names[item["label"]]
            categories[label_name] += 1
        
        total = len(classified_sentences)
        categories_percent = {k: round(v/total*100, 1) for k, v in categories.items()}
        
        logger.info(f"[{file_id}] Generating few-shot summary...")
        
        # 6. Generate few-shot summary with GPT
        summary = generate_fewshot_summary(classified_sentences)
        
        logger.info(f"[{file_id}] Summary generated, analysis complete!")
        
        # 7. Video generation is optional - can be triggered separately
        # This allows immediate results without waiting 2-3 minutes for video
        video_url = None
        logger.info(f"[{file_id}] Video generation skipped (can be generated separately)")
        
        # 10. Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        # 11. Return results
        return {
            "status": "success",
            "data": {
                "file_id": file_id,
                "summary": summary,
                "video_url": video_url,
                "total_sentences": len(classified_sentences),
                "categories": categories,
                "categories_percent": categories_percent,
                "classified_sentences": classified_sentences[:50]  # Return first 50 for display
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{file_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-url")
async def analyze_policy_from_url(request: URLAnalysisRequest):
    """
    Alternative endpoint: Analyze privacy policy from URL
    """
    file_id = str(uuid.uuid4())
    temp_file_path = None
    
    try:
        logger.info(f"[{file_id}] Received URL: {request.url}")
        
        # 1. Validate URL
        validate_url(request.url)
        
        # 2. Fetch content from URL
        logger.info(f"[{file_id}] Fetching content from URL...")
        temp_file_path = fetch_policy_from_url(request.url)
        
        logger.info(f"[{file_id}] Content fetched, extracting text...")
        
        # 3. Extract text and sentences
        sentences = extract_text_from_file(temp_file_path)
        
        if not sentences or len(sentences) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from URL")
        
        logger.info(f"[{file_id}] Extracted {len(sentences)} sentences")
        
        # 4. Classify sentences using PrivBERT
        if USE_REAL_PRIVBERT:
            # Use real PrivBERT model with async threading for concurrency
            from fastapi.concurrency import run_in_threadpool
            classified_sentences = await run_in_threadpool(classify_sentences, sentences)
            logger.info(f"[{file_id}] Classified {len(classified_sentences)} sentences using PrivBERT model")
        else:
            # Use mock data
            classified_sentences = get_mock_classification()
            logger.info(f"[{file_id}] Using mock classification data ({len(classified_sentences)} sentences)")
        
        # 5. Calculate category distribution
        categories = {
            "Collection": 0,
            "Usage": 0, 
            "Sharing": 0,
            "User Control": 0,
            "Other": 0
        }
        
        label_names = ["Collection", "Usage", "Sharing", "User Control", "Other"]
        
        for item in classified_sentences:
            label_name = label_names[item["label"]]
            categories[label_name] += 1
        
        total = len(classified_sentences)
        categories_percent = {k: round(v/total*100, 1) for k, v in categories.items()}
        
        logger.info(f"[{file_id}] Generating few-shot summary...")
        
        # 6. Generate few-shot summary with GPT
        summary = generate_fewshot_summary(classified_sentences)
        
        logger.info(f"[{file_id}] Summary generated, analysis complete!")
        
        # 7. Video generation is optional - can be triggered separately
        # This allows immediate results without waiting 2-3 minutes for video
        video_url = None
        logger.info(f"[{file_id}] Video generation skipped (can be generated separately)")
        
        # 10. Clean up temporary file
        if temp_file_path:
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        # 11. Return results
        return {
            "status": "success",
            "data": {
                "file_id": file_id,
                "source_url": request.url,
                "summary": summary,
                "video_url": video_url,
                "total_sentences": len(classified_sentences),
                "categories": categories,
                "categories_percent": categories_percent,
                "classified_sentences": classified_sentences[:50]  # Return first 50 for display
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{file_id}] Error: {str(e)}", exc_info=True)
        # Clean up temp file on error
        if temp_file_path:
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/generate-video")
async def generate_video_endpoint(request: dict):
    """
    Generate video from existing analysis results
    Allows users to optionally generate video after seeing analysis
    """
    file_id = str(uuid.uuid4())
    
    try:
        summary = request.get("summary")
        if not summary:
            raise HTTPException(status_code=400, detail="Summary text is required")
        
        logger.info(f"[{file_id}] Generating video from summary...")
        
        # Generate video script from summary
        script = generate_video_script(summary)
        logger.info(f"[{file_id}] Script generated, creating video with HeyGen...")
        
        # Generate HeyGen video
        video_filename = generate_video(script, file_id)
        video_url = f"/api/video/{video_filename}"
        
        logger.info(f"[{file_id}] Video generated successfully")
        
        return {
            "status": "success",
            "data": {
                "video_url": video_url,
                "video_id": file_id
            }
        }
    
    except TimeoutError as e:
        logger.error(f"[{file_id}] Video timeout: {str(e)}")
        raise HTTPException(status_code=408, detail="Video generation timed out. Please try again.")
    except Exception as e:
        logger.error(f"[{file_id}] Video error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

@app.get("/api/video/{video_filename}")
async def get_video(video_filename: str):
    """Serve generated video file"""
    video_path = OUTPUT_DIR / video_filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(video_path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

