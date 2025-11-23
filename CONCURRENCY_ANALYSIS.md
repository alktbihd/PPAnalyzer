# Concurrency & Request Handling in PPAnalyzer

## Current Configuration

### Backend Setup
- **Framework**: FastAPI (async-capable)
- **Server**: Uvicorn (single worker)
- **Azure Tier**: Basic (B2) - 2 vCPU, 3.5 GB RAM
- **Container**: Single instance
- **Model Loading**: PrivBERT loaded once at startup (shared across requests)

---

## How Concurrent Requests Are Handled

### 1. **FastAPI Async Architecture**

```python
# Current endpoints are synchronous
@app.post("/api/analyze")
async def analyze_policy(file: UploadFile = File(...)):
    # Even though marked async, the function body is synchronous
    sentences = extract_text_from_file(str(file_path))  # Blocking I/O
    classified_sentences = classify_sentences(sentences)  # CPU-intensive
    summary = generate_fewshot_summary(classified_sentences)  # API call
```

**Current Behavior:**
- ✅ Uvicorn can accept multiple connections
- ❌ CPU-intensive operations (PrivBERT inference) block the event loop
- ❌ Each request processes sequentially for CPU-bound tasks
- ⚠️ Multiple requests will queue and process one at a time

---

### 2. **Request Processing Pipeline**

```
Request 1: Upload → Extract (5s) → Classify (10-30s) → Summarize (5-15s) → Done
Request 2:                          [Waiting...] ────────────────────────→ Start
Request 3:                                         [Waiting...] ─────────→ Start
```

**Total time per request**: 20-50 seconds  
**Concurrent capacity**: Effectively 1 request at a time (CPU-bound)

---

### 3. **Resource Bottlenecks**

| Component | Type | Impact on Concurrency |
|-----------|------|----------------------|
| **Text Extraction** | I/O | ✅ Can be async |
| **PrivBERT Classification** | CPU (PyTorch) | ❌ Blocks event loop |
| **GPT-5 API Call** | Network I/O | ✅ Can be async |
| **HeyGen API Call** | Network I/O | ✅ Can be async |
| **Model in Memory** | RAM | ✅ Shared (476MB) |

**Critical Issue**: PrivBERT inference is CPU-intensive and currently blocks concurrent processing.

---

## Current Limitations

### With 1 Worker (Current Setup)
- ✅ Handles 1 CPU-intensive request at a time
- ✅ Multiple lightweight requests (health checks) work fine
- ❌ 2nd analysis request waits for 1st to complete
- ❌ ~20-50 second wait time if someone else is analyzing

### What Happens with Concurrent Requests?

**Scenario: 3 users analyze policies simultaneously**

```
User 1: Start ──────[Processing 30s]──────→ Done (30s)
User 2:        Wait ──────[Processing 30s]──────→ Done (60s) 
User 3:               Wait ──────[Processing 30s]──────→ Done (90s)
```

---

## Scaling Solutions

### ✅ **Quick Wins (Can Implement Now)**

#### 1. **Add Workers** (Recommended First Step)
```dockerfile
# Current
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# With 2 workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Result**: 
- 2 requests process simultaneously
- Each worker loads model (2 × 476MB = ~1GB RAM)
- Utilizes both vCPUs on B2 tier

**Trade-off**: Doubles memory usage but doubles throughput

#### 2. **Make I/O Operations Truly Async**
```python
# Current (blocking)
response = requests.get(url, headers=headers, timeout=30)

# Async (non-blocking)
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=headers, timeout=30)
```

**Result**: 
- Better concurrency for API calls
- Doesn't help with CPU-bound PrivBERT inference

#### 3. **Use Background Tasks for Video Generation**
```python
from fastapi import BackgroundTasks

@app.post("/api/generate-video")
async def generate_video_endpoint(background_tasks: BackgroundTasks, ...):
    background_tasks.add_task(generate_video, script)
    return {"status": "processing", "job_id": job_id}
```

**Result**: Video generation doesn't block analysis response

---

### 🚀 **Advanced Scaling (For High Traffic)**

#### 1. **Scale Out with Multiple Instances**
```bash
# Azure App Service
az webapp scale --name ppanalyzer-backend \
  --resource-group ppanalyzer-rg \
  --instance-count 3
```

**Cost**: ~$75/month per additional instance  
**Result**: 3× capacity (3-6 concurrent requests)

#### 2. **Add Redis Queue for Long Tasks**
```python
# Queue classification job
job_id = queue.enqueue(classify_sentences, sentences)

# Poll for results
GET /api/status/{job_id}
```

**Result**: 
- Immediate response to user
- Background workers process queue
- Better user experience

#### 3. **Separate Classification Service**
```
Frontend → API Gateway → Text Extraction API
                      ↓
                      → Classification Queue → PrivBERT Workers (3 instances)
                      ↓
                      → GPT Summarization API
```

**Result**: Microservices architecture for horizontal scaling

---

## Recommended Immediate Actions

### **For Current Traffic (< 10 concurrent users)**

1. **Add 2 Workers** (Easy, costs nothing extra)
   - Increases capacity to 2 concurrent requests
   - Uses available CPU cores

2. **Make API Calls Async** (Improves responsiveness)
   - Replace `requests` with `httpx` 
   - Use `asyncio` for concurrent API calls

3. **Add Request Timeout** (Prevent queue buildup)
   - Set max processing time per request
   - Return error if queue is too long

### **Code Changes Needed**

```python
# 1. Add worker configuration
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# 2. Make URL fetching async
async def fetch_policy_from_url_async(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30)
        # ...

# 3. Add concurrency limits
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_concurrent=2):
        super().__init__(app)
        self.max_concurrent = max_concurrent
        self.current = 0
```

---

## Performance Metrics

### Current Capacity
- **Requests per minute**: ~2-3 (single worker)
- **Concurrent users**: 1 effectively
- **Average latency**: 20-50 seconds
- **Queue time**: 0-90 seconds (if others analyzing)

### With 2 Workers
- **Requests per minute**: ~4-6
- **Concurrent users**: 2
- **Average latency**: 20-50 seconds
- **Queue time**: 0-45 seconds

### With 3 Instances + 2 Workers Each
- **Requests per minute**: ~12-18
- **Concurrent users**: 6
- **Average latency**: 20-50 seconds
- **Queue time**: 0-20 seconds

---

## Testing Concurrency

```bash
# Simulate 3 concurrent requests
for i in {1..3}; do
  curl -X POST "https://ppanalyzer-backend.azurewebsites.net/api/analyze-url" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://www.example.com/privacy"}' &
done
wait
```

---

## Summary

**Current State**: Single-threaded processing, one request at a time  
**Bottleneck**: CPU-intensive PrivBERT inference  
**Quick Fix**: Add workers to utilize multiple cores  
**Long-term**: Queue system + horizontal scaling for high traffic

**For your project (academic/demo)**: Current setup is fine for single-user demos. Adding 2 workers would handle classroom demos with multiple simultaneous users.

