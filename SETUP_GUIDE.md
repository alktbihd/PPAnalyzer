# PPAnalyzer Setup Guide

Complete step-by-step guide to get PPAnalyzer running on your machine.

## Prerequisites Checklist

- [ ] Python 3.9 or higher installed
- [ ] Node.js 18 or higher installed
- [ ] OpenAI API key (for GPT-4 summarization)
- [ ] HeyGen API key (for video generation)

## Step 1: Clone/Navigate to Project

```bash
cd /path/to/ppaudit
```

## Step 2: Backend Setup

### 2.1 Navigate to backend directory
```bash
cd backend
```

### 2.2 Create and activate virtual environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2.3 Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2.4 Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 2.5 Create `.env` file

Create a file named `.env` in the `backend/` directory:

```env
OPENAI_API_KEY=sk-proj-your_openai_key_here
HEYGEN_API_KEY=your_heygen_key_here
```

**Important:** Replace the placeholder keys with your actual API keys!

### 2.6 Test backend

```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test by opening: http://localhost:8000/docs (FastAPI Swagger UI)

**Leave this terminal running!**

## Step 3: Frontend Setup

### 3.1 Open a NEW terminal and navigate to frontend

```bash
cd /path/to/ppaudit/frontend
```

### 3.2 Install Node.js dependencies

```bash
npm install
```

This will take 2-3 minutes.

### 3.3 Start development server

```bash
npm run dev
```

You should see:
```
VITE v5.x.x ready in xxx ms
➜  Local:   http://localhost:3000/
```

## Step 4: Access the Application

Open your browser and go to: **http://localhost:3000**

You should see the PPAnalyzer landing page with a file upload interface.

## Step 5: Test the Application

1. **Prepare a test file:**
   - Use any privacy policy PDF or HTML file
   - Or use one from your `pp_policies/` folder

2. **Upload the file:**
   - Drag and drop onto the upload area
   - Or click "Browse Files"

3. **Watch the processing:**
   - You'll see a progress indicator
   - Stages: Uploading → Extracting → Classifying → Summarizing → Video

4. **View results:**
   - Summary text
   - Video player (if video generated successfully)
   - Category breakdown chart
   - Sample classified sentences

## Common Issues

### Issue: ModuleNotFoundError: No module named 'fastapi'

**Solution:** Make sure you activated the virtual environment:
```bash
source backend/venv/bin/activate  # macOS/Linux
backend\venv\Scripts\activate  # Windows
```

### Issue: spacy.cli.download failed

**Solution:** Install manually:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Frontend shows "Network Error"

**Solution:** 
1. Check backend is running on port 8000
2. Check console for CORS errors
3. Restart both servers

### Issue: Video generation fails

**Solution:**
1. Check your HeyGen API key is valid
2. Check you have HeyGen credits
3. Video generation takes 2-3 minutes - be patient

### Issue: npm install fails

**Solution:**
```bash
# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## Running Both Servers Simultaneously

### Option 1: Two Terminal Windows (Recommended for Development)

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Option 2: Background Processes (macOS/Linux)

```bash
# Start backend in background
cd backend
source venv/bin/activate
python main.py > backend.log 2>&1 &

# Start frontend
cd ../frontend
npm run dev
```

## Stopping the Servers

**Backend:**
- Press `Ctrl + C` in the terminal running `python main.py`

**Frontend:**
- Press `Ctrl + C` in the terminal running `npm run dev`

## Next Steps

### 1. Integrate Real PrivBERT Model

Currently using mock data. To use real PrivBERT:

1. Ask your friend for the model server URL
2. Create `backend/services/classifier.py`
3. Update `backend/main.py` to call real classifier instead of `get_mock_classification()`

### 2. Customize

- Edit prompts in `backend/services/summarizer.py`
- Modify UI in `frontend/src/components/`
- Add new features as needed

## API Keys

### Where to Get Them

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Sign up / Log in
3. Create new API key
4. Copy and paste into `.env`

**HeyGen:**
1. Go to https://app.heygen.com/
2. Sign up for account
3. Navigate to Settings → API Keys
4. Create new API key
5. Copy and paste into `.env`

## Need Help?

Check the logs:
- Backend: Terminal running `python main.py`
- Frontend: Browser console (F12)
- Network: Browser DevTools → Network tab

## Success Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Can access http://localhost:3000
- [ ] Can upload a file
- [ ] Processing completes successfully
- [ ] Results display correctly

If all boxes are checked, you're ready to go! 🎉

