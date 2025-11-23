"""
PrivBERT Classification Service
Uses the pre-trained model from backend/model/privbert_final
EXACT same code as ClassificationCode.py - no training, just classification
"""
import os
import torch
import warnings
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

# Disable warnings and W&B
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Check if cuda supported GPU is available, else use cpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dictionary of Labels (same as ClassificationCode.py)
id2label = {0:'Data Collection', 1:'Data Usage', 2:'Data Sharing', 3:'User Control', 4:'Other'}
label2id = {v:k for k,v in id2label.items()}
NUM_LABELS = len(id2label)

# Paths - use pre-trained model in backend/model/privbert_final
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
MODEL_DIR = BASE_DIR / 'model' / 'privbert_final'

# Global variables (loaded once on startup)
tokenizer = None
model = None
BEST_THRESHOLDS = None

def load_model():
    """
    Load pre-trained PrivBERT model from backend/model/privbert_final
    Called once on application startup
    EXACT same loading as ClassificationCode.py
    """
    global tokenizer, model, BEST_THRESHOLDS
    
    print(f"Loading PrivBERT model (device: {DEVICE})...")
    
    # Check if model directory exists
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_DIR}\n"
            f"Make sure privbert_final model is in backend/model/ directory"
        )
    
    print(f"✓ Found model at: {MODEL_DIR}")
    
    # Load model and tokenizer from privbert_final
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Load thresholds (from ClassificationCode.py tuning)
    threshold_file = MODEL_DIR / 'thresholds.npy'
    if threshold_file.exists():
        BEST_THRESHOLDS = np.load(threshold_file)
        print(f"✓ Loaded thresholds:")
        for i, label_name in id2label.items():
            print(f"  {label_name:15s} → {BEST_THRESHOLDS[i]:.2f}")
    else:
        print("⚠ Warning: thresholds.npy not found, using default (0.5)")
        print("  Run ClassificationCode.py threshold tuning to get optimal thresholds")
        BEST_THRESHOLDS = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Move model to GPU/CPU and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    print(f"✓ PrivBERT model loaded successfully on {DEVICE}!")

def classify(texts: List[str]) -> List[Dict]:
    """
    Classify texts using PrivBERT model
    EXACT same function from ClassificationCode.py (lines 920-943)
    
    Args:
        texts: List of sentences to classify
        
    Returns:
        List of dictionaries with sentence, ids, and labels
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Tokenize your texts into inputs for model and put them in GPU
    toks = tokenizer(texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(**toks).logits
        probs = torch.sigmoid(logits).cpu().numpy()  # shape (num_of_samples, num_of_labels)

    # Binarize the predictions (1 or 0) based on the thresholds. If less, 0 if equal or more 1
    binarized = (probs >= BEST_THRESHOLDS[None, :]).astype(int)
    
    # Guarantee at least one label per text
    for i in range(binarized.shape[0]):
        if binarized[i].sum() == 0:  # If no labels, find the highest probability and set the according label to 1, its most likely others
            j = probs[i].argmax()
            binarized[i, j] = 1

    # Convert predictions into readable output
    results = []
    for t, row in zip(texts, binarized):
        ids = np.where(row == 1)[0].tolist()
        labels = [id2label[i] for i in ids]
        results.append({
            'sentence': t,
            'label': ids[0] if ids else 4,  # Return first label for compatibility with backend
            'ids': ids,
            'labels': labels
        })
    return results

def classify_sentences(sentences: List[str], batch_size: int = 32) -> List[Dict]:
    """
    Classify a list of sentences in batches
    
    Args:
        sentences: List of sentences to classify
        batch_size: Number of sentences to process at once
        
    Returns:
        List of dictionaries: [{"sentence": "...", "label": 0, "ids": [...], "labels": [...]}, ...]
    """
    if not sentences:
        return []
    
    all_results = []
    
    # Process in batches
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        batch_results = classify(batch)
        all_results.extend(batch_results)
    
    return all_results

