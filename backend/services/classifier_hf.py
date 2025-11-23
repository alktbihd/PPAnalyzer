"""
Hugging Face Inference API classifier
Uses HF API for fast GPU inference instead of local model
"""
import os
import requests
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Label mappings (same as classifier.py)
id2label = {
    0: "Data Collection",
    1: "Data Usage",
    2: "Data Sharing",
    3: "User Control",
    4: "Other"
}

label2id = {v: k for k, v in id2label.items()}

# Thresholds from trained model
BEST_THRESHOLDS = np.array([0.95, 0.65, 0.85, 0.25, 0.25])

# Hugging Face API configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "saeedalketbi/privbert-final")
HF_API_URL = f"https://router.huggingface.co/models/{HF_MODEL_ID}"  # Updated endpoint

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def classify_batch_hf(texts: List[str]) -> List[Dict]:
    """
    Classify texts using Hugging Face Inference API
    
    Args:
        texts: List of sentences to classify
        
    Returns:
        List of dictionaries with sentence, label, ids, and labels
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set. Please set it in environment variables.")
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": texts,
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    try:
        logger.info(f"Calling HF API for {len(texts)} sentences...")
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # HF API returns list of predictions
        api_results = response.json()
        
        # Process results
        results = []
        for i, text in enumerate(texts):
            # Get logits for this text
            if isinstance(api_results[i], list):
                # API returns list of label scores
                logits = np.array([item['score'] for item in api_results[i]])
            else:
                # Fallback: assume direct logits
                logits = np.array(api_results[i])
            
            # Apply sigmoid and thresholds
            probs = sigmoid(logits)
            binarized = (probs >= BEST_THRESHOLDS).astype(int)
            
            # Guarantee at least one label
            if binarized.sum() == 0:
                j = probs.argmax()
                binarized[j] = 1
            
            # Convert to output format
            ids = np.where(binarized == 1)[0].tolist()
            labels = [id2label[idx] for idx in ids]
            
            results.append({
                'sentence': text,
                'label': ids[0] if ids else 4,
                'ids': ids,
                'labels': labels
            })
        
        logger.info(f"✓ HF API classified {len(results)} sentences successfully")
        return results
        
    except requests.exceptions.Timeout:
        logger.error("HF API request timed out")
        raise RuntimeError("Hugging Face API timeout. Model might be loading, please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API error: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response: {e.response.text}")
        raise RuntimeError(f"Hugging Face API error: {str(e)}")

def classify_sentences(sentences: List[str], batch_size: int = 32) -> List[Dict]:
    """
    Classify a list of sentences in batches using HF API
    
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
        batch_results = classify_batch_hf(batch)
        all_results.extend(batch_results)
        
        logger.info(f"Processed batch {start//batch_size + 1}/{(len(sentences)-1)//batch_size + 1}")
    
    return all_results

def check_hf_api_status():
    """Check if HF API is available and model is loaded"""
    if not HF_API_TOKEN:
        return False, "HF_API_TOKEN not configured"
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.get(HF_API_URL, headers=headers, timeout=5)
        
        if response.status_code == 200:
            return True, "HF API ready"
        else:
            return False, f"HF API returned {response.status_code}"
    except Exception as e:
        return False, f"Cannot reach HF API: {str(e)}"

