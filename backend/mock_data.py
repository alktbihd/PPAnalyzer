"""
Mock classification data loader
Uses sampleoutputbetter.txt as sample data until real PrivBERT model is integrated
"""
import json
import ast
from pathlib import Path

def get_mock_classification() -> list[dict]:
    """
    Load mock classification data from sampleoutputbetter.txt
    
    Returns:
        List of classified sentences: [{"sentence": "...", "label": 0}, ...]
    """
    # Path to the sample data file (one level up from backend/)
    sample_data_path = Path(__file__).parent.parent / "sampleoutputbetter.txt"
    
    try:
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        
        label_map = {
            "Data Collection": 0,
            "Data Usage": 1,
            "Data Sharing": 2,
            "User Control": 3,
            "Other": 4
        }
        
        classified_sentences = []
        
        # Parse the file format (sentence on one line, labels on next line with →)
        for i, line in enumerate(lines):
            if "→" not in line:
                continue
            
            # Previous line is the sentence
            if i == 0:
                continue
            
            sentence = lines[i - 1].strip()
            
            # Parse the label part
            label_part = line.split("→", 1)[1].strip()
            
            try:
                # Evaluate the dictionary
                label_dict = ast.literal_eval(label_part)
                labels = label_dict.get("labels", [])
            except Exception:
                labels = ["Other"]
            
            if not labels:
                labels = ["Other"]
            
            # Use the first label only (for simplicity)
            label_name = labels[0]
            label_id = label_map.get(label_name, 4)
            
            classified_sentences.append({
                "sentence": sentence,
                "label": label_id
            })
        
        if not classified_sentences:
            # Fallback to minimal sample
            classified_sentences = [
                {"sentence": "We collect your data.", "label": 0},
                {"sentence": "We use your data for analytics.", "label": 1},
                {"sentence": "We share data with partners.", "label": 2},
                {"sentence": "You can delete your data.", "label": 3},
                {"sentence": "This is a privacy policy.", "label": 4}
            ]
        
        return classified_sentences
    
    except Exception as e:
        print(f"Warning: Could not load mock data: {e}")
        # Return minimal fallback
        return [
            {"sentence": "We collect your personal information.", "label": 0},
            {"sentence": "We use your data to improve services.", "label": 1},
            {"sentence": "We may share data with third parties.", "label": 2},
            {"sentence": "You have the right to delete your data.", "label": 3},
            {"sentence": "This privacy policy is subject to change.", "label": 4}
        ]

