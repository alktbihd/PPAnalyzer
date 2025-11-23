"""
GPT-based summarization using Few-Shot prompting
Adapted from pp_api.py
"""
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_fewshot_summary(classified_sentences: list[dict]) -> str:
    """
    Generate few-shot summary using GPT
    
    Args:
        classified_sentences: List of dicts with 'sentence' and 'label' keys
        
    Returns:
        Summary text string
    """
    
    # Convert to format GPT expects
    privbert_output = [
        {"sentence": item["sentence"], "label": item["label"]}
        for item in classified_sentences
    ]
    
    privbert_text = json.dumps(privbert_output, indent=2)
    
    # Few-shot prompt (exact copy from pp_api.py)
    prompt = f"""
You are a privacy policy explainer.
Below are examples showing how to summarize machine labeled policies.

Example 1:
Input:
[
  {{"sentence": "We collect your location and contact details.", "label": 0}},
  {{"sentence": "We use your data to improve app performance.", "label": 1}}
]
Output:
Summary of the Policy:
The company collects location and contact data to enhance its services.
Key Observations:
It clearly states the purpose but does not mention retention time.
Recommendations:
Limit permissions if you prefer not to share location data.

Example 2:
Input:
[
  {{"sentence": "We share anonymized analytics with advertisers.", "label": 2}},
  {{"sentence": "You may disable ad tracking in your settings.", "label": 3}}
]
Output:
Summary of the Policy:
The policy allows sharing of anonymized data with advertisers but offers an opt-out.
Key Observations:
Transparency is decent but technical information are not mentioned.
Recommendations:
Review your ad preferences regularly.

Be concise and explain it as if to a high school student.
Make it XR/VR related if applicable.
Now explain the following policy:

PrivBERT Output:
{privbert_text}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {str(e)}")

