"""
Video script generator from privacy policy summary
Converts the summary into a narration script for HeyGen
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_video_script(summary: str) -> str:
    """
    Generate a video narration script from the privacy policy summary
    
    Args:
        summary: The privacy policy summary text
        
    Returns:
        Video script text suitable for narration
    """
    
    prompt = f"""
You are a script writer for short educational videos.

Convert the following privacy policy summary into a 20-30 second narration script.
The script should:
- Be very concise (aim for 50-75 words maximum)
- Highlight only the 2-3 most critical points
- Be spoken in a friendly, clear tone
- Use simple, everyday language
- Be suitable for a 20-30 second video

Privacy Policy Summary:
{summary}

Generate ONLY the narration script.
Keep it short.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0  # Increased timeout for script generation
        )
        
        script = response.choices[0].message.content.strip()
        
        # Limit to 450 characters for 20-30 second video
        # Average speech is ~150 words per minute = 2.5 words/sec
        # 25 seconds × 2.5 = ~60 words = ~450 characters
        if len(script) > 450:
            script = script[:447] + "..."
        
        return script
    
    except Exception as e:
        raise RuntimeError(f"Failed to generate video script: {str(e)}")

