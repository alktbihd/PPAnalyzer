"""
HeyGen video generation
Adapted from pp_video_api.py
"""
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
AVATAR_ID = "Albert_public_3"
VOICE_ID = "f38a635bee7a4d1f9b0a654a31d050d2"

def generate_video(script_text: str, file_id: str) -> str:
    """
    Generate HeyGen video and return filename
    
    Args:
        script_text: The narration script to use
        file_id: Unique identifier for this request
        
    Returns:
        Video filename (not full path)
    """
    
    # Limit text length for short video (20-30 seconds)
    script = script_text[:450]
    
    # 1. Request video generation
    video_id = request_video_generation(script)
    
    # 2. Poll for completion
    video_url = wait_for_video(video_id)
    
    # 3. Download video
    output_filename = f"{file_id}.mp4"
    output_path = Path("outputs") / output_filename
    download_video(video_url, output_path)
    
    return output_filename

def request_video_generation(script: str) -> str:
    """Request HeyGen to generate a video"""
    url = "https://api.heygen.com/v2/video/generate"
    
    payload = {
        "video_inputs": [{
            "character": {
                "type": "avatar",
                "avatar_id": AVATAR_ID,
                "avatar_style": "normal"
            },
            "voice": {
                "type": "text",
                "input_text": script,
                "voice_id": VOICE_ID,
                "speed": 1.0  # Normal speed for short videos
            },
            "background": {"type": "color", "value": "#FFFFFF"}
        }],
        "dimension": {"width": 1280, "height": 720}
    }
    
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        video_id = response.json()["data"]["video_id"]
        return video_id
    
    except Exception as e:
        raise RuntimeError(f"Failed to request video generation: {str(e)}")

def wait_for_video(video_id: str, max_wait: int = 360, check_interval: int = 10) -> str:
    """
    Poll HeyGen API until video is ready
    
    Args:
        video_id: The HeyGen video ID
        max_wait: Maximum seconds to wait
        check_interval: Seconds between checks
        
    Returns:
        URL to download the video
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                "https://api.heygen.com/v1/video_status.get",
                headers={"X-Api-Key": HEYGEN_API_KEY},
                params={"video_id": video_id}
            )
            response.raise_for_status()
            
            data = response.json()["data"]
            status = data["status"]
            
            if status == "completed":
                return data["video_url"]
            elif status == "failed":
                raise RuntimeError("HeyGen video generation failed")
            
            # Still processing, wait and try again
            time.sleep(check_interval)
        
        except Exception as e:
            raise RuntimeError(f"Error checking video status: {str(e)}")
    
    raise TimeoutError(f"Video generation timed out after {max_wait} seconds")

def download_video(video_url: str, output_path: Path):
    """Download video from HeyGen URL to local file"""
    try:
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download video: {str(e)}")

