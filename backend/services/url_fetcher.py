"""
URL fetcher for privacy policies
Fetches and extracts text from privacy policy URLs
"""
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import tempfile

def fetch_policy_from_url(url: str) -> str:
    """
    Fetch privacy policy from URL and extract text
    
    Args:
        url: The URL of the privacy policy
        
    Returns:
        Path to temporary file containing the content
    """
    # Add headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Get content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Create temporary file
        if 'pdf' in content_type:
            # Save as PDF
            temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False)
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            # Save as HTML
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
            temp_file.write(response.text)
            temp_file.close()
            return temp_file.name
            
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {str(e)}")

def validate_url(url: str) -> bool:
    """
    Validate that the URL is properly formatted
    
    Args:
        url: The URL to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        raise ValueError("Invalid URL format")
    
    return True

