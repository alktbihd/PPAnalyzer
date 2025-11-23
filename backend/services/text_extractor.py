"""
Text extraction from PDF and HTML files
Adapted from extract_sentences.py
"""
from pathlib import Path
from bs4 import BeautifulSoup
import PyPDF2
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path: str) -> list[str]:
    """
    Extract sentences from PDF or HTML file
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of sentences
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        text = extract_from_pdf(file_path)
    elif file_path.suffix.lower() in ['.html', '.htm']:
        text = extract_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Split into sentences
    sentences = split_sentences(text)
    return sentences

def extract_from_pdf(file_path: Path) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    return text.strip()

def extract_from_html(file_path: Path) -> str:
    """
    Extract visible text from HTML
    Similar to extract_sentences.py logic
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.extract()
        
        text = soup.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from HTML: {str(e)}")

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using spaCy
    Only keep sentences with more than 10 characters
    """
    if not text or len(text) < 10:
        return []
    
    doc = nlp(text)
    sentences = [
        sent.text.strip() 
        for sent in doc.sents 
        if len(sent.text.strip()) > 10
    ]
    
    return sentences

