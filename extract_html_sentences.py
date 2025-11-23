import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load("en_core_web_sm")

privacy_evaluation = Path("privacy_evaluation")  # input folder (HTML files)
output_dir = Path("privacy_evaluation/text")  # output folder for extracted sentences

def extract_visible_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def process_single_file(html_path: Path, save_to_file: bool = True):
    """Process a single HTML file and save sentences to text file"""
    try:
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
        text = extract_visible_text(raw_html)
        sentences = extract_sentences(text)
        
        print(f"Processing {html_path.name} → {len(sentences)} sentences")
        
        if save_to_file:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to text file (same name as HTML file but with .txt extension)
            output_file = output_dir / f"{html_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            
            print(f"  Saved to: {output_file}")
        
        return sentences
        
    except Exception as e:
        print(f"Error processing {html_path.name}: {str(e)}")
        return []

def process_directory(directory: Path):
    """Process all HTML files in a directory and save to text files"""
    html_files = list(directory.glob("*.html")) + list(directory.glob("*.htm"))
    
    if not html_files:
        print(f"No HTML files found in {directory}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_sentences = 0
    for html_file in sorted(html_files):
        sentences = process_single_file(html_file, save_to_file=True)
        total_sentences += len(sentences)
    
    print(f"\nExtracted {total_sentences} sentences from {len(html_files)} HTML files")
    print(f"All text files saved to: {output_dir}")

def main():
    # If no arguments, process all files in privacy_evaluation directory
    if len(sys.argv) == 1:
        if not privacy_evaluation.exists():
            print(f"Error: Directory not found: {privacy_evaluation}")
            print(f"Usage: python extract_html_sentences.py [path_to_html_file]")
            print(f"   or: python extract_html_sentences.py  (processes all files in privacy_evaluation/)")
            sys.exit(1)
        
        print(f"Processing all HTML files in {privacy_evaluation}/")
        process_directory(privacy_evaluation)
        return
    
    # Process single file
    html_path = Path(sys.argv[1])
    
    if not html_path.exists():
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    if html_path.is_dir():
        print(f"Processing all HTML files in {html_path}/")
        process_directory(html_path)
        return
    
    if html_path.suffix.lower() not in ['.html', '.htm']:
        print(f"Error: File must be an HTML file (.html or .htm)")
        sys.exit(1)
    
    process_single_file(html_path, save_to_file=True)

if __name__ == "__main__":
    main()

