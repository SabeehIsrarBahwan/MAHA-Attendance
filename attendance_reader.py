import json
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

try:
    from pdf2image import convert_from_path
    import ollama
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def pdf_to_images(pdf_path: str, dpi: int =150) -> List[Path]:
    images = convert_from_path(pdf_path, dpi=dpi, fmt='jpeg')
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        img_path = temp_dir / f"page_{i+1}.jpg"
        img.save(img_path, 'JPEG', quality=85)
        paths.append(img_path)
    log.info(f"Converted {len(paths)} pages")
    return paths

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Try to extract JSON from a string that may contain extra text."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        return json.loads(json_str)
    raise ValueError("No JSON object found in response")

def extract_from_image(image_path: Path, model: str = "qwen2.5vl:7b") -> Dict[str, Any]:
    prompt = """Extract monthly attendance from this image.
Return ONLY valid JSON in this format:
{
  "month_year": "Month Year (e.g., January 2026)",
  "employees": [
    {
      "name": "Employee name",
      "attendance": {
        "1": "P/A/L/WO/H",
        "2": "...",
        ...
      }
    }
  ]
}
Use codes: P=present, A=absent, L=late, WO=week off, H=holiday.
Do not add any extra text outside the JSON.
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
            options={'temperature': 0.1}
        )
        content = response['message']['content'].strip()
        return extract_json_from_text(content)
    except Exception as e:
        log.error(f"Error on {image_path.name}: {e}")
        return {"error": str(e), "page": str(image_path)}

def main(pdf_path: str):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        log.error(f"File not found: {pdf_path}")
        return
    images = pdf_to_images(str(pdf_file))
    results = []
    for img in images:
        log.info(f"Processing {img.name}...")
        results.append(extract_from_image(img))
    out_file = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_file}")
    for i, r in enumerate(results):
        print(f"Page {i+1}: {r.get('month_year', 'no month')} -> {len(r.get('employees', []))} employees")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python attendance_reader.py <pdf_file>")
        sys.exit(1)
    main(sys.argv[1])
