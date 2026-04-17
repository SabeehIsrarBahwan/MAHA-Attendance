import json
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from PIL import Image

try:
    from pdf2image import convert_from_path
    import ollama
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  Helper: extract JSON from model response
# -------------------------------------------------------------------
def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        return json.loads(json_str)
    raise ValueError("No JSON object found in response")

# -------------------------------------------------------------------
#  Step 1: PDF to images
# -------------------------------------------------------------------
def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Path]:
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

# -------------------------------------------------------------------
#  Step 2: Extract employee names from the "Name" column region
# -------------------------------------------------------------------
def extract_employee_names(image_path: Path, name_column_bbox: Tuple[int, int, int, int]) -> List[str]:
    """
    name_column_bbox: (left, top, right, bottom) in pixels.
    Returns list of employee names in order from top to bottom.
    """
    img = Image.open(image_path)
    name_region = img.crop(name_column_bbox)
    # Save temporarily for VLM
    temp_name_path = Path("temp_name_col.jpg")
    name_region.save(temp_name_path)
    
    prompt = """Extract all employee names from this image column. 
Return ONLY a JSON array of strings, in order from top to bottom.
Example: ["Name1", "Name2", ...]
Do not add any extra text."""
    
    try:
        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{'role': 'user', 'content': prompt, 'images': [str(temp_name_path)]}],
            options={'temperature': 0.1}
        )
        content = response['message']['content'].strip()
        names = extract_json_from_text(content)
        if not isinstance(names, list):
            raise ValueError("Response is not a list")
        log.info(f"Extracted {len(names)} employee names")
        return names
    except Exception as e:
        log.error(f"Failed to extract names: {e}")
        return []
    finally:
        temp_name_path.unlink(missing_ok=True)

# -------------------------------------------------------------------
#  Step 3: For a given day column crop, extract statuses in order
# -------------------------------------------------------------------
def extract_statuses_for_day(day_image_path: Path, expected_count: int) -> List[str]:
    prompt = f"""This image shows a column of attendance statuses for a single day.
There are {expected_count} rows, in the same order as the employee list.
Return ONLY a JSON array of status strings, e.g. ["P","A","L","WO","H",...].
Use: P=present, A=absent, L=late, WO=week off, H=holiday.
If a cell is empty, use null.
Do not add any extra text."""
    try:
        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{'role': 'user', 'content': prompt, 'images': [str(day_image_path)]}],
            options={'temperature': 0.1}
        )
        content = response['message']['content'].strip()
        statuses = extract_json_from_text(content)
        if not isinstance(statuses, list):
            raise ValueError("Not a list")
        # Pad or trim if necessary
        if len(statuses) < expected_count:
            statuses += [None] * (expected_count - len(statuses))
        return statuses[:expected_count]
    except Exception as e:
        log.error(f"Failed to extract statuses: {e}")
        return [None] * expected_count

# -------------------------------------------------------------------
#  Step 4: Split the page into day columns (using pre‑defined X ranges)
# -------------------------------------------------------------------
def split_into_day_columns(full_image_path: Path, day_x_ranges: List[Tuple[int, int]]) -> List[Path]:
    """
    day_x_ranges: list of (left, right) for each day column (in order: day1, day2, ...)
    Returns list of temporary image file paths for each day.
    """
    img = Image.open(full_image_path)
    day_paths = []
    for i, (x_left, x_right) in enumerate(day_x_ranges, start=1):
        # Crop the full height (top=0, bottom=img.height)
        crop = img.crop((x_left, 0, x_right, img.height))
        day_path = Path(f"temp_day_{i}.jpg")
        crop.save(day_path)
        day_paths.append(day_path)
    return day_paths

# -------------------------------------------------------------------
#  Main orchestration
# -------------------------------------------------------------------
def main(pdf_path: str, name_bbox: Tuple[int, int, int, int], day_x_ranges: List[Tuple[int, int]]):
    """
    name_bbox: (left, top, right, bottom) for the employee name column.
    day_x_ranges: list of (left, right) for each day column (1,2,...,31).
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        log.error(f"File not found: {pdf_path}")
        return
    
    # Convert PDF to images
    image_paths = pdf_to_images(str(pdf_file))
    all_results = []
    
    for img_idx, img_path in enumerate(image_paths, start=1):
        log.info(f"Processing page {img_idx}...")
        
        # 1. Extract employee names from the name column region
        employee_names = extract_employee_names(img_path, name_bbox)
        if not employee_names:
            log.error("No names extracted, skipping page")
            continue
        
        # 2. Split into day columns
        day_images = split_into_day_columns(img_path, day_x_ranges)
        
        # 3. For each day, extract statuses
        day_statuses = []
        for day_idx, day_img in enumerate(day_images, start=1):
            log.info(f"  Processing day {day_idx}...")
            statuses = extract_statuses_for_day(day_img, len(employee_names))
            day_statuses.append(statuses)
            day_img.unlink()  # clean up temp file
        
        # 4. Build employee-wise attendance
        employees_data = []
        for emp_idx, name in enumerate(employee_names):
            attendance = {}
            for day_idx, status_list in enumerate(day_statuses, start=1):
                if emp_idx < len(status_list):
                    attendance[str(day_idx)] = status_list[emp_idx]
                else:
                    attendance[str(day_idx)] = None
            employees_data.append({
                "name": name,
                "attendance": attendance
            })
        
        # 5. Store page result (we need month name; could be extracted from image)
        # For simplicity, use page number as month placeholder; you can later add a prompt to get month.
        all_results.append({
            "page": img_idx,
            "month": f"Month_{img_idx}",
            "employees": employees_data
        })
    
    # Save final JSON
    out_file = f"attendance_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_file}")
    
    # Print summary
    for page in all_results:
        print(f"Page {page['page']}: {len(page['employees'])} employees, days: {len(page['employees'][0]['attendance']) if page['employees'] else 0}")

# -------------------------------------------------------------------
#  Helper to calibrate coordinates (run once to find day column positions)
# -------------------------------------------------------------------
def calibrate_coordinates(image_path: Path):
    """Open the image and print its size. User can manually determine X ranges."""
    img = Image.open(image_path)
    width, height = img.size
    print(f"Image size: {width} x {height}")
    print("Open this image in an editor and note the X pixel values for:")
    print(" - Left edge of the Name column")
    print(" - Right edge of the Name column")
    print(" - For each day column: left and right edges (in order from day1 to day31)")
    print("Example: name_bbox = (150, 0, 350, height)")
    print("day_x_ranges = [(400, 460), (460, 520), ...]")
    print("Then call main() with those values.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python attendance_reader_split.py <pdf_file>")
        sys.exit(1)
    
    # ----- CALIBRATION (RUN ONCE) -----
    # To calibrate, first convert a page to image and examine it.
    # For now, we provide example coordinates for the sample PDF at DPI=200.
    # You MUST adjust these numbers based on your actual image.
    # Use the calibrate_coordinates function if needed.
    
    # Example values for "E HRMS Aastha Jan & Feb 26.pdf" at DPI=200 (found by testing)
    # Name column: left=120, right=420 (width 300)
    # Day columns: each 60px wide, starting at x=430 for day1.
    # We'll generate day_x_ranges for 31 days (adjust number as needed)
    
    name_bbox = (120, 0, 420, 10000)   # (left, top, right, bottom) – bottom can be large
    
    # Generate day column ranges (31 days, each 60px, starting at 430)
    day_start = 430
    day_width = 60
    day_x_ranges = []
    for i in range(31):
        left = day_start + i * day_width
        right = left + day_width
        day_x_ranges.append((left, right))
    
    # If you want to auto-detect the number of days, you can later add logic.
    # For now, we assume 31 days (January). For February, you may need to stop earlier.
    
    main(sys.argv[1], name_bbox, day_x_ranges)