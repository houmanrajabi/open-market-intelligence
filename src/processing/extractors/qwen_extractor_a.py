import base64
import json
import io
import os
import re
import time
from openai import OpenAI
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1"
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
API_KEY = "production-key"

client = OpenAI(base_url=API_URL, api_key=API_KEY)

# --- SPECIALIZED PROMPTS ---
LAYOUT_PROMPT = """You are a Document Structure Layout Engine for financial and analytical documents.

### CRITICAL RULES:

1. **HEADER (Highest Priority)**
   - Section titles, table captions, chart titles
   - Bold/larger text that introduces a block
   - Example: "Table 1: Economic Projections" → HEADER
   
2. **TABLE** 
   - Rows and columns with data
   - Include column headers, exclude title if it's distinct
   
3. **FIGURE**
   - Charts, graphs, plots, diagrams
   
4. **FOOTER**
   - Page numbers, footnotes, "Confidential" markers
   
5. **TEXT**
   - Paragraphs, bullet points, explanatory text

### MULTI-COLUMN DETECTION:
If you detect multiple columns, add a "column" field (1, 2, etc.)

### OUTPUT FORMAT (JSON only, no markdown):
{
  "layout": [
    {"type": "HEADER", "bbox": [50, 50, 100, 950], "hint": "Table 1 Economic Projections"},
    {"type": "TABLE", "bbox": [110, 50, 400, 950], "column": 1},
    {"type": "TEXT", "bbox": [110, 50, 400, 450], "column": 1},
    {"type": "TEXT", "bbox": [110, 460, 400, 950], "column": 2}
  ]
}

Coordinates: [ymin, xmin, ymax, xmax] scaled 0-1000, origin top-left."""

TABLE_PROMPT = """You are a Table Data Extractor. Extract this table as STRICTLY VALID CSV.

### REQUIREMENTS:
1. First row = column headers (no empty headers)
2. Numeric cells: Use the number or leave empty if unclear
3. Multi-level headers: Use underscore "Parent_Child"
4. Ranges like "5.5-6.0": Create TWO columns "Variable_Min" and "Variable_Max"
5. Merged cells: Repeat the value
6. NO markdown, NO explanations, NO code blocks

### EXAMPLE:
Input: Table with "GDP" row showing "2020: -6.5" and range "(-7.6 to -5.5)"
Output:
Variable,Year_2020_Median,Year_2020_Min,Year_2020_Max
GDP,-6.5,-7.6,-5.5

### ERROR HANDLING:
- If cell is illegible: Leave empty
- If structure is unclear: Use best judgment but maintain CSV validity

Output ONLY the CSV text, starting immediately with the header row."""

CHART_PROMPT = """You are a Chart Data Analyst. Extract structured data from this visualization.

### OUTPUT FORMAT (JSON only):
{
  "chart_type": "line|bar|box_plot|scatter|pie",
  "title": "Brief title",
  "axes": {
    "x_label": "Year",
    "y_label": "Percent",
    "x_values": ["2020", "2021", "2022"],
    "y_range": [min, max]
  },
  "data_series": [
    {
      "name": "Median GDP Growth",
      "points": [
        {"x": "2020", "y": -6.5},
        {"x": "2021", "y": 5.0}
      ]
    }
  ],
  "key_insights": [
    "GDP projected to decline 6.5% in 2020",
    "Recovery expected in 2021 with 5% growth"
  ],
  "data_labels": ["Any visible data labels on chart"]
}

CRITICAL: Extract ALL visible numbers from the chart. If exact values aren't shown, estimate from axis."""

TEXT_PROMPT = """Extract and clean this text. Preserve paragraph structure but remove artifacts.

### RULES:
1. Maintain line breaks between paragraphs
2. Fix obvious OCR errors
3. Remove page numbers/headers if present
4. Preserve lists and bullet points
5. Keep numerical data formatted correctly

Output the cleaned text directly, no JSON wrapper."""

# --- HELPER FUNCTIONS ---

def encode_image(image_obj, max_size=1600):
    """Resize and encode image with quality preservation."""
    if max(image_obj.width, image_obj.height) > max_size:
        if image_obj.width > image_obj.height:
            new_width = max_size
            new_height = int(max_size * image_obj.height / image_obj.width)
        else:
            new_height = max_size
            new_width = int(max_size * image_obj.width / image_obj.height)
        
        image_obj = image_obj.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG", quality=92)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def repair_json(json_str: str) -> dict:
    """Robust JSON repair with multiple fallback strategies."""
    json_str = json_str.strip()
    
    # Strategy 1: Remove markdown wrapping
    if json_str.startswith("```"):
        lines = json_str.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        json_str = "\n".join(lines).strip()
    
    # Strategy 2: Direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract JSON object
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # Nested arrays
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, json_str, re.DOTALL)
        for match in reversed(matches):  # Try longest match first
            try:
                return json.loads(match)
            except:
                continue
    
    logger.warning(f"JSON repair failed. Raw output: {json_str[:100]}...")
    return {"layout": []}

def validate_csv(csv_string: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """Validate CSV structure and parseability."""
    if not csv_string or len(csv_string.strip()) < 10:
        return False, None, "CSV string too short"
    
    try:
        df = pd.read_csv(io.StringIO(csv_string))
        
        # Validation checks
        if df.shape[0] < 1:
            return False, None, "No data rows"
        
        if df.shape[1] < 2:
            return False, None, "Less than 2 columns"
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            return False, None, f"Unnamed columns detected: {unnamed_cols}"
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Empty columns found: {empty_cols}")
            df = df.dropna(axis=1, how='all')
        
        return True, df, "Valid CSV"
    
    except Exception as e:
        return False, None, f"Parse error: {str(e)}"

def validate_chart_data(chart_dict: dict) -> Tuple[bool, str]:
    """Validate chart extraction has required fields and data."""
    required_fields = ["chart_type", "title"]
    
    for field in required_fields:
        if field not in chart_dict:
            return False, f"Missing required field: {field}"
    
    # Check for actual data
    has_data = False
    if "data_series" in chart_dict and chart_dict["data_series"]:
        has_data = True
    elif "key_insights" in chart_dict and chart_dict["key_insights"]:
        has_data = True
    
    if not has_data:
        return False, "No data or insights extracted"
    
    return True, "Valid chart data"

# --- MAIN EXTRACTION FUNCTIONS ---

def call_vlm_with_retry(
    image_path: str,
    prompt: str,
    max_attempts: int = 3,
    temperature: float = 0.01
) -> Optional[str]:
    """Call VLM with exponential backoff retry."""
    
    with Image.open(image_path) as img:
        base64_image = encode_image(img)
    
    for attempt in range(max_attempts):
        try:
            # Adjust temperature on retries
            current_temp = temperature + (attempt * 0.05)
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=4096,
                temperature=current_temp
            )
            
            content = response.choices[0].message.content
            return content
        
        except Exception as e:
            logger.error(f"VLM call failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    
    return None

def analyze_layout(image_path_or_obj) -> dict:
    """Phase 1: Layout detection with validation."""
    
    if isinstance(image_path_or_obj, str):
        img_path = image_path_or_obj
    else:
        # Save temporarily if PIL Image
        img_path = "/tmp/temp_layout.jpg"
        image_path_or_obj.save(img_path)
    
    logger.info("🔍 Starting layout analysis...")
    
    content = call_vlm_with_retry(img_path, LAYOUT_PROMPT)
    
    if not content:
        logger.error("Layout detection failed completely")
        return {"layout": []}
    
    layout_data = repair_json(content)
    
    # Validate layout structure
    if "layout" not in layout_data or not layout_data["layout"]:
        logger.warning("No layout elements detected")
        return {"layout": []}
    
    # Quality check
    valid_elements = []
    for elem in layout_data["layout"]:
        if "type" in elem and "bbox" in elem:
            # Ensure bbox has 4 coordinates
            if len(elem["bbox"]) == 4:
                valid_elements.append(elem)
            else:
                logger.warning(f"Invalid bbox for element: {elem}")
    
    logger.info(f"✅ Detected {len(valid_elements)} valid layout elements")
    
    return {"layout": valid_elements}

def extract_table_data(image_path: str) -> dict:
    """Extract table with validation and retry."""
    
    logger.info("📊 Extracting table data...")
    
    for attempt in range(3):
        content = call_vlm_with_retry(image_path, TABLE_PROMPT)
        
        if not content:
            continue
        
        # Clean potential markdown
        csv_text = content.strip()
        if csv_text.startswith("```"):
            csv_text = "\n".join(csv_text.split("\n")[1:-1])
        
        # Validate
        is_valid, df, message = validate_csv(csv_text)
        
        if is_valid:
            logger.info(f"✅ Valid CSV extracted: {df.shape[0]} rows, {df.shape[1]} cols")
            return {
                "type": "TABLE",
                "content": csv_text,
                "summary": f"Table with {df.shape[0]} rows and {df.shape[1]} columns",
                "structured_data": df.to_dict(orient='records'),
                "columns": df.columns.tolist(),
                "extraction_quality": 0.9,
                "validation_status": "PASS"
            }
        else:
            logger.warning(f"Attempt {attempt + 1}: {message}")
    
    # Fallback: Return raw text with low quality score
    logger.error("❌ Failed to extract valid CSV after 3 attempts")
    return {
        "type": "TABLE",
        "content": content if content else "EXTRACTION_FAILED",
        "summary": "Table extraction failed validation",
        "extraction_quality": 0.3,
        "validation_status": "FAIL",
        "error": message
    }

def extract_chart_data(image_path: str) -> dict:
    """Extract chart with structured data validation."""
    
    logger.info("📈 Extracting chart data...")
    
    content = call_vlm_with_retry(image_path, CHART_PROMPT)
    
    if not content:
        return {
            "type": "FIGURE",
            "content": "EXTRACTION_FAILED",
            "extraction_quality": 0.0,
            "validation_status": "FAIL"
        }
    
    chart_data = repair_json(content)
    
    # Validate
    is_valid, message = validate_chart_data(chart_data)
    
    if is_valid:
        # Create text summary from structured data
        summary_parts = [chart_data.get("title", "Chart")]
        
        if "key_insights" in chart_data:
            summary_parts.extend(chart_data["key_insights"][:3])
        
        logger.info(f"✅ Valid chart data extracted: {chart_data.get('chart_type', 'unknown')}")
        
        return {
            "type": "FIGURE",
            "content": "\n".join(summary_parts),
            "summary": chart_data.get("title", "Chart"),
            "structured_data": chart_data,
            "chart_type": chart_data.get("chart_type"),
            "extraction_quality": 0.85,
            "validation_status": "PASS"
        }
    else:
        logger.warning(f"Chart validation failed: {message}")
        return {
            "type": "FIGURE",
            "content": str(chart_data),
            "summary": "Chart data incomplete",
            "extraction_quality": 0.5,
            "validation_status": "PARTIAL"
        }

def extract_page_data(image_path: str, element_type: str = "TABLE") -> dict:
    """Unified extraction router with type-specific handling."""
    
    if element_type in ["TABLE"]:
        return extract_table_data(image_path)
    elif element_type in ["FIGURE", "CHART"]:
        return extract_chart_data(image_path)
    else:
        # Generic text extraction
        content = call_vlm_with_retry(image_path, TEXT_PROMPT)
        return {
            "type": element_type,
            "content": content if content else "EXTRACTION_FAILED",
            "extraction_quality": 0.7 if content else 0.0
        }