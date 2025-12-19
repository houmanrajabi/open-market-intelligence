import base64
import json
import io
import os
import re
from openai import OpenAI
from PIL import Image

# --- CONFIGURATION ---
# Update to your current Ngrok URL
API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1"
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
API_KEY = "production-key" 

client = OpenAI(base_url=API_URL, api_key=API_KEY)

def encode_image(image_obj, max_size=1600):
    """
    Resizes and encodes the image.
    1600px is the sweet spot for Qwen 72B (balance of detail vs token cost).
    """
    if image_obj.width > max_size:
        aspect_ratio = image_obj.height / image_obj.width
        new_height = int(max_size * aspect_ratio)
        image_obj = image_obj.resize((max_size, new_height), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG", quality=90) # Increased quality slightly for OCR
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def repair_json(json_str):
    """
    Robust JSON repair tool. 
    Strips Markdown code blocks and hunts for the outermost list/dict.
    """
    json_str = json_str.strip()
    
    # 1. Strip Markdown wraps
    if json_str.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = json_str.splitlines()
        if len(lines) > 1:
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].strip() == "```": lines = lines[:-1]
            json_str = "\n".join(lines)
    
    # 2. Try Direct Parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # 3. Fallback: Regex Hunt for JSON structure
    try:
        # Look for the outer-most list [...] or dict {...}
        match = re.search(r'(\[.*\]|\{.*\})', json_str, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except:
        pass
        
    print(f"      ⚠️ JSON Repair Failed. Raw output start: {json_str[:50]}...")
    return {"layout": []} # Safe fallback

def analyze_layout(image_path_or_obj):
    """
    Pass 1: Structural Layout Analysis.
    CRITICAL: Detects 'HEADER' types to serve as Semantic Anchors.
    """
    if isinstance(image_path_or_obj, str):
        img = Image.open(image_path_or_obj)
    else:
        img = image_path_or_obj

    base64_image = encode_image(img)

    system_prompt = """
You are a Document Structure Layout Engine.
Your goal is to segment the image into semantic blocks for a RAG pipeline.

### CATEGORY DEFINITIONS (Strict):

1. **HEADER (Crucial for Semantics)**
   - **Definition:** Any distinct **Section Title**, **Chapter Heading**, **Table Title**, or **Page Header** (running head).
   - **Visual Cues:** Bold text, larger font, isolated lines, text centered or starting a new block.
   - **Role:** These will be used as "Anchors". If you see "Table 1: Inflation", box it as HEADER, not TEXT.

2. **TABLE**
   - **Definition:** Rows and columns of data, grids, or financial statements.
   - **Scope:** Include the column headers and the data grid. (Do NOT include the Title if it is distinct enough to be a HEADER).

3. **FIGURE**
   - **Definition:** Charts, graphs, plots, diagrams.
   - **Scope:** Box the visual element.

4. **FOOTER**
   - **Definition:** Page numbers, "Confidential" markers at bottom, footnotes that are NOT part of a table.

5. **TEXT**
   - **Definition:** Standard paragraphs, bullet points, narrative text.

### COORDINATE SYSTEM:
- **Format:** [ymin, xmin, ymax, xmax]
- **Scale:** 0 to 1000 (Normalized).
- **Origin:** Top-Left is [0, 0].

### OUTPUT FORMAT:
Return ONLY valid JSON.
{
  "layout": [
    {"type": "HEADER", "bbox": [50, 50, 100, 950], "hint": "Projection of Inflation"},
    {"type": "TEXT", "bbox": [110, 50, 300, 950], "hint": "The inflation rate is expected..."}
  ]
}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.01 
        )
        
        content = response.choices[0].message.content
        return repair_json(content)

    except Exception as e:
        print(f"   ❌ Layout Scan Error: {e}")
        return {"layout": []}

def extract_page_data(image_path):
    """
    Pass 2: Content Extraction (High-Res Crop).
    Specialized prompt for extracting CSV from tables and Insights from charts.
    """
    # 1. Encode Image
    with Image.open(image_path) as img:
        base64_image = encode_image(img, max_size=1600)

    # 2. Specialized Prompt
    system_prompt = """
    You are a Data Extraction Specialist. 
    Analyze the provided crop (Table or Chart) and extract structured data.

    ### MODE 1: IF TABLE
    - **content**: Output the data in **Strict CSV Format** (header1, header2\nrow1col1, row1col2...).
    - **summary**: A 1-sentence summary of what this table shows (e.g., "Unemployment projections 2020-2023").

    ### MODE 2: IF CHART/FIGURE
    - **content**: Describe the trends, X/Y axes, and key data points. (e.g., "Line chart showing GDP drop in Q2 2020 followed by recovery").
    - **summary**: Title or main takeaway of the chart.

    ### MODE 3: IF TEXT/OTHER
    - Transcribe the text exactly.

    ### OUTPUT FORMAT (JSON LIST):
    [
      {
        "type": "TABLE",
        "content": "Year, Median, Range\n2020, 1.5, 1.2-1.8",
        "summary": "Inflation projections for 2020.",
        "notes": "Includes participants' estimates."
      }
    ]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=4096,
            temperature=0.01
        )
        
        raw_content = response.choices[0].message.content
        data = repair_json(raw_content)
        
        # Normalize output to expected dict format
        if isinstance(data, list):
            return {"tables": data} # 'tables' is just the generic key we use in hybrid_extractor
        elif isinstance(data, dict):
            return {"tables": [data]}
        else:
            return {"tables": []}

    except Exception as e:
        print(f"      ❌ Visual Extractor Failed: {e}")
        return {"tables": []}