# # import os
# # import base64
# # import json
# # import uuid
# # import pandas as pd
# # from pdf2image import convert_from_path
# # from openai import OpenAI
# # from PIL import Image
# # from io import StringIO

# # # --- CONFIGURATION ---
# # # PASTE THE URL FROM COLAB HERE (e.g., https://abc-123.ngrok-free.app/v1)
# # API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1" 
# # API_KEY = "dev" # Can be anything for local/colab
# # INPUT_PDF = "/Users/houmanrajabi/Desktop/Projects/THESIS_RAG/data/raw/fomcprojtabl20200916.pdf" # Put your PDF file name here
# # OUTPUT_DIR = "data/processed_data"

# # # Initialize Client
# # client = OpenAI(base_url=API_URL, api_key=API_KEY)
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # def encode_image(image_path):
# # #     with open(image_path, "rb") as image_file:
# # #         return base64.b64encode(image_file.read()).decode('utf-8')
# # import io # Add this if missing

# # # REPLACE YOUR OLD encode_image FUNCTION WITH THIS:
# # def encode_image(image_path):
# #     # 1. Open the image
# #     with Image.open(image_path) as img:
# #         # 2. Resize if too big (limit width to 1024px to save tokens)
# #         max_width = 1024
# #         if img.width > max_width:
# #             aspect_ratio = img.height / img.width
# #             new_height = int(max_width * aspect_ratio)
# #             img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
# #             print(f"      (Resized image to {max_width}x{new_height} for token efficiency)")

# #         # 3. Save to buffer and encode
# #         buffered = io.BytesIO()
# #         img.save(buffered, format="JPEG", quality=85)
# #         return base64.b64encode(buffered.getvalue()).decode('utf-8')

# # def extract_page_data(image_path):
# #     """Sends image to Qwen and asks for JSON data + Bounding Boxes"""
    
# #     # 1. Encode Image
# #     base64_image = encode_image(image_path)
    
# #     # 2. Define the Strict Prompt
# #     system_prompt = """
# #     You are a financial data assistant. Analyze this image.
# #     Identify ALL tables. For each table, extract:
# #     1. 'summary': A concise description of the table's topic and key trend.
# #     2. 'csv': The full table content in CSV format.
# #     3. 'bbox': The bounding box [ymin, xmin, ymax, xmax] (scale 0-1000).

# #     Output ONLY valid JSON in this format:
# #     {
# #       "tables": [
# #         {
# #           "summary": "Inflation projections for 2024...",
# #           "csv": "Year,GDP\\n2024,2.1...",
# #           "bbox": [150, 50, 500, 950]
# #         }
# #       ]
# #     }
# #     If no tables are present, return {"tables": []}.
# #     """

# #     # 3. Call API
# #     try:
# #         response = client.chat.completions.create(
# #             model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": system_prompt},
# #                         {
# #                             "type": "image_url",
# #                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
# #                         },
# #                     ],
# #                 }
# #             ],
# #             max_tokens=4000,
# #             temperature=0.1 
# #         )
# #         # Clean the response (sometimes models add ```json ... ``` wrappers)
# #         content = response.choices[0].message.content
# #         content = content.replace("```json", "").replace("```", "").strip()
# #         return json.loads(content)
# #     except Exception as e:
# #         print(f"Error calling API: {e}")
# #         # DEBUG: Print the raw content to see why JSON failed
# #         if 'response' in locals():
# #             print(f"RAW RESPONSE: {response.choices[0].message.content}") 
# #         return {"tables": []}

# # def process_pdf(pdf_path):
# #     print(f"📄 Processing: {pdf_path}")
    
# #     # Convert PDF to Images (300 DPI for clarity)
# #     images = convert_from_path(pdf_path, dpi=300)
    
# #     for i, img in enumerate(images):
# #         page_num = i + 1
# #         print(f"   Analyzing Page {page_num}...")
        
# #         # Save temp image for processing
# #         temp_img_path = f"temp_page_{page_num}.jpg"
# #         img.save(temp_img_path)
        
# #         # Get Data from AI
# #         result = extract_page_data(temp_img_path)
        
# #         # Process Tables
# #         if result.get("tables"):
# #             print(f"   Found {len(result['tables'])} tables.")
# #             width, height = img.size
            
# #             for table in result['tables']:
# #                 table_id = str(uuid.uuid4())[:8]
                
# #                 # A. Save Screenshot (The "Evidence")
# #                 # Qwen uses [ymin, xmin, ymax, xmax] on 1000 scale
# #                 y1, x1, y2, x2 = table['bbox']
                
# #                 # Convert to pixels
# #                 crop_box = (
# #                     int(x1 / 1000 * width),  # left
# #                     int(y1 / 1000 * height), # top
# #                     int(x2 / 1000 * width),  # right
# #                     int(y2 / 1000 * height)  # bottom
# #                 )
                
# #                 # Safety check (sometimes model outputs 0 or negative)
# #                 try:
# #                     table_img = img.crop(crop_box)
# #                     img_path = os.path.join(OUTPUT_DIR, f"{table_id}.jpg")
# #                     table_img.save(img_path)
# #                 except Exception as e:
# #                     print(f"      Warning: Could not crop image {table_id}")

# #                 # B. Save CSV (The "Data")
# #                 csv_path = os.path.join(OUTPUT_DIR, f"{table_id}.csv")
# #                 with open(csv_path, "w") as f:
# #                     f.write(table['csv'])
                
# #                 # C. Save Metadata (The "Search Index")
# #                 # We save this as JSON so your RAG pipeline can ingest it later
# #                 meta_path = os.path.join(OUTPUT_DIR, f"{table_id}.json")
# #                 metadata = {
# #                     "id": table_id,
# #                     "summary": table['summary'],
# #                     "source_pdf": pdf_path,
# #                     "page": page_num,
# #                     "csv_path": csv_path,
# #                     "image_path": img_path
# #                 }
# #                 with open(meta_path, "w") as f:
# #                     json.dump(metadata, f, indent=2)
                
# #                 print(f"      ✅ Saved Table {table_id} (CSV + Image + Meta)")
        
# #         # Cleanup temp file
# #         os.remove(temp_img_path)

# # # --- RUN IT ---
# # if __name__ == "__main__":
# #     # Ensure you have a PDF named 'fomc_minutes.pdf' in the folder
# #     if os.path.exists(INPUT_PDF):
# #         process_pdf(INPUT_PDF)
# #     else:
# #         print(f"❌ File {INPUT_PDF} not found. Please add a PDF to test.")

# # import os
# # import base64
# # import json
# # import uuid
# # import io
# # import time
# # from pdf2image import convert_from_path
# # from openai import OpenAI
# # from PIL import Image

# # # --- CONFIGURATION ---
# # API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1" # <--- CHECK THIS IS STILL VALID
# # API_KEY = "dev"
# # INPUT_PDF = "/Users/houmanrajabi/Desktop/Projects/THESIS_RAG/data/raw/fomcprojtabl20200916.pdf"
# # OUTPUT_DIR = "data/processed_data"

# # client = OpenAI(base_url=API_URL, api_key=API_KEY)
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # def encode_image(image_path):
# #     with Image.open(image_path) as img:
# #         # INCREASED LIMIT: 2048px gives the model a much clearer view of tiny numbers
# #         max_width = 2048 
# #         if img.width > max_width:
# #             aspect_ratio = img.height / img.width
# #             new_height = int(max_width * aspect_ratio)
# #             img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
# #             print(f"      (Resized to {max_width}x{new_height} for clarity)")
        
# #         buffered = io.BytesIO()
# #         img.save(buffered, format="JPEG", quality=90) # Increased quality
# #         return base64.b64encode(buffered.getvalue()).decode('utf-8')

# # import re  # <--- ADD THIS AT THE TOP

# # # --- ADD THIS HELPER FUNCTION ---
# # def extract_json(text):
# #     """
# #     Tries to find and parse JSON from the model's output, 
# #     even if it's surrounded by text or missing a closing bracket.
# #     """
# #     # 1. Try to find content between ```json and ```
# #     match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
# #     if match:
# #         text = match.group(1)
    
# #     # 2. If no markdown, find the first '{' and last '}'
# #     else:
# #         start = text.find("{")
# #         end = text.rfind("}")
# #         if start != -1 and end != -1:
# #             text = text[start : end + 1]

# #     # 3. Try parsing
# #     try:
# #         return json.loads(text)
# #     except json.JSONDecodeError:
# #         # 4. Simple Repair: Attempt to close broken brackets
# #         # (7B models often cut off at max_tokens)
# #         try:
# #             return json.loads(text + "}]}") # Try closing list/obj
# #         except:
# #             try:
# #                 return json.loads(text + "]}")
# #             except:
# #                 try:
# #                     return json.loads(text + "}")
# #                 except:
# #                     return None

# # --- REPLACE YOUR extract_page_data FUNCTION WITH THIS ---
# # import re

# # import re

# # def extract_page_data(image_path):
# #     base64_image = encode_image(image_path)
    
# #     # --- UPDATED PROMPT ---
# #     # 1. Added "NOTES" section.
# #     # 2. Added negative constraint ("If text only, output NOTHING").
# #     # 3. Added BBox warning ("Do not return [0,0,1000,1000]").
# #     system_prompt = """
# #     You are a financial data assistant. Analyze this page.
    
# #     TASK:
# #     1. Identify VISUAL TABLES or CHARTS. 
# #        - If the page is just text/paragraphs, output NOTHING. Do not hallucinate tables from text.
# #     2. For each valid table/chart, extract:
# #        - SUMMARY: What is it showing?
# #        - CSV: The data. Flatten nested headers (e.g., "2020 Median").
# #        - NOTES: Transcribe any footnotes/notes located immediately below the table.
# #        - BBOX: The cropping coordinates [ymin, xmin, ymax, xmax] (0-1000 scale).
# #          * IMPORTANT: Do not return [0,0,1000,1000]. Tightly crop the table only.

# #     OUTPUT FORMAT (Use this separator):
    
# #     ---ITEM START---
# #     SUMMARY: [Text]
# #     CSV:
# #     [Data]
# #     NOTES: [Footnotes/Notes below table]
# #     BBOX: [ymin, xmin, ymax, xmax]
# #     ---ITEM END---
# #     """

# #     try:
# #         response = client.chat.completions.create(
# #             model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": system_prompt},
# #                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
# #                     ],
# #                 }
# #             ],
# #             max_tokens=4000,
# #             temperature=0.01 
# #         )
# #         content = response.choices[0].message.content
        
# #         # --- ROBUST PARSING ---
# #         items = []
# #         raw_blocks = content.split("---ITEM START---")
        
# #         for block in raw_blocks:
# #             if "---ITEM END---" not in block:
# #                 continue
                
# #             # Parse Fields using Regex
# #             # Note: We now look for the NOTES section between CSV and BBOX
# #             summary = (re.search(r"SUMMARY:\s*(.*?)\nCSV:", block, re.DOTALL) or [None, ""])[1].strip()
# #             csv_data = (re.search(r"CSV:\s*(.*?)\nNOTES:", block, re.DOTALL) or [None, ""])[1].strip()
# #             notes = (re.search(r"NOTES:\s*(.*?)\nBBOX:", block, re.DOTALL) or [None, ""])[1].strip()
            
# #             # Parse BBox
# #             bbox_match = re.search(r"BBOX:\s*\[?([\d\s,]+)\]?", block)
# #             bbox = [0,0,0,0]
# #             if bbox_match:
# #                 try:
# #                     # Handle both "1, 2, 3, 4" and "1 2 3 4" formats
# #                     coords = [int(x) for x in bbox_match.group(1).replace(',', ' ').split()]
# #                     if len(coords) == 4:
# #                         bbox = coords
# #                 except:
# #                     pass

# #             # --- QUALITY FILTERS ---
# #             # 1. Reject empty CSVs
# #             if not csv_data: 
# #                 continue
            
# #             # 2. Reject "Hallucinated" Tables (Text masquerading as CSV)
# #             # Real tables usually have more than 2 lines
# #             if len(csv_data.split('\n')) < 3: 
# #                 print(f"      ⚠️ Skipping item: CSV too short (likely text hallucination).")
# #                 continue

# #             items.append({
# #                 "summary": summary,
# #                 "csv": csv_data,
# #                 "notes": notes, # <--- Captured!
# #                 "bbox": bbox
# #             })

# #         return {"tables": items}

# #     except Exception as e:
# #         print(f"   ❌ API Failure: {e}")
# #         return {"tables": []}
# # second one 
# import re

# # def extract_page_data(image_path):
# #     base64_image = encode_image(image_path)
    
# #     # --- FINAL PRODUCTION PROMPT ---
# #     # Changes:
# #     # 1. Separated logic for TABLES vs CHARTS/GRAPHS.
# #     # 2. Charts: Ask for "Key Trends" instead of CSV to prevent hallucination.
# #     # 3. Notes: Explicitly told to look for text *below* the visual element.
# #     system_prompt = """
# #     You are a financial data assistant. Analyze this page.
    
# #     TASK:
# #     Identify every VISUAL ELEMENT (Table or Chart).
    
# #     For each element, extract the following fields using the exact separator structure:
    
# #     ---ITEM START---
# #     TYPE: [TABLE or CHART]
# #     SUMMARY: [Detailed description of what is shown]
# #     CSV: 
# #     [If TABLE: Output clean CSV with flattened headers (e.g., '2020_Median').]
# #     [If CHART: Output 'N/A' unless exact numbers are printed. Do NOT estimate data points.]
# #     NOTES: [Transcribe all footnotes/notes found immediately below the element. If none, write 'None'.]
# #     BBOX: [ymin, xmin, ymax, xmax] (0-1000 scale. Tightly crop the element.)
# #     ---ITEM END---
    
# #     NEGATIVE CONSTRAINTS:
# #     - If the page is text-only, output NOTHING.
# #     - Do not output [0,0,1000,1000] unless the table truly covers the entire page.
# #     """

# #     try:
# #         response = client.chat.completions.create(
# #             model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": system_prompt},
# #                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
# #                     ],
# #                 }
# #             ],
# #             max_tokens=4000,
# #             temperature=0.01 
# #         )
# #         content = response.choices[0].message.content
        
# #         # --- PARSING LOGIC ---
# #         items = []
# #         raw_blocks = content.split("---ITEM START---")
        
# #         for block in raw_blocks:
# #             if "---ITEM END---" not in block:
# #                 continue
                
# #             # Regex Extraction
# #             type_val = (re.search(r"TYPE:\s*(.*?)\nSUMMARY:", block, re.DOTALL) or [None, "Table"])[1].strip()
# #             summary = (re.search(r"SUMMARY:\s*(.*?)\nCSV:", block, re.DOTALL) or [None, ""])[1].strip()
# #             csv_data = (re.search(r"CSV:\s*(.*?)\nNOTES:", block, re.DOTALL) or [None, ""])[1].strip()
# #             notes = (re.search(r"NOTES:\s*(.*?)\nBBOX:", block, re.DOTALL) or [None, ""])[1].strip()
            
# #             # BBox Handling
# #             bbox_match = re.search(r"BBOX:\s*\[?([\d\s,]+)\]?", block)
# #             bbox = [0,0,0,0]
# #             if bbox_match:
# #                 try:
# #                     coords = [int(x) for x in bbox_match.group(1).replace(',', ' ').split()]
# #                     if len(coords) == 4:
# #                         bbox = coords
# #                 except:
# #                     pass

# #             # Filter: Skip empty items or "N/A" charts that add no value
# #             if not summary or (type_val == "CHART" and csv_data == "N/A" and notes == "None"):
# #                 continue

# #             items.append({
# #                 "type": type_val,
# #                 "summary": summary,
# #                 "csv": csv_data,
# #                 "notes": notes,
# #                 "bbox": bbox
# #             })

# #         return {"tables": items}

# #     except Exception as e:
# #         print(f"   ❌ API Failure: {e}")
# #         return {"tables": []}

# #therid one
# # import re

# # def extract_page_data(image_path):
# #     # 1. OPTIMIZATION: Lower resolution slightly to save tokens for the CSV output
# #     with Image.open(image_path) as img:
# #         max_width = 1600 # <--- Changed from 2048 to 1600 (Better safety margin)
# #         if img.width > max_width:
# #             aspect_ratio = img.height / img.width
# #             new_height = int(max_width * aspect_ratio)
# #             img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
# #         buffered = io.BytesIO()
# #         img.save(buffered, format="JPEG", quality=85)
# #         base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

# #     system_prompt = """
# #     You are a financial data assistant. Analyze this page.
# #     Identify every VISUAL ELEMENT (Table or Chart).
    
# #     For each element, extract these fields using this separator:
    
# #     ---ITEM START---
# #     TYPE: [TABLE or CHART]
# #     SUMMARY: [Description]
# #     CSV: 
# #     [Output clean CSV data. Flatten headers.]
# #     NOTES: [Transcribe footnotes found below the element.]
# #     BBOX: [ymin, xmin, ymax, xmax]
# #     ---ITEM END---
    
# #     If text-only, output NOTHING.
# #     """

# #     try:
# #         response = client.chat.completions.create(
# #             model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": system_prompt},
# #                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
# #                     ],
# #                 }
# #             ],
# #             max_tokens=4000,
# #             temperature=0.01 
# #         )
# #         content = response.choices[0].message.content
        
# #         # --- RESILIENT PARSING LOGIC ---
# #         items = []
# #         # We split by START, but we do NOT require END to exist
# #         raw_blocks = content.split("---ITEM START---")
        
# #         for block in raw_blocks:
# #             if not block.strip(): continue # Skip empty preamble
            
# #             # Regex Extraction (Safe mode: returns empty string if not found)
# #             type_val = (re.search(r"TYPE:\s*(.*?)\n", block) or [None, "Table"])[1].strip()
# #             summary = (re.search(r"SUMMARY:\s*(.*?)\nCSV:", block, re.DOTALL) or [None, ""])[1].strip()
            
# #             # For CSV, we grab everything after "CSV:" until "NOTES:" OR until end of string
# #             csv_match = re.search(r"CSV:\s*(.*?)(?=\nNOTES:|$)", block, re.DOTALL)
# #             csv_data = csv_match.group(1).strip() if csv_match else ""
            
# #             # For Notes, we grab everything after "NOTES:" until "BBOX:" OR until end of string
# #             notes_match = re.search(r"NOTES:\s*(.*?)(?=\nBBOX:|$)", block, re.DOTALL)
# #             notes = notes_match.group(1).strip() if notes_match else ""
            
# #             # Parse BBox
# #             bbox = [0,0,0,0]
# #             bbox_match = re.search(r"BBOX:\s*\[?([\d\s,]+)\]?", block)
# #             if bbox_match:
# #                 try:
# #                     coords = [int(x) for x in bbox_match.group(1).replace(',', ' ').split()]
# #                     if len(coords) == 4: bbox = coords
# #                 except: pass

# #             # VALIDATION: Only add if we have valid CSV data
# #             if csv_data and len(csv_data) > 10:
# #                 items.append({
# #                     "type": type_val,
# #                     "summary": summary,
# #                     "csv": csv_data,
# #                     "notes": notes,
# #                     "bbox": bbox
# #                 })

# #         # DEBUG: If we found nothing, print what the model actually said
# #         if not items:
# #             print(f"   ⚠️ No items found. RAW OUTPUT START:\n{content[:200]}...\n")

# #         return {"tables": items}

# #     except Exception as e:
# #         print(f"   ❌ API Failure: {e}")
# #         return {"tables": []}
    
# # def process_pdf(pdf_path):
# #     print(f"📄 Processing: {pdf_path}")
# #     images = convert_from_path(pdf_path, dpi=300) # High DPI for source crop
    
# #     for i, img in enumerate(images):
# #         page_num = i + 1
# #         print(f"\n--- Page {page_num} ---")
# #         temp_img_path = f"temp_page_{page_num}.jpg"
# #         img.save(temp_img_path)
        
# #         result = extract_page_data(temp_img_path)
        
# #         if result.get("tables"):
# #             print(f"   ✅ Found {len(result['tables'])} items.")
# #             width, height = img.size
            
# #             for item in result['tables']:
# #                 item_id = str(uuid.uuid4())[:8]
                
# #                 # 1. Save CSV
# #                 with open(f"{OUTPUT_DIR}/{item_id}.csv", "w") as f:
# #                     f.write(item.get('csv', ''))
                
# #                 # 2. Save Screenshot
# #                 try:
# #                     y1, x1, y2, x2 = item['bbox']
# #                     crop = img.crop((
# #                         int(x1/1000 * width), int(y1/1000 * height),
# #                         int(x2/1000 * width), int(y2/1000 * height)
# #                     ))
# #                     crop.save(f"{OUTPUT_DIR}/{item_id}.jpg")
# #                 except:
# #                     pass

# #                 # 3. Save Metadata
# #                 with open(f"{OUTPUT_DIR}/{item_id}.json", "w") as f:
# #                     json.dump(item, f, indent=2)
                
# #                 print(f"      💾 Saved Item: {item_id}")

# #                 meta_path = os.path.join(OUTPUT_DIR, f"{item_id}.json")
# #                 metadata = {
# #                     "id": item_id,
# #                     "summary": item['summary'],
# #                     "notes": item.get('notes', ''), # <--- Save notes here for RAG
# #                     "source_pdf": os.path.basename(pdf_path),
# #                     "page": page_num,
# #                     "csv_path": f"{item_id}.csv",
# #                     "image_path": f"{item_id}.jpg"
# #                 }

# #         else:
# #             print("   (No structured data found)")
        
# #         if os.path.exists(temp_img_path): os.remove(temp_img_path)

# # if __name__ == "__main__":
# #     process_pdf(INPUT_PDF)

# import base64
# import json
# import io
# import os
# from openai import OpenAI
# from pdf2image import convert_from_path
# from PIL import Image

# # --- CONFIGURATION ---
# API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1"
# API_KEY = "dev"
# client = OpenAI(base_url=API_URL, api_key=API_KEY)

# def encode_image(image_obj, max_size=1024):
#     """
#     Resizes and encodes the image for the layout scan.
#     Medium resolution (1024px) is enough for layout detection.
#     """
#     if image_obj.width > max_size:
#         aspect_ratio = image_obj.height / image_obj.width
#         new_height = int(max_size * aspect_ratio)
#         image_obj = image_obj.resize((max_size, new_height), Image.Resampling.LANCZOS)
    
#     buffered = io.BytesIO()
#     image_obj.save(buffered, format="JPEG", quality=85)
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def analyze_layout(image_path_or_obj):
#     """
#     Pass 1: Asks Qwen to categorize the page into bounding boxes.
#     Does NOT ask for full text transcription.
#     """
#     if isinstance(image_path_or_obj, str):
#         img = Image.open(image_path_or_obj)
#     else:
#         img = image_path_or_obj

#     base64_image = encode_image(img)

#     system_prompt = """
#     You are a Document Layout Analyst. Your job is to bounding-box visual elements.
    
#     CRITICAL INSTRUCTION: 
#     - If you see a Chart, Graph, or Plot, DO NOT read the axis text. Mark the WHOLE image area as 'FIGURE'.
#     - If you see a Grid of Numbers (Table), mark the WHOLE grid as 'TABLE'.
    
#     CATEGORIES:
#     - HEADER: Top page titles (e.g. "For release at 2:00 p.m...").
#     - TABLE: Any grid of data/numbers. (PRIORITY: Detect these first).
#     - FIGURE: Charts, Dot Plots, Graphs. (PRIORITY: Detect these first).
#     - TEXT: Narrative paragraphs and lists.
#     - FOOTER: Page numbers/footnotes.

#     OUTPUT FORMAT (JSON):
#     {
#       "layout": [
#         {"type": "TABLE", "bbox": [ymin, xmin, ymax, xmax], "hint": "Table 1 Economic Projections..."},
#         {"type": "FIGURE", "bbox": [ymin, xmin, ymax, xmax], "hint": "Figure 1 Medians and ranges..."}
#       ]
#     }
#     """

#     try:
#         response = client.chat.completions.create(
#             model="Qwen/Qwen2-VL-7B-Instruct-AWQ",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": system_prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#                     ],
#                 }
#             ],
#             max_tokens=2048,
#             temperature=0.01
#         )
        
#         content = response.choices[0].message.content
#         # Robust parsing cleanup
#         content = content.replace("```json", "").replace("```", "").strip()
        
#         # Parse JSON
#         return json.loads(content)

#     except json.JSONDecodeError:
#         print("   ⚠️ Layout Scan: JSON Parse Error")
#         return {"layout": []}
#     except Exception as e:
#         print(f"   ❌ Layout Scan Error: {e}")
#         return {"layout": []}

# # --- TEST FUNCTION ---
# def test_layout_scan(pdf_path):
#     print(f"🔍 Scanning Layout: {pdf_path}")

#     images = convert_from_path(pdf_path, dpi=200, last_page=1)
    
#     if images:
#         result = analyze_layout(images[0])
#         return result
#     else:
#         print("❌ Failed to load PDF image")

# if __name__ == "__main__":
#     # Test on a file likely to have structure (e.g., Summary of Projections)
#     test_pdf = "data/raw/fomcprojtabl20240612.pdf" 
#     if os.path.exists(test_pdf):
#         print(test_layout_scan(test_pdf))
#     else:
#         print(f"File not found: {test_pdf}")
#     print("Module loaded. Run test_layout_scan(pdf_path) to test layout extraction.")


import base64
import json
import io
import os
import re
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

# --- CONFIGURATION ---
# 1. Update the URL to your new Ngrok address
API_URL = "https://sternmost-nonesuriently-trinity.ngrok-free.dev/v1"

# 2. Update the Model Name (Must match exactly what we launched)
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

# 3. Key stays the same
API_KEY = "production-key" 

client = OpenAI(base_url=API_URL, api_key=API_KEY)

def encode_image(image_obj, max_size=1600):
    """
    Resizes and encodes the image.
    Updated to 1600px max_size to leverage 72B model's vision capabilities.
    """
    if image_obj.width > max_size:
        aspect_ratio = image_obj.height / image_obj.width
        new_height = int(max_size * aspect_ratio)
        image_obj = image_obj.resize((max_size, new_height), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def repair_json(json_str):
    """Attempts to fix common JSON errors from LLMs."""
    json_str = json_str.strip()
    # Remove markdown wrappers
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    
    # Attempt to parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: Try to find the first [ or { and last ] or }
        try:
            # Find start/end for List or Dict
            if "{" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
            else:
                start = json_str.find("[")
                end = json_str.rfind("]") + 1
                
            if start != -1 and end != -1:
                return json.loads(json_str[start:end])
        except:
            return []
    return []

def analyze_layout(image_path_or_obj):
    """
    Pass 1: Asks Qwen 72B to categorize the page into bounding boxes.
    """
    if isinstance(image_path_or_obj, str):
        img = Image.open(image_path_or_obj)
    else:
        img = image_path_or_obj

    base64_image = encode_image(img, max_size=1600) # Increased resolution for 72B

    system_prompt = """
You are a High-Precision Layout Detection Engine. Your mode is STRICT STRUCTURAL SEGMENTATION. 
You do not read content for meaning; you analyze visual density and alignment to draw bounding boxes.

### PRIME DIRECTIVE: PRIORITIZE TABLES AND FIGURES
Your specific mission is to stop the model from fragmenting complex objects into text lines. 
If an area looks like a Table or a Figure, you must CAPTURE THE WHOLE BLOCK.

### STRICT CATEGORY DEFINITIONS:

1. **TABLE (Highest Priority)**
   - **Trigger:** Any region with *columnar alignment*, *grids of numbers*, or *row headers* (e.g., "Median", "Central Tendency").
   - **Aggressive Rule:** If you see distinct columns, IT IS A TABLE. Do not label it as TEXT.
   - **Scope:** The bounding box MUST include the **Table Title** (top), the **Column Headers**, the **Data Grid**, and any **Table Footnotes** (bottom text starting with "Note:" or "1.").
   - **Anti-Pattern:** Do NOT split a table into headers and data. Box them as ONE single 'TABLE'.

2. **FIGURE (High Priority)**
   - **Trigger:** Charts, plots, graphs, diagrams, or visual data representations.
   - **Aggressive Rule:** IGNORE all internal text (axis labels, data points). They are "noise".
   - **Scope:** Box the entire visual container, including the Figure Title and the Legend.

3. **HEADER / FOOTER**
   - **Trigger:** Isolated single lines at the absolute top or bottom of the page (e.g., page numbers, "Confidential", release dates).
   - **Restriction:** Do not mistake Table Titles for Page Headers.

4. **TEXT (Lowest Priority)**
   - **Trigger:** Standard prose paragraphs or narrative lists.
   - **Restriction:** Only use this for content that fails the Table/Figure tests.

### OUTPUT PROTOCOL (JSON ONLY)
- Output strictly valid JSON.
- Coordinates: [ymin, xmin, ymax, xmax] (0-1000 scale or normalized 0-1).
- "hint": Provide the first 3-5 words of the block to confirm identity.

Example JSON Structure:
{
  "layout": [
    {"type": "HEADER", "bbox": [10, 50, 40, 950], "hint": "For release at..."},
    {"type": "TABLE", "bbox": [120, 40, 850, 960], "hint": "Table 1. Economic projections..."} 
  ]
}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, # Use 72B Model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=4096, # Increased tokens for safety
            temperature=0.01
        )
        
        content = response.choices[0].message.content
        return repair_json(content)

    except Exception as e:
        print(f"   ❌ Layout Scan Error: {e}")
        return {"layout": []}

def extract_page_data(image_path):
    """
    Pass 2: Visual Extraction (High-Res Crop).
    Returns a list of dictionaries.
    """
    # 1. Encode Image
    with Image.open(image_path) as img:
        # 1600px is safe for 72B on A100 (80GB)
        max_dim = 1600
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 2. Simple JSON Prompt
    system_prompt = """
    Analyze this cropped image (Table or Chart).
    Return a JSON LIST of objects.
    
    Format:
    [
      {
        "type": "TABLE" or "CHART",
        "summary": "Detailed description of the data trend or structure.",
        "content": "For TABLES: Output clean CSV data. For CHARTS: Output 'Visual Chart'.",
        "notes": "Any footnotes or text below the image."
      }
    ]
    
    If the image is empty or blurry, return [].
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
        
        # 3. Robust Parsing
        raw_content = response.choices[0].message.content
        data = repair_json(raw_content)
        
        # Wrap in expected format for the builder
        if isinstance(data, list):
            return {"tables": data}
        elif isinstance(data, dict):
            return {"tables": [data]}
        else:
            return {"tables": []}

    except Exception as e:
        print(f"      ❌ Visual Extractor Failed: {e}")
        return {"tables": []}

# --- TEST FUNCTION ---
def test_layout_scan(pdf_path, page_num=2):
    print(f"🔍 Scanning Layout: {pdf_path} (Page {page_num})")

    # Only convert the specific page we want to test
    images = convert_from_path(pdf_path, dpi=300, first_page=page_num, last_page=page_num)
    
    if images:
        result = analyze_layout(images[0])
        print(json.dumps(result, indent=2))
        return result
    else:
        print("❌ Failed to load PDF image")

if __name__ == "__main__":
    # Test on Page 2 (likely to have a table/figure)
    test_pdf = "data/raw/fomcprojtabl20200610.pdf" 
    if os.path.exists(test_pdf):
        test_layout_scan(test_pdf, page_num=2)
    else:
        print(f"File not found: {test_pdf}")