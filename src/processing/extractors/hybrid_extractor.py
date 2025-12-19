import fitz  # PyMuPDF
import uuid
import os
import json
from PIL import Image, ImageDraw, ImageFont
from src.processing.extractors.qwen_extractor import extract_page_data

def scale_rect_from_1000(bbox_1000, target_w, target_h):
    """ Converts 0-1000 normalized coordinates to absolute pixels/points. """
    ymin, xmin, ymax, xmax = bbox_1000
    return (
        int(xmin / 1000 * target_w),
        int(ymin / 1000 * target_h),
        int(xmax / 1000 * target_w),
        int(ymax / 1000 * target_h)
    )

def create_debug_overlay(image, layout_map, output_path):
    """ Draws colored boxes on a copy of the image for human review. """
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    styles = {
        "TABLE":  {"color": "red", "width": 4},
        "FIGURE": {"color": "blue", "width": 4},
        "HEADER": {"color": "green", "width": 3},
        "TEXT":   {"color": "orange", "width": 2}
    }

    w, h = debug_img.size
    for item in layout_map.get("layout", []):
        etype = item.get("type", "TEXT")
        bbox = item.get("bbox", [0,0,0,0])
        x0, y0, x1, y1 = scale_rect_from_1000(bbox, w, h)
        
        style = styles.get(etype, styles["TEXT"])
        draw.rectangle([x0, y0, x1, y1], outline=style["color"], width=style["width"])
        draw.text((x0 + 5, y0 - 25), etype, fill=style["color"], font=font)

    debug_img.save(output_path)

def extract_hybrid_content(pdf_path, page_num, layout_map, temp_image_path, output_dir, previous_context=None):
    """
    Args:
        previous_context (str): The section anchor from the previous page (to maintain continuity).
    Returns:
        (extracted_elements, last_active_anchor)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pdf_w, pdf_h = page.rect.width, page.rect.height
    
    full_image = Image.open(temp_image_path)
    img_w, img_h = full_image.size

    # 1. Save Visual Assets
    clean_path = os.path.join(output_dir, f"page_{page_num}_clean.jpg")
    debug_path = os.path.join(output_dir, f"page_{page_num}_debug.jpg")
    if not os.path.exists(clean_path): # Optimization: Don't overwrite if exists
        full_image.save(clean_path)
    create_debug_overlay(full_image, layout_map, debug_path)

    extracted_elements = []
    
    # 2. Sort by Reading Order (Top -> Bottom)
    sorted_layout = sorted(layout_map.get("layout", []), key=lambda k: (k['bbox'][0], k['bbox'][1]))

    # 3. Context Logic: Inherit from previous page or default
    current_section_anchor = previous_context if previous_context else "Unspecified Section"
    
    for index, item in enumerate(sorted_layout):
        element_type = item.get("type", "TEXT")
        bbox_1000 = item.get("bbox", [0,0,0,0]) 
        element_id = f"{os.path.basename(pdf_path).replace('.pdf','')}_p{page_num}_{index}"

        # --- ANCHOR DETECTION ---
        if element_type == "HEADER":
            # Extract header text to use as new anchor
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, pdf_w, pdf_h)
            header_rect = fitz.Rect(x0, y0, x1, y1)
            header_text = page.get_text("text", clip=header_rect).strip().replace("\n", " ")
            if len(header_text) > 5:
                current_section_anchor = header_text
                print(f"      ⚓ Anchor Updated: {current_section_anchor[:30]}...")

        # --- VISUAL EXTRACTION (VLM) ---
        if element_type in ["TABLE", "FIGURE", "CHART"]:
            print(f"      📸 Visual Extract: {element_type}")
            # Crop with Padding for Image
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, img_w, img_h)
            pad = 15
            crop_coords = (max(0, x0-pad), max(0, y0-pad), min(img_w, x1+pad), min(img_h, y1+pad))
            
            crop_filename = f"crop_{element_id}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            full_image.crop(crop_coords).save(crop_path)
            
            # Call VLM
            try:
                vlm_data = extract_page_data(crop_path)
                data_item = vlm_data.get("tables", [{}])[0]
                
                extracted_elements.append({
                    "id": element_id,
                    "page": page_num,
                    "type": element_type,
                    "section_anchor": current_section_anchor,
                    "bbox_norm": bbox_1000,
                    "content": data_item.get("content", "") or data_item.get("csv", ""),
                    "summary": data_item.get("summary", ""),
                    "visual_ref": {
                        "clean_page": f"page_{page_num}_clean.jpg",
                        "debug_page": f"page_{page_num}_debug.jpg",
                        "crop_file": crop_filename
                    }
                })
            except Exception as e:
                print(f"      ⚠️ VLM Error: {e}")

        # --- TEXT EXTRACTION (PyMuPDF) ---
        else:
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, pdf_w, pdf_h)
            
            # FIX 1: SAFETY PADDING
            # Expand the box slightly (e.g., 2 points) to catch characters on the edge
            # while respecting page boundaries
            pad = 2 
            safe_rect = fitz.Rect(max(0, x0-pad), max(0, y0-pad), min(pdf_w, x1+pad), min(pdf_h, y1+pad))
            
            # FIX 2: CLEANING
            text_val = page.get_text("text", clip=safe_rect)
            text_val = " ".join(text_val.split()) # Removes \n and extra spaces
            
            if len(text_val) > 5:
                extracted_elements.append({
                    "id": element_id,
                    "page": page_num,
                    "type": element_type,
                    "section_anchor": current_section_anchor,
                    "bbox_norm": bbox_1000,
                    "content": text_val,
                    "visual_ref": {
                        "clean_page": f"page_{page_num}_clean.jpg",
                        "debug_page": f"page_{page_num}_debug.jpg"
                    }
                })

    return extracted_elements, current_section_anchor