import fitz  # PyMuPDF
import uuid
import os
import json
import re
from PIL import Image, ImageDraw, ImageFont
from src.processing.extractors.qwen_extractor import extract_page_data
from src.utils.config import config
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# --- COORDINATE UTILITIES ---

def scale_rect_from_1000(bbox_1000, target_w, target_h):
    """Converts 0-1000 normalized coordinates to absolute pixels/points."""
    ymin, xmin, ymax, xmax = bbox_1000
    return (
        int(xmin / 1000 * target_w),
        int(ymin / 1000 * target_h),
        int(xmax / 1000 * target_w),
        int(ymax / 1000 * target_h)
    )

def calculate_overlap(bbox1, bbox2):
    """Calculate intersection over union (IoU) for two bboxes."""
    y1_min, x1_min, y1_max, x1_max = bbox1
    y2_min, x2_min, y2_max, x2_max = bbox2

    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def refine_bbox_with_text(page: fitz.Page, bbox_norm: List[int], pdf_w: float, pdf_h: float, padding_pct: float = 2.0) -> List[int]:
    """
    Refine bounding box to actual text boundaries using PyMuPDF.

    Args:
        page: PyMuPDF page object
        bbox_norm: Normalized bbox [ymin, xmin, ymax, xmax] in 0-1000 scale
        pdf_w: PDF page width
        pdf_h: PDF page height
        padding_pct: Padding to add around text (percentage of 1000)

    Returns:
        Refined bbox in same normalized format
    """
    # Convert normalized bbox to PDF coordinates
    ymin, xmin, ymax, xmax = bbox_norm
    bbox_pdf = fitz.Rect(
        xmin * pdf_w / 1000,
        ymin * pdf_h / 1000,
        xmax * pdf_w / 1000,
        ymax * pdf_h / 1000
    )

    # Get actual text blocks in this region
    try:
        text_blocks = page.get_text("blocks", clip=bbox_pdf)
    except:
        logger.warning("Failed to get text blocks for bbox refinement")
        return bbox_norm

    if not text_blocks:
        # No text found, return original bbox
        return bbox_norm

    # Find min/max coordinates of actual text
    min_x = min(block[0] for block in text_blocks)
    min_y = min(block[1] for block in text_blocks)
    max_x = max(block[2] for block in text_blocks)
    max_y = max(block[3] for block in text_blocks)

    # Convert back to normalized coordinates
    refined_bbox = [
        int(min_y / pdf_h * 1000),
        int(min_x / pdf_w * 1000),
        int(max_y / pdf_h * 1000),
        int(max_x / pdf_w * 1000)
    ]

    # Add padding (default 2% of page)
    padding = int(padding_pct * 10)  # 2% of 1000 = 20
    refined_bbox[0] = max(0, refined_bbox[0] - padding)
    refined_bbox[1] = max(0, refined_bbox[1] - padding)
    refined_bbox[2] = min(1000, refined_bbox[2] + padding)
    refined_bbox[3] = min(1000, refined_bbox[3] + padding)

    # Calculate improvement
    original_area = (bbox_norm[2] - bbox_norm[0]) * (bbox_norm[3] - bbox_norm[1])
    refined_area = (refined_bbox[2] - refined_bbox[0]) * (refined_bbox[3] - refined_bbox[1])

    if refined_area < original_area * 0.3:
        # Refinement too aggressive (reduced by >70%), keep original
        logger.debug(f"Bbox refinement too aggressive, keeping original")
        return bbox_norm

    logger.debug(f"Bbox refined: area reduced by {((original_area - refined_area) / original_area * 100):.1f}%")

    return refined_bbox

# --- VISUAL DEBUG UTILITIES ---

def create_debug_overlay(image, layout_map, output_path):
    """
    Enhanced debug visualization with confidence scores, labels, and overlap detection.

    Features:
    - Color-coded bounding boxes by element type
    - Element type + confidence score labels
    - Overlap warning visualization
    - Extraction quality color coding
    """
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    # Try to load fonts with fallback
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
            label_font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            label_font = title_font
            small_font = title_font

    styles = {
        "TABLE":  {"color": "red", "width": 5},
        "FIGURE": {"color": "blue", "width": 5},
        "HEADER": {"color": "green", "width": 4},
        "TEXT":   {"color": "orange", "width": 3},
        "FOOTER": {"color": "gray", "width": 2}
    }

    w, h = debug_img.size
    layout_elements = layout_map.get("layout", [])

    # First pass: detect overlaps
    overlaps = []
    for i, item in enumerate(layout_elements):
        for j, other in enumerate(layout_elements):
            if i >= j:
                continue
            overlap = calculate_overlap(item.get("bbox", [0,0,0,0]), other.get("bbox", [0,0,0,0]))
            if overlap > 0.3:  # More than 30% overlap
                overlaps.append((i, j, overlap))

    # Second pass: draw elements
    for idx, item in enumerate(layout_elements):
        etype = item.get("type", "TEXT")
        bbox = item.get("bbox", [0,0,0,0])
        confidence = item.get("confidence", 0.0)
        quality = item.get("extraction_quality", 0.0)

        x0, y0, x1, y1 = scale_rect_from_1000(bbox, w, h)

        style = styles.get(etype, styles["TEXT"])

        # Check if this element has overlaps
        has_overlap = any(idx in (pair[0], pair[1]) for pair in overlaps)

        # Adjust color for low quality
        box_color = style["color"]
        if quality > 0 and quality < 0.5:
            box_color = "darkred"  # Low quality warning
        elif has_overlap:
            box_color = "purple"  # Overlap warning

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=style["width"])

        # Draw overlap warning (X across box)
        if has_overlap:
            draw.line([(x0, y0), (x1, y1)], fill="red", width=2)
            draw.line([(x1, y0), (x0, y1)], fill="red", width=2)

        # Prepare label text
        if confidence > 0:
            label = f"{idx}: {etype} ({confidence:.2f})"
        else:
            label = f"{idx}: {etype}"

        # Add quality indicator
        if quality > 0:
            quality_indicator = "⭐" if quality >= 0.8 else "⚠️" if quality >= 0.5 else "❌"
            label += f" {quality_indicator}"

        # Calculate label position (above bbox)
        label_y = max(5, y0 - 25)

        # Draw label background
        try:
            label_bbox_coords = draw.textbbox((x0 + 5, label_y), label, font=label_font)
            draw.rectangle(label_bbox_coords, fill=box_color, outline="black")
        except:
            # Fallback if textbbox not available
            label_width = len(label) * 8
            label_height = 20
            draw.rectangle(
                [(x0 + 5, label_y), (x0 + 5 + label_width, label_y + label_height)],
                fill=box_color,
                outline="black"
            )

        # Draw label text
        draw.text((x0 + 5, label_y), label, fill="white", font=label_font)

        # Draw confidence bar (if available)
        if confidence > 0:
            bar_width = int(confidence * 100)
            bar_y = y1 - 5
            draw.rectangle([(x0, bar_y), (x0 + bar_width, bar_y + 3)], fill="lime")

    # Draw legend
    legend_x = 10
    legend_y = 10
    legend_items = [
        ("TABLE", "red"),
        ("FIGURE", "blue"),
        ("HEADER", "green"),
        ("TEXT", "orange"),
        ("FOOTER", "gray"),
        ("OVERLAP", "purple"),
        ("LOW QUAL", "darkred")
    ]

    for i, (name, color) in enumerate(legend_items):
        y_pos = legend_y + (i * 25)
        # Draw color box
        draw.rectangle([(legend_x, y_pos), (legend_x + 20, y_pos + 15)], fill=color, outline="black")
        # Draw label
        draw.text((legend_x + 25, y_pos), name, fill="black", font=small_font)

    # Draw summary stats
    stats_text = f"Total: {len(layout_elements)} | Overlaps: {len(overlaps)}"
    draw.text((legend_x, legend_y + 190), stats_text, fill="black", font=label_font)

    debug_img.save(output_path, quality=config.processing.image_quality)
    logger.info(f"Enhanced debug overlay saved: {output_path} ({len(layout_elements)} elements, {len(overlaps)} overlaps)")

# --- READING ORDER LOGIC ---

def detect_multi_column(layout_elements: List[dict], threshold: float = 0.3) -> bool:
    """Detect if layout has multiple columns."""
    if len(layout_elements) < 3:
        return False
    
    # Check horizontal spacing between elements
    x_positions = []
    for elem in layout_elements:
        if elem.get("type") != "HEADER":
            bbox = elem.get("bbox", [0, 0, 0, 0])
            x_center = (bbox[1] + bbox[3]) / 2
            x_positions.append(x_center)
    
    if len(x_positions) < 2:
        return False
    
    # Look for clustering in x positions
    x_positions.sort()
    gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
    
    if not gaps:
        return False
    
    max_gap = max(gaps)
    avg_gap = sum(gaps) / len(gaps)
    
    # If there's a large gap (column separator), it's multi-column
    return max_gap > avg_gap * 3

def sort_reading_order(layout_elements: List[dict]) -> List[dict]:
    """Intelligent reading order sorting."""
    
    if not layout_elements:
        return []
    
    # Separate headers (should come first)
    headers = [e for e in layout_elements if e.get("type") == "HEADER"]
    non_headers = [e for e in layout_elements if e.get("type") != "HEADER"]
    
    # Sort headers by vertical position
    headers.sort(key=lambda e: e["bbox"][0])
    
    # Check for multi-column
    if detect_multi_column(non_headers):
        logger.info("Multi-column layout detected")
        non_headers = sort_multi_column(non_headers)
    else:
        # Simple top-to-bottom, left-to-right
        non_headers.sort(key=lambda e: (e["bbox"][0], e["bbox"][1]))
    
    return headers + non_headers

def sort_multi_column(elements: List[dict]) -> List[dict]:
    """Sort multi-column layout by columns, then by vertical position."""
    
    # Find column boundaries
    x_centers = [(e["bbox"][1] + e["bbox"][3]) / 2 for e in elements]
    x_centers.sort()
    
    # Use k-means like clustering (simplified)
    if len(x_centers) < 2:
        return sorted(elements, key=lambda e: e["bbox"][0])
    
    mid_point = (max(x_centers) + min(x_centers)) / 2
    
    left_col = [e for e in elements if (e["bbox"][1] + e["bbox"][3]) / 2 < mid_point]
    right_col = [e for e in elements if (e["bbox"][1] + e["bbox"][3]) / 2 >= mid_point]
    
    # Sort each column vertically
    left_col.sort(key=lambda e: e["bbox"][0])
    right_col.sort(key=lambda e: e["bbox"][0])
    
    # Interleave based on vertical position
    result = []
    i, j = 0, 0
    
    while i < len(left_col) and j < len(right_col):
        if left_col[i]["bbox"][0] < right_col[j]["bbox"][0]:
            result.append(left_col[i])
            i += 1
        else:
            result.append(right_col[j])
            j += 1
    
    result.extend(left_col[i:])
    result.extend(right_col[j:])
    
    return result

# --- TEXT EXTRACTION WITH STRUCTURE PRESERVATION ---

def extract_text_with_structure(page, bbox_1000, pdf_w, pdf_h) -> str:
    """Extract text preserving paragraph structure."""
    
    x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, pdf_w, pdf_h)
    
    # Add safety padding
    pad = 3
    safe_rect = fitz.Rect(
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(pdf_w, x1 + pad),
        min(pdf_h, y1 + pad)
    )
    
    # Extract with layout preservation
    text_val = page.get_text("text", clip=safe_rect)
    
    # Intelligent cleaning
    # 1. Normalize line breaks (preserve paragraphs)
    text_val = re.sub(r'\n{3,}', '\n\n', text_val)  # Max 2 consecutive newlines
    
    # 2. Fix hyphenated words at line breaks
    text_val = re.sub(r'-\n', '', text_val)
    
    # 3. Collapse single line breaks within paragraphs
    lines = text_val.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            # Preserve empty lines (paragraph breaks)
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append('')
        else:
            # Check if this continues previous line
            if (cleaned_lines and 
                cleaned_lines[-1] and 
                not line[0].isupper() and 
                not cleaned_lines[-1].endswith(('.', ':', '!', '?'))):
                # Continue previous line
                cleaned_lines[-1] += ' ' + line
            else:
                cleaned_lines.append(line)
    
    text_val = '\n'.join(cleaned_lines)
    
    # 4. Clean up excessive spaces
    text_val = re.sub(r' +', ' ', text_val)
    
    return text_val.strip()

# --- METADATA EXTRACTION ---

def extract_keywords(text: str, content_type: str) -> List[str]:
    """Extract keywords from text based on content type."""
    
    # Domain-specific keyword patterns
    patterns = {
        "economic": r'\b(GDP|inflation|unemployment|rate|growth|projection|estimate|forecast|basis points?|percent)\b',
        "financial": r'\b(revenue|profit|loss|margin|assets|liabilities|equity|earnings)\b',
        "dates": r'\b(\d{4}|Q[1-4]|\d{1,2}/\d{1,2}/\d{2,4})\b'
    }
    
    keywords = set()
    text_lower = text.lower()
    
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        keywords.update(matches)
    
    # Limit to top 10 most relevant
    return list(keywords)[:10]

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Simple entity extraction (can be enhanced with NER models)."""
    
    entities = {
        "organizations": [],
        "locations": [],
        "dates": []
    }
    
    # Pattern-based extraction
    org_patterns = [
        r'\b(Federal Reserve|FOMC|Board of Governors|Federal Reserve Bank)\b',
        r'\b([A-Z][a-z]+ [A-Z][a-z]+ (Committee|Board|Bank|Corporation|Inc\.))\b'
    ]
    
    for pattern in org_patterns:
        matches = re.findall(pattern, text)
        entities["organizations"].extend([m if isinstance(m, str) else m[0] for m in matches])
    
    # Date extraction
    date_pattern = r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
    entities["dates"] = re.findall(date_pattern, text)
    
    return entities

def calculate_quality_score(element: dict) -> float:
    """Calculate extraction quality score."""
    
    base_score = element.get("extraction_quality", 0.7)
    
    # Penalties
    content = element.get("content", "")
    
    if len(content) < 10:
        base_score -= 0.2
    
    if "EXTRACTION_FAILED" in content:
        return 0.0
    
    if element.get("validation_status") == "FAIL":
        base_score -= 0.3
    elif element.get("validation_status") == "PARTIAL":
        base_score -= 0.1
    
    # Bonuses
    if element.get("structured_data"):
        base_score += 0.1
    
    return max(0.0, min(1.0, base_score))

# --- IMAGE DETECTION FALLBACK ---

def detect_image_blocks_pymupdf(page: fitz.Page, pdf_w: float, pdf_h: float) -> List[dict]:
    """
    Fallback image detection using PyMuPDF.

    Detects embedded images that VLM might have missed.

    Args:
        page: PyMuPDF page object
        pdf_w: PDF page width
        pdf_h: PDF page height

    Returns:
        List of image elements with normalized bboxes
    """
    image_elements = []

    try:
        # Get all images on the page
        image_list = page.get_images(full=True)

        if not image_list:
            return []

        logger.debug(f"PyMuPDF detected {len(image_list)} embedded images")

        for img_index, img in enumerate(image_list):
            xref = img[0]  # Image xref number

            # Get image bounding box
            try:
                # Get all locations where this image appears
                img_rects = page.get_image_rects(xref)

                if not img_rects:
                    continue

                # Use first occurrence
                bbox_pdf = img_rects[0]

                # Convert to normalized coordinates [ymin, xmin, ymax, xmax] in 0-1000
                bbox_norm = [
                    int(bbox_pdf.y0 / pdf_h * 1000),
                    int(bbox_pdf.x0 / pdf_w * 1000),
                    int(bbox_pdf.y1 / pdf_h * 1000),
                    int(bbox_pdf.x1 / pdf_w * 1000)
                ]

                # Calculate image area (percentage of page)
                area = (bbox_norm[2] - bbox_norm[0]) * (bbox_norm[3] - bbox_norm[1])
                area_pct = area / (1000 * 1000) * 100

                # Skip very small images (likely logos, icons)
                if area_pct < 1.0:  # Less than 1% of page
                    logger.debug(f"Skipping small image ({area_pct:.1f}% of page)")
                    continue

                image_elements.append({
                    "type": "FIGURE",
                    "bbox": bbox_norm,
                    "hint": f"Image {img_index + 1} (detected by PyMuPDF)",
                    "source": "pymupdf_fallback",
                    "confidence": 0.7,  # Lower confidence for fallback detection
                    "metadata": {
                        "xref": xref,
                        "area_percentage": area_pct,
                        "detection_method": "pymupdf_image_block"
                    }
                })

                logger.debug(f"Added image: {area_pct:.1f}% of page at {bbox_norm}")

            except Exception as e:
                logger.warning(f"Failed to get bbox for image {img_index}: {e}")
                continue

        return image_elements

    except Exception as e:
        logger.error(f"PyMuPDF image detection failed: {e}")
        return []


def merge_layout_with_fallback_images(
    layout_elements: List[dict],
    fallback_images: List[dict],
    overlap_threshold: float = 0.5
) -> List[dict]:
    """
    Merge fallback-detected images into layout, avoiding duplicates.

    Args:
        layout_elements: Elements from VLM
        fallback_images: Images detected by PyMuPDF
        overlap_threshold: IoU threshold to consider as duplicate

    Returns:
        Merged element list
    """
    if not fallback_images:
        return layout_elements

    # Check if VLM already detected FIGUREs
    existing_figures = [e for e in layout_elements if e.get("type") == "FIGURE"]

    if not existing_figures:
        # No figures detected by VLM, add all fallback images
        logger.info(f"Adding {len(fallback_images)} PyMuPDF-detected images (VLM found none)")
        return layout_elements + fallback_images

    # Check for overlaps with existing figures
    images_to_add = []
    for img in fallback_images:
        is_duplicate = False
        for fig in existing_figures:
            overlap = calculate_overlap(img["bbox"], fig["bbox"])
            if overlap > overlap_threshold:
                is_duplicate = True
                logger.debug(f"Skipping duplicate image (IoU: {overlap:.2f})")
                break

        if not is_duplicate:
            images_to_add.append(img)

    if images_to_add:
        logger.info(f"Adding {len(images_to_add)} additional PyMuPDF-detected images")
        return layout_elements + images_to_add
    else:
        logger.debug("All PyMuPDF images already covered by VLM detection")
        return layout_elements


# --- MAIN EXTRACTION FUNCTION ---

def extract_hybrid_content(
    pdf_path: str,
    page_num: int,
    layout_map: dict,
    temp_image_path: str,
    output_dir: str,
    previous_context: Optional[str] = None,
    document_metadata: Optional[dict] = None
) -> Tuple[List[dict], str]:
    """
    Enhanced hybrid extraction with validation, metadata, and quality scoring.
    
    Returns:
        (extracted_elements, last_active_anchor)
    """
    
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pdf_w, pdf_h = page.rect.width, page.rect.height
    
    full_image = Image.open(temp_image_path)
    img_w, img_h = full_image.size
    
    # 1. Save visual assets
    clean_path = os.path.join(output_dir, f"page_{page_num}_clean.jpg")
    debug_path = os.path.join(output_dir, f"page_{page_num}_debug.jpg")
    
    if not os.path.exists(clean_path):
        full_image.save(clean_path, quality=config.processing.image_quality)
    
    create_debug_overlay(full_image, layout_map, debug_path)
    
    extracted_elements = []
    
    # 2. Sort by intelligent reading order
    sorted_layout = sort_reading_order(layout_map.get("layout", []))

    # 3. Apply PyMuPDF image detection fallback
    fallback_images = detect_image_blocks_pymupdf(page, pdf_w, pdf_h)
    if fallback_images:
        sorted_layout = merge_layout_with_fallback_images(sorted_layout, fallback_images)
        # Re-sort after adding fallback images
        sorted_layout = sort_reading_order(sorted_layout)

    logger.info(f"Processing {len(sorted_layout)} elements in reading order")
    
    # 3. Context management with reset logic
    current_section_anchor = previous_context or "Document Content"
    section_start_page = page_num if not previous_context else None
    
    for index, item in enumerate(sorted_layout):
        element_type = item.get("type", "TEXT")
        bbox_1000 = item.get("bbox", [0, 0, 0, 0])
        element_id = f"{os.path.basename(pdf_path).replace('.pdf', '')}_p{page_num}_{index}"
        
        # --- ANCHOR DETECTION & MANAGEMENT ---
        if element_type == "HEADER":
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, pdf_w, pdf_h)
            header_rect = fitz.Rect(x0, y0, x1, y1)
            header_text = page.get_text("text", clip=header_rect).strip().replace("\n", " ")
            
            if len(header_text) > 5:
                current_section_anchor = header_text
                section_start_page = page_num
                logger.info(f"⚓ New Section: {current_section_anchor[:50]}...")
        
        # --- FOOTER DETECTION (triggers anchor reset) ---
        elif element_type == "FOOTER":
            # Check if footer indicates end of section
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, pdf_w, pdf_h)
            footer_rect = fitz.Rect(x0, y0, x1, y1)
            footer_text = page.get_text("text", clip=footer_rect).lower()
            
            if any(marker in footer_text for marker in ["end of section", "***", "---"]):
                logger.info("🔄 Section end detected, resetting anchor")
                current_section_anchor = "Document Content"
        
        # --- VISUAL EXTRACTION (VLM) ---
        if element_type in ["TABLE", "FIGURE", "CHART"]:
            logger.info(f"🖼️  Extracting {element_type} at element {index}")
            
            # Crop with padding
            x0, y0, x1, y1 = scale_rect_from_1000(bbox_1000, img_w, img_h)
            pad = 20
            crop_coords = (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(img_w, x1 + pad),
                min(img_h, y1 + pad)
            )
            
            crop_filename = f"crop_{element_id}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            full_image.crop(crop_coords).save(crop_path, quality=config.processing.image_quality)

            
            # Extract with VLM
            try:
                vlm_result = extract_page_data(crop_path, element_type)
                
                # Build element with rich metadata
                element = {
                    "id": element_id,
                    "page": page_num,
                    "type": element_type,
                    "section_anchor": current_section_anchor,
                    "bbox_norm": bbox_1000,
                    "content": vlm_result.get("content", ""),
                    "summary": vlm_result.get("summary", ""),
                    "structured_data": vlm_result.get("structured_data"),
                    "extraction_method": "VLM",
                    "extraction_quality": calculate_quality_score(vlm_result),
                    "validation_status": vlm_result.get("validation_status", "UNKNOWN"),
                    "metadata": {
                        "keywords": extract_keywords(vlm_result.get("content", ""), "economic"),
                        "contains_numerical_data": bool(re.search(r'\d+\.?\d*', vlm_result.get("content", ""))),
                        "chart_type": vlm_result.get("chart_type"),
                        "column_count": len(vlm_result.get("columns", [])) if vlm_result.get("columns") else None
                    },
                    "visual_ref": {
                        "clean_page": f"page_{page_num}_clean.jpg",
                        "debug_page": f"page_{page_num}_debug.jpg",
                        "crop_file": crop_filename
                    }
                }
                
                extracted_elements.append(element)
                
            except Exception as e:
                logger.error(f"❌ VLM extraction failed for {element_id}: {e}")
                # Add placeholder element to maintain document structure
                extracted_elements.append({
                    "id": element_id,
                    "page": page_num,
                    "type": element_type,
                    "section_anchor": current_section_anchor,
                    "bbox_norm": bbox_1000,
                    "content": f"[Extraction failed: {str(e)}]",
                    "extraction_quality": 0.0,
                    "validation_status": "FAIL",
                    "error": str(e)
                })
        
        # --- TEXT EXTRACTION (PyMuPDF) ---
        else:
            # Refine bbox for TEXT and HEADER elements
            refined_bbox = bbox_1000
            if element_type in ["TEXT", "HEADER"]:
                refined_bbox = refine_bbox_with_text(page, bbox_1000, pdf_w, pdf_h)

            text_val = extract_text_with_structure(page, refined_bbox, pdf_w, pdf_h)

            if len(text_val) > 5:
                entities = extract_entities(text_val)

                element = {
                    "id": element_id,
                    "page": page_num,
                    "type": element_type,
                    "section_anchor": current_section_anchor,
                    "bbox_norm": refined_bbox,
                    "content": text_val,
                    "extraction_method": "PyMuPDF",
                    "extraction_quality": 0.9,  # High confidence for direct text extraction
                    "metadata": {
                        "word_count": len(text_val.split()),
                        "char_count": len(text_val),
                        "keywords": extract_keywords(text_val, "economic"),
                        "entities": entities,
                        "contains_numerical_data": bool(re.search(r'\d+', text_val)),
                        "bbox_refined": refined_bbox != bbox_1000
                    },
                    "visual_ref": {
                        "clean_page": f"page_{page_num}_clean.jpg",
                        "debug_page": f"page_{page_num}_debug.jpg"
                    }
                }

                extracted_elements.append(element)
    
    doc.close()
    
    logger.info(f"✅ Page {page_num}: Extracted {len(extracted_elements)} elements")
    
    return extracted_elements, current_section_anchor