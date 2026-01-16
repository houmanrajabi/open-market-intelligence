import os
import io
import requests
import fitz  # PyMuPDF
from PIL import Image
from typing import List, Dict, Any, Tuple
import logging

# Internal imports
from src.utils.config import config
from src.processing.extractors.qwen_extractor import (
    call_vlm_with_retry, 
    TABLE_PROMPT, 
    CHART_PROMPT
)

logger = logging.getLogger(__name__)

class HybridSuryaExtractor:
    """
    Tier 1 Hybrid Extractor (CLIENT VERSION).
    
    Instead of running Surya locally, this client sends page images to the 
    Vast.ai instance via the SSH tunnel (localhost:8002).
    """

    def __init__(self):
        logger.info("🚀 Initializing HybridSuryaExtractor (Remote Client)...")
        # We no longer load models here. We just check the API config.
        self.api_url = config.processing.surya_api_url  # Ensure this is set in your .env
        if not self.api_url:
            # Fallback if config is missing, assuming standard tunnel
            self.api_url = "http://localhost:8002/v1/ocr" 
            logger.warning(f"⚠️ SURYA_API_URL not found in config. Defaulting to {self.api_url}")

    # --- LAYER 1: DETECTION (REMOTE) ---
    def detect_layout(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Send image to Remote API for layout detection.
        """
        try:
            # 1. Convert PIL Image to Bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            # 2. Call the Remote API (Vast.ai Instance 2)
            # Assuming your surya_api.py expects a file upload named 'file'
            response = requests.post(
            self.api_url,
            files={'file': ('page.jpg', img_byte_arr, 'image/jpeg')},
            data={'langs': '["en"]'},  # <--- Add this line!
            timeout=600
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Remote Surya API Error: {response.text}")
                return []

            # 3. Parse Response
            # The API should return the normalized bboxes directly
            layout_result = response.json()
            return layout_result.get("elements", [])

        except Exception as e:
            logger.error(f"❌ Failed to connect to Surya API at {self.api_url}: {e}")
            return []

    # --- LAYER 2: GEOMETRIC CLUSTERING (LOCAL) ---
    # This logic is fast and mathematical, so we keep it on the Mac
    def cluster_elements(self, elements: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Associate captions with tables/figures and group conceptually.
        """
        clustered = []
        used_indices = set()
        
        # Separate visual assets from text/captions
        visuals = [(i, e) for i, e in enumerate(elements) if e["type"] in ["TABLE", "FIGURE"]]
        texts = [(i, e) for i, e in enumerate(elements) if e["type"] in ["TEXT", "CAPTION"]]

        for v_idx, visual in visuals:
            if v_idx in used_indices: continue
            
            # Search for nearest text/caption (Threshold: 5% of page height)
            threshold_y = 50  
            best_caption = None
            best_dist = float('inf')
            best_c_idx = -1

            v_bbox = visual["bbox"] # [ymin, xmin, ymax, xmax]

            for t_idx, text_elem in texts:
                if t_idx in used_indices: continue
                
                t_bbox = text_elem["bbox"]
                
                # Horizontal alignment check
                v_center_x = (v_bbox[1] + v_bbox[3]) / 2
                t_center_x = (t_bbox[1] + t_bbox[3]) / 2
                if abs(v_center_x - t_center_x) > 200: 
                    continue

                # Vertical proximity check
                dist_below = t_bbox[0] - v_bbox[2]
                dist_above = v_bbox[0] - t_bbox[2]
                dist = min(d for d in [dist_below, dist_above] if d >= -10)
                
                if 0 <= dist < threshold_y:
                    if dist < best_dist:
                        best_dist = dist
                        best_caption = text_elem
                        best_c_idx = t_idx

            if best_caption:
                visual["detected_caption_bbox"] = best_caption["bbox"]
                used_indices.add(best_c_idx)
            
            clustered.append(visual)
            used_indices.add(v_idx)

        # Add remaining elements
        for i, e in enumerate(elements):
            if i not in used_indices:
                if e["type"] == "CAPTION": e["type"] = "TEXT"
                clustered.append(e)

        return clustered

    # --- LAYER 3: HIERARCHY (LOCAL) ---
    def build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign section context based on reading order."""
        sorted_elements = sorted(elements, key=lambda x: (x["bbox"][0], x["bbox"][1]))
        current_header = "Document Start"

        for elem in sorted_elements:
            if elem["type"] == "HEADER":
                elem["is_header"] = True
                elem["section_context"] = current_header
            else:
                elem["section_context"] = current_header
                
        return sorted_elements

    # --- LAYER 4: ENRICHMENT (REMOTE QWEN + LOCAL PYMUPDF) ---
    def extract_content(
        self, 
        elements: List[Dict[str, Any]], 
        page_image: Image.Image, 
        pdf_page: fitz.Page,
        page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Execute extraction: PyMuPDF (Local) + Qwen VLM (Remote).
        """
        final_elements = []
        w, h = page_image.size
        pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height
        
        current_section_text = "Document Start"

        for idx, elem in enumerate(elements):
            bbox_norm = elem["bbox"]
            
            # Map coordinates to PDF points
            rect_pdf = fitz.Rect(
                bbox_norm[1] * pdf_w / 1000,
                bbox_norm[0] * pdf_h / 1000,
                bbox_norm[3] * pdf_w / 1000,
                bbox_norm[2] * pdf_h / 1000
            )

            # 1. TEXT / HEADER (Local is faster/better for simple text)
            if elem["type"] in ["TEXT", "HEADER", "FOOTER"]:
                text = pdf_page.get_text("text", clip=rect_pdf).strip()
                
                if elem["type"] == "HEADER" and len(text) > 3:
                    current_section_text = text
                
                if not text: continue

                final_elements.append({
                    "id": f"p{page_num}_{idx}_{elem['type']}",
                    "type": elem["type"],
                    "content": text,
                    "bbox_norm": bbox_norm,
                    "page": page_num,
                    "section_anchor": current_section_text,
                    "metadata": {"extraction_method": "PyMuPDF_Surya"}
                })

            # 2. VISUALS (Send to Qwen VLM on Instance 1)
            elif elem["type"] in ["TABLE", "FIGURE"]:
                caption_text = ""
                if "detected_caption_bbox" in elem:
                    c_bbox = elem["detected_caption_bbox"]
                    c_rect = fitz.Rect(
                        c_bbox[1] * pdf_w / 1000,
                        c_bbox[0] * pdf_h / 1000,
                        c_bbox[3] * pdf_w / 1000,
                        c_bbox[2] * pdf_h / 1000
                    )
                    caption_text = pdf_page.get_text("text", clip=c_rect).strip()
                
                # Crop logic
                crop_box = (
                    int(bbox_norm[1] * w / 1000),
                    int(bbox_norm[0] * h / 1000),
                    int(bbox_norm[3] * w / 1000),
                    int(bbox_norm[2] * h / 1000)
                )
                crop_box = (
                    max(0, crop_box[0]-10), max(0, crop_box[1]-10),
                    min(w, crop_box[2]+10), min(h, crop_box[3]+10)
                )
                
                crop_img = page_image.crop(crop_box)
                crop_path = f"/tmp/crop_{page_num}_{idx}.jpg"
                crop_img.save(crop_path)
                
                context_str = f"Context: This item appears in section '{current_section_text}'."
                if caption_text:
                    context_str += f" The caption is: '{caption_text}'."
                
                try:
                    logger.info(f"🧠 Calling Qwen VLM (Remote) for {elem['type']}")
                    
                    if elem["type"] == "TABLE":
                        prompt = f"{context_str}\n{TABLE_PROMPT}"
                        vlm_result = self._extract_with_vlm(crop_path, prompt, "TABLE")
                    else:
                        prompt = f"{context_str}\n{CHART_PROMPT}"
                        vlm_result = self._extract_with_vlm(crop_path, prompt, "FIGURE")
                        
                    vlm_result["metadata"]["caption"] = caption_text
                    
                    final_elements.append({
                        "id": f"p{page_num}_{idx}_{elem['type']}",
                        "page": page_num,
                        "bbox_norm": bbox_norm,
                        "section_anchor": current_section_text,
                        **vlm_result
                    })
                    
                except Exception as e:
                    logger.error(f"VLM Failed: {e}")
                finally:
                    if os.path.exists(crop_path):
                        os.remove(crop_path)

        return final_elements

    def _extract_with_vlm(self, image_path: str, prompt: str, type_label: str) -> Dict[str, Any]:
        """Calls the Qwen VLM (which is already configured to use the remote API in qwen_extractor.py)"""
        content = call_vlm_with_retry(image_path, prompt)
        return {
            "type": type_label,
            "content": content,
            "extraction_method": "Surya_Layout_Qwen_Content",
            "metadata": {}
        }

    # --- MAIN ENTRY POINT ---
    def process_document(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Process entire PDF page by page.
        """
        doc = fitz.open(pdf_path)
        all_elements = []
        
        for page_num, page in enumerate(doc):
            try:
                # Rasterize for API sending
                pix = page.get_pixmap(dpi=config.processing.pdf_dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # 1. Detection (NOW REMOTE via API)
                layout_elements = self.detect_layout(img)
                
                if not layout_elements:
                    logger.warning(f"No layout detected on page {page_num+1}")
                    continue

                # 2. Clustering (Local)
                clustered = self.cluster_elements(layout_elements, img.size)
                
                # 3. Hierarchy (Local)
                ordered = self.build_hierarchy(clustered)
                
                # 4. Extraction (Mixed Remote/Local)
                page_content = self.extract_content(ordered, img, page, page_num + 1)
                
                all_elements.extend(page_content)
                logger.info(f"✅ Page {page_num+1} processed: {len(page_content)} elements")

            except Exception as e:
                logger.error(f"❌ Failed to process page {page_num+1}: {e}")
        
        return all_elements
    

if __name__ == "__main__":
    print("🚀 Starting Document Processing...")
    
    # 1. Define your input
    my_document_path = "data/raw/fomcprojtabl20200610.pdf" 
    processed_data = HybridSuryaExtractor().process_document(my_document_path, output_dir="data/ts_processed/")

    
    # 3. Output the result
    print(f"✅ Processing Complete. Found {len(processed_data)} chunks.")
    print("--- Preview ---")
    # print(processed_data[0])