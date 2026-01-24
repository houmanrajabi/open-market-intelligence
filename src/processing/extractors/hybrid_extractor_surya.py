import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

# Internal imports
from src.utils.config import config
# Import VLM caller directly to inject custom context-aware prompts
from src.processing.extractors.qwen_extractor import (
    call_vlm_with_retry, 
    TABLE_PROMPT, 
    CHART_PROMPT
)

# Surya Imports
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor  #

logger = logging.getLogger(__name__)

class HybridSuryaExtractor:
    """
    Tier 1 Hybrid Extractor using the 4-Layer Architecture:
    1. Detection (Surya)
    2. Geometric Clustering (Captions/Grouping)
    3. Hierarchy (Section/Header Tree)
    4. Content Enrichment (Qwen VLM + PyMuPDF)
    """

    def __init__(self):
        logger.info("🚀 Initializing HybridSuryaExtractor...")
        
        # Load Surya Models (Layer 1)
        # 1. Initialize Foundation Predictor (Shared Encoder)
        # This prevents reloading the heavy encoder for different tasks
        self.foundation_predictor = FoundationPredictor() #
        
        # 2. Initialize Layout Predictor with the foundation
        self.layout_predictor = LayoutPredictor(self.foundation_predictor) #
        
        logger.info("✅ Surya Layout model loaded.")

    # --- LAYER 1: DETECTION ---
    def detect_layout(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Use Surya to detect layout elements with high precision.
        Returns normalized bboxes (0-1000 scale).
        """
        # Surya expects a list of images
        layout_result = self.layout_predictor([image])[0]
        
        # Convert Surya bboxes to normalized format [ymin, xmin, ymax, xmax] 0-1000
        width, height = image.size
        normalized_elements = []

        for item in layout_result.bboxes:
            bbox = item.bbox
            label = item.label
            
            # Normalize to 0-1000
            norm_bbox = [
                int((bbox[1] / height) * 1000), # ymin
                int((bbox[0] / width) * 1000),  # xmin
                int((bbox[3] / height) * 1000), # ymax
                int((bbox[2] / width) * 1000)   # xmax
            ]

            normalized_elements.append({
                "type": self._map_surya_label(label),
                "bbox": norm_bbox,
                "confidence": getattr(item, 'confidence', 1.0),
                "original_bbox": bbox # Keep raw pixels for cropping if needed
            })
            
        return normalized_elements

    def _map_surya_label(self, label: str) -> str:
        """Map Surya labels to our internal schema."""
        label_map = {
            "Table": "TABLE",
            "Figure": "FIGURE",
            "Chart": "FIGURE",
            "Picture": "FIGURE",
            "Title": "HEADER",
            "Section-header": "HEADER",
            "Page-header": "HEADER",
            "Page-footer": "FOOTER",
            "Text": "TEXT",
            "List-item": "TEXT",
            "Caption": "CAPTION", # Temporary type for Layer 2 clustering
            "Formula": "TEXT"
        }
        return label_map.get(label, "TEXT")

    # --- LAYER 2: GEOMETRIC CLUSTERING ---
    def cluster_elements(self, elements: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Associate captions with tables/figures and group conceptually.
        Strategy: Look for 'CAPTION' or 'TEXT' blocks immediately above/below Figures/Tables.
        """
        w, h = image_size
        clustered = []
        used_indices = set()
        
        # Separate main visual assets from potential captions
        visuals = [(i, e) for i, e in enumerate(elements) if e["type"] in ["TABLE", "FIGURE"]]
        texts = [(i, e) for i, e in enumerate(elements) if e["type"] in ["TEXT", "CAPTION"]]

        for v_idx, visual in visuals:
            if v_idx in used_indices: continue
            
            # Search for nearest text/caption within threshold (e.g., 5% of page height)
            threshold_y = 50  # 50/1000 = 5% 
            best_caption = None
            best_dist = float('inf')
            best_c_idx = -1

            v_bbox = visual["bbox"] # [ymin, xmin, ymax, xmax]

            for t_idx, text_elem in texts:
                if t_idx in used_indices: continue
                
                t_bbox = text_elem["bbox"]
                
                # Check horizontal alignment (center points aligned)
                v_center_x = (v_bbox[1] + v_bbox[3]) / 2
                t_center_x = (t_bbox[1] + t_bbox[3]) / 2
                if abs(v_center_x - t_center_x) > 200: # Alignment tolerance
                    continue

                # Check vertical proximity (above or below)
                # Distance from Visual Bottom to Text Top (Caption Below)
                dist_below = t_bbox[0] - v_bbox[2]
                # Distance from Text Bottom to Visual Top (Caption Above)
                dist_above = v_bbox[0] - t_bbox[2]

                # We accept positive small distances
                dist = min(d for d in [dist_below, dist_above] if d >= -10) # -10 tolerance for slight overlap
                
                if 0 <= dist < threshold_y:
                    if dist < best_dist:
                        best_dist = dist
                        best_caption = text_elem
                        best_c_idx = t_idx

            # If caption found, merge it into the visual element's metadata
            if best_caption:
                visual["detected_caption_bbox"] = best_caption["bbox"]
                # We will extract the actual text later in Layer 4
                used_indices.add(best_c_idx)
            
            clustered.append(visual)
            used_indices.add(v_idx)

        # Add remaining unused text elements
        for i, e in enumerate(elements):
            if i not in used_indices:
                # Convert temp CAPTION type back to TEXT if it wasn't attached
                if e["type"] == "CAPTION":
                    e["type"] = "TEXT"
                clustered.append(e)

        return clustered

    # --- LAYER 3: HIERARCHICAL STRUCTURE ---
    def build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign a 'section_context' to every element based on the preceding Header.
        Sorts elements by reading order.
        """
        # 1. Sort by vertical position (Y-min)
        # TODO: Add multi-column sorting logic here if needed (Left-Right then Top-Down)
        sorted_elements = sorted(elements, key=lambda x: (x["bbox"][0], x["bbox"][1]))
        
        current_header = "Document Start"

        for elem in sorted_elements:
            if elem["type"] == "HEADER":
                # We will extract text later, but for now we mark this element as a header
                # In Layer 4 we will update 'current_header' with the actual text
                elem["is_header"] = True
                elem["section_context"] = current_header # Headers belong to previous context until defined
            else:
                elem["section_context"] = current_header # Placeholder, will be filled with text ID later
                
        return sorted_elements

    # --- LAYER 4: CONTENT EXTRACTION & ENRICHMENT ---
    def extract_content(
        self, 
        elements: List[Dict[str, Any]], 
        page_image: Image.Image, 
        pdf_page: fitz.Page,
        page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Execute the extraction strategy:
        - TEXT: Use PyMuPDF (fast, accurate for text)
        - TABLE/FIGURE: Crop -> Qwen VLM (smart)
        """
        final_elements = []
        w, h = page_image.size
        pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height
        
        current_section_text = "Document Start"

        for idx, elem in enumerate(elements):
            bbox_norm = elem["bbox"] # 0-1000
            
            # Convert 1000-scale to PDF points for text extraction
            rect_pdf = fitz.Rect(
                bbox_norm[1] * pdf_w / 1000,
                bbox_norm[0] * pdf_h / 1000,
                bbox_norm[3] * pdf_w / 1000,
                bbox_norm[2] * pdf_h / 1000
            )

            # 1. TEXT / HEADER Extraction (PyMuPDF)
            if elem["type"] in ["TEXT", "HEADER", "FOOTER"]:
                text = pdf_page.get_text("text", clip=rect_pdf).strip()
                
                # Update Context State if this is a header
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

            # 2. VISUAL Extraction (VLM)
            elif elem["type"] in ["TABLE", "FIGURE"]:
                # Caption Extraction first (if linked in Layer 2)
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
                
                # Prepare Crop
                # Use original pixel bbox from Surya if available for better resolution
                # mapping back to PIL image coordinates
                crop_box = (
                    int(bbox_norm[1] * w / 1000),
                    int(bbox_norm[0] * h / 1000),
                    int(bbox_norm[3] * w / 1000),
                    int(bbox_norm[2] * h / 1000)
                )
                # Pad crop slightly
                crop_box = (
                    max(0, crop_box[0]-10), max(0, crop_box[1]-10),
                    min(w, crop_box[2]+10), min(h, crop_box[3]+10)
                )
                
                crop_img = page_image.crop(crop_box)
                
                # Save temp crop
                crop_path = f"/tmp/crop_{page_num}_{idx}.jpg"
                crop_img.save(crop_path)
                
                # Context-Aware Prompt Construction
                context_str = f"Context: This item appears in section '{current_section_text}'."
                if caption_text:
                    context_str += f" The caption is: '{caption_text}'."
                
                # VLM Call
                logger.info(f"🧠 Calling Qwen VLM for {elem['type']} (Context: {current_section_text[:20]}...)")
                
                try:
                    if elem["type"] == "TABLE":
                        # Inject context into table prompt
                        prompt = f"{context_str}\n{TABLE_PROMPT}"
                        vlm_result = self._extract_with_vlm(crop_path, prompt, "TABLE")
                    else:
                        # Inject context into chart prompt
                        prompt = f"{context_str}\n{CHART_PROMPT}"
                        vlm_result = self._extract_with_vlm(crop_path, prompt, "FIGURE")
                        
                    # Add caption to metadata
                    vlm_result["metadata"]["caption"] = caption_text
                    
                    final_elements.append({
                        "id": f"p{page_num}_{idx}_{elem['type']}",
                        "page": page_num,
                        "bbox_norm": bbox_norm,
                        "section_anchor": current_section_text,
                        **vlm_result
                    })
                    
                except Exception as e:
                    logger.error(f"VLM Failed for element {idx}: {e}")
                    final_elements.append({
                        "type": elem["type"],
                        "content": f"[Extraction Failed] {caption_text}",
                        "error": str(e)
                    })
                finally:
                    if os.path.exists(crop_path):
                        os.remove(crop_path)

        return final_elements

    def _extract_with_vlm(self, image_path: str, prompt: str, type_label: str) -> Dict[str, Any]:
        """Helper to call existing Qwen interface with raw prompt."""
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
        Process entire PDF document page by page.
        """
        doc = fitz.open(pdf_path)
        all_elements = []
        
        for page_num, page in enumerate(doc):
            try:
                # 0. Rasterize Page for Surya/VLM
                pix = page.get_pixmap(dpi=config.processing.pdf_dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Layer 1: Detection
                layout_elements = self.detect_layout(img)
                
                # Layer 2: Clustering
                clustered = self.cluster_elements(layout_elements, img.size)
                
                # Layer 3: Hierarchy
                ordered = self.build_hierarchy(clustered)
                
                # Layer 4: Extraction
                page_content = self.extract_content(ordered, img, page, page_num + 1)
                
                all_elements.extend(page_content)
                logger.info(f"✅ Page {page_num+1} processed: {len(page_content)} elements")

            except Exception as e:
                logger.error(f"❌ Failed to process page {page_num+1}: {e}")
        
        return all_elements

# Factory function for easy integration
def extract_hybrid_content_surya(pdf_path: str) -> List[Dict[str, Any]]:
    extractor = HybridSuryaExtractor()
    return extractor.process_document(pdf_path, "/tmp")