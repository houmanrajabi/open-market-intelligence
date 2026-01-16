import fitz  # PyMuPDF
import os
import uuid
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Generator

# Surya imports
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor

from src.utils.models import DocumentChunk, ChunkType
from src.utils.config import config

logger = logging.getLogger(__name__)

class SectionTracker:
    """Tracks hierarchical section context (H1 -> H2 -> H3)."""
    def __init__(self):
        self.hierarchy = {"Title": None, "Section-header": None, "Subsection-header": None}
        self.current_header = "Document Start"

    def update(self, text: str, label: str):
        """Update hierarchy based on label specificity."""
        clean_text = text.strip()
        if not clean_text:
            return

        if label == "Title":
            self.hierarchy = {"Title": clean_text, "Section-header": None, "Subsection-header": None}
            self.current_header = clean_text
        elif label == "Section-header":
            self.hierarchy["Section-header"] = clean_text
            self.hierarchy["Subsection-header"] = None
            # Combine Title > Section for rich context
            self.current_header = f"{self.hierarchy.get('Title') or ''} > {clean_text}".strip(' >')
        
    def get_context(self) -> str:
        return self.current_header

class SuryaExtractor:
    """
    Refined High-precision extractor using Surya.
    Features:
    - Memory-efficient streaming
    - Smart line-to-layout assignment (handling nesting)
    - Asset cropping (Tables/Figures) for downstream VLM
    - Hierarchical context tracking
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.data.processed_data_dir / "assets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading Surya models...")
        self.foundation_predictor = FoundationPredictor()
        self.layout_predictor = LayoutPredictor(self.foundation_predictor)
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
        logger.info("✅ Surya models loaded.")

    def _pdf_page_generator(self, pdf_path: str, dpi: int = 144) -> Generator[Tuple[int, Image.Image], None, None]:
        """Yields (page_num, image) to save memory."""
        try:
            doc = fitz.open(pdf_path)
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            
            logger.info(f"Processing {len(doc)} pages from {Path(pdf_path).name}")
            
            for page_num, page in enumerate(doc):
                try:
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    yield page_num, img
                except Exception as e:
                    logger.error(f"Failed to rasterize page {page_num}: {e}")
                    continue
            doc.close()
        except Exception as e:
            logger.critical(f"Could not open PDF {pdf_path}: {e}")

    def _calculate_overlap(self, inner_box: List[float], outer_box: List[float]) -> float:
        """Calculate what percentage of inner_box is inside outer_box."""
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        
        x_left = max(ix1, ox1)
        y_top = max(iy1, oy1)
        x_right = min(ix2, ox2)
        y_bottom = min(iy2, oy2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        inner_area = (ix2 - ix1) * (iy2 - iy1)
        
        return intersection_area / inner_area if inner_area > 0 else 0.0

    def _assign_line_to_layout(self, line_bbox: List[float], layout_boxes: List[Any]) -> Optional[Any]:
        """
        Assign a text line to the best fitting layout box.
        Prioritizes the *smallest* box that contains the line (handling nesting).
        """
        candidates = []
        for box in layout_boxes:
            overlap = self._calculate_overlap(line_bbox, box.bbox)
            if overlap > 0.5:  # Threshold: Line is >50% inside the box
                # Calculate area to prioritize smaller, nested boxes (e.g., caption inside figure)
                width = box.bbox[2] - box.bbox[0]
                height = box.bbox[3] - box.bbox[1]
                area = width * height
                candidates.append((box, area))
        
        if not candidates:
            return None
            
        # Sort by area ascending (smallest containing box wins)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _is_header_or_footer(self, bbox: List[float], img_height: int) -> bool:
        """Check if bbox is in top 5% or bottom 5% of page."""
        _, y1, _, y2 = bbox
        if y1 < (img_height * 0.05): return True  # Header
        if y2 > (img_height * 0.95): return True  # Footer
        return False

    def _find_caption(self, target_bbox: List[float], text_lines: List[Any], 
                     img_size: Tuple[int, int], search_radius: int = 100) -> Optional[str]:
        """Look for nearby text starting with 'Figure' or 'Table'."""
        tx1, ty1, tx2, ty2 = target_bbox
        w, h = img_size
        
        # Define search area (expanded box)
        search_box = [
            max(0, tx1 - 50), max(0, ty1 - search_radius),
            min(w, tx2 + 50), min(h, ty2 + search_radius)
        ]
        
        candidates = []
        for line in text_lines:
            # Check if line is within search radius
            if self._calculate_overlap(line.bbox, search_box) > 0.5:
                text = line.text.strip().lower()
                # Basic heuristic check
                if text.startswith(("figure", "fig", "table", "chart", "exhibit")):
                    candidates.append(line.text)
        
        return " | ".join(candidates) if candidates else None

    def extract_document(self, pdf_path: str) -> List[DocumentChunk]:
        """Main extraction pipeline."""
        all_chunks = []
        section_tracker = SectionTracker()
        pdf_stem = Path(pdf_path).stem

        for page_num, image in self._pdf_page_generator(pdf_path):
            try:
                logger.info(f"Analyzing Page {page_num + 1}...")
                img_w, img_h = image.size
                
                # 1. Inference
                layout_result = self.layout_predictor([image])[0]
                ocr_result = self.rec_predictor([image], [["en"]])[0]
                
                # 2. Filter & Sort Layouts (Reading Order: Top-Left -> Bottom-Right)
                # Filter out low-confidence boxes if necessary (Surya usually good)
                sorted_layouts = sorted(layout_result.bboxes, key=lambda x: (x.bbox[1], x.bbox[0]))
                
                # 3. Map Text Lines to Layouts
                layout_content: Dict[int, List[str]] = {i: [] for i in range(len(sorted_layouts))}
                processed_line_indices = set()
                
                for idx, line in enumerate(ocr_result.text_lines):
                    assigned_box = self._assign_line_to_layout(line.bbox, sorted_layouts)
                    
                    if assigned_box:
                        # Find index of this box in our sorted list
                        box_idx = sorted_layouts.index(assigned_box)
                        layout_content[box_idx].append(line.text)
                        processed_line_indices.add(idx)
                    else:
                        # Check for Header/Footer to discard
                        if self._is_header_or_footer(line.bbox, img_h):
                            processed_line_indices.add(idx)  # Mark processed, don't store
                
                # 4. Process Layout Regions into Chunks
                for i, layout_box in enumerate(sorted_layouts):
                    label = layout_box.label
                    bbox = layout_box.bbox
                    text_lines = layout_content[i]
                    full_text = "\n".join(text_lines).strip()
                    
                    # Generate Unique ID
                    chunk_id = f"{pdf_stem}_p{page_num+1}_{label}_{i}"
                    
                    # --- Logic per Type ---
                    
                    if label in ["Title", "Section-header"]:
                        section_tracker.update(full_text, label)
                        # We still create a chunk for the header text
                        all_chunks.append(DocumentChunk(
                            unique_id=chunk_id,
                            content=full_text,
                            chunk_type=ChunkType.TEXT,
                            metadata={"page": page_num + 1, "is_header": True}
                        ))

                    elif label == "Table":
                        # Crop Table Image
                        crop_path = self.output_dir / f"{chunk_id}.jpg"
                        image.crop(bbox).save(crop_path)
                        
                        # Find Caption
                        caption = self._find_caption(bbox, ocr_result.text_lines, (img_w, img_h))
                        
                        all_chunks.append(DocumentChunk(
                            unique_id=chunk_id,
                            content=full_text if full_text else "[TABLE DATA]",
                            chunk_type=ChunkType.TABLE,
                            metadata={
                                "page": page_num + 1,
                                "bbox": bbox,
                                "section_context": section_tracker.get_context(),
                                "image_path": str(crop_path),
                                "caption": caption,
                                "requires_vlm_processing": True  # Flag for downstream
                            }
                        ))

                    elif label in ["Figure", "Picture", "Image", "Chart"]:
                        # Crop Figure Image
                        crop_path = self.output_dir / f"{chunk_id}.jpg"
                        image.crop(bbox).save(crop_path)
                        
                        caption = self._find_caption(bbox, ocr_result.text_lines, (img_w, img_h))

                        all_chunks.append(DocumentChunk(
                            unique_id=chunk_id,
                            content=caption or f"[{label}]", # Content is caption for figures
                            chunk_type=ChunkType.FIGURE,
                            metadata={
                                "page": page_num + 1,
                                "bbox": bbox,
                                "section_context": section_tracker.get_context(),
                                "image_path": str(crop_path),
                                "caption": caption
                            }
                        ))

                    elif label == "Text":
                        if full_text:
                            all_chunks.append(DocumentChunk(
                                unique_id=chunk_id,
                                content=full_text,
                                chunk_type=ChunkType.TEXT,
                                metadata={
                                    "page": page_num + 1,
                                    "bbox": bbox,
                                    "section_context": section_tracker.get_context()
                                }
                            ))
                
                # 5. Handle Orphans (Text not in any layout box)
                orphan_text = []
                for idx, line in enumerate(ocr_result.text_lines):
                    if idx not in processed_line_indices:
                        # Double check it's not a header/footer
                        if not self._is_header_or_footer(line.bbox, img_h):
                            orphan_text.append(line.text)
                
                if orphan_text:
                    all_chunks.append(DocumentChunk(
                        unique_id=f"{pdf_stem}_p{page_num+1}_misc",
                        content="\n".join(orphan_text),
                        chunk_type=ChunkType.TEXT,
                        metadata={
                            "page": page_num + 1,
                            "note": "Orphan text",
                            "section_context": section_tracker.get_context()
                        }
                    ))

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}", exc_info=True)
                continue

        return all_chunks

if __name__ == "__main__":
    # Test execution
    test_pdf = config.data.raw_data_dir / "fomcprojtabl20200610.pdf"
    if test_pdf.exists():
        extractor = SuryaExtractor()
        chunks = extractor.extract_document(str(test_pdf))
        
        print(f"Extracted {len(chunks)} chunks.")
        for c in chunks:
            print(f"[{c.chunk_type.value}] {c.content[:40]}... (Context: {c.metadata.get('section_context')})")