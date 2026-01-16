# -*- coding: utf-8 -*-
"""
Refactored Document Layout Analyzer (Cluster-First Architecture)
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from surya.detection import DetectionPredictor

class RegionType(Enum):
    TEXT = "text"           # Standard prose
    TITLE = "title"         # Section headers/Page titles
    TABLE = "table"         # Tabular data
    FIGURE = "figure"       # General images
    CHART = "chart"         # Data visualizations (bars, lines, dots)
    HEADER = "header"       # Page artifacts
    FOOTER = "footer"       # Page artifacts
    CAPTION = "caption"     # Meta-text for visuals
    UNKNOWN = "unknown"

@dataclass
class LayoutElement:
    """A raw atomic element (e.g., a single text line or a dot in a chart)"""
    x1: float
    y1: float
    x2: float
    y2: float
    raw_type: str = "unknown"  # 'text_line' or 'graphic'
    
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return ((self.x1 + self.x2)/2, (self.y1 + self.y2)/2)
    @property
    def aspect_ratio(self): return self.width / self.height if self.height > 0 else 0

@dataclass
class LayoutCluster:
    """A semantic group of elements (e.g., a paragraph, a table, a chart)"""
    id: int
    elements: List[LayoutElement] = field(default_factory=list)
    region_type: RegionType = RegionType.UNKNOWN
    confidence: float = 0.0
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.elements: return (0,0,0,0)
        return (
            min(e.x1 for e in self.elements),
            min(e.y1 for e in self.elements),
            max(e.x2 for e in self.elements),
            max(e.y2 for e in self.elements)
        )
    
    @property
    def width(self): return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]
    
    def get_density(self) -> float:
        """Percentage of the cluster area covered by elements"""
        cluster_area = self.width * self.height
        if cluster_area == 0: return 0
        element_area = sum(e.area for e in self.elements)
        return element_area / cluster_area

    def analyze_grid_structure(self, alignment_threshold=10) -> float:
        """
        Returns a score (0-1) indicating how 'grid-like' the elements are.
        High score = Table.
        """
        if len(self.elements) < 4: return 0.0
        
        # Collect all x and y starts
        x_starts = np.array([e.x1 for e in self.elements])
        y_starts = np.array([e.y1 for e in self.elements])
        
        # Count alignments
        x_matches = 0
        for x in x_starts:
            # Count how many other elements share this X within threshold
            matches = np.sum(np.abs(x_starts - x) < alignment_threshold)
            if matches > 1: x_matches += 1
            
        y_matches = 0
        for y in y_starts:
            matches = np.sum(np.abs(y_starts - y) < alignment_threshold)
            if matches > 1: y_matches += 1
            
        # Normalize
        score = (x_matches + y_matches) / (2 * len(self.elements))
        return min(1.0, score)

class DocumentLayoutAnalyzer:
    def __init__(self):
        print("Loading Surya detection model...")
        self.predictor = DetectionPredictor()
        
    def analyze_pdf(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        results = []
        for i, page in enumerate(doc):
            print(f"Processing Page {i+1}...")
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            results.append(self.analyze_page(img, i))
        return results

    def analyze_page(self, image: Image.Image, page_num: int):
        # 1. Detect Raw Elements
        predictions = self.predictor([image])[0]
        elements = []
        
        # 2. Atomic Classification (The "Is this text?" check)
        # Text lines are typically wide (AR > 2) and short. 
        # Chart elements (dots) are square (AR ~ 1).
        avg_height = np.median([b.bbox[3] - b.bbox[1] for b in predictions.bboxes]) if predictions.bboxes else 10
        
        for bbox in predictions.bboxes:
            x1, y1, x2, y2 = bbox.bbox
            elem = LayoutElement(x1, y1, x2, y2)
            
            # Heuristic: Atomic Typing
            ar = elem.aspect_ratio
            if ar > 2.0 and elem.height < (avg_height * 2):
                elem.raw_type = "text_line"
            elif ar < 1.5 and elem.height < (avg_height * 3):
                elem.raw_type = "graphic" # Likely a dot or bar segment
            else:
                elem.raw_type = "block" # Could be a large merged block or image
            
            elements.append(elem)

        # 3. Structural Clustering
        clusters = self._cluster_elements(elements, image.width, image.height)
        
        # 4. Cluster Classification
        self._classify_clusters(clusters, image.width, image.height)
        
        return {
            "page": page_num,
            "clusters": clusters,
            "width": image.width,
            "height": image.height
        }

    def _cluster_elements(self, elements: List[LayoutElement], page_w, page_h) -> List[LayoutCluster]:
        """
        Groups elements that are spatially close.
        Uses a larger vertical threshold to link paragraphs and broken tables.
        """
        if not elements: return []
        
        # Distance Thresholds
        H_THRESH = 30  # Horizontal gap allowed
        V_THRESH = 25  # Vertical gap allowed (line spacing)
        
        clusters = []
        assigned = [False] * len(elements)
        
        for i, elem in enumerate(elements):
            if assigned[i]: continue
            
            # Start new cluster
            current_cluster = [elem]
            assigned[i] = True
            changed = True
            
            while changed:
                changed = False
                # Try to add unassigned elements to this cluster
                # Optimization: In production, use spatial index (R-tree)
                for j, candidate in enumerate(elements):
                    if assigned[j]: continue
                    
                    # Check connection to ANY element in current cluster
                    # (Slow O(N^2) but fine for single page)
                    is_close = False
                    for existing in current_cluster:
                        # Check Overlap or Proximity
                        h_dist = max(0, max(existing.x1, candidate.x1) - min(existing.x2, candidate.x2))
                        v_dist = max(0, max(existing.y1, candidate.y1) - min(existing.y2, candidate.y2))
                        
                        # Logic: If vertically aligned (like a column), allow larger vertical gap
                        vertically_aligned = min(existing.x2, candidate.x2) > max(existing.x1, candidate.x1)
                        
                        thresh_v = V_THRESH * 1.5 if vertically_aligned else V_THRESH
                        
                        if h_dist < H_THRESH and v_dist < thresh_v:
                            is_close = True
                            break
                    
                    if is_close:
                        current_cluster.append(candidate)
                        assigned[j] = True
                        changed = True
            
            clusters.append(LayoutCluster(id=len(clusters), elements=current_cluster))
            
        return clusters

    def _classify_clusters(self, clusters: List[LayoutCluster], page_w, page_h):
        """
        Determines the type of the ENTIRE cluster based on its constituents and shape.
        """
        for cluster in clusters:
            bbox = cluster.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Metrics
            num_elems = len(cluster.elements)
            num_text = sum(1 for e in cluster.elements if e.raw_type == "text_line")
            num_graphic = sum(1 for e in cluster.elements if e.raw_type == "graphic")
            text_ratio = num_text / num_elems if num_elems > 0 else 0
            grid_score = cluster.analyze_grid_structure()
            
            # --- RULES ENGINE ---
            
            # 1. Header / Footer (Strict Position)
            if cy < (page_h * 0.08):
                cluster.region_type = RegionType.HEADER
                continue
            if cy > (page_h * 0.92):
                cluster.region_type = RegionType.FOOTER
                continue
                
            # 2. Chart Detection (Key Fix for Page 3)
            # If cluster has many non-text graphic elements (dots, bars)
            if num_graphic > 5 and text_ratio < 0.5:
                cluster.region_type = RegionType.CHART
                continue
                
            # 3. Table Detection (Key Fix for Page 1)
            # High grid score OR moderate grid + 'Table' keyword in content (if we had OCR)
            # We assume Tables are mostly text but highly aligned
            if num_elems > 10 and grid_score > 0.6:
                cluster.region_type = RegionType.TABLE
                continue
            
            # 4. Title Detection
            # Single line, centered, top half of page, large font relative to others?
            # (Simplification: If it's a small cluster (1-2 lines) and centered near top)
            if num_elems <= 3 and cy < (page_h * 0.4):
                # Check centering
                dev_from_center = abs(cx - (page_w/2))
                if dev_from_center < (page_w * 0.1): # Within 10% of center
                    cluster.region_type = RegionType.TITLE
                    continue

            # 5. Text Block vs Figure (Key Fix for Page 4/5)
            # Previous code failed here by calling large text blocks "Figures"
            if text_ratio > 0.8:
                cluster.region_type = RegionType.TEXT
            else:
                # If it's large and NOT mostly text lines, it's a Figure
                area_ratio = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / (page_w * page_h)
                if area_ratio > 0.1:
                    cluster.region_type = RegionType.FIGURE
                else:
                    cluster.region_type = RegionType.TEXT # Default fallback

    def visualize(self, result: Dict, output_path: str):
        img = result['clusters'][0].elements[0] # dummy
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        
        # Load original image again for plotting (inefficient but safe)
        # In real code pass the image object
        
        color_map = {
            RegionType.TEXT: 'blue',
            RegionType.TABLE: 'purple',
            RegionType.CHART: 'green',
            RegionType.FIGURE: 'red',
            RegionType.TITLE: 'orange',
            RegionType.HEADER: 'grey',
            RegionType.FOOTER: 'grey'
        }
        
        # Draw clusters
        for cluster in result['clusters']:
            x1, y1, x2, y2 = cluster.bbox
            c_type = cluster.region_type
            color = color_map.get(c_type, 'black')
            
            # Draw Box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw Label
            ax.text(x1, y1-5, c_type.value.upper(), color=color, fontsize=8, weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_title(f"Page {result['page']+1} Analysis")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# Example Usage
if __name__ == "__main__":
    analyzer = DocumentLayoutAnalyzer()
    pdf_path = "data/raw/fomcprojtabl20200610.pdf"
    results = analyzer.analyze_pdf(pdf_path)
    
    # Print and visualize each page
    for i,analysis in enumerate(results):
        # analyzer.print_analysis(analysis)
        analyzer.visualize(
            analysis, 
            output_path=f"page_{i + 1 }_classified.png",
        )
# # Alias for backward compatibility
# DocumentLayoutAnalyzer = ImprovedAnalyzer

# if __name__ == "__main__":
#     # Initialize analyzer
#     analyzer = DocumentLayoutAnalyzer()
    
#     # Example: Analyze a PDF
#     pdf_path = "data/raw/fomcprojtabl20200610.pdf"
    
#     if os.path.exists(pdf_path):
#         # Analyze all pages
#         results = analyzer.analyze_pdf(pdf_path)
        
#         # Print and visualize each page
#         for analysis in results:
#             analyzer.print_analysis(analysis)
#             analyzer.visualize(
#                 analysis, 
#                 pdf_path=pdf_path,
#                 output_path=f"page_{analysis.page_num + 1}_classified.png",
#                 show=False
#             )
#     else:
#         print(f"Example PDF not found: {pdf_path}")
#         print("Please provide a valid PDF path to test the analyzer.")
