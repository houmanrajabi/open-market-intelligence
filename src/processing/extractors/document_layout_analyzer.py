# -*- coding: utf-8 -*-
"""
Document Layout Analyzer
Detects and classifies document regions (text, figures, charts, tables) using Surya OCR
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from surya.detection import DetectionPredictor


class RegionType(Enum):
    """Types of document regions"""
    TEXT = "text"
    FIGURE = "figure"
    CHART = "chart"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


@dataclass
class BBoxInfo:
    """Enhanced bbox with classification and metrics"""
    x1: float
    y1: float
    x2: float
    y2: float
    region_type: RegionType = RegionType.UNKNOWN
    confidence: float = 0.0
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def distance_to(self, other: 'BBoxInfo') -> float:
        """Euclidean distance between centers"""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx2 - cx1)**2 + (cy2 - cy1)**2)**0.5
    
    def vertical_gap_to(self, other: 'BBoxInfo') -> float:
        """Vertical gap between bboxes (negative if overlapping)"""
        if self.y2 < other.y1:
            return other.y1 - self.y2
        elif other.y2 < self.y1:
            return self.y1 - other.y2
        return 0  # Overlapping
    
    def horizontal_gap_to(self, other: 'BBoxInfo') -> float:
        """Horizontal gap between bboxes"""
        if self.x2 < other.x1:
            return other.x1 - self.x2
        elif other.x2 < self.x1:
            return self.x1 - other.x2
        return 0  # Overlapping


@dataclass
class ClassificationRules:
    """Configurable rules for region classification"""
    # Text detection
    text_min_aspect_ratio: float = 2.0  # Width/height ratio for text lines
    text_max_height_ratio: float = 0.05  # Max height relative to page height
    text_vertical_gap_threshold: float = 30.0  # Gap for text line grouping
    
    # Figure detection
    figure_min_area_ratio: float = 0.1  # Min area relative to page area
    figure_min_aspect_ratio: float = 0.5
    figure_max_aspect_ratio: float = 3.0
    
    # Chart detection
    chart_min_area_ratio: float = 0.08
    chart_regularity_threshold: float = 0.7  # Grid pattern regularity
    
    # Table detection
    table_min_bboxes: int = 4  # Min bboxes to form a table
    table_alignment_threshold: float = 20.0  # Pixel threshold for alignment
    
    # Header/Footer detection
    header_y_threshold: float = 0.1  # Top 10% of page
    footer_y_threshold: float = 0.9  # Bottom 10% of page


@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    page_num: int
    page_width: float
    page_height: float
    bboxes: List[BBoxInfo] = field(default_factory=list)
    
    @property
    def page_area(self) -> float:
        return self.page_width * self.page_height
    
    def get_by_type(self, region_type: RegionType) -> List[BBoxInfo]:
        """Filter bboxes by region type"""
        return [bbox for bbox in self.bboxes if bbox.region_type == region_type]
    
    def summary(self) -> Dict[str, int]:
        """Count regions by type"""
        summary = {rt.value: 0 for rt in RegionType}
        for bbox in self.bboxes:
            summary[bbox.region_type.value] += 1
        return summary


class DocumentLayoutAnalyzer:
    """
    Analyzes document layout by detecting and classifying regions.
    
    Features:
    - Detects bounding boxes using Surya OCR
    - Classifies regions: text, figures, charts, tables, headers, footers
    - Configurable classification rules
    - Supports PDF and image inputs
    """
    
    def __init__(self, rules: Optional[ClassificationRules] = None):
        """
        Initialize the analyzer.
        
        Args:
            rules: Custom classification rules (uses defaults if None)
        """
        print("Loading Surya detection models...")
        self.predictor = DetectionPredictor()
        self.rules = rules or ClassificationRules()
        print("Models loaded successfully")
    
    def analyze_pdf(self, pdf_path: str) -> List[PageAnalysis]:
        """
        Analyze all pages in a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PageAnalysis objects, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
        
        print(f"Analyzing {num_pages} pages from {pdf_path.name}")
        
        results = []
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1}/{num_pages}...")
            analysis = self.analyze_page(pdf_path, page_num)
            results.append(analysis)
        
        return results
    
    def analyze_page(self, pdf_path: str, page_num: int) -> PageAnalysis:
        """
        Analyze a single PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Zero-indexed page number
            
        Returns:
            PageAnalysis object
        """
        # Convert page to image
        img = self._pdf_page_to_image(pdf_path, page_num)
        
        # Detect bboxes
        predictions = self.predictor([img])
        result = predictions[0]
        
        # Create analysis object
        analysis = PageAnalysis(
            page_num=page_num,
            page_width=img.width,
            page_height=img.height
        )
        
        # Convert raw bboxes to BBoxInfo objects
        for bbox in result.bboxes:
            x1, y1, x2, y2 = bbox.bbox
            bbox_info = BBoxInfo(x1=x1, y1=y1, x2=x2, y2=y2)
            analysis.bboxes.append(bbox_info)
        
        # Classify all bboxes
        self._classify_regions(analysis)
        
        return analysis
    
    def analyze_image(self, image: Image.Image) -> PageAnalysis:
        """
        Analyze a single image.
        
        Args:
            image: PIL Image object
            
        Returns:
            PageAnalysis object
        """
        # Detect bboxes
        predictions = self.predictor([image])
        result = predictions[0]
        
        # Create analysis object
        analysis = PageAnalysis(
            page_num=0,
            page_width=image.width,
            page_height=image.height
        )
        
        # Convert raw bboxes to BBoxInfo objects
        for bbox in result.bboxes:
            x1, y1, x2, y2 = bbox.bbox
            bbox_info = BBoxInfo(x1=x1, y1=y1, x2=x2, y2=y2)
            analysis.bboxes.append(bbox_info)
        
        # Classify all bboxes
        self._classify_regions(analysis)
        
        return analysis
    
    def _classify_regions(self, analysis: PageAnalysis) -> None:
        """
        Classify all bboxes in the analysis using rule-based logic.
        
        Args:
            analysis: PageAnalysis object to classify (modified in-place)
        """
        if not analysis.bboxes:
            return
        
        # Sort bboxes by vertical position for sequential analysis
        sorted_bboxes = sorted(analysis.bboxes, key=lambda b: b.y1)
        
        # First pass: classify individual bbox characteristics
        for bbox in analysis.bboxes:
            self._classify_single_bbox(bbox, analysis)
        
        # Second pass: refine using spatial relationships
        self._refine_classification_by_clustering(analysis)
    
    def _classify_single_bbox(self, bbox: BBoxInfo, analysis: PageAnalysis) -> None:
        """
        Classify a single bbox based on its properties.
        
        Args:
            bbox: BBoxInfo object to classify
            analysis: PageAnalysis containing page context
        """
        # Calculate relative metrics
        rel_height = bbox.height / analysis.page_height
        rel_area = bbox.area / analysis.page_area
        rel_y = bbox.y1 / analysis.page_height
        
        # Check for header/footer first (positional)
        if rel_y < self.rules.header_y_threshold:
            bbox.region_type = RegionType.HEADER
            bbox.confidence = 0.8
            return
        elif rel_y > self.rules.footer_y_threshold:
            bbox.region_type = RegionType.FOOTER
            bbox.confidence = 0.8
            return
        
        # Text detection: high aspect ratio, small height
        if (bbox.aspect_ratio > self.rules.text_min_aspect_ratio and 
            rel_height < self.rules.text_max_height_ratio):
            bbox.region_type = RegionType.TEXT
            bbox.confidence = 0.7
            return
        
        # Figure detection: medium-large area, reasonable aspect ratio
        if (rel_area > self.rules.figure_min_area_ratio and
            self.rules.figure_min_aspect_ratio < bbox.aspect_ratio < self.rules.figure_max_aspect_ratio):
            bbox.region_type = RegionType.FIGURE
            bbox.confidence = 0.6
            return
        
        # Chart detection: medium area (charts are often smaller than figures)
        if rel_area > self.rules.chart_min_area_ratio:
            bbox.region_type = RegionType.CHART
            bbox.confidence = 0.5
            return
        
        # Default to unknown
        bbox.region_type = RegionType.UNKNOWN
        bbox.confidence = 0.3
    
    def _refine_classification_by_clustering(self, analysis: PageAnalysis) -> None:
        """
        Refine classifications by analyzing spatial clustering patterns.
        
        Args:
            analysis: PageAnalysis to refine (modified in-place)
        """
        # Group text bboxes that are close vertically (likely paragraphs)
        text_bboxes = [b for b in analysis.bboxes if b.region_type == RegionType.TEXT]
        
        for i, bbox in enumerate(text_bboxes):
            # Check if this bbox is part of a text block
            nearby_text = [
                other for j, other in enumerate(text_bboxes)
                if i != j and bbox.vertical_gap_to(other) < self.rules.text_vertical_gap_threshold
            ]
            
            if nearby_text:
                bbox.confidence = min(0.9, bbox.confidence + 0.2)
        
        # Detect table patterns: aligned bboxes in grid
        self._detect_tables(analysis)
    
    def _detect_tables(self, analysis: PageAnalysis) -> None:
        """
        Detect table structures based on bbox alignment.
        
        Args:
            analysis: PageAnalysis to update (modified in-place)
        """
        # Look for groups of bboxes with similar x-coordinates (columns)
        potential_table_bboxes = [
            b for b in analysis.bboxes 
            if b.region_type in [RegionType.UNKNOWN, RegionType.TEXT]
        ]
        
        if len(potential_table_bboxes) < self.rules.table_min_bboxes:
            return
        
        # Check for vertical alignment (columns)
        x_positions = [b.x1 for b in potential_table_bboxes]
        x_positions.sort()
        
        # Find clusters of aligned x-positions
        aligned_groups = []
        current_group = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_group[-1] < self.rules.table_alignment_threshold:
                current_group.append(x)
            else:
                if len(current_group) >= 2:
                    aligned_groups.append(current_group)
                current_group = [x]
        
        if len(current_group) >= 2:
            aligned_groups.append(current_group)
        
        # If we found multiple aligned columns, mark as table
        if len(aligned_groups) >= 2:
            for bbox in potential_table_bboxes:
                for group in aligned_groups:
                    if any(abs(bbox.x1 - x) < self.rules.table_alignment_threshold for x in group):
                        bbox.region_type = RegionType.TABLE
                        bbox.confidence = 0.75
                        break
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 144) -> Image.Image:
        """
        Convert PDF page to PIL Image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Zero-indexed page number
            dpi: Resolution (default 144)
            
        Returns:
            PIL Image object
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Calculate zoom factor for desired DPI (72 is default)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        return img
    
    def visualize(
        self, 
        analysis: PageAnalysis, 
        image: Optional[Image.Image] = None,
        pdf_path: Optional[str] = None,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize detected and classified regions.
        
        Args:
            analysis: PageAnalysis object to visualize
            image: PIL Image (if None, will load from pdf_path)
            pdf_path: Path to PDF (used if image is None)
            output_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        # Get image if not provided
        if image is None:
            if pdf_path is None:
                raise ValueError("Either image or pdf_path must be provided")
            image = self._pdf_page_to_image(pdf_path, analysis.page_num)
        
        # Color map for region types
        color_map = {
            RegionType.TEXT: 'blue',
            RegionType.FIGURE: 'red',
            RegionType.CHART: 'green',
            RegionType.TABLE: 'purple',
            RegionType.HEADER: 'orange',
            RegionType.FOOTER: 'orange',
            RegionType.UNKNOWN: 'gray'
        }
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(15, 20))
        ax.imshow(image)
        
        # Draw bboxes
        for bbox in analysis.bboxes:
            color = color_map[bbox.region_type]
            rect = patches.Rectangle(
                (bbox.x1, bbox.y1), 
                bbox.width, 
                bbox.height,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{bbox.region_type.value} ({bbox.confidence:.2f})"
            ax.text(
                bbox.x1, bbox.y1 - 5,
                label,
                fontsize=8,
                color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color)
            )
        
        # Title with summary
        summary = analysis.summary()
        title = f"Page {analysis.page_num + 1} - "
        title += ", ".join([f"{k}: {v}" for k, v in summary.items() if v > 0])
        ax.set_title(title, fontsize=14, pad=20)
        ax.axis('off')
        
        # Save if requested
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_analysis(self, analysis: PageAnalysis) -> None:
        """
        Print detailed analysis results.
        
        Args:
            analysis: PageAnalysis object to print
        """
        print(f"\n{'='*70}")
        print(f"Page {analysis.page_num + 1} Analysis")
        print(f"{'='*70}")
        print(f"Page dimensions: {analysis.page_width:.0f} x {analysis.page_height:.0f} px")
        print(f"Total regions detected: {len(analysis.bboxes)}")
        print(f"\nRegion Summary:")
        
        summary = analysis.summary()
        for region_type, count in summary.items():
            if count > 0:
                print(f"  {region_type:12s}: {count}")
        
        print(f"\nDetailed Regions:")
        for i, bbox in enumerate(analysis.bboxes, 1):
            print(f"\n  Region {i}: {bbox.region_type.value.upper()}")
            print(f"    Position: [{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]")
            print(f"    Size: {bbox.width:.1f} x {bbox.height:.1f} px")
            print(f"    Area: {bbox.area:.0f} px² ({bbox.area/analysis.page_area*100:.1f}% of page)")
            print(f"    Aspect ratio: {bbox.aspect_ratio:.2f}")
            print(f"    Confidence: {bbox.confidence:.2f}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DocumentLayoutAnalyzer()
    
    # Example: Analyze a PDF
    pdf_path = "data/raw/fomcprojtabl20200610.pdf"
    
    if os.path.exists(pdf_path):
        # Analyze all pages
        results = analyzer.analyze_pdf(pdf_path)
        
        # Print and visualize each page
        for analysis in results:
            analyzer.print_analysis(analysis)
            analyzer.visualize(
                analysis, 
                pdf_path=pdf_path,
                output_path=f"page_{analysis.page_num + 1}_classified.png",
                show=False
            )
    else:
        print(f"Example PDF not found: {pdf_path}")
        print("Please provide a valid PDF path to test the analyzer.")
