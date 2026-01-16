# -*- coding: utf-8 -*-
"""
Improved Document Layout Analyzer
Enhanced classification logic based on real-world Federal Reserve document analysis
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

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
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
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
    cluster_id: Optional[int] = None  # For grouping related bboxes
    
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
        """Vertical gap between bboxes"""
        if self.y2 < other.y1:
            return other.y1 - self.y2
        elif other.y2 < self.y1:
            return self.y1 - other.y2
        return 0
    
    def horizontal_gap_to(self, other: 'BBoxInfo') -> float:
        """Horizontal gap between bboxes"""
        if self.x2 < other.x1:
            return other.x1 - self.x2
        elif other.x2 < self.x1:
            return self.x1 - other.x2
        return 0
    
    def vertical_overlap(self, other: 'BBoxInfo') -> float:
        """Calculate vertical overlap ratio (0-1)"""
        overlap_start = max(self.y1, other.y1)
        overlap_end = min(self.y2, other.y2)
        overlap = max(0, overlap_end - overlap_start)
        min_height = min(self.height, other.height)
        return overlap / min_height if min_height > 0 else 0
    
    def horizontal_overlap(self, other: 'BBoxInfo') -> float:
        """Calculate horizontal overlap ratio (0-1)"""
        overlap_start = max(self.x1, other.x1)
        overlap_end = min(self.x2, other.x2)
        overlap = max(0, overlap_end - overlap_start)
        min_width = min(self.width, other.width)
        return overlap / min_width if min_width > 0 else 0
    
    def is_horizontally_aligned(self, other: 'BBoxInfo', tolerance: float = 20.0) -> bool:
        """Check if bboxes are horizontally aligned (same y-position)"""
        return abs(self.y1 - other.y1) < tolerance or abs(self.y2 - other.y2) < tolerance
    
    def is_vertically_aligned(self, other: 'BBoxInfo', tolerance: float = 20.0) -> bool:
        """Check if bboxes are vertically aligned (same x-position)"""
        return abs(self.x1 - other.x1) < tolerance or abs(self.x2 - other.x2) < tolerance


@dataclass
class ClassificationRules:
    """Enhanced classification rules based on Federal Reserve document analysis"""
    
    # Text detection - more conservative
    text_min_aspect_ratio: float = 3.0  # Increased from 2.0 - text lines are wider
    text_max_height_ratio: float = 0.04  # Decreased from 0.05 - text is usually small
    text_min_width_ratio: float = 0.15  # New: text should span reasonable width
    text_vertical_gap_threshold: float = 20.0  # Decreased for tighter grouping
    
    # Title detection - new
    title_min_font_size_ratio: float = 1.3  # Relative to average text height
    title_center_threshold: float = 0.3  # Center alignment tolerance (0-0.5)
    title_top_margin_ratio: float = 0.2  # Titles are usually in top 20%
    
    # Caption detection - new
    caption_max_height_ratio: float = 0.025  # Smaller than regular text
    caption_near_figure_distance: float = 50.0  # Distance to figure/chart
    
    # Figure detection - more strict
    figure_min_area_ratio: float = 0.08  # Decreased - figures can be smaller
    figure_max_density: float = 0.3  # New: figures have few internal bboxes
    figure_isolation_threshold: float = 50.0  # Must be isolated from other regions
    
    # Chart detection - completely new approach
    chart_min_area_ratio: float = 0.1  # Charts are substantial
    chart_bbox_density_min: int = 5  # Charts have multiple internal elements
    chart_bbox_density_max: int = 100  # But not too many (that's a table)
    chart_has_axes_pattern: bool = True  # Look for axis-like arrangements
    
    # Table detection - much more conservative
    table_min_bboxes: int = 9  # Increased from 4 - tables have many cells
    table_min_rows: int = 3  # New: must have multiple rows
    table_min_cols: int = 2  # New: must have multiple columns
    table_alignment_threshold: float = 15.0  # Stricter alignment
    table_max_area_ratio: float = 0.7  # New: tables don't cover entire page
    table_regularity_threshold: float = 0.6  # New: require grid regularity
    
    # Header/Footer detection
    header_y_threshold: float = 0.08  # Stricter - top 8% only
    footer_y_threshold: float = 0.92  # Stricter - bottom 8% only
    
    # Clustering parameters - new
    cluster_distance_threshold: float = 100.0  # Group nearby bboxes
    cluster_min_density: int = 3  # Minimum bboxes for a dense cluster


@dataclass
class BBoxCluster:
    """Represents a group of related bboxes"""
    bboxes: List[BBoxInfo]
    cluster_id: int
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Get overall bounding box of cluster"""
        if not self.bboxes:
            return (0, 0, 0, 0)
        x1 = min(b.x1 for b in self.bboxes)
        y1 = min(b.y1 for b in self.bboxes)
        x2 = max(b.x2 for b in self.bboxes)
        y2 = max(b.y2 for b in self.bboxes)
        return (x1, y1, x2, y2)
    
    @property
    def area(self) -> float:
        """Total area of cluster bounding box"""
        x1, y1, x2, y2 = self.bounding_box
        return (x2 - x1) * (y2 - y1)
    
    @property
    def density(self) -> float:
        """Number of bboxes per unit area"""
        area = self.area
        return len(self.bboxes) / area if area > 0 else 0
    
    @property
    def avg_bbox_size(self) -> float:
        """Average bbox area in cluster"""
        if not self.bboxes:
            return 0
        return sum(b.area for b in self.bboxes) / len(self.bboxes)
    
    def has_grid_pattern(self, tolerance: float = 20.0) -> Tuple[bool, int, int]:
        """
        Check if cluster has grid-like arrangement.
        Returns (has_grid, num_rows, num_cols)
        """
        if len(self.bboxes) < 4:
            return (False, 0, 0)
        
        # Group by y-coordinate (rows)
        y_groups = defaultdict(list)
        for bbox in self.bboxes:
            # Find existing group within tolerance
            matched = False
            for y_key in list(y_groups.keys()):
                if abs(bbox.y1 - y_key) < tolerance:
                    y_groups[y_key].append(bbox)
                    matched = True
                    break
            if not matched:
                y_groups[bbox.y1].append(bbox)
        
        # Group by x-coordinate (columns)
        x_groups = defaultdict(list)
        for bbox in self.bboxes:
            matched = False
            for x_key in list(x_groups.keys()):
                if abs(bbox.x1 - x_key) < tolerance:
                    x_groups[x_key].append(bbox)
                    matched = True
                    break
            if not matched:
                x_groups[bbox.x1].append(bbox)
        
        num_rows = len(y_groups)
        num_cols = len(x_groups)
        
        # Check if it's a reasonable grid
        expected_cells = num_rows * num_cols
        actual_cells = len(self.bboxes)
        
        # Grid should have most cells filled (allow some missing)
        regularity = actual_cells / expected_cells if expected_cells > 0 else 0
        has_grid = regularity > 0.5 and num_rows >= 2 and num_cols >= 2
        
        return (has_grid, num_rows, num_cols)


@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    page_num: int
    page_width: float
    page_height: float
    bboxes: List[BBoxInfo] = field(default_factory=list)
    clusters: List[BBoxCluster] = field(default_factory=list)
    
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


class ImprovedDocumentLayoutAnalyzer:
    """
    Improved document layout analyzer with enhanced classification logic.
    
    Key improvements:
    - Multi-pass classification with context awareness
    - Clustering to identify complex regions (charts, tables)
    - Better distinction between tables and charts
    - Title and caption detection
    - More conservative table detection
    """
    
    def __init__(self, rules: Optional[ClassificationRules] = None):
        """Initialize the analyzer with improved rules."""
        print("Loading Surya detection models...")
        self.predictor = DetectionPredictor()
        self.rules = rules or ClassificationRules()
        print("Models loaded successfully")
    
    def analyze_pdf(self, pdf_path: str) -> List[PageAnalysis]:
        """Analyze all pages in a PDF document."""
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
        """Analyze a single PDF page with improved classification."""
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
        
        # Multi-pass classification
        self._classify_regions_improved(analysis)
        
        return analysis
    
    def analyze_image(self, image: Image.Image) -> PageAnalysis:
        """Analyze a single image."""
        predictions = self.predictor([image])
        result = predictions[0]
        
        analysis = PageAnalysis(
            page_num=0,
            page_width=image.width,
            page_height=image.height
        )
        
        for bbox in result.bboxes:
            x1, y1, x2, y2 = bbox.bbox
            bbox_info = BBoxInfo(x1=x1, y1=y1, x2=x2, y2=y2)
            analysis.bboxes.append(bbox_info)
        
        self._classify_regions_improved(analysis)
        
        return analysis
    
    def _classify_regions_improved(self, analysis: PageAnalysis) -> None:
        """
        Improved multi-pass classification strategy.
        
        Classification order:
        1. Cluster bboxes into groups
        2. Identify headers/footers (positional)
        3. Identify charts (cluster analysis)
        4. Identify tables (grid pattern + cluster analysis)
        5. Identify figures (large isolated regions)
        6. Identify titles (large, centered text)
        7. Identify captions (small text near figures/charts)
        8. Identify regular text (everything else with text characteristics)
        """
        if not analysis.bboxes:
            return
        
        # Step 1: Cluster bboxes
        self._cluster_bboxes(analysis)
        
        # Step 2: Headers and footers (easy, positional)
        self._classify_headers_footers(analysis)
        
        # Step 3: Charts (before tables, to avoid misclassification)
        self._classify_charts(analysis)
        
        # Step 4: Tables (conservative, grid-based)
        self._classify_tables(analysis)
        
        # Step 5: Figures (isolated large regions)
        self._classify_figures(analysis)
        
        # Step 6: Titles (large centered text)
        self._classify_titles(analysis)
        
        # Step 7: Captions (small text near figures/charts)
        self._classify_captions(analysis)
        
        # Step 8: Regular text (remaining bboxes with text characteristics)
        self._classify_text(analysis)
    
    def _cluster_bboxes(self, analysis: PageAnalysis) -> None:
        """Cluster nearby bboxes using distance-based grouping."""
        if not analysis.bboxes:
            return
        
        # Simple clustering: group bboxes within threshold distance
        unassigned = set(range(len(analysis.bboxes)))
        cluster_id = 0
        
        while unassigned:
            # Start new cluster with first unassigned bbox
            seed_idx = min(unassigned)
            cluster_indices = {seed_idx}
            unassigned.remove(seed_idx)
            
            # Grow cluster by adding nearby bboxes
            changed = True
            while changed:
                changed = False
                for idx in list(unassigned):
                    bbox = analysis.bboxes[idx]
                    # Check if close to any bbox in current cluster
                    for cluster_idx in cluster_indices:
                        cluster_bbox = analysis.bboxes[cluster_idx]
                        if bbox.distance_to(cluster_bbox) < self.rules.cluster_distance_threshold:
                            cluster_indices.add(idx)
                            unassigned.remove(idx)
                            changed = True
                            break
            
            # Create cluster
            cluster_bboxes = [analysis.bboxes[i] for i in cluster_indices]
            for bbox in cluster_bboxes:
                bbox.cluster_id = cluster_id
            
            cluster = BBoxCluster(bboxes=cluster_bboxes, cluster_id=cluster_id)
            analysis.clusters.append(cluster)
            cluster_id += 1
    
    def _classify_headers_footers(self, analysis: PageAnalysis) -> None:
        """Classify headers and footers based on position."""
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            rel_y = bbox.y1 / analysis.page_height
            
            if rel_y < self.rules.header_y_threshold:
                bbox.region_type = RegionType.HEADER
                bbox.confidence = 0.9
            elif rel_y > self.rules.footer_y_threshold:
                bbox.region_type = RegionType.FOOTER
                bbox.confidence = 0.9
    
    def _classify_charts(self, analysis: PageAnalysis) -> None:
        """
        Classify charts based on cluster analysis.
        Charts have:
        - Moderate number of internal bboxes (not too few, not too many)
        - Substantial area
        - May have axis-like patterns
        """
        for cluster in analysis.clusters:
            # Skip if too few or too many bboxes
            if not (self.rules.chart_bbox_density_min <= len(cluster.bboxes) <= 
                    self.rules.chart_bbox_density_max):
                continue
            
            # Check area
            rel_area = cluster.area / analysis.page_area
            if rel_area < self.rules.chart_min_area_ratio:
                continue
            
            # Check for grid pattern (charts often have regular spacing)
            has_grid, rows, cols = cluster.has_grid_pattern(tolerance=30.0)
            
            # Charts might have some grid-like structure but not perfect
            # (unlike tables which are very regular)
            has_chart_pattern = (
                (has_grid and rows <= 15 and cols <= 15) or  # Some structure but not dense
                (not has_grid and len(cluster.bboxes) >= 8)  # Or irregular with enough elements
            )
            
            if has_chart_pattern:
                for bbox in cluster.bboxes:
                    if bbox.region_type == RegionType.UNKNOWN:
                        bbox.region_type = RegionType.CHART
                        bbox.confidence = 0.75
    
    def _classify_tables(self, analysis: PageAnalysis) -> None:
        """
        Conservative table classification based on strict grid patterns.
        Tables must have:
        - Clear grid structure with rows and columns
        - Multiple cells (minimum threshold)
        - Regular spacing
        - Not covering entire page
        """
        for cluster in analysis.clusters:
            # Must have enough bboxes
            if len(cluster.bboxes) < self.rules.table_min_bboxes:
                continue
            
            # Check area (tables shouldn't cover entire page)
            rel_area = cluster.area / analysis.page_area
            if rel_area > self.rules.table_max_area_ratio:
                continue
            
            # Must have clear grid pattern
            has_grid, rows, cols = cluster.has_grid_pattern(
                tolerance=self.rules.table_alignment_threshold
            )
            
            if not has_grid:
                continue
            
            # Must have minimum rows and columns
            if rows < self.rules.table_min_rows or cols < self.rules.table_min_cols:
                continue
            
            # Check regularity
            expected_cells = rows * cols
            actual_cells = len(cluster.bboxes)
            regularity = actual_cells / expected_cells
            
            if regularity < self.rules.table_regularity_threshold:
                continue
            
            # This is a table!
            for bbox in cluster.bboxes:
                if bbox.region_type == RegionType.UNKNOWN:
                    bbox.region_type = RegionType.TABLE
                    bbox.confidence = 0.85
    
    def _classify_figures(self, analysis: PageAnalysis) -> None:
        """
        Classify figures as large, isolated regions.
        Figures are typically:
        - Large area
        - Single bbox or very few bboxes
        - Isolated from other regions
        """
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            rel_area = bbox.area / analysis.page_area
            
            # Must be substantial size
            if rel_area < self.rules.figure_min_area_ratio:
                continue
            
            # Check isolation - count nearby bboxes
            nearby_count = sum(
                1 for other in analysis.bboxes
                if other != bbox and bbox.distance_to(other) < self.rules.figure_isolation_threshold
            )
            
            # Figures should be relatively isolated
            if nearby_count < 5:  # Very few neighbors
                bbox.region_type = RegionType.FIGURE
                bbox.confidence = 0.7
    
    def _classify_titles(self, analysis: PageAnalysis) -> None:
        """
        Classify titles as large, centered text near top of page.
        """
        # Calculate average text height for comparison
        text_heights = [
            bbox.height for bbox in analysis.bboxes
            if bbox.region_type == RegionType.UNKNOWN
        ]
        
        if not text_heights:
            return
        
        avg_height = sum(text_heights) / len(text_heights)
        
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            # Must be larger than average
            if bbox.height < avg_height * self.rules.title_min_font_size_ratio:
                continue
            
            # Must be near top
            rel_y = bbox.y1 / analysis.page_height
            if rel_y > self.rules.title_top_margin_ratio:
                continue
            
            # Check if centered (x-position near page center)
            center_x = bbox.center[0]
            page_center_x = analysis.page_width / 2
            rel_offset = abs(center_x - page_center_x) / analysis.page_width
            
            if rel_offset < self.rules.title_center_threshold:
                bbox.region_type = RegionType.TITLE
                bbox.confidence = 0.8
    
    def _classify_captions(self, analysis: PageAnalysis) -> None:
        """
        Classify captions as small text near figures or charts.
        """
        # Get figures and charts
        figures_and_charts = [
            bbox for bbox in analysis.bboxes
            if bbox.region_type in [RegionType.FIGURE, RegionType.CHART]
        ]
        
        if not figures_and_charts:
            return
        
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            # Must be small
            rel_height = bbox.height / analysis.page_height
            if rel_height > self.rules.caption_max_height_ratio:
                continue
            
            # Must be text-like (high aspect ratio)
            if bbox.aspect_ratio < 2.0:
                continue
            
            # Must be near a figure or chart
            for fig_chart in figures_and_charts:
                distance = bbox.distance_to(fig_chart)
                if distance < self.rules.caption_near_figure_distance:
                    bbox.region_type = RegionType.CAPTION
                    bbox.confidence = 0.7
                    break
    
    def _classify_text(self, analysis: PageAnalysis) -> None:
        """
        Classify remaining bboxes as text if they have text characteristics.
        """
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            # Text characteristics
            rel_height = bbox.height / analysis.page_height
            rel_width = bbox.width / analysis.page_width
            
            # Text lines: wide, short, reasonable size
            is_text = (
                bbox.aspect_ratio > self.rules.text_min_aspect_ratio and
                rel_height < self.rules.text_max_height_ratio and
                rel_width > self.rules.text_min_width_ratio
            )
            
            if is_text:
                bbox.region_type = RegionType.TEXT
                bbox.confidence = 0.65
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 144) -> Image.Image:
        """Convert PDF page to PIL Image."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
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
        show: bool = True,
        show_clusters: bool = False
    ) -> None:
        """Visualize detected and classified regions."""
        if image is None:
            if pdf_path is None:
                raise ValueError("Either image or pdf_path must be provided")
            image = self._pdf_page_to_image(pdf_path, analysis.page_num)
        
        color_map = {
            RegionType.TEXT: 'blue',
            RegionType.FIGURE: 'red',
            RegionType.CHART: 'green',
            RegionType.TABLE: 'purple',
            RegionType.TITLE: 'darkblue',
            RegionType.CAPTION: 'cyan',
            RegionType.HEADER: 'orange',
            RegionType.FOOTER: 'orange',
            RegionType.UNKNOWN: 'gray'
        }
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 20))
        ax.imshow(image)
        
        # Draw cluster boundaries if requested
        if show_clusters:
            for cluster in analysis.clusters:
                x1, y1, x2, y2 = cluster.bounding_box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1,
                    edgecolor='yellow',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.3
                )
                ax.add_patch(rect)
        
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
            label = f"{bbox.region_type.value}"
            if bbox.confidence > 0:
                label += f" ({bbox.confidence:.2f})"
            
            ax.text(
                bbox.x1, bbox.y1 - 5,
                label,
                fontsize=7,
                color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=1)
            )
        
        # Title with summary
        summary = analysis.summary()
        title = f"Page {analysis.page_num + 1} - "
        title += ", ".join([f"{k}: {v}" for k, v in summary.items() if v > 0])
        ax.set_title(title, fontsize=14, pad=20)
        ax.axis('off')
        
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_analysis(self, analysis: PageAnalysis, verbose: bool = False) -> None:
        """Print detailed analysis results."""
        print(f"\n{'='*70}")
        print(f"Page {analysis.page_num + 1} Analysis")
        print(f"{'='*70}")
        print(f"Page dimensions: {analysis.page_width:.0f} x {analysis.page_height:.0f} px")
        print(f"Total regions detected: {len(analysis.bboxes)}")
        print(f"Number of clusters: {len(analysis.clusters)}")
        
        print(f"\nRegion Summary:")
        summary = analysis.summary()
        for region_type, count in summary.items():
            if count > 0:
                print(f"  {region_type:12s}: {count}")
        
        if verbose:
            print(f"\nCluster Analysis:")
            for cluster in analysis.clusters:
                has_grid, rows, cols = cluster.has_grid_pattern()
                print(f"  Cluster {cluster.cluster_id}:")
                print(f"    Bboxes: {len(cluster.bboxes)}")
                print(f"    Area: {cluster.area:.0f} px²")
                print(f"    Density: {cluster.density:.6f}")
                print(f"    Grid pattern: {has_grid} ({rows}x{cols})")
            
            print(f"\nDetailed Regions:")
            for i, bbox in enumerate(analysis.bboxes, 1):
                print(f"\n  Region {i}: {bbox.region_type.value.upper()}")
                print(f"    Position: [{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]")
                print(f"    Size: {bbox.width:.1f} x {bbox.height:.1f} px")
                print(f"    Aspect ratio: {bbox.aspect_ratio:.2f}")
                print(f"    Confidence: {bbox.confidence:.2f}")
                print(f"    Cluster: {bbox.cluster_id}")


# Backward compatibility - keep old class name as alias
DocumentLayoutAnalyzer = ImprovedDocumentLayoutAnalyzer


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
