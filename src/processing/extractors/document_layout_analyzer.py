# -*- coding: utf-8 -*-
"""
Final Improved Document Layout Analyzer
Based on critical analysis of Federal Reserve document failures

Key Improvements:
1. Line Merging: Merge fragmented text lines into coherent blocks
2. Alignment Scoring: Distinguish tables (high alignment) from charts (low alignment)
3. No Density Cap: Charts can have any number of elements
4. Proper Classification Order
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

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
    cluster_id: Optional[int] = None
    is_merged: bool = False  # NEW: Track if this bbox was created by merging
    merged_from: List[int] = field(default_factory=list)  # NEW: Original bbox indices
    
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
    
    def horizontal_overlap(self, other: 'BBoxInfo') -> float:
        """Calculate horizontal overlap ratio (0-1)"""
        overlap_start = max(self.x1, other.x1)
        overlap_end = min(self.x2, other.x2)
        overlap = max(0, overlap_end - overlap_start)
        min_width = min(self.width, other.width)
        return overlap / min_width if min_width > 0 else 0
    
    def vertical_overlap(self, other: 'BBoxInfo') -> float:
        """Calculate vertical overlap ratio (0-1)"""
        overlap_start = max(self.y1, other.y1)
        overlap_end = min(self.y2, other.y2)
        overlap = max(0, overlap_end - overlap_start)
        min_height = min(self.height, other.height)
        return overlap / min_height if min_height > 0 else 0


@dataclass
class ClassificationRules:
    """Improved classification rules based on analysis"""
    
    # Line merging parameters - NEW!
    merge_vertical_gap_multiplier: float = 1.5  # Max gap = line_height * multiplier
    merge_horizontal_overlap_threshold: float = 0.3  # Min overlap to merge
    merge_height_similarity_threshold: float = 0.5  # Max height difference ratio
    
    # Alignment scoring - NEW!
    alignment_tolerance: float = 5.0  # Pixels tolerance for alignment
    alignment_min_matches: int = 2  # Min matches to count as aligned
    
    # Table detection
    table_alignment_score_threshold: float = 0.7  # 70% of boxes must be aligned
    table_min_bboxes: int = 9
    table_min_rows: int = 3
    table_min_cols: int = 2
    table_max_area_ratio: float = 0.8
    
    # Chart detection - REMOVED density max!
    chart_min_area_ratio: float = 0.08
    chart_bbox_density_min: int = 5
    chart_alignment_score_max: float = 0.4  # Charts have LOW alignment
    
    # Text detection
    text_min_aspect_ratio: float = 2.5
    text_max_height_ratio: float = 0.05
    text_min_width_ratio: float = 0.15
    
    # Title detection
    title_min_font_size_ratio: float = 1.3
    title_center_threshold: float = 0.3
    title_top_margin_ratio: float = 0.2
    
    # Caption detection
    caption_max_height_ratio: float = 0.025
    caption_near_figure_distance: float = 60.0
    
    # Figure detection
    figure_min_area_ratio: float = 0.08
    figure_max_internal_bboxes: int = 3  # Figures are mostly isolated
    
    # Header/Footer detection
    header_y_threshold: float = 0.08
    footer_y_threshold: float = 0.92
    
    # Clustering
    cluster_distance_threshold: float = 100.0


@dataclass
class BBoxCluster:
    """Represents a group of related bboxes with alignment analysis"""
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
    
    def calculate_alignment_score(self, tolerance: float = 5.0, min_matches: int = 2) -> Tuple[float, float, float]:
        """
        Calculate alignment score for the cluster.
        
        Returns:
            (x_alignment_score, y_alignment_score, combined_score)
            Each score is between 0.0 (chaotic) and 1.0 (perfectly aligned)
        """
        if len(self.bboxes) < 3:
            return (0.0, 0.0, 0.0)
        
        # X-alignment (vertical alignment - same left/right edges)
        x_aligned_count = 0
        for bbox in self.bboxes:
            # Count how many other boxes share similar x-coordinates
            left_matches = sum(
                1 for other in self.bboxes 
                if other != bbox and abs(bbox.x1 - other.x1) < tolerance
            )
            right_matches = sum(
                1 for other in self.bboxes
                if other != bbox and abs(bbox.x2 - other.x2) < tolerance
            )
            
            if left_matches >= min_matches or right_matches >= min_matches:
                x_aligned_count += 1
        
        x_alignment_score = x_aligned_count / len(self.bboxes)
        
        # Y-alignment (horizontal alignment - same top/bottom edges)
        y_aligned_count = 0
        for bbox in self.bboxes:
            top_matches = sum(
                1 for other in self.bboxes
                if other != bbox and abs(bbox.y1 - other.y1) < tolerance
            )
            bottom_matches = sum(
                1 for other in self.bboxes
                if other != bbox and abs(bbox.y2 - other.y2) < tolerance
            )
            
            if top_matches >= min_matches or bottom_matches >= min_matches:
                y_aligned_count += 1
        
        y_alignment_score = y_aligned_count / len(self.bboxes)
        
        # Combined score (both alignments matter for tables)
        combined_score = (x_alignment_score + y_alignment_score) / 2
        
        return (x_alignment_score, y_alignment_score, combined_score)
    
    def has_grid_structure(self, tolerance: float = 5.0) -> Tuple[bool, int, int]:
        """
        Check if cluster has grid structure using fuzzy alignment.
        
        Returns:
            (has_grid, num_rows, num_cols)
        """
        if len(self.bboxes) < 4:
            return (False, 0, 0)
        
        # Fuzzy grouping by y-coordinate (rows)
        y_groups = []
        for bbox in self.bboxes:
            # Find existing group
            matched = False
            for group in y_groups:
                if abs(bbox.y1 - group[0]) < tolerance:
                    group.append(bbox.y1)
                    matched = True
                    break
            if not matched:
                y_groups.append([bbox.y1])
        
        # Fuzzy grouping by x-coordinate (columns)
        x_groups = []
        for bbox in self.bboxes:
            matched = False
            for group in x_groups:
                if abs(bbox.x1 - group[0]) < tolerance:
                    group.append(bbox.x1)
                    matched = True
                    break
            if not matched:
                x_groups.append([bbox.x1])
        
        num_rows = len(y_groups)
        num_cols = len(x_groups)
        
        # Check if it's a reasonable grid
        expected_cells = num_rows * num_cols
        actual_cells = len(self.bboxes)
        regularity = actual_cells / expected_cells if expected_cells > 0 else 0
        
        # More lenient - 40% filled is enough for a grid
        has_grid = regularity > 0.4 and num_rows >= 2 and num_cols >= 2
        
        return (has_grid, num_rows, num_cols)


@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    page_num: int
    page_width: float
    page_height: float
    bboxes: List[BBoxInfo] = field(default_factory=list)
    original_bboxes: List[BBoxInfo] = field(default_factory=list)  # NEW: Before merging
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


class FinalImprovedAnalyzer:
    """
    Final improved document layout analyzer addressing all critical issues.
    
    Key Features:
    1. Line Merging: Merges fragmented text lines into blocks
    2. Alignment Scoring: Distinguishes tables (aligned) from charts (chaotic)
    3. No Density Caps: Charts can have unlimited elements
    4. Proper multi-pass classification
    """
    
    def __init__(self, rules: Optional[ClassificationRules] = None):
        """Initialize the analyzer."""
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
        """Analyze a single PDF page."""
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
            analysis.original_bboxes.append(bbox_info)
        
        # CRITICAL: Merge lines FIRST (before any classification)
        print(f"  Merging fragmented lines...")
        self._merge_text_lines(analysis)
        print(f"    {len(analysis.original_bboxes)} bboxes → {len(analysis.bboxes)} merged bboxes")
        
        # Multi-pass classification
        self._classify_regions_final(analysis)
        
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
            analysis.original_bboxes.append(bbox_info)
        
        self._merge_text_lines(analysis)
        self._classify_regions_final(analysis)
        
        return analysis
    
    def _merge_text_lines(self, analysis: PageAnalysis) -> None:
        """
        Merge fragmented text lines into coherent blocks.
        
        Algorithm:
        1. Sort bboxes by y-coordinate
        2. For each bbox, check if it should merge with previous
        3. Merge if: close vertically, overlap horizontally, similar height
        """
        if not analysis.original_bboxes:
            return
        
        # Sort by vertical position
        sorted_bboxes = sorted(analysis.original_bboxes, key=lambda b: (b.y1, b.x1))
        
        # Track which bboxes have been merged
        merged_indices = set()
        merged_bboxes = []
        
        i = 0
        while i < len(sorted_bboxes):
            if i in merged_indices:
                i += 1
                continue
            
            current = sorted_bboxes[i]
            current_group = [current]
            current_indices = [i]
            
            # Look ahead to find mergeable bboxes
            j = i + 1
            while j < len(sorted_bboxes):
                if j in merged_indices:
                    j += 1
                    continue
                
                candidate = sorted_bboxes[j]
                
                # Check if candidate should merge with current group
                # Use the last bbox in group for comparison
                last_in_group = current_group[-1]
                
                # 1. Check vertical gap
                v_gap = candidate.y1 - last_in_group.y2
                avg_height = (last_in_group.height + candidate.height) / 2
                max_gap = avg_height * self.rules.merge_vertical_gap_multiplier
                
                if v_gap > max_gap:
                    # Too far vertically, stop looking
                    break
                
                # 2. Check horizontal overlap
                h_overlap = last_in_group.horizontal_overlap(candidate)
                
                if h_overlap < self.rules.merge_horizontal_overlap_threshold:
                    # No horizontal overlap, check next
                    j += 1
                    continue
                
                # 3. Check height similarity
                height_ratio = min(last_in_group.height, candidate.height) / max(last_in_group.height, candidate.height)
                
                if height_ratio < self.rules.merge_height_similarity_threshold:
                    # Different font sizes, skip
                    j += 1
                    continue
                
                # All checks passed - merge!
                current_group.append(candidate)
                current_indices.append(j)
                merged_indices.add(j)
                
                j += 1
            
            # Create merged bbox
            if len(current_group) == 1:
                # No merging happened
                merged_bboxes.append(current)
            else:
                # Merge into single bbox
                x1 = min(b.x1 for b in current_group)
                y1 = min(b.y1 for b in current_group)
                x2 = max(b.x2 for b in current_group)
                y2 = max(b.y2 for b in current_group)
                
                merged = BBoxInfo(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    is_merged=True,
                    merged_from=current_indices
                )
                merged_bboxes.append(merged)
            
            merged_indices.add(i)
            i += 1
        
        # Update analysis with merged bboxes
        analysis.bboxes = merged_bboxes
    
    def _classify_regions_final(self, analysis: PageAnalysis) -> None:
        """
        Final multi-pass classification with improved logic.
        
        Order:
        1. Cluster bboxes
        2. Headers/Footers (positional)
        3. Charts (LOW alignment + high density)
        4. Tables (HIGH alignment + grid structure)
        5. Figures (large isolated regions)
        6. Titles (large centered text)
        7. Captions (small text near figures/charts)
        8. Text (merged lines with text characteristics)
        """
        if not analysis.bboxes:
            return
        
        # Step 1: Cluster
        self._cluster_bboxes(analysis)
        
        # Step 2: Headers/Footers
        self._classify_headers_footers(analysis)
        
        # Step 3: Charts (BEFORE tables!)
        self._classify_charts_final(analysis)
        
        # Step 4: Tables
        self._classify_tables_final(analysis)
        
        # Step 5: Figures
        self._classify_figures(analysis)
        
        # Step 6: Titles
        self._classify_titles(analysis)
        
        # Step 7: Captions
        self._classify_captions(analysis)
        
        # Step 8: Text
        self._classify_text(analysis)
    
    def _cluster_bboxes(self, analysis: PageAnalysis) -> None:
        """Cluster nearby bboxes using distance-based grouping."""
        if not analysis.bboxes:
            return
        
        unassigned = set(range(len(analysis.bboxes)))
        cluster_id = 0
        
        while unassigned:
            seed_idx = min(unassigned)
            cluster_indices = {seed_idx}
            unassigned.remove(seed_idx)
            
            # Grow cluster
            changed = True
            while changed:
                changed = False
                for idx in list(unassigned):
                    bbox = analysis.bboxes[idx]
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
    
    def _classify_charts_final(self, analysis: PageAnalysis) -> None:
        """
        Classify charts using alignment score.
        
        Charts have:
        - LOW alignment score (< 0.4) - elements are not grid-aligned
        - High density OR moderate density
        - Substantial area
        """
        for cluster in analysis.clusters:
            # Must have enough elements
            if len(cluster.bboxes) < self.rules.chart_bbox_density_min:
                continue
            
            # Check area
            rel_area = cluster.area / analysis.page_area
            if rel_area < self.rules.chart_min_area_ratio:
                continue
            
            # CRITICAL: Calculate alignment score
            x_align, y_align, combined_align = cluster.calculate_alignment_score(
                tolerance=self.rules.alignment_tolerance,
                min_matches=self.rules.alignment_min_matches
            )
            
            # Charts have LOW alignment (scatter plots, dot plots, etc.)
            if combined_align < self.rules.chart_alignment_score_max:
                # This is a chart!
                for bbox in cluster.bboxes:
                    if bbox.region_type == RegionType.UNKNOWN:
                        bbox.region_type = RegionType.CHART
                        bbox.confidence = 0.80
                
                print(f"    Detected CHART (cluster {cluster.cluster_id}): "
                      f"{len(cluster.bboxes)} elements, "
                      f"alignment={combined_align:.2f}")
    
    def _classify_tables_final(self, analysis: PageAnalysis) -> None:
        """
        Classify tables using alignment score and grid structure.
        
        Tables have:
        - HIGH alignment score (> 0.7) - elements are grid-aligned
        - Grid structure with rows and columns
        - Multiple elements
        """
        for cluster in analysis.clusters:
            # Must have enough bboxes
            if len(cluster.bboxes) < self.rules.table_min_bboxes:
                continue
            
            # Check area (tables shouldn't cover entire page)
            rel_area = cluster.area / analysis.page_area
            if rel_area > self.rules.table_max_area_ratio:
                continue
            
            # CRITICAL: Calculate alignment score
            x_align, y_align, combined_align = cluster.calculate_alignment_score(
                tolerance=self.rules.alignment_tolerance,
                min_matches=self.rules.alignment_min_matches
            )
            
            # Tables have HIGH alignment
            if combined_align < self.rules.table_alignment_score_threshold:
                continue
            
            # Check grid structure
            has_grid, rows, cols = cluster.has_grid_structure(
                tolerance=self.rules.alignment_tolerance
            )
            
            if not has_grid:
                continue
            
            # Must have minimum rows and columns
            if rows < self.rules.table_min_rows or cols < self.rules.table_min_cols:
                continue
            
            # This is a table!
            for bbox in cluster.bboxes:
                if bbox.region_type == RegionType.UNKNOWN:
                    bbox.region_type = RegionType.TABLE
                    bbox.confidence = 0.85
            
            print(f"    Detected TABLE (cluster {cluster.cluster_id}): "
                  f"{rows}x{cols} grid, "
                  f"alignment={combined_align:.2f}")
    
    def _classify_figures(self, analysis: PageAnalysis) -> None:
        """Classify figures as large, isolated regions."""
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            rel_area = bbox.area / analysis.page_area
            
            if rel_area < self.rules.figure_min_area_ratio:
                continue
            
            # Check isolation
            nearby_count = sum(
                1 for other in analysis.bboxes
                if other != bbox and 
                other.region_type == RegionType.UNKNOWN and
                bbox.distance_to(other) < 100
            )
            
            if nearby_count < self.rules.figure_max_internal_bboxes:
                bbox.region_type = RegionType.FIGURE
                bbox.confidence = 0.75
    
    def _classify_titles(self, analysis: PageAnalysis) -> None:
        """Classify titles as large, centered text near top."""
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
            
            if bbox.height < avg_height * self.rules.title_min_font_size_ratio:
                continue
            
            rel_y = bbox.y1 / analysis.page_height
            if rel_y > self.rules.title_top_margin_ratio:
                continue
            
            center_x = bbox.center[0]
            page_center_x = analysis.page_width / 2
            rel_offset = abs(center_x - page_center_x) / analysis.page_width
            
            if rel_offset < self.rules.title_center_threshold:
                bbox.region_type = RegionType.TITLE
                bbox.confidence = 0.80
    
    def _classify_captions(self, analysis: PageAnalysis) -> None:
        """Classify captions as small text near figures or charts."""
        figures_and_charts = [
            bbox for bbox in analysis.bboxes
            if bbox.region_type in [RegionType.FIGURE, RegionType.CHART]
        ]
        
        if not figures_and_charts:
            return
        
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            rel_height = bbox.height / analysis.page_height
            if rel_height > self.rules.caption_max_height_ratio:
                continue
            
            if bbox.aspect_ratio < 2.0:
                continue
            
            for fig_chart in figures_and_charts:
                distance = bbox.distance_to(fig_chart)
                if distance < self.rules.caption_near_figure_distance:
                    bbox.region_type = RegionType.CAPTION
                    bbox.confidence = 0.70
                    break
    
    def _classify_text(self, analysis: PageAnalysis) -> None:
        """Classify remaining bboxes as text if they have text characteristics."""
        for bbox in analysis.bboxes:
            if bbox.region_type != RegionType.UNKNOWN:
                continue
            
            rel_height = bbox.height / analysis.page_height
            rel_width = bbox.width / analysis.page_width
            
            is_text = (
                bbox.aspect_ratio > self.rules.text_min_aspect_ratio and
                rel_height < self.rules.text_max_height_ratio and
                rel_width > self.rules.text_min_width_ratio
            )
            
            if is_text:
                bbox.region_type = RegionType.TEXT
                bbox.confidence = 0.70
    
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
        show_original: bool = False
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
        
        # Optionally show original (pre-merge) bboxes
        if show_original and analysis.original_bboxes:
            for bbox in analysis.original_bboxes:
                rect = patches.Rectangle(
                    (bbox.x1, bbox.y1), 
                    bbox.width, 
                    bbox.height,
                    linewidth=0.5,
                    edgecolor='lightgray',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.3
                )
                ax.add_patch(rect)
        
        # Draw classified bboxes
        for bbox in analysis.bboxes:
            color = color_map[bbox.region_type]
            
            # Thicker border for merged bboxes
            linewidth = 3 if bbox.is_merged else 2
            
            rect = patches.Rectangle(
                (bbox.x1, bbox.y1), 
                bbox.width, 
                bbox.height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{bbox.region_type.value}"
            if bbox.confidence > 0:
                label += f" ({bbox.confidence:.2f})"
            if bbox.is_merged:
                label += f" [merged from {len(bbox.merged_from)}]"
            
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
        title += f"\n[{len(analysis.original_bboxes)} original → {len(analysis.bboxes)} after merge]"
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
        print(f"Original bboxes: {len(analysis.original_bboxes)}")
        print(f"After merging: {len(analysis.bboxes)}")
        print(f"Merge ratio: {len(analysis.bboxes)/len(analysis.original_bboxes)*100:.1f}%")
        print(f"Number of clusters: {len(analysis.clusters)}")
        
        print(f"\nRegion Summary:")
        summary = analysis.summary()
        for region_type, count in summary.items():
            if count > 0:
                print(f"  {region_type:12s}: {count}")
        
        if verbose:
            print(f"\nCluster Analysis:")
            for cluster in analysis.clusters:
                x_align, y_align, combined = cluster.calculate_alignment_score(
                    self.rules.alignment_tolerance,
                    self.rules.alignment_min_matches
                )
                has_grid, rows, cols = cluster.has_grid_structure(
                    self.rules.alignment_tolerance
                )
                
                print(f"\n  Cluster {cluster.cluster_id}:")
                print(f"    Bboxes: {len(cluster.bboxes)}")
                print(f"    Alignment: X={x_align:.2f}, Y={y_align:.2f}, Combined={combined:.2f}")
                print(f"    Grid: {has_grid} ({rows}x{cols})")
                print(f"    Area: {cluster.area:.0f} px² ({cluster.area/analysis.page_area*100:.1f}% of page)")


# Alias for backward compatibility
DocumentLayoutAnalyzer = FinalImprovedAnalyzer

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
