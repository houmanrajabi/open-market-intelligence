# Document Layout Analyzer - Usage Examples

"""
This file demonstrates various ways to use the DocumentLayoutAnalyzer class.
"""

from document_layout_analyzer import (
    DocumentLayoutAnalyzer,
    ClassificationRules,
    RegionType,
    PageAnalysis
)
from PIL import Image
import os


# ============================================================================
# EXAMPLE 1: Basic PDF Analysis with Default Rules
# ============================================================================

def example_basic_analysis():
    """Analyze a PDF with default classification rules."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic PDF Analysis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = DocumentLayoutAnalyzer()
    
    # Analyze PDF
    pdf_path = "sample.pdf"
    results = analyzer.analyze_pdf(pdf_path)
    
    # Print results for each page
    for analysis in results:
        analyzer.print_analysis(analysis)
        
        # Visualize (save without showing)
        analyzer.visualize(
            analysis,
            pdf_path=pdf_path,
            output_path=f"output/page_{analysis.page_num + 1}.png",
            show=False
        )


# ============================================================================
# EXAMPLE 2: Custom Classification Rules
# ============================================================================

def example_custom_rules():
    """Use custom rules for more aggressive text detection."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Classification Rules")
    print("="*70)
    
    # Define custom rules
    custom_rules = ClassificationRules(
        # Make text detection more sensitive
        text_min_aspect_ratio=1.5,  # Lower threshold (default: 2.0)
        text_max_height_ratio=0.08,  # Allow taller text (default: 0.05)
        text_vertical_gap_threshold=50.0,  # Larger grouping (default: 30.0)
        
        # Make figure detection more strict
        figure_min_area_ratio=0.15,  # Require larger area (default: 0.1)
        
        # Table detection parameters
        table_min_bboxes=6,  # Require more bboxes (default: 4)
        table_alignment_threshold=15.0,  # Stricter alignment (default: 20.0)
    )
    
    # Initialize with custom rules
    analyzer = DocumentLayoutAnalyzer(rules=custom_rules)
    
    # Analyze
    pdf_path = "sample.pdf"
    results = analyzer.analyze_pdf(pdf_path)
    
    for analysis in results:
        analyzer.print_analysis(analysis)


# ============================================================================
# EXAMPLE 3: Single Page Analysis
# ============================================================================

def example_single_page():
    """Analyze just one page of a PDF."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Page Analysis")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    
    # Analyze specific page (zero-indexed)
    pdf_path = "sample.pdf"
    page_num = 0  # First page
    
    analysis = analyzer.analyze_page(pdf_path, page_num)
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Visualize
    analyzer.visualize(
        analysis,
        pdf_path=pdf_path,
        output_path=f"single_page_analysis.png"
    )


# ============================================================================
# EXAMPLE 4: Image Analysis
# ============================================================================

def example_image_analysis():
    """Analyze a standalone image (not from PDF)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Image Analysis")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    
    # Load image
    image_path = "document_scan.jpg"
    image = Image.open(image_path)
    
    # Analyze
    analysis = analyzer.analyze_image(image)
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Visualize
    analyzer.visualize(
        analysis,
        image=image,
        output_path="image_analysis.png"
    )


# ============================================================================
# EXAMPLE 5: Filter by Region Type
# ============================================================================

def example_filter_regions():
    """Extract specific types of regions from analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Filter Regions by Type")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    pdf_path = "sample.pdf"
    
    results = analyzer.analyze_pdf(pdf_path)
    
    # Process each page
    for analysis in results:
        print(f"\nPage {analysis.page_num + 1}:")
        
        # Get all text regions
        text_regions = analysis.get_by_type(RegionType.TEXT)
        print(f"  Found {len(text_regions)} text regions")
        for i, bbox in enumerate(text_regions, 1):
            print(f"    Text {i}: size={bbox.width:.0f}x{bbox.height:.0f}, "
                  f"aspect={bbox.aspect_ratio:.2f}")
        
        # Get all figures
        figures = analysis.get_by_type(RegionType.FIGURE)
        print(f"  Found {len(figures)} figures")
        for i, bbox in enumerate(figures, 1):
            print(f"    Figure {i}: area={bbox.area:.0f}px² "
                  f"({bbox.area/analysis.page_area*100:.1f}% of page)")
        
        # Get all tables
        tables = analysis.get_by_type(RegionType.TABLE)
        print(f"  Found {len(tables)} table cells")


# ============================================================================
# EXAMPLE 6: Spatial Analysis
# ============================================================================

def example_spatial_analysis():
    """Analyze spatial relationships between regions."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Spatial Analysis")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    pdf_path = "sample.pdf"
    
    analysis = analyzer.analyze_page(pdf_path, 0)
    
    # Analyze distances between regions
    if len(analysis.bboxes) >= 2:
        print("\nSpatial Relationships:")
        
        for i in range(len(analysis.bboxes) - 1):
            bbox1 = analysis.bboxes[i]
            bbox2 = analysis.bboxes[i + 1]
            
            distance = bbox1.distance_to(bbox2)
            v_gap = bbox1.vertical_gap_to(bbox2)
            h_gap = bbox1.horizontal_gap_to(bbox2)
            
            print(f"\nRegion {i+1} ({bbox1.region_type.value}) to "
                  f"Region {i+2} ({bbox2.region_type.value}):")
            print(f"  Center distance: {distance:.1f}px")
            print(f"  Vertical gap: {v_gap:.1f}px")
            print(f"  Horizontal gap: {h_gap:.1f}px")


# ============================================================================
# EXAMPLE 7: Batch Processing
# ============================================================================

def example_batch_processing():
    """Process multiple PDFs and aggregate statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Batch Processing")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    
    # List of PDFs to process
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    # Aggregate statistics
    total_stats = {rt.value: 0 for rt in RegionType}
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_path} (not found)")
            continue
        
        print(f"\nProcessing: {pdf_path}")
        results = analyzer.analyze_pdf(pdf_path)
        
        # Aggregate counts
        for analysis in results:
            summary = analysis.summary()
            for region_type, count in summary.items():
                total_stats[region_type] += count
    
    # Print aggregate statistics
    print("\n" + "="*50)
    print("AGGREGATE STATISTICS")
    print("="*50)
    for region_type, count in total_stats.items():
        if count > 0:
            print(f"{region_type:12s}: {count}")


# ============================================================================
# EXAMPLE 8: Export Results to JSON
# ============================================================================

def example_export_json():
    """Export analysis results to JSON format."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Export to JSON")
    print("="*70)
    
    import json
    
    analyzer = DocumentLayoutAnalyzer()
    pdf_path = "sample.pdf"
    
    results = analyzer.analyze_pdf(pdf_path)
    
    # Convert to JSON-serializable format
    json_data = []
    for analysis in results:
        page_data = {
            "page_num": analysis.page_num + 1,
            "page_dimensions": {
                "width": analysis.page_width,
                "height": analysis.page_height
            },
            "summary": analysis.summary(),
            "regions": []
        }
        
        for bbox in analysis.bboxes:
            region_data = {
                "type": bbox.region_type.value,
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "dimensions": {
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": bbox.area,
                    "aspect_ratio": bbox.aspect_ratio
                },
                "confidence": bbox.confidence,
                "center": list(bbox.center)
            }
            page_data["regions"].append(region_data)
        
        json_data.append(page_data)
    
    # Save to file
    output_file = "analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Exported results to {output_file}")


# ============================================================================
# EXAMPLE 9: Academic Paper Analysis
# ============================================================================

def example_academic_paper():
    """Specialized rules for academic papers with equations and figures."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Academic Paper Analysis")
    print("="*70)
    
    # Rules optimized for academic papers
    academic_rules = ClassificationRules(
        # Text is dominant in papers
        text_min_aspect_ratio=1.8,
        text_max_height_ratio=0.06,
        text_vertical_gap_threshold=25.0,
        
        # Papers often have smaller figures
        figure_min_area_ratio=0.05,
        figure_min_aspect_ratio=0.4,
        figure_max_aspect_ratio=4.0,
        
        # Tables are common
        table_min_bboxes=3,
        table_alignment_threshold=25.0,
        
        # Headers/footers with page numbers
        header_y_threshold=0.08,
        footer_y_threshold=0.92
    )
    
    analyzer = DocumentLayoutAnalyzer(rules=academic_rules)
    
    pdf_path = "research_paper.pdf"
    results = analyzer.analyze_pdf(pdf_path)
    
    # Analyze paper structure
    print("\nPaper Structure Analysis:")
    for analysis in results:
        figures = analysis.get_by_type(RegionType.FIGURE)
        tables = analysis.get_by_type(RegionType.TABLE)
        
        print(f"\nPage {analysis.page_num + 1}:")
        print(f"  Figures: {len(figures)}")
        print(f"  Tables: {len(tables)}")
        print(f"  Text density: {len(analysis.get_by_type(RegionType.TEXT))}")


# ============================================================================
# EXAMPLE 10: Error Handling and Validation
# ============================================================================

def example_error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Error Handling")
    print("="*70)
    
    analyzer = DocumentLayoutAnalyzer()
    
    # Handle missing file
    try:
        results = analyzer.analyze_pdf("nonexistent.pdf")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Handle invalid page number
    try:
        pdf_path = "sample.pdf"
        analysis = analyzer.analyze_page(pdf_path, 999)  # Invalid page
    except Exception as e:
        print(f"Error: {e}")
    
    # Validate analysis results
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        results = analyzer.analyze_pdf(pdf_path)
        
        for analysis in results:
            # Check if any regions were detected
            if not analysis.bboxes:
                print(f"Warning: No regions detected on page {analysis.page_num + 1}")
            
            # Check for unknown regions
            unknown = analysis.get_by_type(RegionType.UNKNOWN)
            if unknown:
                print(f"Warning: {len(unknown)} regions could not be classified "
                      f"on page {analysis.page_num + 1}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOCUMENT LAYOUT ANALYZER - USAGE EXAMPLES")
    print("="*70)
    
    # Uncomment the examples you want to run:
    
    example_basic_analysis()
    example_custom_rules()
    example_single_page()
    example_image_analysis()
    example_filter_regions()
    example_spatial_analysis()
    example_batch_processing()
    example_export_json()
    example_academic_paper()
    example_error_handling()
    
    print("\nNote: Uncomment the examples you want to run in the main block.")
    print("Make sure to provide valid PDF paths for testing.")
