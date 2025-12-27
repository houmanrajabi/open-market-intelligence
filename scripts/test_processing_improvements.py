"""
Test Processing Improvements

Tests the bbox refinement and enhanced debug visualization on a sample document.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.pdf_processor import PDFProcessor
from src.utils.logger import logger
import json


def test_single_document(pdf_name: str = "fomcprojtabl20221214"):
    """
    Test processing improvements on a single document.

    Args:
        pdf_name: Name of PDF file (without .pdf extension)
    """
    logger.info("="*70)
    logger.info("TESTING PROCESSING IMPROVEMENTS")
    logger.info("="*70)

    # Paths
    pdf_path = Path(f"data/raw/{pdf_name}.pdf")
    output_dir = Path(f"data/processed_test/{pdf_name}")

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Available PDFs:")
        for p in Path("data/raw").glob("*.pdf"):
            logger.info(f"  - {p.stem}")
        return

    # Create output directory
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Initialize processor with custom output directory
    processor = PDFProcessor(output_dir=output_dir.parent)

    # Process document
    logger.info(f"\n📄 Processing: {pdf_name}")
    logger.info(f"   Output: {output_dir}")

    result = processor.process_document(pdf_path)

    # Analysis
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)

    # Load full_structure.json
    structure_path = output_dir / "full_structure.json"
    if structure_path.exists():
        with open(structure_path, 'r') as f:
            data = json.load(f)

        elements = data.get("elements", [])

        # Count element types
        type_counts = {}
        refined_count = 0

        for elem in elements:
            etype = elem.get("type", "UNKNOWN")
            type_counts[etype] = type_counts.get(etype, 0) + 1

            # Check if bbox was refined
            if elem.get("metadata", {}).get("bbox_refined"):
                refined_count += 1

        # Print statistics
        logger.info(f"\n📊 Element Statistics:")
        logger.info(f"   Total elements: {len(elements)}")
        for etype, count in sorted(type_counts.items()):
            logger.info(f"   {etype}: {count}")

        logger.info(f"\n🎯 Bbox Refinement:")
        logger.info(f"   Elements refined: {refined_count}/{len(elements)} ({refined_count/len(elements)*100:.1f}%)")

        logger.info(f"\n🎨 Debug Visualizations:")
        debug_files = list(output_dir.glob("*_debug.jpg"))
        logger.info(f"   Generated {len(debug_files)} debug images")
        logger.info(f"   Location: {output_dir}")

        # Show sample debug image path
        if debug_files:
            logger.info(f"\n   Sample: {debug_files[0]}")
            logger.info(f"   Open in finder: open {output_dir}")

    else:
        logger.error(f"Structure file not found: {structure_path}")

    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE")
    logger.info("="*70)

    return result


def compare_with_original(test_doc: str = "fomcprojtabl20221214"):
    """
    Compare new processing with original.

    Args:
        test_doc: Document name
    """
    original_dir = Path(f"data/processed/{test_doc}")
    new_dir = Path(f"data/processed_test/{test_doc}")

    if not original_dir.exists():
        logger.warning(f"Original not found: {original_dir}")
        return

    if not new_dir.exists():
        logger.warning(f"New processing not found: {new_dir}")
        return

    # Load both structures
    with open(original_dir / "full_structure.json", 'r') as f:
        original = json.load(f)

    with open(new_dir / "full_structure.json", 'r') as f:
        new = json.load(f)

    original_elements = original.get("elements", [])
    new_elements = new.get("elements", [])

    logger.info("\n" + "="*70)
    logger.info("COMPARISON: Original vs New Processing")
    logger.info("="*70)

    logger.info(f"\n📊 Element Count:")
    logger.info(f"   Original: {len(original_elements)}")
    logger.info(f"   New:      {len(new_elements)}")
    logger.info(f"   Difference: {len(new_elements) - len(original_elements)}")

    # Compare bbox sizes
    original_avg_area = sum((e["bbox_norm"][2] - e["bbox_norm"][0]) * (e["bbox_norm"][3] - e["bbox_norm"][1])
                            for e in original_elements) / len(original_elements)
    new_avg_area = sum((e["bbox_norm"][2] - e["bbox_norm"][0]) * (e["bbox_norm"][3] - e["bbox_norm"][1])
                       for e in new_elements) / len(new_elements)

    logger.info(f"\n📐 Average Bbox Area:")
    logger.info(f"   Original: {original_avg_area:.0f}")
    logger.info(f"   New:      {new_avg_area:.0f}")
    logger.info(f"   Reduction: {(original_avg_area - new_avg_area) / original_avg_area * 100:.1f}%")

    # Check for refinement metadata
    refined_elements = [e for e in new_elements if e.get("metadata", {}).get("bbox_refined")]
    logger.info(f"\n🎯 Refined Elements: {len(refined_elements)} ({len(refined_elements)/len(new_elements)*100:.1f}%)")

    logger.info("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test processing improvements")
    parser.add_argument("--doc", type=str, default="fomcprojtabl20221214",
                       help="Document name to test")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with original processing")

    args = parser.parse_args()

    # Run test
    test_single_document(args.doc)

    # Compare if requested
    if args.compare:
        compare_with_original(args.doc)
