"""
PDF Validator

Validates downloaded PDFs for integrity and extractability.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pymupdf as fitz

from src.utils.logger import logger
from src.utils.models import DocumentMetadata


class PDFValidator:
    """
    Validates PDF documents for integrity and extractability

    Checks:
    - File can be opened
    - Has pages
    - Pages contain extractable content
    - Not corrupted
    """

    def __init__(self):
        """Initialize validator"""
        self.validation_results: Dict[str, Dict] = {}

    def validate_pdf(self, file_path: Path) -> Tuple[bool, Dict]:
        """
        Validate a single PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "file_path": str(file_path),
            "exists": False,
            "can_open": False,
            "num_pages": 0,
            "has_text": False,
            "has_images": False,
            "file_size": 0,
            "errors": []
        }

        # Check file exists
        if not file_path.exists():
            validation_info["errors"].append("File does not exist")
            return False, validation_info

        validation_info["exists"] = True
        validation_info["file_size"] = file_path.stat().st_size

        # Try to open PDF
        try:
            doc = fitz.open(file_path)
            validation_info["can_open"] = True
            validation_info["num_pages"] = len(doc)

            if len(doc) == 0:
                validation_info["errors"].append("PDF has no pages")
                doc.close()
                return False, validation_info

            # Check first few pages for content
            pages_to_check = min(3, len(doc))
            text_found = False
            images_found = False

            for page_num in range(pages_to_check):
                page = doc[page_num]

                # Check for text
                text = page.get_text().strip()
                if len(text) > 50:  # At least 50 characters
                    text_found = True

                # Check for images
                images = page.get_images()
                if images:
                    images_found = True

            validation_info["has_text"] = text_found
            validation_info["has_images"] = images_found

            doc.close()

            # Determine if valid
            is_valid = (
                validation_info["can_open"] and
                validation_info["num_pages"] > 0 and
                (validation_info["has_text"] or validation_info["has_images"])
            )

            if not is_valid and not validation_info["has_text"]:
                validation_info["errors"].append("No extractable text found")

            return is_valid, validation_info

        except Exception as e:
            validation_info["errors"].append(f"Error opening PDF: {str(e)}")
            return False, validation_info

    def validate_batch(
        self,
        file_paths: List[Path]
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Validate multiple PDF files

        Args:
            file_paths: List of PDF file paths

        Returns:
            Dictionary mapping file path to (is_valid, validation_info)
        """
        results = {}

        logger.info(f"Validating {len(file_paths)} PDF files...")

        for file_path in file_paths:
            is_valid, info = self.validate_pdf(file_path)
            results[str(file_path)] = (is_valid, info)

            if is_valid:
                logger.debug(f"✓ Valid: {file_path.name} ({info['num_pages']} pages)")
            else:
                logger.warning(f"✗ Invalid: {file_path.name} - {info['errors']}")

        # Summary
        valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
        invalid_count = len(results) - valid_count

        logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")

        return results

    def validate_from_metadata(
        self,
        metadata: Dict[str, DocumentMetadata]
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Validate PDFs from metadata dictionary

        Args:
            metadata: Dictionary of doc_id -> DocumentMetadata

        Returns:
            Dictionary mapping doc_id to (is_valid, validation_info)
        """
        file_paths = [meta.file_path for meta in metadata.values()]
        path_to_id = {str(meta.file_path): doc_id for doc_id, meta in metadata.items()}

        results = self.validate_batch(file_paths)

        # Remap to doc_id
        return {
            path_to_id[path]: result
            for path, result in results.items()
        }
    
import pdfplumber

def advanced_pdf_detection(pdf_path):
    """More detailed analysis using pdfplumber"""
    logger.debug(f"Performing standard and scaned pdf detection on {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        
        # Check for actual text objects
        chars = first_page.chars  # Text characters as objects
        images = first_page.images
        
        logger.debug(f"First page analysis for {pdf_path}:")
        logger.debug(f"Text objects: {len(chars)}")
        logger.debug(f"Image objects: {len(images)}")   

        result_details = {
            "text_chars": len(chars),
            "image_objects": len(images)
        }
        
        if len(chars) > 50:
            return "standard", result_details
        elif len(images) > 0:
            return "scanned", result_details
        else:
            return "unknown", result_details


def main():
    """CLI entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate FOMC PDFs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PDFs"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output JSON file for validation results"
    )

    parser.add_argument(
        "--pdf-standard-scan-detect",
        action="store_true",
        help="Perform standard vs scanned pdf detection using pdfplumber"
    )

    args = parser.parse_args()

    # Find all PDFs
    pdf_files = list(args.input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_dir}")
        return

    # Validate
    validator = PDFValidator()
    results = validator.validate_batch(pdf_files)

    # Store scan results in a dictionary
    scan_results = {}
    if args.pdf_standard_scan_detect:
        for pdf_path in pdf_files:
            scan_type, result_details = advanced_pdf_detection(pdf_path)  # Unpack tuple
            scan_results[str(pdf_path)] = {
                "scan_type": scan_type,
                "details": result_details
            }
            logger.debug(f"{pdf_path.name}: {scan_type}")

    # Save results if requested
    if args.output_file:
        output_data = {
            str(path): {
                "valid": is_valid,
                "info": info,
                "scan_info": scan_results.get(str(path), None)  # Add scan results here
            }
            for path, (is_valid, info) in results.items()
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
#testing code 
# python -m src.ingestion.validator --input-dir data/raw --pdf-standard-scan-detect --output-file data/output/ingestion/ingestion_validation_results.json