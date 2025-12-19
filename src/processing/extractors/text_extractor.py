"""
Text Extractor

Extracts text content from PDF documents using PyMuPDF and PDFPlumber.
"""

from pathlib import Path
from typing import List, Dict, Any

import pymupdf as fitz
import pdfplumber

from src.utils.logger import logger


class TextExtractor:
    """
    Extracts text content from PDF documents

    Uses PyMuPDF as primary extractor, with PDFPlumber as fallback.
    """

    def __init__(self, preserve_layout: bool = True):
        """
        Initialize text extractor

        Args:
            preserve_layout: Whether to preserve text layout
        """
        self.preserve_layout = preserve_layout

    def extract_from_page_pymupdf(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        try:
            # OPTION A: Text with Layout Preservation (The Quality Upgrade)
            # "blocks" sorts text by vertical/horizontal position, not stream order.
            # This fixes the #1 quality issue with PDFs: mixed-up columns.
            blocks = page.get_text("blocks", sort=True)
            
            # Reconstruct text from blocks
            text_content = []
            for b in blocks:
                # b[4] is the text content of the block
                if b[6] == 0:  # b[6] is block type (0=text, 1=image)
                    text_content.append(b[4])
            
            text = "\n\n".join(text_content)

            return {
                "page_num": page_num,
                "text": text,
                "char_count": len(text),
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "extractor": "pymupdf_blocks"
            }

        except Exception as e:
            # Fallback remains valuable!
            logger.warning(f"PyMuPDF block extraction failed: {e}")
            return {"error": str(e)}

    def extract_from_page_pdfplumber(
        self,
        page: pdfplumber.page.Page,
        page_num: int
    ) -> Dict[str, Any]:
        """
        Extract text from a single page using PDFPlumber

        Args:
            page: PDFPlumber page object
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Extract text
            text = page.extract_text() or ""

            return {
                "page_num": page_num,
                "text": text,
                "char_count": len(text),
                "page_width": page.width,
                "page_height": page.height,
                "extractor": "pdfplumber"
            }

        except Exception as e:
            logger.error(f"PDFPlumber extraction failed for page {page_num}: {e}")
            return {
                "page_num": page_num,
                "text": "",
                "error": str(e),
                "extractor": "pdfplumber"
            }

    def extract_from_pdf(
        self,
        pdf_path: Path,
        method: str = "pymupdf"
    ) -> List[Dict[str, Any]]:
        """
        Extract text from entire PDF

        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pymupdf' or 'pdfplumber')

        Returns:
            List of page extraction results
        """
        pages_data = []

        if method == "pymupdf":
            try:
                doc = fitz.open(pdf_path)

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_data = self.extract_from_page_pymupdf(page, page_num)
                    pages_data.append(page_data)

                doc.close()

            except Exception as e:
                logger.error(f"Failed to open PDF with PyMuPDF: {e}")
                return []

        elif method == "pdfplumber":
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_data = self.extract_from_page_pdfplumber(page, page_num)
                        pages_data.append(page_data)

            except Exception as e:
                logger.error(f"Failed to open PDF with PDFPlumber: {e}")
                return []

        else:
            raise ValueError(f"Unknown extraction method: {method}")

        return pages_data

    def extract_with_fallback(
        self,
        pdf_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Extract text with fallback strategy

        Tries PyMuPDF first, falls back to PDFPlumber if needed.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page extraction results
        """
        logger.debug(f"Extracting text from {pdf_path.name}")

        # Try PyMuPDF first (faster)
        pages_data = self.extract_from_pdf(pdf_path, method="pymupdf")

        # Check if extraction was successful
        total_chars = sum(p.get("char_count", 0) for p in pages_data)

        if total_chars < 100:  # Very little text extracted
            logger.warning(f"PyMuPDF extracted little text ({total_chars} chars), trying PDFPlumber")
            pages_data = self.extract_from_pdf(pdf_path, method="pdfplumber")

        total_chars = sum(p.get("char_count", 0) for p in pages_data)
        logger.debug(f"Extracted {total_chars} characters from {len(pages_data)} pages")

        return pages_data

    def merge_pages(
        self,
        pages_data: List[Dict[str, Any]],
        separator: str = "\n\n"
    ) -> str:
        """
        Merge all pages into single text

        Args:
            pages_data: List of page extraction results
            separator: Separator between pages

        Returns:
            Merged text
        """
        texts = [p["text"] for p in pages_data if p.get("text")]
        return separator.join(texts)


if __name__ == "__main__": 
    logger.info("Running TextExtractor standalone test")
    extractor = TextExtractor(preserve_layout=True)


    pages_data_plum = extractor.extract_from_pdf(
        pdf_path =  'data/raw/fomcminutes20200129.pdf',
        method = "pdfplumber"
    )
    print( print(str(pages_data_plum[7])) )

