"""
Figure Extractor

Extracts figures and charts from PDF documents.
"""

from pathlib import Path
from typing import List, Dict, Any
import io

import pymupdf as fitz
from PIL import Image

from src.utils.logger import logger


class FigureExtractor:
    """
    Extracts figures and images from PDF documents

    Uses PyMuPDF to extract embedded images.
    """

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize figure extractor

        Args:
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            output_dir: Directory to save extracted images (optional)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.output_dir = output_dir

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a single page

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            doc_id: Document identifier

        Returns:
            List of image metadata dictionaries
        """
        images_data = []

        try:
            # Get images from page
            image_list = page.get_images()

            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]

                    # Extract image
                    base_image = page.parent.extract_image(xref)

                    if not base_image:
                        continue

                    # Get image data
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter small images
                    if width < self.min_width or height < self.min_height:
                        continue

                    # Create metadata
                    img_id = f"{doc_id}_p{page_num}_img{img_index}"

                    img_data = {
                        "image_id": img_id,
                        "page": page_num + 1,  # 1-indexed
                        "width": width,
                        "height": height,
                        "format": image_ext,
                        "size_bytes": len(image_bytes),
                        "xref": xref
                    }

                    # Save image if output directory specified
                    if self.output_dir:
                        img_path = self.output_dir / f"{img_id}.{image_ext}"
                        img_path.write_bytes(image_bytes)
                        img_data["file_path"] = str(img_path)

                    # Store image bytes if not saving to disk
                    if not self.output_dir:
                        img_data["image_bytes"] = image_bytes

                    images_data.append(img_data)

                except Exception as e:
                    logger.error(f"Error extracting image {img_index} from page {page_num}: {e}")

        except Exception as e:
            logger.error(f"Error getting images from page {page_num}: {e}")

        return images_data

    def extract_from_pdf(
        self,
        pdf_path: Path,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all images from PDF

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier

        Returns:
            List of image metadata dictionaries
        """
        logger.debug(f"Extracting figures from {pdf_path.name}")

        all_images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images = self.extract_from_page(page, page_num, doc_id)
                all_images.extend(page_images)

            doc.close()

        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            return []

        logger.debug(f"Extracted {len(all_images)} figures")

        return all_images

    def get_image_summary(self, image_data: Dict[str, Any]) -> str:
        """
        Generate text summary of image

        Args:
            image_data: Image metadata dictionary

        Returns:
            Text summary
        """
        summary = f"Figure on page {image_data['page']}"
        summary += f" ({image_data['width']}x{image_data['height']} pixels)"
        summary += f" [Format: {image_data['format']}]"

        return summary


# Optional import for type hint
from typing import Optional
