"""
Main PDF Processor

Orchestrates extraction and chunking of PDF documents.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .extractors.text_extractor import TextExtractor
from .extractors.table_extractor import TableExtractor
from .extractors.figure_extractor import FigureExtractor
from .chunker import DocumentChunker

from ..utils.logger import logger
from ..utils.models import DocumentChunk, DocumentMetadata
from ..utils.config import config


class PDFProcessor:
    """
    Main PDF processor that extracts and chunks documents

    Workflow:
    1. Extract text from all pages
    2. Extract tables from document
    3. Extract figures from document
    4. Chunk text content
    5. Create chunks for tables and figures
    6. Return all chunks with metadata
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        extract_figures: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize PDF processor

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            extract_figures: Whether to extract figures
            output_dir: Directory for processed output
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_figures = extract_figures

        self.output_dir = output_dir or config.data.processed_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if extract_figures:
            figures_dir = self.output_dir / "figures"
            self.figure_extractor = FigureExtractor(output_dir=figures_dir)
        else:
            self.figure_extractor = None

    def process_document(
        self,
        pdf_path: Path,
        doc_metadata: Optional[DocumentMetadata] = None
    ) -> Dict[str, Any]:
        """
        Process a single PDF document

        Args:
            pdf_path: Path to PDF file
            doc_metadata: Optional document metadata

        Returns:
            Dictionary with all extracted chunks and statistics
        """
        logger.info(f"Processing: {pdf_path.name}")

        doc_id = pdf_path.stem
        if doc_metadata:
            doc_id = doc_metadata.doc_id

        all_chunks = []
        stats = {
            "doc_id": doc_id,
            "file_path": str(pdf_path),
            "text_chunks": 0,
            "table_chunks": 0,
            "figure_chunks": 0,
            "total_chunks": 0
        }

        # Base metadata for all chunks
        base_metadata = {
            "doc_id": doc_id,
            "file_path": str(pdf_path)
        }

        if doc_metadata:
            base_metadata.update({
                "doc_type": doc_metadata.doc_type.value,
                "date": doc_metadata.date.isoformat()
            })

        # 1. Extract text
        logger.debug("Extracting text...")
        pages_data = self.text_extractor.extract_with_fallback(pdf_path)

        # 2. Extract tables
        logger.debug("Extracting tables...")
        tables = self.table_extractor.extract_from_pdf(pdf_path)
        stats["table_chunks"] = len(tables)

        # Build set of pages with tables (to avoid double-processing)
        table_pages = set(t["page"] for t in tables)

        # 3. Extract figures (optional)
        figures = []
        if self.extract_figures and self.figure_extractor:
            logger.debug("Extracting figures...")
            figures = self.figure_extractor.extract_from_pdf(pdf_path, doc_id)
            stats["figure_chunks"] = len(figures)

        # 4. Chunk text content
        logger.debug("Chunking text...")
        for page_data in pages_data:
            page_num = page_data["page_num"]
            text = page_data["text"]

            if not text or len(text.strip()) < 50:
                continue

            # Skip pages that are mostly tables
            if page_num + 1 in table_pages:
                # Still process if there's significant text
                if len(text.strip()) < 200:
                    continue

            # Create chunks for this page
            page_metadata = {
                **base_metadata,
                "page": page_num + 1
            }

            page_chunks = self.chunker.chunk_text(
                text=text,
                doc_id=doc_id,
                page_num=page_num + 1,
                metadata=page_metadata
            )

            all_chunks.extend(page_chunks)
            stats["text_chunks"] += len(page_chunks)

        # 5. Add table chunks
        logger.debug("Creating table chunks...")
        for i, table in enumerate(tables):
            table_metadata = {
                **base_metadata,
                "page": table["page"],
                "extractor": table["extractor"],
                "shape": table["shape"]
            }

            table_chunk = self.chunker.chunk_table(
                table_markdown=table["markdown"],
                doc_id=doc_id,
                table_num=i,
                page_num=table["page"],
                metadata=table_metadata
            )

            all_chunks.append(table_chunk)

        # 6. Add figure chunks
        if figures:
            logger.debug("Creating figure chunks...")
            for i, figure in enumerate(figures):
                figure_metadata = {
                    **base_metadata,
                    "page": figure["page"]
                }

                figure_chunk = self.chunker.chunk_figure(
                    figure_data=figure,
                    doc_id=doc_id,
                    figure_num=i,
                    metadata=figure_metadata
                )

                all_chunks.append(figure_chunk)

        stats["total_chunks"] = len(all_chunks)

        logger.info(
            f"✓ Processed {pdf_path.name}: "
            f"{stats['text_chunks']} text, "
            f"{stats['table_chunks']} tables, "
            f"{stats['figure_chunks']} figures"
        )

        return {
            "doc_id": doc_id,
            "chunks": all_chunks,
            "stats": stats,
            "tables": tables,
            "figures": figures
        }

    def save_processed_document(
        self,
        processed_data: Dict[str, Any],
        output_subdir: str = "chunks"
    ) -> Path:
        """
        Save processed document to disk

        Args:
            processed_data: Output from process_document()
            output_subdir: Subdirectory for output

        Returns:
            Path to saved file
        """
        doc_id = processed_data["doc_id"]

        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{doc_id}.json"

        # Convert chunks to dict format
        chunks_data = [
            chunk.to_dict() for chunk in processed_data["chunks"]
        ]

        # Save
        save_data = {
            "doc_id": doc_id,
            "stats": processed_data["stats"],
            "chunks": chunks_data
        }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.debug(f"Saved processed document to {output_file}")

        return output_file

    def load_processed_document(
        self,
        doc_id: str,
        input_subdir: str = "chunks"
    ) -> Dict[str, Any]:
        """
        Load processed document from disk

        Args:
            doc_id: Document identifier
            input_subdir: Subdirectory for input

        Returns:
            Dictionary with chunks and stats
        """
        input_file = self.output_dir / input_subdir / f"{doc_id}.json"

        if not input_file.exists():
            raise FileNotFoundError(f"Processed document not found: {input_file}")

        with open(input_file, 'r') as f:
            data = json.load(f)

        # Convert chunks back to DocumentChunk objects
        chunks = [
            DocumentChunk.from_dict(chunk_data)
            for chunk_data in data["chunks"]
        ]

        return {
            "doc_id": data["doc_id"],
            "chunks": chunks,
            "stats": data["stats"]
        }

    def process_all_documents(
        self,
        pdf_dir: Path,
        metadata: Optional[Dict[str, DocumentMetadata]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory

        Args:
            pdf_dir: Directory containing PDFs
            metadata: Optional metadata dictionary

        Returns:
            List of processed document results
        """
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []

        logger.info(f"Processing {len(pdf_files)} documents...")

        results = []

        for pdf_path in pdf_files:
            doc_id = pdf_path.stem
            doc_metadata = metadata.get(doc_id) if metadata else None

            try:
                processed = self.process_document(pdf_path, doc_metadata)
                self.save_processed_document(processed)
                results.append(processed)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        logger.info(f"Processing complete: {len(results)}/{len(pdf_files)} succeeded")

        return results


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Process FOMC PDFs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.data.raw_data_dir,
        help="Input directory with PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.data.processed_data_dir,
        help="Output directory"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.processing.chunk_size,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.processing.chunk_overlap,
        help="Chunk overlap in tokens"
    )
    parser.add_argument(
        "--extract-figures",
        action="store_true",
        help="Extract figures from PDFs"
    )

    args = parser.parse_args()

    # Load metadata if available
    metadata_file = args.input_dir / "metadata.json"
    metadata = None

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata_data = json.load(f)
            metadata = {
                doc_id: DocumentMetadata.from_dict(meta)
                for doc_id, meta in metadata_data.items()
            }

    # Process documents
    processor = PDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extract_figures=args.extract_figures,
        output_dir=args.output_dir
    )

    processor.process_all_documents(args.input_dir, metadata)


if __name__ == "__main__":
    main()
