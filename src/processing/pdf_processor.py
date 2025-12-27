"""
Main PDF Processor 

"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# External libraries for PDF processing
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from src.processing.extractors.qwen_extractor import analyze_layout
from src.processing.extractors.hybrid_extractor import extract_hybrid_content
from src.processing.chunker import DocumentChunker
from src.utils.logger import logger
from src.utils.models import DocumentMetadata
from src.utils.config import config

class PDFProcessor:
    """
    Production PDF processor that orchestrates VLM extraction and semantic chunking.
    
    **NEW**: All configuration now loaded from config.processing.*

    Workflow:
    1.  **Convert** PDF pages to high-res images.
    2.  **Analyze** layout using VLM (Qwen2-VL) to detect headers, tables, figures.
    3.  **Extract** content using a hybrid approach (OCR for text, VLM for visuals).
    4.  **Chunk** elements using section-aware logic (preserving table/figure context).
    5.  **Save** rich metadata and structured JSON outputs.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        output_dir: Optional[Path] = None,
        extract_figures: Optional[bool] = None,
        strategy: Optional[str] = None,
        # New parameters for override capability
        preserve_large_tables: Optional[bool] = None,
        small_table_threshold: Optional[int] = None,
        multimodal_chunks: Optional[bool] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize the processor with configuration.
        
        All parameters are optional and default to config.processing values.
        Pass explicit values to override config settings.
        
        Args:
            chunk_size: Override config.processing.chunk_size
            chunk_overlap: Override config.processing.chunk_overlap
            output_dir: Override config.data.processed_data_dir
            extract_figures: Override config.processing.extract_figures
            strategy: Override config.processing.strategy
            preserve_large_tables: Override config.processing.preserve_large_tables
            small_table_threshold: Override config.processing.small_table_threshold
            multimodal_chunks: Override config.processing.multimodal_chunks
            similarity_threshold: Override config.processing.similarity_threshold
        """
        
        # Load settings from config with override capability
        self.chunk_size = chunk_size or config.processing.chunk_size
        self.chunk_overlap = chunk_overlap or config.processing.chunk_overlap
        self.output_dir = output_dir or config.data.processed_data_dir
        self.extract_figures = extract_figures if extract_figures is not None else config.processing.extract_figures
        self.strategy = strategy or config.processing.strategy
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the enhanced chunker with config values
        self.chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=config.processing.min_chunk_size,
            max_chunk_size=config.processing.max_chunk_size,
            strategy=self.strategy,
            quality_threshold=config.processing.quality_threshold,
            preserve_large_tables=preserve_large_tables if preserve_large_tables is not None 
                                 else config.processing.preserve_large_tables,
            small_table_threshold=small_table_threshold or config.processing.small_table_threshold,
            multimodal_chunks=multimodal_chunks if multimodal_chunks is not None 
                            else config.processing.multimodal_chunks,
            similarity_threshold=similarity_threshold or config.processing.similarity_threshold
        )
        
        # Store config references for easy access
        self.pdf_dpi = config.processing.pdf_dpi
        self.image_quality = config.processing.image_quality
        
        logger.info(f"📄 PDFProcessor initialized with config:")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info(f"   Chunk size: {self.chunk_size} tokens")
        logger.info(f"   Strategy: {self.strategy}")
        logger.info(f"   Extract figures: {self.extract_figures}")
        logger.info(f"   PDF DPI: {self.pdf_dpi}")

    def process_document(
        self,
        pdf_path: Path,
        doc_metadata: Optional[DocumentMetadata] = None
    ) -> Dict[str, Any]:
        """
        Process a single PDF document through the full VLM pipeline.

        Args:
            pdf_path: Path to the input PDF file.
            doc_metadata: Optional metadata (title, date, etc.).

        Returns:
            Dictionary containing stats, full extracted elements, and final chunks.
        """
        start_time = time.time()
        doc_id = pdf_path.stem
        logger.info(f"🚀 Starting processing for: {pdf_path.name} (ID: {doc_id})")

        # 1. Setup Output Directory for this specific document
        doc_output_dir = self.output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Extract Document-Level Metadata (Basic)
        base_metadata = self._extract_pdf_metadata(pdf_path)
        if doc_metadata:
            base_metadata.update(doc_metadata.to_dict())

        # 3. Main Extraction Loop (Page by Page)
        all_elements = []
        last_anchor_context = None
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            logger.error(f"❌ Failed to open PDF {pdf_path}: {e}")
            return {}

        for page_num in range(1, total_pages + 1):
            logger.info(f"📄 Processing Page {page_num}/{total_pages}...")
            
            try:
                # A. Convert Page to Image (using config DPI)
                logger.debug(f"   - Rasterizing page at {self.pdf_dpi} DPI...")
                images = convert_from_path(
                    str(pdf_path),
                    dpi=self.pdf_dpi,  # ✅ Now from config
                    first_page=page_num,
                    last_page=page_num
                )
                
                if not images:
                    logger.warning(f"⚠️  Failed to rasterize page {page_num}")
                    continue
                
                temp_img_path = doc_output_dir / f"temp_page_{page_num}.jpg"
                images[0].save(temp_img_path, quality=self.image_quality)  # ✅ Now from config

                # B. Phase 1: VLM Layout Analysis
                logger.debug(f"   - Analyzing layout for page {page_num}...")
                layout_map = analyze_layout(str(temp_img_path))
                
                if not layout_map.get("layout"):
                    logger.warning(f"   - ⚠️  No layout detected on page {page_num}. Falling back to text.")

                # C. Phase 2: Hybrid Extraction
                logger.debug(f"   - Extracting hybrid content for page {page_num}...")
                page_elements, last_anchor_context = extract_hybrid_content(
                    pdf_path=str(pdf_path),
                    page_num=page_num,
                    layout_map=layout_map,
                    temp_image_path=str(temp_img_path),
                    output_dir=str(doc_output_dir),
                    previous_context=last_anchor_context
                )
                
                all_elements.extend(page_elements)
                logger.debug(f"   - ✅ Extracted {len(page_elements)} elements from page {page_num}")
                
                # Cleanup temp image
                if temp_img_path.exists():
                    temp_img_path.unlink()
                    
            except Exception as e:
                logger.error(f"❌ Error processing page {page_num}: {e}", exc_info=True)
                continue

        # 4. Phase 3: Semantic Chunking
        logger.info(f"🧩 Chunking {len(all_elements)} extracted elements...")
        chunks = self.chunker.chunk_elements(all_elements, doc_id=doc_id)

        # 5. Compile Statistics
        stats = self._compile_stats(
            start_time=start_time,
            total_pages=total_pages,
            num_elements=len(all_elements),
            num_chunks=len(chunks),
            chunker_stats=self.chunker.stats
        )
        
        logger.info(
            f"✅ Completed {doc_id}: {len(chunks)} chunks created in {stats['processing_time_seconds']}s. "
            f"(Tables preserved: {stats.get('tables_preserved', 0)})"
        )

        # 6. Construct Final Output
        result = {
            "doc_id": doc_id,
            "metadata": base_metadata,
            "stats": stats,
            "chunks": chunks,
            "elements": all_elements  # Optional: keep raw elements for debugging
        }

        # 7. Save to Disk
        self.save_processed_document(result, doc_id)
        
        return result

    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract basic technical metadata from PDF."""
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            page_count = doc.page_count
            doc.close()
            
            return {
                "title": meta.get("title", pdf_path.stem),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creation_date": meta.get("creationDate", ""),
                "file_size_bytes": pdf_path.stat().st_size,
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "page_count": page_count,
                "processed_date": datetime.now().isoformat(),
                "source_file": str(pdf_path.name)
            }
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            return {
                "processed_date": datetime.now().isoformat(),
                "source_file": str(pdf_path.name)
            }

    def _compile_stats(
        self,
        start_time: float,
        total_pages: int,
        num_elements: int,
        num_chunks: int,
        chunker_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile comprehensive processing statistics."""
        processing_time = time.time() - start_time
        
        return {
            "processing_time_seconds": round(processing_time, 2),
            "pages_per_second": round(total_pages / processing_time, 2) if processing_time > 0 else 0,
            "total_pages": total_pages,
            "total_elements": num_elements,
            "total_chunks": num_chunks,
            "elements_per_page": round(num_elements / total_pages, 2) if total_pages > 0 else 0,
            "chunks_per_page": round(num_chunks / total_pages, 2) if total_pages > 0 else 0,
            "tables_preserved": chunker_stats.get("tables_preserved", 0),
            "figures_processed": chunker_stats.get("figures_processed", 0),
            "splits_performed": chunker_stats.get("splits_performed", 0),
            "chunking_strategy": self.strategy,
            "avg_chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    def save_processed_document(self, processed_data: Dict[str, Any], doc_id: str):
        """Save the processed outputs (JSONs) to the document's output directory."""
        output_dir = self.output_dir / doc_id
        
        # Save RAG-ready chunks (Optimized for Vector DB)
        chunks_data = [chunk.to_dict() for chunk in processed_data["chunks"]]
        rag_output = {
            "doc_id": doc_id,
            "metadata": processed_data["metadata"],
            "stats": processed_data["stats"],
            "chunks": chunks_data,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "strategy": self.strategy,
                "pdf_dpi": self.pdf_dpi
            }
        }
        
        rag_chunks_path = output_dir / "rag_chunks.json"
        with open(rag_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(rag_output, f, indent=2, ensure_ascii=False)
        logger.debug(f"💾 Saved RAG chunks: {rag_chunks_path}")
            
        # Save Full Structure (For debugging/auditing)
        full_structure_path = output_dir / "full_structure.json"
        with open(full_structure_path, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_id": doc_id,
                "metadata": processed_data["metadata"],
                "elements": processed_data["elements"]
            }, f, indent=2, ensure_ascii=False)
        logger.debug(f"💾 Saved full structure: {full_structure_path}")

    def process_all_documents(
        self,
        pdf_dir: Path,
        metadata: Optional[Dict[str, DocumentMetadata]] = None,
        skip_existing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            metadata: Optional dict mapping doc_id -> DocumentMetadata
            skip_existing: If True, skip PDFs that already have rag_chunks.json
        
        Returns:
            List of processing results
        """
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {pdf_dir}")
            return []

        logger.info(f"📚 Processing batch of {len(pdf_files)} documents...")
        results = []
        skipped = 0

        for idx, pdf_path in enumerate(pdf_files, 1):
            doc_id = pdf_path.stem
            doc_meta = metadata.get(doc_id) if metadata else None
            
            # Check if already processed
            output_file = self.output_dir / doc_id / "rag_chunks.json"
            if skip_existing and output_file.exists():
                logger.info(f"⏭️  Skipping {doc_id} ({idx}/{len(pdf_files)}) - already processed")
                skipped += 1
                continue

            try:
                logger.info(f"📄 Processing {doc_id} ({idx}/{len(pdf_files)})...")
                result = self.process_document(pdf_path, doc_meta)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path.name}: {e}", exc_info=True)
                continue

        logger.info(f"✅ Batch processing complete: {len(results)} processed, {skipped} skipped")
        return results

    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for a batch processing run."""
        if not results:
            return {"error": "No results to summarize"}
        
        total_pages = sum(r["stats"]["total_pages"] for r in results)
        total_chunks = sum(r["stats"]["total_chunks"] for r in results)
        total_elements = sum(r["stats"]["total_elements"] for r in results)
        total_time = sum(r["stats"]["processing_time_seconds"] for r in results)
        total_tables = sum(r["stats"]["tables_preserved"] for r in results)
        
        return {
            "documents_processed": len(results),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_elements": total_elements,
            "total_tables_preserved": total_tables,
            "total_processing_time_seconds": round(total_time, 2),
            "avg_pages_per_doc": round(total_pages / len(results), 2),
            "avg_chunks_per_doc": round(total_chunks / len(results), 2),
            "avg_processing_time_per_doc": round(total_time / len(results), 2),
            "pages_per_second": round(total_pages / total_time, 2) if total_time > 0 else 0
        }


def main():
    """CLI entry point with enhanced argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process FOMC PDFs with VLM Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        default=None,  # Will use config value if not provided
        help="Target chunk size in tokens (overrides config)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in tokens (overrides config)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["section_aware", "fixed_size"],
        help="Chunking strategy (overrides config)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed documents"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess all documents"
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Path to metadata JSON file"
    )

    args = parser.parse_args()

    # Load optional external metadata
    metadata = None
    metadata_file = args.metadata_file or (args.input_dir / "metadata.json")
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = {k: DocumentMetadata.from_dict(v) for k, v in data.items()}
            logger.info(f"📋 Loaded metadata for {len(metadata)} documents")
        except Exception as e:
            logger.warning(f"⚠️  Could not load metadata.json: {e}")

    # Initialize processor (with optional overrides)
    processor = PDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
        strategy=args.strategy
    )

    # Process all documents
    results = processor.process_all_documents(
        args.input_dir, 
        metadata,
        skip_existing=args.skip_existing
    )
    
    # Print summary
    if results:
        summary = processor.get_processing_summary(results)
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        for key, value in summary.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()