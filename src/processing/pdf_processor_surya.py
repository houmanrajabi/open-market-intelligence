"""
Main PDF Processor (v2.0)

Orchestrates the Tier 1 Hybrid Extraction pipeline (Surya + PyMuPDF + Qwen VLM)
and Semantic Chunking for RAG ingestion.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Internal imports
import fitz  # PyMuPDF for metadata
from src.processing.extractors.hybrid_extractor_surya import extract_hybrid_content_surya
from src.processing.chunker import DocumentChunker
from src.utils.logger import logger
from src.utils.models import DocumentMetadata
from src.utils.config import config

class PDFProcessor:
    """
    Production PDF processor integrating Hybrid Surya Extraction and Semantic Chunking.
    
    Workflow:
    1.  **Extract Content**: Delegate to HybridSuryaExtractor (Layers 1-4).
        - Detection (Surya) -> Clustering -> Hierarchy -> Content Enrichment (VLM/PyMuPDF).
    2.  **Chunk Content**: Use DocumentChunker to group elements by section and semantic context.
    3.  **Save Output**: Persist structured JSONs (RAG chunks + full lineage).
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        output_dir: Optional[Path] = None,
        strategy: Optional[str] = None,
        # Override parameters
        preserve_large_tables: Optional[bool] = None,
        small_table_threshold: Optional[int] = None,
        multimodal_chunks: Optional[bool] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize the processor with configuration.
        """
        # Load settings from config with override capability
        self.chunk_size = chunk_size or config.processing.chunk_size
        self.chunk_overlap = chunk_overlap or config.processing.chunk_overlap
        self.output_dir = output_dir or config.data.processed_data_dir
        self.strategy = strategy or config.processing.strategy
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the semantic chunker
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
        
        logger.info(f"📄 PDFProcessor initialized (Hybrid Surya + {self.strategy} Chunking)")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info(f"   Chunk size: {self.chunk_size} | Overlap: {self.chunk_overlap}")

    def process_document(
        self,
        pdf_path: Path,
        doc_metadata: Optional[DocumentMetadata] = None
    ) -> Dict[str, Any]:
        """
        Process a single PDF document through the Tier 1 pipeline.

        Args:
            pdf_path: Path to the input PDF file.
            doc_metadata: Optional external metadata.

        Returns:
            Dictionary containing stats, extracted elements, and final chunks.
        """
        start_time = time.time()
        doc_id = pdf_path.stem
        logger.info(f"🚀 Starting processing for: {pdf_path.name} (ID: {doc_id})")

        # 1. Setup Output Directory
        doc_output_dir = self.output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Extract Document-Level Metadata
        base_metadata = self._extract_pdf_metadata(pdf_path)
        if doc_metadata:
            base_metadata.update(doc_metadata.to_dict())

        # 3. Hybrid Extraction (Surya + VLM)
        # Delegates the entire page loop, detection, and enrichment to the optimized extractor
        try:
            logger.info("⚡ invoking HybridSuryaExtractor...")
            # Note: The extractor handles its own temp files internally
            all_elements = extract_hybrid_content_surya(str(pdf_path))
            
            if not all_elements:
                logger.warning(f"⚠️  No elements extracted from {doc_id}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_id}: {e}", exc_info=True)
            return {}

        # 4. Semantic Chunking
        # Groups the extracted elements based on the hierarchy built in Layer 3 of the extractor
        logger.info(f"🧩 Chunking {len(all_elements)} extracted elements...")
        chunks = self.chunker.chunk_elements(all_elements, doc_id=doc_id)

        # 5. Compile Statistics
        # Calculate total pages for stats (re-open briefly as extraction is decoupled)
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = doc.page_count
        except:
            total_pages = 0

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

        # 6. Construct Final Output Structure
        result = {
            "doc_id": doc_id,
            "metadata": base_metadata,
            "stats": stats,
            "chunks": chunks,
            "elements": all_elements 
        }

        # 7. Save to Disk
        self.save_processed_document(result, doc_id)
        
        return result

    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract basic technical metadata from PDF using PyMuPDF."""
        try:
            with fitz.open(pdf_path) as doc:
                meta = doc.metadata
                page_count = doc.page_count
            
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
            "chunking_strategy": self.strategy
        }

    def save_processed_document(self, processed_data: Dict[str, Any], doc_id: str):
        """Save the processed outputs (RAG chunks + lineage) to disk."""
        output_dir = self.output_dir / doc_id
        
        # 1. Save RAG-ready chunks (Optimized for Vector DB ingestion)
        chunks_data = [chunk.to_dict() for chunk in processed_data["chunks"]]
        rag_output = {
            "doc_id": doc_id,
            "metadata": processed_data["metadata"],
            "stats": processed_data["stats"],
            "chunks": chunks_data,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "strategy": self.strategy
            }
        }
        
        rag_chunks_path = output_dir / "rag_chunks.json"
        with open(rag_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(rag_output, f, indent=2, ensure_ascii=False)
        logger.debug(f"💾 Saved RAG chunks: {rag_chunks_path}")
            
        # 2. Save Full Lineage (For debugging/auditing)
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
            
            # Check skip logic
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
        
        return {
            "documents_processed": len(results),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_elements": total_elements,
            "total_processing_time_seconds": round(total_time, 2),
            "avg_time_per_doc": round(total_time / len(results), 2),
            "pages_per_second": round(total_pages / total_time, 2) if total_time > 0 else 0
        }

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process FOMC PDFs with Hybrid Surya Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=config.data.raw_data_dir, help="Input directory")
    parser.add_argument("--output-dir", type=Path, default=config.data.processed_data_dir, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size (tokens)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap (tokens)")
    parser.add_argument("--strategy", type=str, choices=["section_aware", "fixed_size"], default=None, help="Chunking strategy")
    parser.add_argument("--no-skip", action="store_false", dest="skip_existing", help="Force reprocess all")
    parser.add_argument("--metadata-file", type=Path, default=None, help="Path to metadata JSON")

    args = parser.parse_args()

    # Load metadata
    metadata = None
    metadata_file = args.metadata_file or (args.input_dir / "metadata.json")
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = {k: DocumentMetadata.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"⚠️  Could not load metadata: {e}")

    # Run Processor
    processor = PDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
        strategy=args.strategy
    )

    results = processor.process_all_documents(
        args.input_dir, 
        metadata,
        skip_existing=args.skip_existing
    )
    
    if results:
        summary = processor.get_processing_summary(results)
        logger.info(f"\n{'='*60}\n📊 PROCESSING SUMMARY\n{'='*60}")
        for k, v in summary.items():
            logger.info(f"{k.replace('_', ' ').title()}: {v}")

if __name__ == "__main__":
    main()