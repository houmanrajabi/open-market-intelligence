import os
import json
import time
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from typing import List, Dict
import logging
from datetime import datetime
from collections import defaultdict

# Import your modules
from src.processing.extractors.qwen_extractor_a import analyze_layout
from src.processing.extractors.hybrid_extractor_a import extract_hybrid_content

import src.processing.extractors.hybrid_extractor as hybrid_module
print(f"🔍 Using hybrid_extractor from: {hybrid_module.__file__}")
print(f"🔍 Function signature: {hybrid_module.extract_hybrid_content.__code__.co_varnames}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/output/processing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunking configuration
TARGET_CHUNK_SIZE = 512  # tokens
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1024

# --- UTILITY FUNCTIONS ---

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)."""
    return len(text) // 4

def extract_document_metadata(pdf_path: str) -> Dict:
    """Extract metadata from PDF."""
    metadata = {
        "filename": os.path.basename(pdf_path),
        "processing_date": datetime.now().isoformat(),
        "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024)
    }
    
    try:
        doc = fitz.open(pdf_path)
        metadata.update({
            "page_count": doc.page_count,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "producer": doc.metadata.get("producer", "")
        })
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract PDF metadata: {e}")
    
    return metadata

# --- CHUNKING SYSTEM ---

class IntelligentChunker:
    """Chunks document elements for optimal RAG performance."""
    
    def __init__(self, target_size=512, min_size=100, max_size=1024):
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
    
    def should_merge(self, current_tokens: int, next_tokens: int) -> bool:
        """Decide if next element should be merged into current chunk."""
        combined = current_tokens + next_tokens
        
        # Don't exceed max size
        if combined > self.max_size:
            return False
        
        # If current chunk is too small, merge
        if current_tokens < self.min_size:
            return True
        
        # If adding next element keeps us near target, merge
        if combined <= self.target_size * 1.2:
            return True
        
        return False
    
    def create_chunk_metadata(self, elements: List[Dict]) -> Dict:
        """Generate metadata for a chunk."""
        
        # Aggregate metadata from constituent elements
        all_keywords = set()
        all_entities = {"organizations": set(), "locations": set(), "dates": set()}
        has_numerical = False
        quality_scores = []
        
        for elem in elements:
            elem_meta = elem.get("metadata", {})
            
            # Keywords
            if elem_meta.get("keywords"):
                all_keywords.update(elem_meta["keywords"])
            
            # Entities
            if elem_meta.get("entities"):
                for ent_type, ent_list in elem_meta["entities"].items():
                    if ent_type in all_entities:
                        all_entities[ent_type].update(ent_list)
            
            # Numerical data
            if elem_meta.get("contains_numerical_data"):
                has_numerical = True
            
            # Quality
            if elem.get("extraction_quality"):
                quality_scores.append(elem["extraction_quality"])
        
        return {
            "element_count": len(elements),
            "element_ids": [e["id"] for e in elements],
            "element_types": list(set(e["type"] for e in elements)),
            "pages": sorted(set(e["page"] for e in elements)),
            "section_anchor": elements[0].get("section_anchor", "Unknown"),
            "keywords": list(all_keywords)[:15],
            "entities": {k: list(v)[:10] for k, v in all_entities.items()},
            "contains_numerical_data": has_numerical,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "extraction_methods": list(set(e.get("extraction_method", "unknown") for e in elements))
        }
    
    def chunk_elements(self, elements: List[Dict]) -> List[Dict]:
        """Create intelligent chunks from extracted elements."""
        
        if not elements:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        
        for elem in elements:
            content = elem.get("content", "")
            elem_tokens = estimate_tokens(content)
            
            # Handle oversized elements
            if elem_tokens > self.max_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._finalize_chunk(current_chunk, chunk_id))
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large element
                split_chunks = self._split_large_element(elem, chunk_id)
                chunks.extend(split_chunks)
                chunk_id += len(split_chunks)
                continue
            
            # Decide whether to merge or create new chunk
            if current_chunk and not self.should_merge(current_tokens, elem_tokens):
                # Finalize current chunk
                chunks.append(self._finalize_chunk(current_chunk, chunk_id))
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            # Add element to current chunk
            current_chunk.append(elem)
            current_tokens += elem_tokens
        
        # Finalize last chunk
        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk, chunk_id))
        
        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks
    
    def _finalize_chunk(self, elements: List[Dict], chunk_id: int) -> Dict:
        """Convert element list to final chunk format."""
        
        # Combine content
        content_parts = []
        for elem in elements:
            elem_type = elem.get("type", "TEXT")
            content = elem.get("content", "")
            
            # Add type prefix for context
            if elem_type in ["TABLE", "FIGURE"]:
                prefix = f"[{elem_type}]"
                content_parts.append(f"{prefix} {content}")
            else:
                content_parts.append(content)
        
        combined_content = "\n\n".join(content_parts)
        
        return {
            "chunk_id": chunk_id,
            "content": combined_content,
            "token_count": estimate_tokens(combined_content),
            "metadata": self.create_chunk_metadata(elements)
        }
    
    def _split_large_element(self, element: Dict, start_chunk_id: int) -> List[Dict]:
        """Split an oversized element into multiple chunks."""
        
        content = element.get("content", "")
        
        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        
        chunks = []
        current_chunk_parts = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            
            if current_tokens + para_tokens > self.target_size and current_chunk_parts:
                # Create chunk
                chunks.append({
                    "chunk_id": start_chunk_id + len(chunks),
                    "content": "\n\n".join(current_chunk_parts),
                    "token_count": current_tokens,
                    "metadata": {
                        "element_count": 1,
                        "element_ids": [element["id"]],
                        "element_types": [element["type"]],
                        "pages": [element["page"]],
                        "section_anchor": element.get("section_anchor", "Unknown"),
                        "is_split": True,
                        "split_index": len(chunks)
                    }
                })
                current_chunk_parts = []
                current_tokens = 0
            
            current_chunk_parts.append(para)
            current_tokens += para_tokens
        
        # Final chunk
        if current_chunk_parts:
            chunks.append({
                "chunk_id": start_chunk_id + len(chunks),
                "content": "\n\n".join(current_chunk_parts),
                "token_count": current_tokens,
                "metadata": {
                    "element_count": 1,
                    "element_ids": [element["id"]],
                    "element_types": [element["type"]],
                    "pages": [element["page"]],
                    "section_anchor": element.get("section_anchor", "Unknown"),
                    "is_split": True,
                    "split_index": len(chunks)
                }
            })
        
        return chunks

# --- QUALITY MONITORING ---

class QualityMonitor:
    """Track extraction quality across the pipeline."""
    
    def __init__(self):
        self.stats = {
            "total_elements": 0,
            "by_type": defaultdict(int),
            "quality_scores": [],
            "validation_status": defaultdict(int),
            "extraction_methods": defaultdict(int),
            "failures": []
        }
    
    def record_element(self, element: Dict):
        """Record stats for an extracted element."""
        self.stats["total_elements"] += 1
        self.stats["by_type"][element.get("type", "UNKNOWN")] += 1
        
        if "extraction_quality" in element:
            self.stats["quality_scores"].append(element["extraction_quality"])
        
        if "validation_status" in element:
            self.stats["validation_status"][element["validation_status"]] += 1
        
        if "extraction_method" in element:
            self.stats["extraction_methods"][element["extraction_method"]] += 1
        
        # Track failures
        if element.get("extraction_quality", 1.0) < 0.3:
            self.stats["failures"].append({
                "id": element.get("id"),
                "type": element.get("type"),
                "page": element.get("page"),
                "quality": element.get("extraction_quality")
            })
    
    def get_report(self) -> Dict:
        """Generate quality report."""
        scores = self.stats["quality_scores"]
        
        return {
            "total_elements": self.stats["total_elements"],
            "elements_by_type": dict(self.stats["by_type"]),
            "average_quality": sum(scores) / len(scores) if scores else 0.0,
            "min_quality": min(scores) if scores else 0.0,
            "max_quality": max(scores) if scores else 0.0,
            "validation_status": dict(self.stats["validation_status"]),
            "extraction_methods": dict(self.stats["extraction_methods"]),
            "failure_count": len(self.stats["failures"]),
            "failures": self.stats["failures"][:10]  # Top 10 failures
        }
    
    def print_summary(self):
        """Print quality summary to console."""
        report = self.get_report()
        
        logger.info("=" * 60)
        logger.info("EXTRACTION QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Elements: {report['total_elements']}")
        logger.info(f"Average Quality: {report['average_quality']:.2%}")
        logger.info(f"\nElements by Type:")
        for elem_type, count in report['elements_by_type'].items():
            logger.info(f"  {elem_type}: {count}")
        
        logger.info(f"\nValidation Status:")
        for status, count in report['validation_status'].items():
            logger.info(f"  {status}: {count}")
        
        if report['failures']:
            logger.warning(f"\n⚠️  {report['failure_count']} low-quality extractions detected")
            logger.warning("Review pipeline.log for details")
        
        logger.info("=" * 60)

# --- MAIN PROCESSING FUNCTION ---

def process_document(filename: str):
    """Process PDF with full pipeline."""
    
    pdf_path = os.path.join(INPUT_DIR, filename)
    doc_id = filename.replace(".pdf", "")
    
    # Create output directory
    doc_output_dir = os.path.join(OUTPUT_DIR, doc_id)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    # Initialize
    quality_monitor = QualityMonitor()
    chunker = IntelligentChunker(
        target_size=TARGET_CHUNK_SIZE,
        min_size=MIN_CHUNK_SIZE,
        max_size=MAX_CHUNK_SIZE
    )
    
    # Extract document metadata
    doc_metadata = extract_document_metadata(pdf_path)
    
    final_doc_data = {
        "doc_id": doc_id,
        "original_file": filename,
        "metadata": doc_metadata,
        "processing_info": {
            "pipeline_version": "2.0",
            "processing_timestamp": datetime.now().isoformat()
        },
        "pages": [],
        "chunks": []
    }
    
    logger.info("=" * 60)
    logger.info(f"🚀 Processing Document: {filename}")
    logger.info(f"📄 Pages: {doc_metadata.get('page_count', 'unknown')}")
    logger.info("=" * 60)
    
    try:
        total_pages = doc_metadata.get("page_count", 0)
        if total_pages == 0:
            logger.error("Unable to determine page count")
            return
    except Exception as e:
        logger.error(f"Failed to open PDF {filename}: {e}")
        return
    
    # Initialize context
    last_anchor_context = None
    all_elements = []
    
    # Process page by page
    for page_num in range(1, total_pages + 1):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"📄 Processing Page {page_num}/{total_pages}")
        logger.info(f"{'─' * 60}")
        
        try:
            # Convert page to image
            images = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=page_num,
                last_page=page_num
            )
            
            if not images:
                logger.warning(f"No image generated for page {page_num}")
                continue
            
            current_img = images[0]
            
            # Save temp image
            temp_img_path = os.path.join(doc_output_dir, f"temp_proc_{page_num}.jpg")
            current_img.save(temp_img_path, quality=95)
            
            # Phase 1: Layout Detection
            logger.info("🔍 Phase 1: Layout Detection")
            layout_map = analyze_layout(temp_img_path)
            
            if not layout_map.get("layout"):
                logger.warning(f"No layout detected on page {page_num}")
                continue
            
            # Phase 2: Hybrid Extraction
            logger.info("⚙️  Phase 2: Content Extraction")
            page_elements, last_anchor_context = extract_hybrid_content(
                pdf_path=pdf_path,
                page_num=page_num,
                layout_map=layout_map,
                temp_image_path=temp_img_path,
                output_dir=doc_output_dir,
                previous_context=last_anchor_context,
                document_metadata=doc_metadata
            )
            
            # Record quality stats
            for elem in page_elements:
                quality_monitor.record_element(elem)
            
            # Store page data
            final_doc_data["pages"].append({
                "page_num": page_num,
                "element_count": len(page_elements),
                "elements": page_elements
            })
            
            # Accumulate for chunking
            all_elements.extend(page_elements)
            
            # Cleanup temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            logger.info(f"✅ Page {page_num} complete: {len(page_elements)} elements extracted")
        
        except Exception as e:
            logger.error(f"❌ Error processing page {page_num}: {e}", exc_info=True)
            continue
    
    # Phase 3: Intelligent Chunking
    logger.info("\n" + "=" * 60)
    logger.info("🧩 Phase 3: Creating RAG Chunks")
    logger.info("=" * 60)
    
    chunks = chunker.chunk_elements(all_elements)
    final_doc_data["chunks"] = chunks
    final_doc_data["processing_info"]["chunk_count"] = len(chunks)
    
    # Add quality report
    quality_report = quality_monitor.get_report()
    final_doc_data["quality_report"] = quality_report
    
    # Save outputs
    logger.info("\n💾 Saving outputs...")
    
    # Full structure JSON
    full_json_path = os.path.join(doc_output_dir, "full_structure.json")
    with open(full_json_path, "w", encoding='utf-8') as f:
        json.dump(final_doc_data, f, indent=2, ensure_ascii=False)
    
    # RAG-optimized chunks JSON (for vector DB ingestion)
    chunks_json_path = os.path.join(doc_output_dir, "rag_chunks.json")
    with open(chunks_json_path, "w", encoding='utf-8') as f:
        json.dump({
            "doc_id": doc_id,
            "metadata": doc_metadata,
            "chunks": chunks
        }, f, indent=2, ensure_ascii=False)
    
    # Quality report
    quality_report_path = os.path.join(doc_output_dir, "quality_report.json")
    with open(quality_report_path, "w", encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    quality_monitor.print_summary()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Processing Complete: {doc_id}")
    logger.info(f"📁 Output Directory: {doc_output_dir}")
    logger.info(f"📊 Total Elements: {len(all_elements)}")
    logger.info(f"🧩 Total Chunks: {len(chunks)}")
    logger.info(f"📈 Avg Quality: {quality_report['average_quality']:.2%}")
    logger.info("=" * 60)

# --- RUNNER ---

if __name__ == "__main__":
    target_file = "fomcprojtabl20200610.pdf"
    
    if os.path.exists(os.path.join(INPUT_DIR, target_file)):
        start_time = time.time()
        
        try:
            process_document(target_file)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        logger.info(f"\n⏱️  Total Processing Time: {elapsed:.2f}s")
    else:
        logger.error(f"❌ File not found: {os.path.join(INPUT_DIR, target_file)}")