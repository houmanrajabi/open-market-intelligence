"""
Enhanced Document Chunker - Production Ready (v2.1)

A robust, section-aware chunking engine for RAG pipelines.
Handles multimodal content, respects document hierarchy, and preserves semantic context.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from collections import defaultdict

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Configure module-level logger
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Enumeration of content types for chunks."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    MULTIMODAL = "multimodal"

@dataclass
class DocumentChunk:
    """
    Represents a discrete chunk of a document with lineage tracking.
    """
    chunk_id: str
    content: str
    chunk_type: ChunkType
    token_count: int
    section_anchor: str = "Document Content"
    pages: List[int] = field(default_factory=list)
    element_ids: List[str] = field(default_factory=list)
    element_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "token_count": self.token_count,
            "section_anchor": self.section_anchor,
            "pages": self.pages,
            "element_ids": self.element_ids,
            "element_types": self.element_types,
            "metadata": self.metadata
        }

class SentenceSplitter:
    """
    Rule-based sentence splitter designed for noisy PDF text.
    Handles abbreviations and common header formatting issues.
    """
    
    # Abbreviations that rarely end a sentence
    ABBREVIATIONS = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'inc', 'ltd', 'corp', 'co',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'u.s', 'u.k', 'e.g', 'i.e', 'etc', 'vs', 'ph.d', 'b.a', 'm.a',
        'fig', 'no', 'vol', 'pp', 'ed', 'al'
    }
    
    def __init__(self):
        # 1. Split on punctuation followed by space and Uppercase/Quote
        self.split_pattern = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+(?=[A-Z"\'])',
            re.MULTILINE
        )
        # 2. Safety check for abbreviations
        self.abbrev_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(abbr) for abbr in self.ABBREVIATIONS) + r')\.$',
            re.IGNORECASE
        )

    def split(self, text: str) -> List[str]:
        if not text:
            return []
            
        # Pre-process: Treat double newlines as hard breaks (often headers)
        # This fixes cases like "4.0 Conclusion\nThe project..."
        segments = text.split('\n\n')
        final_sentences = []
        
        for segment in segments:
            # Clean single newlines within paragraphs
            clean_segment = segment.replace('\n', ' ').strip()
            if not clean_segment:
                continue
                
            raw_sentences = self.split_pattern.split(clean_segment)
            
            # Merge back false positives (abbreviations)
            buffer = ""
            for s in raw_sentences:
                s = s.strip()
                if not s: continue
                
                buffer += " " + s if buffer else s
                
                # If buffer ends with abbreviation, keep accumulating
                if self.abbrev_pattern.search(buffer):
                    continue
                else:
                    final_sentences.append(buffer)
                    buffer = ""
            
            if buffer:
                final_sentences.append(buffer)
                
        return final_sentences

class DocumentChunker:
    """
    Production-ready Document Chunker.
    
    Key Features:
    - Section Semantic Awareness: Groups text by headings.
    - Smart Overlap: Context flows across chunks but respects hard boundaries.
    - Multimodal: Preserves tables/figures as distinct entities or merged context.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        encoding_name: str = "cl100k_base",
        strategy: str = "section_aware",
        quality_threshold: float = 0.5,
        preserve_large_tables: bool = True,
        small_table_threshold: int = 150,
        multimodal_chunks: bool = True,
        similarity_threshold: float = 0.7
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.preserve_large_tables = preserve_large_tables
        self.small_table_threshold = small_table_threshold
        self.multimodal_chunks = multimodal_chunks
        self.similarity_threshold = similarity_threshold
        
        # Tokenizer Setup
        self.tokenizer = None
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Tiktoken error: {e}. Falling back to char count.")
        
        self.sentence_splitter = SentenceSplitter()
        
        # Internal Stats
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            "chunks_created": 0,
            "elements_processed": 0,
            "tables_preserved": 0,
            "figures_processed": 0,
            "splits_performed": 0
        }

    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text: return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Robust fallback

    def chunk_elements(self, elements: List[Dict[str, Any]], doc_id: str = "doc") -> List[DocumentChunk]:
        """
        Primary execution method.
        """
        self._reset_stats()
        
        if not elements:
            logger.warning(f"No elements provided for {doc_id}")
            return []

        # Optimization: Shallow copy + Pre-calculate tokens in place
        # We assume elements is a list of dicts. We won't mutate the *structure* of input,
        # but we might add the 'token_count' key to the dicts for performance.
        # If strict immutability is required, use elements = [e.copy() for e in elements]
        processed_elements = []
        for e in elements:
            # Safe shallow copy for internal use
            elem = e.copy()
            if "token_count" not in elem:
                elem["token_count"] = self.count_tokens(elem.get("content", ""))
            processed_elements.append(elem)

        if self.strategy == "section_aware":
            chunks = self._chunk_by_sections(processed_elements, doc_id)
        elif self.strategy == "fixed_size":
            chunks = self._chunk_fixed_size(processed_elements, doc_id)
        else:
            # Fallback for unknown strategies
            chunks = self._chunk_by_sections(processed_elements, doc_id)

        self.stats["chunks_created"] = len(chunks)
        self.stats["elements_processed"] = len(elements)
        return self._validate_chunks(chunks)

    def _chunk_by_sections(self, elements: List[Dict[str, Any]], doc_id: str) -> List[DocumentChunk]:
        """Group by section, then chunk, maintaining overlap context between groups."""
        chunks = []
        chunk_counter = 0
        
        # Grouping
        sections = []
        current_section = None
        current_elements = []
        
        for elem in elements:
            anchor = elem.get("section_anchor", "Document Content")
            if anchor != current_section:
                if current_elements:
                    sections.append((current_section, current_elements))
                current_section = anchor
                current_elements = [elem]
            else:
                current_elements.append(elem)
        if current_elements:
            sections.append((current_section, current_elements))
            
        # Processing with Overlap Flow
        previous_overlap_text = None
        
        for section_anchor, section_elems in sections:
            section_chunks, next_overlap = self._process_section(
                section_elems, 
                section_anchor, 
                doc_id, 
                chunk_counter, 
                previous_overlap_text
            )
            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)
            previous_overlap_text = next_overlap
            
        return chunks

    def _process_section(
        self, 
        elements: List[Dict[str, Any]], 
        section_anchor: str, 
        doc_id: str, 
        start_id: int, 
        overlap_text: Optional[str]
    ) -> Tuple[List[DocumentChunk], Optional[str]]:
        
        chunks = []
        current_group = []
        current_tokens = 0
        chunk_id = start_id
        
        # Inject Overlap (Context from previous section)
        if overlap_text:
            overlap_tokens = self.count_tokens(overlap_text)
            if overlap_tokens < self.chunk_overlap * 1.5: # Sanity check size
                current_group.append({
                    "type": "TEXT", "content": overlap_text, 
                    "token_count": overlap_tokens, "is_overlap": True,
                    "metadata": {}
                })
                current_tokens += overlap_tokens

        for elem in elements:
            elem_type = elem.get("type", "TEXT")
            elem_tokens = elem.get("token_count", 0)
            
            if elem_tokens == 0: continue

            # --- 1. Special Element Handling (Tables/Figures) ---
            if elem_type in ["TABLE", "FIGURE"]:
                preserve = False
                if elem_type == "TABLE":
                    if self.preserve_large_tables and elem_tokens > self.small_table_threshold:
                        preserve = True
                        self.stats["tables_preserved"] += 1
                elif elem_type == "FIGURE":
                     if not self.multimodal_chunks: 
                         preserve = True

                if preserve:
                    # Flush pending text
                    if current_group:
                        chunks.append(self._create_chunk(current_group, doc_id, chunk_id, section_anchor))
                        chunk_id += 1
                        current_group = []
                        current_tokens = 0
                    
                    # Create standalone element chunk
                    chunks.append(self._create_chunk([elem], doc_id, chunk_id, section_anchor))
                    chunk_id += 1
                    continue
            
            # --- 2. Oversized Element Handling ---
            if elem_tokens > self.max_chunk_size:
                if current_group:
                    chunks.append(self._create_chunk(current_group, doc_id, chunk_id, section_anchor))
                    chunk_id += 1
                    current_group = []
                    current_tokens = 0
                
                splits = self._split_large_element(elem, doc_id, chunk_id, section_anchor)
                chunks.extend(splits)
                chunk_id += len(splits)
                self.stats["splits_performed"] += 1
                continue

            # --- 3. Merging Logic ---
            if not self._should_merge(current_group, current_tokens, elem, elem_tokens):
                # Finalize current chunk
                chunks.append(self._create_chunk(current_group, doc_id, chunk_id, section_anchor))
                chunk_id += 1
                
                # Setup next chunk with overlap
                # We need to extract the last N tokens from the *Text* of the just-created chunk
                overlap_content = self._extract_overlap_from_group(current_group)
                current_group = []
                current_tokens = 0
                
                if overlap_content:
                    o_tokens = self.count_tokens(overlap_content)
                    current_group.append({
                        "type": "TEXT", "content": overlap_content,
                        "token_count": o_tokens, "is_overlap": True, "metadata": {}
                    })
                    current_tokens = o_tokens
            
            current_group.append(elem)
            current_tokens += elem_tokens
            
        # Final flush
        if current_group:
            chunks.append(self._create_chunk(current_group, doc_id, chunk_id, section_anchor))
            
        # Generate overlap for NEXT section
        next_overlap = None
        if chunks:
            # Only use text from the last chunk for overlap, ignore trailing tables
            last_chunk = chunks[-1]
            if last_chunk.chunk_type == ChunkType.TEXT or last_chunk.chunk_type == ChunkType.MULTIMODAL:
                next_overlap = self._extract_trailing_text(last_chunk.content, self.chunk_overlap)
                
        return chunks, next_overlap

    def _should_merge(self, current_group, current_tokens, new_elem, new_tokens) -> bool:
        """Decide whether to add new_elem to current_group."""
        if not current_group: return True
        
        combined = current_tokens + new_tokens
        
        if combined > self.max_chunk_size: return False
        if combined <= self.chunk_size: return True
        
        # Similarity Booster: If semantically similar, allow 20% overflow
        if combined <= self.chunk_size * 1.2:
            sim = self._calculate_similarity(current_group[-1], new_elem)
            if sim > self.similarity_threshold:
                return True
                
        return False

    def _calculate_similarity(self, elem1: Dict, elem2: Dict) -> float:
        """Heuristic semantic similarity."""
        score = 0.0
        if elem1.get("section_anchor") == elem2.get("section_anchor"):
            score += 0.4
        
        # Keyword overlap
        k1 = set(elem1.get("metadata", {}).get("keywords", []))
        k2 = set(elem2.get("metadata", {}).get("keywords", []))
        if k1 and k2:
            overlap = len(k1 & k2) / len(k1 | k2)
            score += overlap * 0.4
            
        return score

    def _extract_overlap_from_group(self, group: List[Dict]) -> Optional[str]:
        """Extract text from the end of a group of elements."""
        # Iterate backwards to find the last valid text element
        for elem in reversed(group):
            if elem.get("type") == "TEXT" and not elem.get("is_overlap"):
                return self._extract_trailing_text(elem.get("content", ""), self.chunk_overlap)
        return None

    def _extract_trailing_text(self, text: str, target_tokens: int) -> Optional[str]:
        """Get the last N tokens of text using sentence boundaries."""
        sentences = self.sentence_splitter.split(text)
        if not sentences: return None
        
        buffer = []
        count = 0
        for sent in reversed(sentences):
            t = self.count_tokens(sent)
            if count + t > target_tokens and buffer:
                break
            buffer.insert(0, sent)
            count += t
        return " ".join(buffer) if buffer else None

    def _split_large_element(self, elem: Dict, doc_id: str, start_id: int, anchor: str) -> List[DocumentChunk]:
        """Splits large text into chunks."""
        content = elem.get("content", "")
        sentences = self.sentence_splitter.split(content)
        chunks = []
        
        current_sents = []
        current_toks = 0
        
        for sent in sentences:
            t = self.count_tokens(sent)
            if current_toks + t > self.chunk_size and current_sents:
                # Flush
                text = " ".join(current_sents)
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{start_id + len(chunks)}",
                    content=text, chunk_type=ChunkType.TEXT, token_count=current_toks,
                    section_anchor=anchor,
                    element_ids=[elem.get("id")],
                    metadata=elem.get("metadata", {})
                ))
                # Overlap logic for splitting
                overlap_sent = current_sents[-1]
                current_sents = [overlap_sent]
                current_toks = self.count_tokens(overlap_sent)
            
            current_sents.append(sent)
            current_toks += t
            
        if current_sents:
             chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{start_id + len(chunks)}",
                content=" ".join(current_sents), chunk_type=ChunkType.TEXT, 
                token_count=current_toks, section_anchor=anchor,
                element_ids=[elem.get("id")],
                metadata=elem.get("metadata", {})
            ))
        return chunks

    def _create_chunk(self, elements: List[Dict], doc_id: str, chunk_id: int, anchor: str) -> DocumentChunk:
        """Assemble a DocumentChunk."""
        parts = []
        types = set()
        pages = set()
        ids = []
        
        for e in elements:
            if e.get("is_overlap"): 
                # Don't track overlap markers in metadata, just content
                parts.append(e["content"])
                continue
                
            ctype = e.get("type", "TEXT")
            types.add(ctype)
            if e.get("page"): pages.add(e["page"])
            ids.append(e.get("id"))
            
            content = e.get("content", "")
            if ctype == "TABLE":
                parts.append(f"[TABLE]\n{content}")
            elif ctype == "FIGURE":
                parts.append(f"[FIGURE]\n{content}")
            else:
                parts.append(content)
                
        # Determine strict type
        if len(types) > 1: final_type = ChunkType.MULTIMODAL
        elif "TABLE" in types: final_type = ChunkType.TABLE
        elif "FIGURE" in types: final_type = ChunkType.FIGURE
        else: final_type = ChunkType.TEXT
            
        full_text = "\n\n".join(parts)
        
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_id}",
            content=full_text,
            chunk_type=final_type,
            token_count=self.count_tokens(full_text),
            section_anchor=anchor,
            pages=sorted(list(pages)),
            element_ids=[i for i in ids if i],
            element_types=list(types),
            metadata=self._aggregate_metadata(elements)
        )

    def _aggregate_metadata(self, elements: List[Dict]) -> Dict[str, Any]:
        """Aggregate metadata from source elements (excluding overlap markers)."""
        meta_acc = defaultdict(list)
        keywords = set()
        
        real_elements = [e for e in elements if not e.get("is_overlap")]
        
        for e in real_elements:
            m = e.get("metadata", {})
            if "keywords" in m: keywords.update(m["keywords"])
            # Add other aggregations as needed
            
        return {
            "source_count": len(real_elements),
            "keywords": list(keywords)[:10]
        }

    def _validate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter out empty or broken chunks."""
        return [c for c in chunks if c.content and len(c.content.strip()) > 5]

    def _chunk_fixed_size(self, elements: List[Dict], doc_id: str) -> List[DocumentChunk]:
        """Fallback: Flatten everything and chunk by size."""
        # Simple implementation that ignores section boundaries
        full_text = "\n\n".join([e.get("content", "") for e in elements])
        # Wrap in a dummy element to reuse the split logic
        dummy_elem = {"content": full_text, "type": "TEXT", "metadata": {}, "id": "full_doc"}
        return self._split_large_element(dummy_elem, doc_id, 0, "Full Document")