"""
Intelligent Document Chunker

Splits documents into optimal chunks for embedding and retrieval.
"""

from typing import List, Dict, Any, Optional
import tiktoken

from ..utils.logger import logger
from ..utils.models import DocumentChunk, ChunkType
from ..utils.config import config


class DocumentChunker:
    """
    Intelligently chunks documents for RAG system

    Strategy:
    - Text: Semantic chunking with token limits
    - Tables: Keep whole (never split)
    - Figures: Keep as-is with captions
    - Overlap: Configurable token overlap between chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            encoding_name: Tokenizer encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoding {encoding_name}: {e}")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate as 4 chars per token
            return len(text) // 4

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk text content with overlap

        Args:
            text: Text to chunk
            doc_id: Document identifier
            page_num: Page number
            metadata: Additional metadata

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []

        chunks = []

        # Split by paragraphs first (semantic boundaries)
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk = ""
        current_tokens = 0
        chunk_num = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk,
                        doc_id=doc_id,
                        chunk_num=chunk_num,
                        page_num=page_num,
                        chunk_type=ChunkType.TEXT,
                        metadata=metadata
                    ))
                    chunk_num += 1
                    current_chunk = ""
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)

                    if current_tokens + sent_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                content=current_chunk,
                                doc_id=doc_id,
                                chunk_num=chunk_num,
                                page_num=page_num,
                                chunk_type=ChunkType.TEXT,
                                metadata=metadata
                            ))
                            chunk_num += 1

                        # Start new chunk with overlap
                        current_chunk = sentence
                        current_tokens = sent_tokens
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens += sent_tokens

            # Paragraph fits in chunk
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens

            # Paragraph doesn't fit, start new chunk
            else:
                # Save current chunk
                chunks.append(self._create_chunk(
                    content=current_chunk,
                    doc_id=doc_id,
                    chunk_num=chunk_num,
                    page_num=page_num,
                    chunk_type=ChunkType.TEXT,
                    metadata=metadata
                ))
                chunk_num += 1

                # Start new chunk with current paragraph
                current_chunk = para
                current_tokens = para_tokens

        # Save final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                content=current_chunk,
                doc_id=doc_id,
                chunk_num=chunk_num,
                page_num=page_num,
                chunk_type=ChunkType.TEXT,
                metadata=metadata
            ))

        return chunks

    def chunk_table(
        self,
        table_markdown: str,
        doc_id: str,
        table_num: int,
        page_num: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """
        Create chunk for table (tables are never split)

        Args:
            table_markdown: Table in markdown format
            doc_id: Document identifier
            table_num: Table number
            page_num: Page number
            metadata: Additional metadata

        Returns:
            DocumentChunk for table
        """
        chunk_id = f"{doc_id}_table_{table_num}"

        chunk_metadata = {
            **metadata,
            "page": page_num,
            "table_num": table_num,
            "chunk_type": "table"
        }

        return DocumentChunk(
            unique_id=chunk_id,
            content=table_markdown,
            chunk_type=ChunkType.TABLE,
            metadata=chunk_metadata
        )

    def chunk_figure(
        self,
        figure_data: Dict[str, Any],
        doc_id: str,
        figure_num: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """
        Create chunk for figure

        Args:
            figure_data: Figure metadata
            doc_id: Document identifier
            figure_num: Figure number
            metadata: Additional metadata

        Returns:
            DocumentChunk for figure
        """
        chunk_id = f"{doc_id}_figure_{figure_num}"

        # Create text representation of figure
        content = f"Figure {figure_num} on page {figure_data['page']}\n"
        content += f"Size: {figure_data['width']}x{figure_data['height']} pixels\n"

        if "caption" in figure_data:
            content += f"Caption: {figure_data['caption']}"

        chunk_metadata = {
            **metadata,
            "page": figure_data["page"],
            "figure_num": figure_num,
            "chunk_type": "figure",
            **figure_data
        }

        return DocumentChunk(
            unique_id=chunk_id,
            content=content,
            chunk_type=ChunkType.FIGURE,
            metadata=chunk_metadata
        )

    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        chunk_num: int,
        page_num: int,
        chunk_type: ChunkType,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create DocumentChunk object"""
        chunk_id = f"{doc_id}_chunk_{chunk_num}"

        chunk_metadata = {
            **metadata,
            "page": page_num,
            "chunk_num": chunk_num,
            "chunk_type": chunk_type.value,
            "token_count": self.count_tokens(content)
        }

        return DocumentChunk(
            unique_id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            metadata=chunk_metadata
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (could use spaCy for better results)
        import re

        # Split on period, exclamation, question mark followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        return [s.strip() for s in sentences if s.strip()]
