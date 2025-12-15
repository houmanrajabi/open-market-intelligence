"""Data models for FOMC RAG system"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path


class ChunkType(Enum):
    """Type of document chunk"""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class DocumentType(Enum):
    """Type of FOMC document"""
    STATEMENT = "statement"
    MINUTES = "minutes"
    PRESS_CONFERENCE = "presconf"
    SEP = "sep"
    IMPLEMENTATION = "implementation"


@dataclass
class DocumentChunk:
    """
    Represents a chunk of document content

    Attributes:
        unique_id: Unique identifier for the chunk
        content: The actual text/table/figure content
        chunk_type: Type of chunk (text, table, figure)
        metadata: Additional metadata
    """
    unique_id: str
    content: str
    chunk_type: ChunkType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert chunk_type to enum if string"""
        if isinstance(self.chunk_type, str):
            self.chunk_type = ChunkType(self.chunk_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "unique_id": self.unique_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary"""
        return cls(
            unique_id=data["unique_id"],
            content=data["content"],
            chunk_type=ChunkType(data["chunk_type"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class DocumentMetadata:
    """
    Metadata for FOMC documents

    Attributes:
        doc_id: Unique document identifier
        doc_type: Type of document
        date: Document date
        file_path: Path to source file
        meeting_date: FOMC meeting date
        pages: Number of pages
        download_date: When document was downloaded
    """
    doc_id: str
    doc_type: DocumentType
    date: datetime
    file_path: Path
    meeting_date: Optional[datetime] = None
    pages: int = 0
    download_date: datetime = field(default_factory=datetime.now)
    url: Optional[str] = None

    def __post_init__(self):
        """Convert types if needed"""
        if isinstance(self.doc_type, str):
            self.doc_type = DocumentType(self.doc_type)
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "date": self.date.isoformat(),
            "file_path": str(self.file_path),
            "meeting_date": self.meeting_date.isoformat() if self.meeting_date else None,
            "pages": self.pages,
            "download_date": self.download_date.isoformat(),
            "url": self.url
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary"""
        return cls(
            doc_id=data["doc_id"],
            doc_type=DocumentType(data["doc_type"]),
            date=datetime.fromisoformat(data["date"]),
            file_path=Path(data["file_path"]),
            meeting_date=datetime.fromisoformat(data["meeting_date"]) if data.get("meeting_date") else None,
            pages=data.get("pages", 0),
            download_date=datetime.fromisoformat(data.get("download_date", datetime.now().isoformat())),
            url=data.get("url")
        )


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    distances: List[float]
    latency: float
    approach: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "distances": self.distances,
            "latency": self.latency,
            "approach": self.approach
        }


@dataclass
class EvalQuestion:
    """Single test question with ground truth"""
    question_id: str
    query: str
    expected_answer: str
    question_type: str  # factual, comparison, trend, calculation, conceptual, multi-hop
    difficulty: int  # 1-5
    requires_table: bool
    requires_multi_doc: bool
    source_documents: List[str]  # Document IDs
    key_values: Optional[List[str]] = None  # For numerical verification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_id": self.question_id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "requires_table": self.requires_table,
            "requires_multi_doc": self.requires_multi_doc,
            "source_documents": self.source_documents,
            "key_values": self.key_values
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalQuestion":
        """Create from dictionary"""
        return cls(
            question_id=data["question_id"],
            query=data["query"],
            expected_answer=data["expected_answer"],
            question_type=data["question_type"],
            difficulty=data["difficulty"],
            requires_table=data["requires_table"],
            requires_multi_doc=data["requires_multi_doc"],
            source_documents=data["source_documents"],
            key_values=data.get("key_values")
        )

if __name__ == "__main__":
    print("This module defines data models for the FOMC RAG system.")