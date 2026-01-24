"""
Ingest Processed Documents to Vector Database

This script loads processed RAG chunks from the ingestion pipeline and stores them
in ChromaDB with embeddings for retrieval.

Usage:
    python -m scripts.ingest_to_vectordb --processed-dir data/processed --reset
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from src.retrieval.embedder import Embedder
from src.retrieval.vector_db import VectorDB
from src.utils.config import config
from src.utils.logger import logger


class DocumentIngester:
    """
    Handles ingestion of processed documents into vector database.
    """

    def __init__(
        self,
        collection_name: str = None,
        reset_db: bool = False
    ):
        """
        Initialize the ingester.

        Args:
            collection_name: Name of the vector DB collection
            reset_db: If True, clears existing collection
        """
        self.collection_name = collection_name or config.vector_db.collection_name

        # Initialize embedder
        logger.info("Initializing embedder...")
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            batch_size=config.embedding.batch_size
        )

        # Initialize vector DB
        logger.info(f"Initializing vector DB: {self.collection_name}")
        self.vector_db = VectorDB(
            collection_name=self.collection_name,
            persist_dir=config.vector_db.persist_dir,
            embedding_dim=self.embedder.embedding_dim,
            distance_metric=config.vector_db.distance_metric
        )

        if reset_db:
            logger.warning("Resetting collection...")
            self.vector_db.reset_collection()

        self.stats = {
            "documents_processed": 0,
            "chunks_ingested": 0,
            "failed_documents": 0,
            "total_tokens": 0
        }

    def load_processed_document(self, doc_dir: Path) -> Dict[str, Any]:
        """
        Load a processed document from its directory.

        Args:
            doc_dir: Directory containing rag_chunks.json

        Returns:
            Dictionary with doc_id, metadata, and chunks
        """
        rag_chunks_path = doc_dir / "rag_chunks.json"

        if not rag_chunks_path.exists():
            logger.warning(f"No rag_chunks.json found in {doc_dir}")
            return None

        try:
            with open(rag_chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug(f"Loaded document: {data['doc_id']} ({len(data['chunks'])} chunks)")
            return data

        except Exception as e:
            logger.error(f"Failed to load {rag_chunks_path}: {e}")
            return None

    def prepare_chunks_for_ingestion(
        self,
        doc_data: Dict[str, Any]
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Prepare chunks for vector DB ingestion.

        Args:
            doc_data: Document data from rag_chunks.json

        Returns:
            Tuple of (chunk_texts, chunk_metadatas, chunk_ids)
        """
        doc_id = doc_data["doc_id"]
        doc_metadata = doc_data["metadata"]

        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []

        for chunk in doc_data["chunks"]:
            # Extract text content
            chunk_texts.append(chunk["content"])

            # Build rich metadata
            metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk["chunk_type"],
                "section_anchor": chunk["section_anchor"],
                "token_count": chunk["token_count"],

                # Document-level metadata
                "doc_title": doc_metadata.get("title", ""),
                "doc_date": doc_metadata.get("creation_date", ""),
                "source_file": doc_metadata.get("source_file", ""),

                # Page and element tracking
                "pages": json.dumps(chunk["pages"]),  # JSON string for Chroma
                "element_ids": json.dumps(chunk["element_ids"]),
                "element_types": json.dumps(chunk["element_types"]),

                # Chunk-specific metadata
                "keywords": json.dumps(chunk["metadata"].get("keywords", [])),
                "source_count": chunk["metadata"].get("source_count", 0)
            }

            chunk_metadatas.append(metadata)
            chunk_ids.append(chunk["chunk_id"])

        return chunk_texts, chunk_metadatas, chunk_ids

    def ingest_document(self, doc_dir: Path) -> bool:
        """
        Ingest a single document into the vector database.

        Args:
            doc_dir: Path to processed document directory

        Returns:
            True if successful, False otherwise
        """
        doc_data = self.load_processed_document(doc_dir)

        if not doc_data:
            self.stats["failed_documents"] += 1
            return False

        try:
            # Prepare chunks
            chunk_texts, chunk_metadatas, chunk_ids = self.prepare_chunks_for_ingestion(doc_data)

            if not chunk_texts:
                logger.warning(f"No chunks to ingest for {doc_data['doc_id']}")
                return False

            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")

            # Generate embeddings
            embeddings = self.embedder.encode_documents(chunk_texts)

            # Convert to list format for ChromaDB
            embeddings_list = embeddings.tolist()

            logger.info(f"Upserting {len(chunk_texts)} chunks to vector DB...")

            # Upsert to vector DB
            # Note: We need to convert our chunks to DocumentChunk format
            from src.utils.models import DocumentChunk, ChunkType

            doc_chunks = []
            for i, (chunk_id, text, metadata) in enumerate(zip(chunk_ids, chunk_texts, chunk_metadatas)):
                doc_chunk = DocumentChunk(
                    unique_id=chunk_id,
                    content=text,
                    chunk_type=ChunkType(metadata["chunk_type"]),
                    metadata=metadata
                )
                doc_chunks.append(doc_chunk)

            self.vector_db.upsert_documents(
                chunks=doc_chunks,
                embeddings=embeddings_list,
                batch_size=config.vector_db.batch_size
            )

            # Update stats
            self.stats["documents_processed"] += 1
            self.stats["chunks_ingested"] += len(chunk_texts)
            self.stats["total_tokens"] += sum(m["token_count"] for m in chunk_metadatas)

            logger.info(
                f"✅ Successfully ingested {doc_data['doc_id']}: "
                f"{len(chunk_texts)} chunks, "
                f"{sum(m['token_count'] for m in chunk_metadatas)} tokens"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to ingest {doc_dir.name}: {e}", exc_info=True)
            self.stats["failed_documents"] += 1
            return False

    def ingest_all(self, processed_dir: Path) -> Dict[str, Any]:
        """
        Ingest all processed documents from a directory.

        Args:
            processed_dir: Directory containing processed document subdirectories

        Returns:
            Dictionary with ingestion statistics
        """
        if not processed_dir.exists():
            logger.error(f"Processed directory not found: {processed_dir}")
            return self.stats

        # Find all document directories
        doc_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if not doc_dirs:
            logger.warning(f"No document directories found in {processed_dir}")
            return self.stats

        logger.info(f"Found {len(doc_dirs)} documents to ingest")

        # Process each document with progress bar
        for doc_dir in tqdm(doc_dirs, desc="Ingesting documents"):
            self.ingest_document(doc_dir)

        # Final statistics
        logger.info("\n" + "=" * 70)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Documents processed: {self.stats['documents_processed']}")
        logger.info(f"Documents failed: {self.stats['failed_documents']}")
        logger.info(f"Total chunks ingested: {self.stats['chunks_ingested']}")
        logger.info(f"Total tokens: {self.stats['total_tokens']}")
        logger.info(f"Vector DB size: {self.vector_db.count()} documents")
        logger.info("=" * 70)

        return self.stats

    def verify_ingestion(self, sample_queries: List[str] = None):
        """
        Verify ingestion by running sample queries.

        Args:
            sample_queries: List of test queries (optional)
        """
        if sample_queries is None:
            sample_queries = [
                "What was the GDP growth projection for 2023?",
                "What did Chair Powell say about inflation?",
                "What are the unemployment rate projections?"
            ]

        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION: Sample Queries")
        logger.info("=" * 70)

        for query in sample_queries:
            logger.info(f"\nQuery: {query}")

            # Embed query
            query_embedding = self.embedder.encode_queries([query])[0]

            # Search
            results = self.vector_db.query(
                query_embedding=query_embedding.tolist(),
                n_results=3,
                include_distances=True
            )

            # Display results
            for i, result in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    Document: {result['metadata'].get('doc_id', 'Unknown')}")
                logger.info(f"    Section: {result['metadata'].get('section_anchor', 'Unknown')}")
                logger.info(f"    Score: {result['score']:.4f}")
                logger.info(f"    Content preview: {result['content'][:150]}...")

        logger.info("\n" + "=" * 70)


def main():
    """Main entry point for the ingestion script."""

    parser = argparse.ArgumentParser(
        description="Ingest processed documents into vector database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.data.processed_data_dir,
        help="Directory containing processed documents"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=config.vector_db.collection_name,
        help="Vector DB collection name"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset collection before ingestion (deletes existing data)"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification queries after ingestion"
    )

    parser.add_argument(
        "--doc-id",
        type=str,
        help="Process only specific document ID"
    )

    args = parser.parse_args()

    # Initialize ingester
    ingester = DocumentIngester(
        collection_name=args.collection,
        reset_db=args.reset
    )

    # Ingest documents
    if args.doc_id:
        # Single document
        doc_dir = args.processed_dir / args.doc_id
        if not doc_dir.exists():
            logger.error(f"Document directory not found: {doc_dir}")
            return

        logger.info(f"Processing single document: {args.doc_id}")
        ingester.ingest_document(doc_dir)
    else:
        # All documents
        ingester.ingest_all(args.processed_dir)

    # Verification
    if args.verify:
        ingester.verify_ingestion()

    logger.info("\n✅ Ingestion script completed successfully!")


if __name__ == "__main__":
    main()
