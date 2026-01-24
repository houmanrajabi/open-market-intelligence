"""
Vector Database Interface (Production Ready)

Robust abstraction over ChromaDB for high-performance vector storage and retrieval.
Includes smart batching, upsert logic, and advanced filtering.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import math

import chromadb
from chromadb.config import Settings
# from chromadb.api.types import Include # Not strictly necessary if using strings

from src.utils.logger import logger
from src.utils.config import config
from src.utils.models import DocumentChunk


class VectorDB:
    """
    Features:
    - Smart batching for large ingestions
    - "Upsert" logic (delete-then-insert) for document updates
    - Advanced filtering and querying
    - Persistent storage management
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: Optional[Path] = None,
        embedding_dim: int = None,
        distance_metric: str = None
    ):
        """
        Initialize the Vector DB.

        Args:
            collection_name: Unique name for the collection
            persist_dir: Path to store database files
            embedding_dim: Expected dimension of embeddings (for validation)
            distance_metric: 'cosine', 'l2', or 'ip'
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir or config.vector_db.persist_dir
        self.embedding_dim = embedding_dim

        # Ensure persistence directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Initializing Vector DB at {self.persist_dir}")

        # Initialize Client with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        # Get or Create Collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            count = self.collection.count()
            logger.info(f"✅ Connected to collection '{collection_name}' ({count} documents)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            raise

    def upsert_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 500
    ):
        """
        Safely update documents in the DB (Idempotent).
        
        Logic:
        1. Identifies unique source documents (doc_id) from the new chunks.
        2. Deletes ALL existing chunks associated with those doc_ids to prevent duplicates.
        3. Inserts the new chunks in batches.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: Corresponding embedding vectors
            batch_size: Number of records to insert per batch
        """
        if not chunks:
            logger.warning("No chunks provided for upsert.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        # 1. Clean up old versions of these documents
        # We assume metadata contains 'doc_id'. If not, we skip deletion (append only).
        unique_doc_ids = set(chunk.metadata.get("doc_id") for chunk in chunks if chunk.metadata.get("doc_id"))
        
        if unique_doc_ids:
            logger.info(f"🔄 Cleaning up old chunks for {len(unique_doc_ids)} documents...")
            try:
                # Delete chunks where 'doc_id' is in our list
                self.collection.delete(
                    where={"doc_id": {"$in": list(unique_doc_ids)}}
                )
            except Exception as e:
                logger.warning(f"Delete failed (might be first run or empty DB): {e}")

        # 2. Prepare Data
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        total_chunks = len(chunks)
        batches = math.ceil(total_chunks / batch_size)

        logger.info(f"📥 Ingesting {total_chunks} chunks in {batches} batches...")

        # 3. Batch Insert
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_chunks)
            
            self.collection.add(
                ids=ids[start_idx:end_idx],
                embeddings=embeddings[start_idx:end_idx],
                documents=documents[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx]
            )
            logger.debug(f"   - Batch {i+1}/{batches} processed")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search.

        Args:
            query_embedding: The vector to search for.
            n_results: Max number of results.
            where: Metadata filters (e.g., {"doc_id": "123", "year": {"$gte": 2020}})
            where_document: Content filters (e.g., {"$contains": "inflation"})
            include_distances: Whether to return the similarity score.

        Returns:
            List of dictionaries containing the result data, sorted by relevance.
        """
        include = ["metadatas", "documents"]
        if include_distances:
            include.append("distances")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

        # Reformat Chroma's column-based response into a row-based list of dicts
        parsed_results = []
        if results["ids"]:
            num_hits = len(results["ids"][0])
            for i in range(num_hits):
                item = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if "distances" in results else None
                }
                parsed_results.append(item)

        return parsed_results

    def delete(self, where: Dict[str, Any]):
        """Delete documents matching a filter."""
        self.collection.delete(where=where)
        logger.info(f"Deleted documents matching: {where}")

    def count(self) -> int:
        """Return total document count."""
        return self.collection.count()

    def reset_collection(self):
        """DANGER: Deletes all data in the collection."""
        logger.warning(f"⚠️ Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )