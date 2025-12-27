"""
End-to-End RAG System with Uncertainty-Aware Retrieval

This module implements the complete RAG pipeline with:
1. Vector-based retrieval (ChromaDB)
2. Uncertainty quantification (entropy-based)
3. Adaptive retrieval expansion (k=5 -> k=10)
4. Abstention mechanism (high uncertainty)

Architecture:
    Query → Embed → Retrieve(k=5) → Generate → Check Entropy
                                        ↓
                                    High Entropy?
                                   /           \
                              Yes /             \ No
                                 ↓               ↓
                        Expand(k=10)         Return Answer
                        Generate Again
                              ↓
                        Check Entropy Again
                              ↓
                        Very High? → Abstain
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from src.retrieval.embedder import Embedder
from src.retrieval.vector_db import VectorDB
from src.retrieval.llm_interface import LlamaInterface, GenerationResult
from src.retrieval.uncertainty_engine import UncertaintyGate, EntropyResult, UncertaintyLevel
from src.utils.config import config
from src.utils.logger import logger


@dataclass
class RAGResponse:
    """
    Complete RAG response with uncertainty metrics.

    Attributes:
        question: Original user query
        answer: Generated answer text
        sources: Retrieved document chunks
        uncertainty_level: LOW, MEDIUM, or HIGH
        entropy_score: Average sequence entropy
        retrieval_expanded: Whether retrieval was expanded
        abstained: Whether system abstained from answering
        retrieval_count: Number of chunks retrieved (5 or 10)
        confidence: Confidence level (HIGH, MEDIUM, LOW)
        latency: Total response time in seconds
        metadata: Additional debug information
    """
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    uncertainty_level: str
    entropy_score: float
    retrieval_expanded: bool
    abstained: bool
    retrieval_count: int
    confidence: str
    latency: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "uncertainty_level": self.uncertainty_level,
            "entropy_score": self.entropy_score,
            "retrieval_expanded": self.retrieval_expanded,
            "abstained": self.abstained,
            "retrieval_count": self.retrieval_count,
            "confidence": self.confidence,
            "latency": self.latency,
            "metadata": self.metadata
        }


class UncertaintyAwareRAG:
    """
    Main RAG system with uncertainty quantification and adaptive retrieval.

    This implements the core research contribution:
    - Token-level entropy calculation
    - Adaptive retrieval based on uncertainty
    - Explicit abstention mechanism
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        llm_api_url: Optional[str] = None,
        enable_uncertainty: bool = True,
        enable_expansion: bool = True
    ):
        """
        Initialize the RAG system.

        Args:
            collection_name: Vector DB collection name
            llm_api_url: vLLM server URL (overrides config)
            enable_uncertainty: Enable entropy calculation
            enable_expansion: Enable adaptive retrieval expansion
        """
        self.enable_uncertainty = enable_uncertainty
        self.enable_expansion = enable_expansion

        logger.info("Initializing Uncertainty-Aware RAG System...")

        # 1. Initialize Embedder
        logger.info("Loading embedder...")
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            batch_size=config.embedding.batch_size
        )

        # 2. Initialize Vector DB
        logger.info("Connecting to vector database...")
        self.vector_db = VectorDB(
            collection_name=collection_name or config.vector_db.collection_name,
            persist_dir=config.vector_db.persist_dir,
            embedding_dim=self.embedder.embedding_dim,
            distance_metric=config.vector_db.distance_metric
        )

        # Verify DB has documents
        doc_count = self.vector_db.count()
        if doc_count == 0:
            logger.warning("⚠️  Vector database is empty! Run ingestion first.")
        else:
            logger.info(f"✅ Vector DB ready: {doc_count} chunks available")

        # 3. Initialize LLM Interface
        logger.info("Connecting to LLM...")
        self.llm = LlamaInterface(
            api_base_url=llm_api_url or config.llm.api_base_url,
            enable_entropy=self.enable_uncertainty
        )

        # Test LLM connection
        if not self.llm.check_connection():
            logger.error("❌ Failed to connect to LLM server!")
            logger.error(f"Expected URL: {self.llm.api_base_url}")
            logger.error("Make sure vLLM server is running")

        # 4. Initialize Uncertainty Gate
        if self.enable_uncertainty:
            self.uncertainty_gate = UncertaintyGate(
                expansion_threshold=config.llm.entropy_expansion_threshold,
                abstention_threshold=config.llm.entropy_abstention_threshold
            )
            logger.info(
                f"Uncertainty thresholds: "
                f"expand={config.llm.entropy_expansion_threshold}, "
                f"abstain={config.llm.entropy_abstention_threshold}"
            )

        # Statistics
        self.stats = {
            "total_queries": 0,
            "expansions_triggered": 0,
            "abstentions": 0,
            "avg_entropy": 0.0,
            "avg_latency": 0.0
        }

        logger.info("✅ RAG System initialized successfully!")

    def retrieve(
        self,
        query: str,
        k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from vector database.

        Args:
            query: User question
            k: Number of chunks to retrieve (default from config)
            filters: Optional metadata filters

        Returns:
            List of retrieved chunks with metadata
        """
        k = k or config.retrieval.top_k

        # Embed query
        query_embedding = self.embedder.encode_queries([query])[0]

        # Search vector DB
        results = self.vector_db.query(
            query_embedding=query_embedding.tolist(),
            n_results=k,
            where=filters,
            include_distances=True
        )

        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")

        return results

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        extract_entropy: bool = True
    ) -> GenerationResult:
        """
        Generate answer using LLM with retrieved context.

        Args:
            query: User question
            context_chunks: Retrieved document chunks
            extract_entropy: Whether to calculate entropy

        Returns:
            GenerationResult with answer and entropy metrics
        """
        # Extract text content from chunks
        context_texts = [chunk["content"] for chunk in context_chunks]

        # Generate with RAG context
        result = self.llm.generate_with_rag_context(
            query=query,
            context_chunks=context_texts,
            max_tokens=config.llm.max_tokens
        )

        return result

    def answer_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> RAGResponse:
        """
        Main entry point: Answer a query with uncertainty-aware retrieval.

        Pipeline:
        1. Retrieve initial chunks (k=5)
        2. Generate answer with entropy tracking
        3. If entropy is medium (1.5-2.0):
           - Expand retrieval to k=10
           - Regenerate answer
        4. If entropy is high (>2.0):
           - Return abstention response
        5. Return answer with confidence metrics

        Args:
            query: User question
            filters: Optional metadata filters (e.g., {"doc_id": "fomcprojtabl20221214"})
            return_sources: Include source chunks in response

        Returns:
            RAGResponse with answer, sources, and uncertainty metrics
        """
        start_time = time.time()

        logger.info(f"\n{'='*70}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*70}")

        # Track state
        retrieval_expanded = False
        abstained = False
        initial_k = config.retrieval.top_k
        expanded_k = config.retrieval.expanded_top_k

        # STEP 1: Initial Retrieval
        logger.info(f"📥 Retrieving top-{initial_k} chunks...")
        initial_chunks = self.retrieve(query, k=initial_k, filters=filters)

        if not initial_chunks:
            logger.warning("⚠️  No chunks retrieved! Database may be empty.")
            return self._create_error_response(
                query=query,
                error="No relevant documents found",
                latency=time.time() - start_time
            )

        # STEP 2: Initial Generation
        logger.info(f"🤖 Generating answer with {len(initial_chunks)} chunks...")
        generation_result = self.generate_answer(
            query=query,
            context_chunks=initial_chunks,
            extract_entropy=self.enable_uncertainty
        )

        current_answer = generation_result.text
        current_chunks = initial_chunks
        entropy_result = generation_result.entropy_result

        # STEP 3: Uncertainty Assessment
        if self.enable_uncertainty and entropy_result:
            seq_entropy = entropy_result.sequence_entropy
            uncertainty_level = entropy_result.uncertainty_level

            logger.info(
                f"📊 Entropy: {seq_entropy:.3f} | "
                f"Uncertainty: {uncertainty_level.value.upper()}"
            )

            # STEP 4: Adaptive Retrieval Expansion
            if (self.enable_expansion and
                entropy_result.should_expand and
                not entropy_result.should_abstain):

                logger.info(f"🔄 EXPANDING RETRIEVAL: {initial_k} → {expanded_k}")
                self.stats["expansions_triggered"] += 1

                # Retrieve additional chunks
                expanded_chunks = self.retrieve(query, k=expanded_k, filters=filters)

                # Regenerate with expanded context
                logger.info(f"🤖 Regenerating with {len(expanded_chunks)} chunks...")
                generation_result = self.generate_answer(
                    query=query,
                    context_chunks=expanded_chunks,
                    extract_entropy=True
                )

                current_answer = generation_result.text
                current_chunks = expanded_chunks
                entropy_result = generation_result.entropy_result
                retrieval_expanded = True

                logger.info(
                    f"📊 New Entropy: {entropy_result.sequence_entropy:.3f} | "
                    f"Uncertainty: {entropy_result.uncertainty_level.value.upper()}"
                )

            # STEP 5: Abstention Check
            if entropy_result.should_abstain:
                logger.warning("🚫 HIGH UNCERTAINTY DETECTED - ABSTAINING")
                self.stats["abstentions"] += 1
                abstained = True

                current_answer = (
                    "INSUFFICIENT INFORMATION: The available context does not contain "
                    "enough information to answer this question confidently. "
                    "The system has detected high uncertainty in the response."
                )

        else:
            # No uncertainty tracking
            entropy_result = None

        # STEP 6: Prepare Response
        latency = time.time() - start_time

        # Update statistics
        self.stats["total_queries"] += 1
        if entropy_result:
            self.stats["avg_entropy"] = (
                (self.stats["avg_entropy"] * (self.stats["total_queries"] - 1) +
                 entropy_result.sequence_entropy) / self.stats["total_queries"]
            )
        self.stats["avg_latency"] = (
            (self.stats["avg_latency"] * (self.stats["total_queries"] - 1) +
             latency) / self.stats["total_queries"]
        )

        # Format sources
        sources = []
        if return_sources:
            for chunk in current_chunks[:5]:  # Limit to top 5 for display
                sources.append({
                    "doc_id": chunk["metadata"].get("doc_id", "Unknown"),
                    "section": chunk["metadata"].get("section_anchor", "Unknown"),
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "score": chunk.get("score", 0.0),
                    "chunk_type": chunk["metadata"].get("chunk_type", "text")
                })

        # Determine confidence
        if abstained:
            confidence = "NONE"
        elif entropy_result:
            if entropy_result.uncertainty_level == UncertaintyLevel.LOW:
                confidence = "HIGH"
            elif entropy_result.uncertainty_level == UncertaintyLevel.MEDIUM:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "UNKNOWN"

        response = RAGResponse(
            question=query,
            answer=current_answer,
            sources=sources,
            uncertainty_level=entropy_result.uncertainty_level.value if entropy_result else "unknown",
            entropy_score=entropy_result.sequence_entropy if entropy_result else 0.0,
            retrieval_expanded=retrieval_expanded,
            abstained=abstained,
            retrieval_count=len(current_chunks),
            confidence=confidence,
            latency=latency,
            metadata={
                "initial_k": initial_k,
                "expanded_k": expanded_k if retrieval_expanded else None,
                "token_count": generation_result.usage["completion_tokens"] if generation_result.usage else 0,
                "finish_reason": generation_result.finish_reason
            }
        )

        logger.info(f"✅ Response generated in {latency:.2f}s")
        logger.info(f"   Confidence: {confidence} | Expanded: {retrieval_expanded} | Abstained: {abstained}")

        return response

    def _create_error_response(
        self,
        query: str,
        error: str,
        latency: float
    ) -> RAGResponse:
        """Create an error response"""
        return RAGResponse(
            question=query,
            answer=f"ERROR: {error}",
            sources=[],
            uncertainty_level="unknown",
            entropy_score=0.0,
            retrieval_expanded=False,
            abstained=False,
            retrieval_count=0,
            confidence="NONE",
            latency=latency,
            metadata={"error": error}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            "vector_db_size": self.vector_db.count(),
            "expansion_rate": (
                self.stats["expansions_triggered"] / self.stats["total_queries"]
                if self.stats["total_queries"] > 0 else 0.0
            ),
            "abstention_rate": (
                self.stats["abstentions"] / self.stats["total_queries"]
                if self.stats["total_queries"] > 0 else 0.0
            )
        }


# Example usage
if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("Uncertainty-Aware RAG System - Example Usage")
    print("=" * 70)

    print("\nInitializing system...")
    print("NOTE: Requires:")
    print("  1. Vector DB populated (run scripts/ingest_to_vectordb.py)")
    print("  2. vLLM server running with Llama-3-8B-Instruct")
    print("     Example: vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000")

    # Initialize RAG system
    try:
        rag = UncertaintyAwareRAG()

        # Example queries
        queries = [
            "What was the GDP growth projection for 2023?",
            "What did Chair Powell say about inflation in December 2023?",
            "What are the long-term unemployment rate expectations?"
        ]

        print("\n" + "=" * 70)
        print("Running Example Queries")
        print("=" * 70)

        for query in queries:
            response = rag.answer_query(query)

            print(f"\n{'─'*70}")
            print(f"Q: {response.question}")
            print(f"{'─'*70}")
            print(f"A: {response.answer}")
            print(f"\nConfidence: {response.confidence}")
            print(f"Entropy: {response.entropy_score:.3f}")
            print(f"Expanded: {response.retrieval_expanded}")
            print(f"Latency: {response.latency:.2f}s")

            if response.sources:
                print(f"\nSources ({len(response.sources)}):")
                for i, source in enumerate(response.sources[:2], 1):
                    print(f"  [{i}] {source['doc_id']} (score: {source['score']:.3f})")

        # Print statistics
        stats = rag.get_statistics()
        print("\n" + "=" * 70)
        print("System Statistics")
        print("=" * 70)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Expansions: {stats['expansions_triggered']} ({stats['expansion_rate']:.1%})")
        print(f"Abstentions: {stats['abstentions']} ({stats['abstention_rate']:.1%})")
        print(f"Avg Entropy: {stats['avg_entropy']:.3f}")
        print(f"Avg Latency: {stats['avg_latency']:.2f}s")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to:")
        print("  1. Run vector DB ingestion first")
        print("  2. Start vLLM server")
        print("  3. Configure LLM__API_BASE_URL in .env")
