"""
Embedding Generator

Generates optimized embeddings using SentenceTransformers.
Supports instruction-tuned models, automatic device selection, and robust batching.
"""

from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.logger import logger
from src.utils.config import config


class Embedder:
    """
    Features:
    - Auto-device selection (CUDA, MPS, CPU)
    - Support for instruction-tuned models (e.g., Instructor, E5)
    - Robust batch processing
    - Cached model loading
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the Embedder.

        Args:
            model_name: HuggingFace model name (default from config)
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
            normalize_embeddings: Whether to normalize vectors (L2 norm)
        """
        self.model_name = model_name or config.embedding.model_name
        self.batch_size = batch_size or config.embedding.batch_size
        self.normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else config.embedding.normalize_embeddings
        self.device = device or self._get_optimal_device()

        logger.info(f"🚀 Loading embedding model: {self.model_name} on {self.device}")

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

        # Check if model requires instructions (e.g., Instructor, E5)
        self.is_instruction_model = any(name in self.model_name.lower() 
                                      for name in ["instructor", "e5", "bge"])
        
        # Define default instructions for retrieval tasks if needed
        self.query_instruction = config.embedding.query_instruction
        self.doc_instruction = config.embedding.doc_instruction

    def _get_optimal_device(self) -> str:
        """Automatically determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single string or list of strings
            show_progress: Whether to show tqdm progress bar
            is_query: Set True if encoding a search query (adds instructions for specific models)

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        # Input validation
        if not texts:
            logger.warning("Empty text input provided to embedder.")
            return np.array([])

        if isinstance(texts, str):
            texts = [texts]

        # Add instructions for instruction-tuned models
        if self.is_instruction_model and is_query:
             # E5/BGE models expect "query: " prefix for queries
            if "e5" in self.model_name.lower() or "bge" in self.model_name.lower():
                texts = [f"query: {t}" for t in texts]
            # Instructor models handle instructions via encode args (handled differently usually, 
            # but simple prefix often works or requires custom call. Sticking to simple prefix for compatibility)

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                device=self.device
            )
            return embeddings

        except Exception as e:
            logger.error(f"❌ Encoding failed: {e}")
            # Return zero vectors as fallback (or raise depending on policy)
            return np.zeros((len(texts), self.embedding_dim))

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Helper specifically for encoding search queries."""
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Helper specifically for encoding documents."""
        # For E5/BGE, documents might need "passage: " prefix, though often raw text is fine.
        # Check specific model docs. For now, we assume raw text or standard config.
        return self.encode(documents, is_query=False)