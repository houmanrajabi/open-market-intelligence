"""
LLM Interface for Llama-3 with Logit Extraction

This module provides an interface to Llama-3-8B-Instruct hosted on remote infrastructure
(VastAI, RunPod, etc.) via vLLM OpenAI-compatible API. It extracts per-token logits for
uncertainty quantification.

Key Features:
- Remote inference via vLLM API (OpenAI-compatible)
- Token-level logit extraction for entropy calculation
- Support for both streaming and non-streaming generation
- Integration with UncertaintyEngine for adaptive retrieval
- Prompt template management for RAG
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from openai import OpenAI

from src.retrieval.uncertainty_engine import (
    EntropyCalculator,
    UncertaintyGate,
    EntropyResult
)
from src.utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Container for LLM generation output with entropy metrics.

    Attributes:
        text: Generated text response
        entropy_result: Entropy analysis from UncertaintyGate
        logprobs: Raw log probabilities for each token (optional)
        tokens: Decoded tokens (optional)
        finish_reason: Reason for generation termination
        usage: Token usage statistics
    """
    text: str
    entropy_result: Optional[EntropyResult] = None
    logprobs: Optional[List[Dict[str, Any]]] = None
    tokens: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "entropy_result": self.entropy_result.to_dict() if self.entropy_result else None,
            "finish_reason": self.finish_reason,
            "usage": self.usage
        }


class LlamaInterface:
    """
    Interface to Llama-3-8B-Instruct via vLLM remote API.

    This class handles:
    1. Remote API communication with vLLM server
    2. Prompt formatting for RAG tasks
    3. Logit extraction and entropy calculation
    4. Integration with uncertainty quantification
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        enable_entropy: bool = True,
        uncertainty_gate: Optional[UncertaintyGate] = None
    ):
        """
        Initialize the Llama interface.

        Args:
            api_base_url: Base URL for vLLM server (e.g., "http://vastai-instance:8000/v1")
            api_key: API key for authentication (can be dummy for vLLM)
            model_name: Model identifier (default from config)
            enable_entropy: Whether to calculate entropy metrics
            uncertainty_gate: Custom UncertaintyGate instance (optional)
        """
        self.api_base_url = api_base_url or config.llm.api_base_url
        self.api_key = api_key or config.llm.api_key
        self.model_name = model_name or config.llm.model_name
        self.enable_entropy = enable_entropy

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key
        )

        # Initialize entropy components
        if self.enable_entropy:
            self.entropy_calculator = EntropyCalculator()
            self.uncertainty_gate = uncertainty_gate or UncertaintyGate(
                expansion_threshold=config.llm.entropy_expansion_threshold,
                abstention_threshold=config.llm.entropy_abstention_threshold
            )
        else:
            self.entropy_calculator = None
            self.uncertainty_gate = None

        logger.info(
            f"LlamaInterface initialized: "
            f"api_base={self.api_base_url}, "
            f"model={self.model_name}, "
            f"entropy_enabled={self.enable_entropy}"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extract_logprobs: bool = True,
        top_logprobs: int = 20
    ) -> GenerationResult:
        """
        Generate text from Llama-3 with optional logit extraction.

        Args:
            messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            extract_logprobs: Whether to extract logprobs for entropy calculation
            top_logprobs: Number of top logprobs to return per token

        Returns:
            GenerationResult with text and entropy metrics
        """
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature if temperature is not None else config.llm.temperature
        top_p = top_p if top_p is not None else config.llm.top_p

        try:
            # Call vLLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=extract_logprobs,
                top_logprobs=top_logprobs if extract_logprobs else None
            )

            # Extract response
            choice = response.choices[0]
            generated_text = choice.message.content
            finish_reason = choice.finish_reason

            # Extract usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None

            # Process logprobs for entropy calculation
            entropy_result = None
            logprobs_data = None
            tokens = None

            if extract_logprobs and choice.logprobs and self.enable_entropy:
                logprobs_data = choice.logprobs.content
                entropy_result, tokens = self._calculate_entropy_from_logprobs(logprobs_data)

            result = GenerationResult(
                text=generated_text,
                entropy_result=entropy_result,
                logprobs=logprobs_data,
                tokens=tokens,
                finish_reason=finish_reason,
                usage=usage
            )

            logger.debug(
                f"Generation completed: {len(generated_text)} chars, "
                f"{usage['completion_tokens'] if usage else 0} tokens"
            )

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    def _calculate_entropy_from_logprobs(
        self,
        logprobs_data: List[Any]
    ) -> Tuple[Optional[EntropyResult], Optional[List[str]]]:
        """
        Calculate entropy from vLLM logprobs output.

        Args:
            logprobs_data: Logprobs content from vLLM response

        Returns:
            Tuple of (EntropyResult, tokens)
        """
        try:
            token_entropies = []
            tokens = []

            for token_data in logprobs_data:
                # Extract token
                token = token_data.token
                tokens.append(token)

                # Extract top logprobs for this token
                top_logprobs = token_data.top_logprobs

                if not top_logprobs:
                    continue

                # Convert logprobs to probabilities
                # vLLM returns dict of {token: logprob}
                logprobs_dict = {item.token: item.logprob for item in top_logprobs}
                logprobs_values = np.array(list(logprobs_dict.values()))

                # Convert log probabilities to probabilities
                probs = np.exp(logprobs_values)

                # Normalize (in case top_k doesn't sum to 1)
                probs = probs / np.sum(probs)

                # Calculate entropy for this token
                entropy = self.entropy_calculator.compute_token_entropy_from_probs(probs)
                token_entropies.append(entropy)

            # Use uncertainty gate to make decisions
            if token_entropies and self.uncertainty_gate:
                entropy_result = self.uncertainty_gate.evaluate(
                    token_entropies=token_entropies,
                    tokens=tokens
                )
                return entropy_result, tokens

            return None, tokens

        except Exception as e:
            logger.error(f"Entropy calculation failed: {e}", exc_info=True)
            return None, None

    def generate_with_rag_context(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate answer using RAG context with proper prompt formatting.

        Args:
            query: User question
            context_chunks: Retrieved document chunks
            system_prompt: Custom system prompt (optional)
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with answer and entropy metrics
        """
        # Build context string
        context_str = "\n\n".join([
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = config.llm.system_prompt

        # Build user message with context
        user_message = f"""Context Information:
{context_str}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- Cite specific documents using [Document N] format
- If the context doesn't contain enough information, say "INSUFFICIENT INFORMATION"
- Be precise and concise

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return self.generate(messages, max_tokens=max_tokens)

    def generate_with_expanded_context(
        self,
        query: str,
        initial_chunks: List[str],
        expanded_chunks: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate answer with expanded retrieval context (k=5 -> k=10).

        This is called when initial generation has medium uncertainty.

        Args:
            query: User question
            initial_chunks: Original top-k chunks
            expanded_chunks: Additional chunks from expanded retrieval
            system_prompt: Custom system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with answer from expanded context
        """
        logger.info("Generating with expanded context (uncertainty detected)")

        # Combine initial and expanded chunks
        all_chunks = initial_chunks + expanded_chunks

        return self.generate_with_rag_context(
            query=query,
            context_chunks=all_chunks,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

    def check_connection(self) -> bool:
        """
        Check if vLLM server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("vLLM server connection successful")
            return True
        except Exception as e:
            logger.error(f"vLLM server connection failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        try:
            # vLLM typically exposes model info via /v1/models endpoint
            models = self.client.models.list()
            return {
                "model_name": self.model_name,
                "api_base": self.api_base_url,
                "available_models": [m.id for m in models.data] if models else []
            }
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {
                "model_name": self.model_name,
                "api_base": self.api_base_url
            }


class PromptTemplate:
    """
    Prompt templates for different RAG scenarios.
    """

    @staticmethod
    def rag_qa_template(
        query: str,
        context: str,
        require_citation: bool = True
    ) -> str:
        """
        Standard RAG QA prompt.

        Args:
            query: User question
            context: Retrieved context
            require_citation: Whether to enforce citation requirement

        Returns:
            Formatted prompt string
        """
        citation_instruction = "- Cite specific documents using [Document N] format" if require_citation else ""

        return f"""Context Information:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
{citation_instruction}
- If the context doesn't contain enough information, say "INSUFFICIENT INFORMATION"
- Be precise and concise

Answer:"""

    @staticmethod
    def abstention_template() -> str:
        """
        Template for generating abstention response.

        Returns:
            Abstention message
        """
        return "INSUFFICIENT INFORMATION: The available context does not contain enough information to answer this question confidently."

    @staticmethod
    def system_prompt_rag() -> str:
        """
        System prompt for RAG tasks.

        Returns:
            System prompt string
        """
        return """You are a precise question-answering assistant for Federal Reserve FOMC documents.

Your core principles:
1. Answer ONLY based on provided context
2. Always cite your sources with [Document N] format
3. If uncertain or information is missing, say "INSUFFICIENT INFORMATION"
4. Never make up or infer information not explicitly stated
5. Be concise and direct in your responses

Remember: Accuracy and citation precision are paramount. It's better to say "I don't know" than to guess."""


# Example usage and testing
if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("LLM Interface - Example Usage")
    print("=" * 70)

    # Note: This requires a running vLLM server
    # Example server command:
    # vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

    print("\nTo use this interface, you need:")
    print("1. A vLLM server running with Llama-3-8B-Instruct")
    print("2. Configure LLM settings in config.py or .env")
    print("\nExample .env configuration:")
    print("  LLM__API_BASE_URL=http://your-vastai-instance:8000/v1")
    print("  LLM__API_KEY=dummy-key")
    print("  LLM__MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct")

    print("\n" + "=" * 70)
    print("Example API Usage")
    print("=" * 70)

    # Example code (will fail without actual server)
    print("""
# Initialize interface
llm = LlamaInterface()

# Check connection
if llm.check_connection():
    print("Connected to vLLM server!")

    # Generate with RAG context
    result = llm.generate_with_rag_context(
        query="What was the unemployment rate projection for 2023?",
        context_chunks=[
            "The FOMC projects unemployment to be 4.5% in 2023...",
            "Economic projections show stable labor markets..."
        ]
    )

    print(f"Answer: {result.text}")

    if result.entropy_result:
        print(f"Entropy: {result.entropy_result.sequence_entropy:.3f}")
        print(f"Should Expand: {result.entropy_result.should_expand}")
        print(f"Should Abstain: {result.entropy_result.should_abstain}")
""")

    print("\n" + "=" * 70)
llm = LlamaInterface()

# Check connection
if llm.check_connection():
    print("Connected to vLLM server!")

    # Generate with RAG context
    result = llm.generate_with_rag_context(
        query="What was the unemployment rate projection for 2023?",
        context_chunks=[
            "The FOMC projects unemployment to be 4.5% in 2023...",
            "Economic projections show stable labor markets..."
        ]
    )

    print(f"Answer: {result.text}")

    if result.entropy_result:
        print(f"Entropy: {result.entropy_result.sequence_entropy:.3f}")
        print(f"Should Expand: {result.entropy_result.should_expand}")
        print(f"Should Abstain: {result.entropy_result.should_abstain}")