"""
Uncertainty Engine for Token-Level Entropy Calculation

This module implements the core uncertainty quantification mechanism for the RAG system.
It calculates Shannon entropy for each generated token and provides decision gates for
adaptive retrieval and abstention.

Key Components:
1. EntropyCalculator: Computes token-level and sequence-level entropy
2. UncertaintyGate: Decision engine for retrieval expansion and abstention
3. EntropyMetrics: Tracking and analysis utilities

Theory:
    Shannon Entropy: H(t) = -Σ p(t_i) * log₂(p(t_i))
    - Low entropy (< 1.0): Model is confident
    - Medium entropy (1.0-2.0): Model is uncertain, expand retrieval
    - High entropy (> 2.0): Model should abstain
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UncertaintyLevel(Enum):
    """Classification of uncertainty levels"""
    LOW = "low"           # H < 1.0: Confident
    MEDIUM = "medium"     # 1.0 <= H < 2.0: Uncertain
    HIGH = "high"         # H >= 2.0: Should abstain


@dataclass
class EntropyResult:
    """
    Container for entropy calculation results

    Attributes:
        token_entropies: Entropy value for each generated token
        sequence_entropy: Aggregated entropy for the full sequence
        uncertainty_level: Classification of uncertainty
        should_expand: Whether to trigger retrieval expansion
        should_abstain: Whether to abstain from answering
        token_ids: Token IDs corresponding to entropies (optional)
        tokens: Decoded tokens (optional, for debugging)
    """
    token_entropies: List[float]
    sequence_entropy: float
    uncertainty_level: UncertaintyLevel
    should_expand: bool
    should_abstain: bool
    token_ids: Optional[List[int]] = None
    tokens: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "token_entropies": self.token_entropies,
            "sequence_entropy": self.sequence_entropy,
            "uncertainty_level": self.uncertainty_level.value,
            "should_expand": self.should_expand,
            "should_abstain": self.should_abstain,
            "token_ids": self.token_ids,
            "tokens": self.tokens,
            "metadata": self.metadata
        }


class EntropyCalculator:
    """
    Calculates Shannon entropy for token probability distributions.

    This class provides methods to compute entropy at both token and sequence levels,
    with support for different aggregation strategies.
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize the entropy calculator.

        Args:
            epsilon: Small constant to avoid log(0) errors
        """
        self.epsilon = epsilon
        logger.debug(f"EntropyCalculator initialized with epsilon={epsilon}")

    def compute_token_entropy(
        self,
        logits: np.ndarray,
        temperature: float = 1.0
    ) -> float:
        """
        Calculate Shannon entropy for a single token's probability distribution.

        H(t) = -Σ p(t_i) * log₂(p(t_i))

        Args:
            logits: Raw logits from the model (shape: [vocab_size])
            temperature: Temperature for softmax (default: 1.0)

        Returns:
            Entropy value in bits

        Example:
            >>> logits = np.array([2.0, 1.0, 0.5])
            >>> calc = EntropyCalculator()
            >>> entropy = calc.compute_token_entropy(logits)
            >>> print(f"Entropy: {entropy:.3f} bits")
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Convert to probabilities via softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Add epsilon to avoid log(0)
        probs = np.clip(probs, self.epsilon, 1.0)

        # Calculate Shannon entropy (using log2 for bits)
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def compute_token_entropy_from_probs(
        self,
        probs: np.ndarray
    ) -> float:
        """
        Calculate entropy directly from probabilities.

        Args:
            probs: Probability distribution (shape: [vocab_size])

        Returns:
            Entropy value in bits
        """
        # Ensure probabilities are valid
        probs = np.clip(probs, self.epsilon, 1.0)
        probs = probs / np.sum(probs)  # Normalize

        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def compute_sequence_entropy(
        self,
        token_entropies: List[float],
        aggregation: str = "mean"
    ) -> float:
        """
        Aggregate token-level entropies into a sequence-level metric.

        Args:
            token_entropies: List of entropy values for each token
            aggregation: Method to aggregate entropies
                - "mean": Average entropy (default)
                - "max": Maximum entropy (pessimistic)
                - "weighted_mean": Weight recent tokens more heavily
                - "median": Median entropy (robust to outliers)

        Returns:
            Aggregated entropy value
        """
        if not token_entropies:
            logger.warning("Empty token_entropies list provided")
            return 0.0

        entropies = np.array(token_entropies)

        if aggregation == "mean":
            return float(np.mean(entropies))

        elif aggregation == "max":
            return float(np.max(entropies))

        elif aggregation == "weighted_mean":
            # Weight recent tokens more heavily (exponential decay)
            n = len(entropies)
            weights = np.exp(np.linspace(-1, 0, n))  # More weight to recent tokens
            weights = weights / np.sum(weights)
            return float(np.sum(entropies * weights))

        elif aggregation == "median":
            return float(np.median(entropies))

        else:
            logger.warning(f"Unknown aggregation method: {aggregation}. Using 'mean'.")
            return float(np.mean(entropies))

    def compute_entropy_statistics(
        self,
        token_entropies: List[float]
    ) -> Dict[str, float]:
        """
        Compute comprehensive statistics for entropy analysis.

        Args:
            token_entropies: List of entropy values

        Returns:
            Dictionary with statistical measures
        """
        if not token_entropies:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "q25": 0.0,
                "q75": 0.0
            }

        entropies = np.array(token_entropies)

        return {
            "mean": float(np.mean(entropies)),
            "median": float(np.median(entropies)),
            "std": float(np.std(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
            "q25": float(np.percentile(entropies, 25)),
            "q75": float(np.percentile(entropies, 75))
        }

    def detect_uncertainty_spikes(
        self,
        token_entropies: List[float],
        window_size: int = 5,
        spike_threshold: float = 2.0
    ) -> List[int]:
        """
        Detect positions where entropy suddenly spikes.

        Useful for identifying specific uncertain phrases or hallucinations.

        Args:
            token_entropies: List of entropy values
            window_size: Size of sliding window for local average
            spike_threshold: Standard deviations above local mean to classify as spike

        Returns:
            List of token indices where spikes occur
        """
        if len(token_entropies) < window_size:
            return []

        entropies = np.array(token_entropies)
        spike_indices = []

        for i in range(window_size, len(entropies)):
            # Local window average
            window = entropies[i - window_size:i]
            local_mean = np.mean(window)
            local_std = np.std(window)

            # Check if current token is a spike
            if local_std > 0 and (entropies[i] - local_mean) > spike_threshold * local_std:
                spike_indices.append(i)

        return spike_indices


class UncertaintyGate:
    """
    Decision gate for adaptive retrieval and abstention based on entropy.

    This class implements the core decision logic:
    1. Should we expand retrieval context (k=5 -> k=10)?
    2. Should we abstain from answering?
    """

    def __init__(
        self,
        expansion_threshold: float = 1.5,
        abstention_threshold: float = 2.0,
        aggregation_method: str = "mean",
        min_tokens_for_decision: int = 10
    ):
        """
        Initialize the uncertainty gate with decision thresholds.

        Args:
            expansion_threshold: Entropy threshold for triggering retrieval expansion
            abstention_threshold: Entropy threshold for abstention
            aggregation_method: How to aggregate token entropies (mean, max, etc.)
            min_tokens_for_decision: Minimum tokens before making decisions
        """
        self.expansion_threshold = expansion_threshold
        self.abstention_threshold = abstention_threshold
        self.aggregation_method = aggregation_method
        self.min_tokens_for_decision = min_tokens_for_decision

        self.entropy_calculator = EntropyCalculator()

        logger.info(
            f"UncertaintyGate initialized: "
            f"expansion_threshold={expansion_threshold}, "
            f"abstention_threshold={abstention_threshold}"
        )

    def evaluate(
        self,
        token_entropies: List[float],
        token_ids: Optional[List[int]] = None,
        tokens: Optional[List[str]] = None
    ) -> EntropyResult:
        """
        Evaluate uncertainty and make decisions.

        Args:
            token_entropies: List of entropy values for generated tokens
            token_ids: Optional token IDs for reference
            tokens: Optional decoded tokens for debugging

        Returns:
            EntropyResult with decisions and metadata
        """
        # Compute sequence-level entropy
        sequence_entropy = self.entropy_calculator.compute_sequence_entropy(
            token_entropies,
            aggregation=self.aggregation_method
        )

        # Classify uncertainty level
        uncertainty_level = self._classify_uncertainty(sequence_entropy)

        # Make decisions
        should_expand = self._should_expand_retrieval(
            sequence_entropy,
            len(token_entropies)
        )

        should_abstain = self._should_abstain(
            sequence_entropy,
            len(token_entropies)
        )

        # Compute statistics
        stats = self.entropy_calculator.compute_entropy_statistics(token_entropies)

        # Detect spikes
        spikes = self.entropy_calculator.detect_uncertainty_spikes(token_entropies)

        result = EntropyResult(
            token_entropies=token_entropies,
            sequence_entropy=sequence_entropy,
            uncertainty_level=uncertainty_level,
            should_expand=should_expand,
            should_abstain=should_abstain,
            token_ids=token_ids,
            tokens=tokens,
            metadata={
                "statistics": stats,
                "spike_indices": spikes,
                "num_tokens": len(token_entropies),
                "expansion_threshold": self.expansion_threshold,
                "abstention_threshold": self.abstention_threshold
            }
        )

        logger.debug(
            f"Entropy evaluation: seq_entropy={sequence_entropy:.3f}, "
            f"uncertainty={uncertainty_level.value}, "
            f"expand={should_expand}, abstain={should_abstain}"
        )

        return result

    def _classify_uncertainty(self, entropy: float) -> UncertaintyLevel:
        """Classify entropy into uncertainty levels"""
        if entropy < self.expansion_threshold:
            return UncertaintyLevel.LOW
        elif entropy < self.abstention_threshold:
            return UncertaintyLevel.MEDIUM
        else:
            return UncertaintyLevel.HIGH

    def _should_expand_retrieval(
        self,
        sequence_entropy: float,
        num_tokens: int
    ) -> bool:
        """
        Decide whether to expand retrieval context.

        Logic:
        - Entropy must exceed expansion threshold
        - Must have generated minimum number of tokens
        - Should not have already crossed abstention threshold
        """
        if num_tokens < self.min_tokens_for_decision:
            return False

        return (
            self.expansion_threshold <= sequence_entropy < self.abstention_threshold
        )

    def _should_abstain(
        self,
        sequence_entropy: float,
        num_tokens: int
    ) -> bool:
        """
        Decide whether to abstain from answering.

        Logic:
        - Entropy must exceed abstention threshold
        - Must have generated minimum number of tokens
        """
        if num_tokens < self.min_tokens_for_decision:
            return False

        return sequence_entropy >= self.abstention_threshold


class SemanticUncertaintyCalculator:
    """
    Calculates semantic uncertainty by comparing multiple generations.

    Instead of looking at token-level entropy, this approach:
    1. Generates multiple responses (e.g., 3-5 times)
    2. Embeds each response
    3. Measures semantic similarity/diversity
    4. Low similarity → High uncertainty (hallucination/disagreement)
    5. High similarity → Low uncertainty (consistent/confident)

    This is more robust to:
    - Paraphrasing (different words, same meaning)
    - Refusals (semantically consistent even with different wording)
    - Hallucinations (semantically inconsistent answers)
    """

    def __init__(
        self,
        embedder=None,
        num_generations: int = 3,
        similarity_threshold: float = 0.85,
        uncertainty_threshold: float = 0.7
    ):
        """
        Initialize semantic uncertainty calculator.

        Args:
            embedder: Embedder instance for encoding text
            num_generations: Number of responses to generate
            similarity_threshold: Cosine similarity threshold for agreement
            uncertainty_threshold: Average similarity below this = uncertain
        """
        self.embedder = embedder
        self.num_generations = num_generations
        self.similarity_threshold = similarity_threshold
        self.uncertainty_threshold = uncertainty_threshold

        logger.info(
            f"SemanticUncertaintyCalculator initialized: "
            f"num_generations={num_generations}, "
            f"similarity_threshold={similarity_threshold}"
        )

    def compute_semantic_uncertainty(
        self,
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Compute semantic uncertainty from multiple generations.

        Args:
            responses: List of generated responses (same question, multiple times)

        Returns:
            Dictionary with uncertainty metrics
        """
        if len(responses) < 2:
            logger.warning("Need at least 2 responses for semantic uncertainty")
            return {
                "semantic_uncertainty": 0.0,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "disagreement_rate": 0.0,
                "should_expand": False,
                "should_abstain": False
            }

        # Embed all responses
        embeddings = self.embedder.encode_queries(responses)

        # Compute pairwise cosine similarities
        similarities = []
        n = len(responses)
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(float(sim))

        avg_similarity = float(np.mean(similarities))
        min_similarity = float(np.min(similarities))

        # Semantic uncertainty = 1 - avg_similarity
        # High similarity → Low uncertainty
        # Low similarity → High uncertainty
        semantic_uncertainty = 1.0 - avg_similarity

        # Count disagreements (pairs below threshold)
        disagreements = sum(1 for s in similarities if s < self.similarity_threshold)
        disagreement_rate = disagreements / len(similarities)

        # Decision making
        should_expand = avg_similarity < self.uncertainty_threshold
        should_abstain = avg_similarity < (self.uncertainty_threshold - 0.15)

        result = {
            "semantic_uncertainty": semantic_uncertainty,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities)),
            "disagreement_rate": disagreement_rate,
            "num_responses": len(responses),
            "should_expand": should_expand,
            "should_abstain": should_abstain,
            "all_similarities": similarities
        }

        logger.debug(
            f"Semantic uncertainty: {semantic_uncertainty:.3f}, "
            f"avg_sim={avg_similarity:.3f}, "
            f"expand={should_expand}, abstain={should_abstain}"
        )

        return result

    def classify_response_type(
        self,
        responses: List[str]
    ) -> str:
        """
        Classify the type of response pattern.

        Returns:
            - "consistent_answer": High similarity, all providing answers
            - "consistent_refusal": High similarity, all refusing
            - "hallucination": Low similarity, different factual claims
            - "mixed": Some answer, some refuse
        """
        if len(responses) < 2:
            return "unknown"

        result = self.compute_semantic_uncertainty(responses)
        avg_sim = result["avg_similarity"]

        # Check if responses are refusals
        refusal_keywords = ["cannot", "insufficient", "unable", "don't know", "not enough"]
        refusal_count = sum(
            1 for r in responses
            if any(kw in r.lower() for kw in refusal_keywords)
        )

        if avg_sim >= self.similarity_threshold:
            if refusal_count >= len(responses) * 0.7:
                return "consistent_refusal"
            else:
                return "consistent_answer"
        else:
            if refusal_count > 0 and refusal_count < len(responses):
                return "mixed"
            else:
                return "hallucination"


class EntropyTracker:
    """
    Utility class for tracking and analyzing entropy over multiple queries.

    Useful for:
    - Analyzing entropy patterns across a dataset
    - Calibrating thresholds
    - Generating reports and visualizations
    """

    def __init__(self):
        """Initialize the entropy tracker"""
        self.records: List[Dict[str, Any]] = []
        logger.info("EntropyTracker initialized")

    def record(
        self,
        query: str,
        entropy_result: EntropyResult,
        response: Optional[str] = None,
        ground_truth: Optional[str] = None,
        was_correct: Optional[bool] = None
    ):
        """
        Record entropy data for a query.

        Args:
            query: The input query
            entropy_result: EntropyResult from evaluation
            response: Generated response (optional)
            ground_truth: Expected answer (optional)
            was_correct: Whether the response was correct (optional)
        """
        record = {
            "query": query,
            "sequence_entropy": entropy_result.sequence_entropy,
            "uncertainty_level": entropy_result.uncertainty_level.value,
            "should_expand": entropy_result.should_expand,
            "should_abstain": entropy_result.should_abstain,
            "num_tokens": len(entropy_result.token_entropies),
            "response": response,
            "ground_truth": ground_truth,
            "was_correct": was_correct,
            "statistics": entropy_result.metadata.get("statistics", {}),
            "spike_count": len(entropy_result.metadata.get("spike_indices", []))
        }

        self.records.append(record)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all recorded queries.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.records:
            return {}

        entropies = [r["sequence_entropy"] for r in self.records]

        # Uncertainty level distribution
        uncertainty_dist = {
            "low": sum(1 for r in self.records if r["uncertainty_level"] == "low"),
            "medium": sum(1 for r in self.records if r["uncertainty_level"] == "medium"),
            "high": sum(1 for r in self.records if r["uncertainty_level"] == "high")
        }

        # Decision statistics
        expansion_rate = sum(1 for r in self.records if r["should_expand"]) / len(self.records)
        abstention_rate = sum(1 for r in self.records if r["should_abstain"]) / len(self.records)

        # Accuracy by uncertainty level (if ground truth available)
        accuracy_by_uncertainty = {}
        for level in ["low", "medium", "high"]:
            level_records = [r for r in self.records if r["uncertainty_level"] == level and r["was_correct"] is not None]
            if level_records:
                accuracy_by_uncertainty[level] = sum(r["was_correct"] for r in level_records) / len(level_records)

        return {
            "total_queries": len(self.records),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_median": float(np.median(entropies)),
            "entropy_std": float(np.std(entropies)),
            "uncertainty_distribution": uncertainty_dist,
            "expansion_rate": expansion_rate,
            "abstention_rate": abstention_rate,
            "accuracy_by_uncertainty": accuracy_by_uncertainty
        }

    def export_to_dict(self) -> List[Dict[str, Any]]:
        """Export all records as a list of dictionaries"""
        return self.records.copy()

    def clear(self):
        """Clear all recorded data"""
        self.records.clear()
        logger.info("EntropyTracker cleared")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("Uncertainty Engine - Example Usage")
    print("=" * 70)

    # Example 1: Calculate token entropy
    print("\nExample 1: Token Entropy Calculation")
    print("-" * 70)

    calc = EntropyCalculator()

    # Simulate logits for a confident prediction
    confident_logits = np.array([10.0, 1.0, 0.5, 0.3])
    entropy_confident = calc.compute_token_entropy(confident_logits)
    print(f"Confident prediction logits: {confident_logits}")
    print(f"Entropy: {entropy_confident:.3f} bits (LOW - model is confident)\n")

    # Simulate logits for an uncertain prediction
    uncertain_logits = np.array([2.0, 1.8, 1.5, 1.4])
    entropy_uncertain = calc.compute_token_entropy(uncertain_logits)
    print(f"Uncertain prediction logits: {uncertain_logits}")
    print(f"Entropy: {entropy_uncertain:.3f} bits (HIGH - model is uncertain)\n")

    # Example 2: Sequence entropy and decision making
    print("\nExample 2: Uncertainty Gate Decision Making")
    print("-" * 70)

    gate = UncertaintyGate(
        expansion_threshold=1.5,
        abstention_threshold=2.0
    )

    # Case 1: Low uncertainty (confident answer)
    low_entropy_sequence = [0.5, 0.6, 0.7, 0.6, 0.5, 0.6, 0.7, 0.6, 0.5, 0.6]
    result_low = gate.evaluate(low_entropy_sequence)
    print(f"Case 1 - Confident Answer:")
    print(f"  Average Entropy: {result_low.sequence_entropy:.3f}")
    print(f"  Uncertainty Level: {result_low.uncertainty_level.value}")
    print(f"  Should Expand Retrieval: {result_low.should_expand}")
    print(f"  Should Abstain: {result_low.should_abstain}\n")

    # Case 2: Medium uncertainty (should expand retrieval)
    medium_entropy_sequence = [1.2, 1.4, 1.6, 1.5, 1.7, 1.6, 1.5, 1.6, 1.7, 1.6]
    result_medium = gate.evaluate(medium_entropy_sequence)
    print(f"Case 2 - Uncertain Answer (Expand Retrieval):")
    print(f"  Average Entropy: {result_medium.sequence_entropy:.3f}")
    print(f"  Uncertainty Level: {result_medium.uncertainty_level.value}")
    print(f"  Should Expand Retrieval: {result_medium.should_expand}")
    print(f"  Should Abstain: {result_medium.should_abstain}\n")

    # Case 3: High uncertainty (should abstain)
    high_entropy_sequence = [2.5, 2.3, 2.4, 2.6, 2.5, 2.4, 2.5, 2.6, 2.5, 2.4]
    result_high = gate.evaluate(high_entropy_sequence)
    print(f"Case 3 - Very Uncertain (Abstain):")
    print(f"  Average Entropy: {result_high.sequence_entropy:.3f}")
    print(f"  Uncertainty Level: {result_high.uncertainty_level.value}")
    print(f"  Should Expand Retrieval: {result_high.should_expand}")
    print(f"  Should Abstain: {result_high.should_abstain}\n")

    # Example 3: Entropy tracking
    print("\nExample 3: Entropy Tracking Across Multiple Queries")
    print("-" * 70)

    tracker = EntropyTracker()

    # Simulate tracking 3 queries
    tracker.record("Query 1", result_low, was_correct=True)
    tracker.record("Query 2", result_medium, was_correct=True)
    tracker.record("Query 3", result_high, was_correct=False)

    summary = tracker.get_summary_statistics()
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Average Entropy: {summary['entropy_mean']:.3f}")
    print(f"Expansion Rate: {summary['expansion_rate']:.2%}")
    print(f"Abstention Rate: {summary['abstention_rate']:.2%}")
    print(f"Uncertainty Distribution: {summary['uncertainty_distribution']}")

    print("\n" + "=" * 70)
    print("Module ready for integration with LLM interface!")
    print("=" * 70)
