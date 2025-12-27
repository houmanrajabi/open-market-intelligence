"""
Evaluation Metrics for RAG System

This module implements metrics for measuring:
1. Retrieval Quality (Precision, Recall, MRR, NDCG)
2. Answer Quality (Accuracy, Faithfulness, Citation Precision)
3. Uncertainty Calibration (Entropy correlation with correctness)
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from src.utils.logger import logger


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    total_queries: int
    avg_score: float


@dataclass
class AnswerMetrics:
    """Metrics for answer quality evaluation"""
    accuracy: float  # Fraction of correct answers
    faithfulness: float  # Answer grounded in context
    citation_precision: float  # Citations are accurate
    hallucination_rate: float  # Unsupported claims
    abstention_rate: float  # Explicit "I don't know"
    total_queries: int


@dataclass
class UncertaintyMetrics:
    """Metrics for uncertainty calibration"""
    entropy_correct_mean: float  # Avg entropy for correct answers
    entropy_incorrect_mean: float  # Avg entropy for incorrect answers
    correlation: float  # Correlation between entropy and correctness
    auc_roc: float  # ROC AUC for uncertainty as binary classifier
    ece: float  # Expected Calibration Error
    total_queries: int


class RetrievalEvaluator:
    """
    Evaluates retrieval quality against ground truth.
    """

    def __init__(self):
        """Initialize retrieval evaluator"""
        self.results = []

    def evaluate_query(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        scores: Optional[List[float]] = None,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query.

        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: List of ground truth relevant document IDs
            scores: Optional relevance scores for retrieved docs
            k: Top-k to evaluate

        Returns:
            Dictionary with precision@k, recall@k, MRR, NDCG
        """
        retrieved_docs = retrieved_docs[:k]

        # Precision@k
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0

        # Recall@k
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                mrr = 1.0 / i
                break

        # NDCG@k
        if scores is None:
            # Binary relevance (1 if relevant, 0 otherwise)
            relevance = [1.0 if doc in relevant_docs else 0.0 for doc in retrieved_docs]
        else:
            relevance = scores[:k]

        ndcg = self._compute_ndcg(relevance, relevant_docs, k)

        metrics = {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr,
            "ndcg_at_k": ndcg
        }

        self.results.append(metrics)
        return metrics

    def _compute_ndcg(
        self,
        relevance_scores: List[float],
        ideal_docs: List[str],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.

        DCG = Σ (rel_i / log2(i+1)) for i in 1..k
        NDCG = DCG / IDCG
        """
        if not relevance_scores:
            return 0.0

        # DCG
        dcg = sum(
            rel / np.log2(i + 2)  # +2 because indexing starts at 0
            for i, rel in enumerate(relevance_scores)
        )

        # IDCG (ideal DCG with perfect ranking)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_relevance)
        )

        return dcg / idcg if idcg > 0 else 0.0

    def aggregate_metrics(self) -> RetrievalMetrics:
        """Aggregate metrics across all queries"""
        if not self.results:
            return RetrievalMetrics(0, 0, 0, 0, 0, 0)

        avg_precision = np.mean([r["precision_at_k"] for r in self.results])
        avg_recall = np.mean([r["recall_at_k"] for r in self.results])
        avg_mrr = np.mean([r["mrr"] for r in self.results])
        avg_ndcg = np.mean([r["ndcg_at_k"] for r in self.results])

        # Compute average score if available
        avg_score = 0.0  # Placeholder

        return RetrievalMetrics(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg,
            total_queries=len(self.results),
            avg_score=avg_score
        )


class AnswerEvaluator:
    """
    Evaluates answer quality: accuracy, faithfulness, citations.
    """

    def __init__(self, use_llm_judge: bool = False):
        """
        Initialize answer evaluator.

        Args:
            use_llm_judge: If True, use LLM for semantic evaluation
        """
        self.use_llm_judge = use_llm_judge
        self.results = []

    def evaluate_answer(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
        context: List[str],
        abstained: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single answer.

        Args:
            question: User query
            generated_answer: System's answer
            ground_truth: Expected correct answer
            context: Retrieved context chunks
            abstained: Whether system abstained

        Returns:
            Dictionary with accuracy, faithfulness, citation metrics
        """
        # 1. Accuracy (exact match or semantic similarity)
        accuracy = self._compute_accuracy(generated_answer, ground_truth)

        # 2. Faithfulness (answer grounded in context)
        faithfulness = self._compute_faithfulness(generated_answer, context)

        # 3. Citation Precision
        citation_precision = self._compute_citation_precision(generated_answer, context)

        # 4. Hallucination Detection
        hallucination_rate = 1.0 - faithfulness

        metrics = {
            "accuracy": accuracy,
            "faithfulness": faithfulness,
            "citation_precision": citation_precision,
            "hallucination_rate": hallucination_rate,
            "abstained": 1.0 if abstained else 0.0
        }

        self.results.append(metrics)
        return metrics

    def _compute_accuracy(self, generated: str, ground_truth: str) -> float:
        """
        Compute accuracy score.

        For now, uses simple heuristics:
        - Exact match → 1.0
        - Contains key facts → 0.5-0.9
        - Completely wrong → 0.0
        """
        generated_lower = generated.lower()
        ground_truth_lower = ground_truth.lower()

        # Exact match
        if generated_lower.strip() == ground_truth_lower.strip():
            return 1.0

        # Extract numerical facts
        gen_numbers = self._extract_numbers(generated)
        gt_numbers = self._extract_numbers(ground_truth)

        if gen_numbers and gt_numbers:
            # Check if key numbers match
            matching_numbers = set(gen_numbers) & set(gt_numbers)
            if matching_numbers:
                return 0.8  # Contains correct numbers
            else:
                return 0.2  # Numbers don't match

        # Token overlap (simple baseline)
        gen_tokens = set(generated_lower.split())
        gt_tokens = set(ground_truth_lower.split())

        if not gt_tokens:
            return 0.0

        overlap = len(gen_tokens & gt_tokens) / len(gt_tokens)

        if overlap > 0.7:
            return 0.7
        elif overlap > 0.5:
            return 0.5
        elif overlap > 0.3:
            return 0.3
        else:
            return 0.0

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numerical values from text"""
        # Match integers, floats, percentages
        pattern = r'\d+\.?\d*%?'
        numbers = re.findall(pattern, text)
        return numbers

    def _compute_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Compute faithfulness score (answer grounded in context).

        Checks if claims in answer are supported by context.
        """
        if "INSUFFICIENT INFORMATION" in answer:
            return 1.0  # Abstention is faithful

        # Simple heuristic: check if answer tokens are in context
        answer_tokens = set(answer.lower().split())
        context_text = " ".join(context).lower()
        context_tokens = set(context_text.split())

        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
        answer_tokens -= stop_words
        context_tokens -= stop_words

        if not answer_tokens:
            return 0.5  # Neutral

        # Fraction of answer tokens in context
        supported_tokens = answer_tokens & context_tokens
        faithfulness = len(supported_tokens) / len(answer_tokens)

        return faithfulness

    def _compute_citation_precision(self, answer: str, context: List[str]) -> float:
        """
        Compute citation precision.

        Checks if citations ([Document N]) reference correct sources.
        """
        # Extract citations
        citations = re.findall(r'\[Document (\d+)\]', answer)

        if not citations:
            # No citations provided
            return 0.0

        # Simple check: citations are within valid range
        valid_citations = [int(c) for c in citations if 1 <= int(c) <= len(context)]

        precision = len(valid_citations) / len(citations) if citations else 0.0

        return precision

    def aggregate_metrics(self) -> AnswerMetrics:
        """Aggregate metrics across all answers"""
        if not self.results:
            return AnswerMetrics(0, 0, 0, 0, 0, 0)

        avg_accuracy = np.mean([r["accuracy"] for r in self.results])
        avg_faithfulness = np.mean([r["faithfulness"] for r in self.results])
        avg_citation = np.mean([r["citation_precision"] for r in self.results])
        avg_hallucination = np.mean([r["hallucination_rate"] for r in self.results])
        abstention_rate = np.mean([r["abstained"] for r in self.results])

        return AnswerMetrics(
            accuracy=avg_accuracy,
            faithfulness=avg_faithfulness,
            citation_precision=avg_citation,
            hallucination_rate=avg_hallucination,
            abstention_rate=abstention_rate,
            total_queries=len(self.results)
        )


class UncertaintyCalibrationEvaluator:
    """
    Evaluates uncertainty calibration: entropy vs correctness.
    """

    def __init__(self):
        """Initialize uncertainty evaluator"""
        self.results = []

    def evaluate_prediction(
        self,
        entropy: float,
        is_correct: bool,
        confidence: str
    ):
        """
        Record a single prediction with entropy and correctness.

        Args:
            entropy: Sequence entropy score
            is_correct: Whether answer was correct
            confidence: System confidence (HIGH, MEDIUM, LOW)
        """
        self.results.append({
            "entropy": entropy,
            "is_correct": is_correct,
            "confidence": confidence
        })

    def aggregate_metrics(self) -> UncertaintyMetrics:
        """Compute uncertainty calibration metrics"""
        if not self.results:
            return UncertaintyMetrics(0, 0, 0, 0, 0, 0)

        entropies = np.array([r["entropy"] for r in self.results])
        correctness = np.array([r["is_correct"] for r in self.results])

        # Separate entropy for correct vs incorrect
        correct_entropies = entropies[correctness == True]
        incorrect_entropies = entropies[correctness == False]

        entropy_correct_mean = np.mean(correct_entropies) if len(correct_entropies) > 0 else 0.0
        entropy_incorrect_mean = np.mean(incorrect_entropies) if len(incorrect_entropies) > 0 else 0.0

        # Correlation (Pearson)
        if len(entropies) > 1:
            correlation = np.corrcoef(entropies, ~correctness)[0, 1]  # ~correctness inverts (higher entropy → incorrect)
        else:
            correlation = 0.0

        # ROC AUC (treat entropy as predictor for incorrectness)
        auc_roc = self._compute_auc(entropies, ~correctness)

        # Expected Calibration Error
        ece = self._compute_ece()

        return UncertaintyMetrics(
            entropy_correct_mean=entropy_correct_mean,
            entropy_incorrect_mean=entropy_incorrect_mean,
            correlation=correlation,
            auc_roc=auc_roc,
            ece=ece,
            total_queries=len(self.results)
        )

    def _compute_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute ROC AUC.

        Simple trapezoidal approximation.
        """
        if len(set(labels)) < 2:
            return 0.5  # No discrimination possible

        # Sort by scores
        sorted_indices = np.argsort(scores)
        sorted_labels = labels[sorted_indices]

        # Compute TPR and FPR at different thresholds
        n_pos = np.sum(sorted_labels)
        n_neg = len(sorted_labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(~sorted_labels) / n_neg

        # Trapezoidal integration
        auc = np.trapz(tpr, fpr)

        return abs(auc)  # Ensure positive

    def _compute_ece(self, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures calibration: does confidence match accuracy?
        """
        if not self.results:
            return 0.0

        # Map confidence to probability
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3, "NONE": 0.0}
        confidences = np.array([confidence_map.get(r["confidence"], 0.5) for r in self.results])
        correctness = np.array([r["is_correct"] for r in self.results])

        # Bin predictions by confidence
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(bin_mask) == 0:
                continue

            bin_confidence = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(correctness[bin_mask])
            bin_size = np.sum(bin_mask)

            ece += (bin_size / len(self.results)) * abs(bin_confidence - bin_accuracy)

        return ece


# Helper functions for evaluation

def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """Compute hit rate (at least one relevant doc in top-k)"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Evaluation Metrics - Example Usage")
    print("=" * 70)

    # Example 1: Retrieval Evaluation
    print("\n1. Retrieval Evaluation")
    print("-" * 70)

    retrieval_eval = RetrievalEvaluator()

    # Simulate evaluation
    retrieval_eval.evaluate_query(
        retrieved_docs=["doc1", "doc2", "doc3", "doc4", "doc5"],
        relevant_docs=["doc1", "doc3", "doc6"],
        k=5
    )

    retrieval_eval.evaluate_query(
        retrieved_docs=["doc7", "doc8", "doc1", "doc9", "doc10"],
        relevant_docs=["doc1", "doc3"],
        k=5
    )

    metrics = retrieval_eval.aggregate_metrics()
    print(f"Precision@5: {metrics.precision_at_k:.3f}")
    print(f"Recall@5: {metrics.recall_at_k:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")
    print(f"NDCG@5: {metrics.ndcg_at_k:.3f}")

    # Example 2: Answer Evaluation
    print("\n2. Answer Evaluation")
    print("-" * 70)

    answer_eval = AnswerEvaluator()

    answer_eval.evaluate_answer(
        question="What was the GDP growth?",
        generated_answer="The GDP growth projection for 2023 is 1.6% [Document 1]",
        ground_truth="GDP growth was projected at 1.6% for 2023",
        context=["FOMC projects GDP growth of 1.6% in 2023"]
    )

    metrics = answer_eval.aggregate_metrics()
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Faithfulness: {metrics.faithfulness:.3f}")
    print(f"Citation Precision: {metrics.citation_precision:.3f}")

    # Example 3: Uncertainty Calibration
    print("\n3. Uncertainty Calibration")
    print("-" * 70)

    uncertainty_eval = UncertaintyCalibrationEvaluator()

    # Correct answers tend to have lower entropy
    uncertainty_eval.evaluate_prediction(entropy=0.8, is_correct=True, confidence="HIGH")
    uncertainty_eval.evaluate_prediction(entropy=0.9, is_correct=True, confidence="HIGH")
    uncertainty_eval.evaluate_prediction(entropy=1.8, is_correct=False, confidence="MEDIUM")
    uncertainty_eval.evaluate_prediction(entropy=2.2, is_correct=False, confidence="LOW")

    metrics = uncertainty_eval.aggregate_metrics()
    print(f"Entropy (Correct): {metrics.entropy_correct_mean:.3f}")
    print(f"Entropy (Incorrect): {metrics.entropy_incorrect_mean:.3f}")
    print(f"Correlation: {metrics.correlation:.3f}")
    print(f"ROC AUC: {metrics.auc_roc:.3f}")

    print("\n" + "=" * 70)
