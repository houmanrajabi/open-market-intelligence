"""
RAG System Evaluator

Comprehensive evaluation runner that:
1. Loads test questions with ground truth
2. Runs RAG system on each question
3. Computes all metrics (retrieval, answer quality, uncertainty)
4. Generates evaluation report
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.rag_system import UncertaintyAwareRAG, RAGResponse
from src.evaluation.metrics import (
    RetrievalEvaluator,
    AnswerEvaluator,
    UncertaintyCalibrationEvaluator,
    compute_f1_score
)
from src.utils.config import config
from src.utils.logger import logger


@dataclass
class EvaluationQuestion:
    """
    Test question with ground truth.

    Attributes:
        question_id: Unique identifier
        query: User question
        expected_answer: Ground truth answer
        relevant_docs: List of document IDs that should be retrieved
        expected_values: Key facts/numbers in expected answer
        question_type: Type of question (factual, comparison, etc.)
        difficulty: 1-5 scale
    """
    question_id: str
    query: str
    expected_answer: str
    relevant_docs: List[str]
    expected_values: Optional[List[str]] = None
    question_type: str = "factual"
    difficulty: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationQuestion":
        """Create from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Results for a single question"""
    question_id: str
    query: str
    rag_response: Dict[str, Any]
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    is_correct: bool
    latency: float


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.
    """

    def __init__(
        self,
        rag_system: Optional[UncertaintyAwareRAG] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator.

        Args:
            rag_system: RAG system to evaluate (creates new one if None)
            output_dir: Directory to save evaluation results (defaults to config.evaluation.eval_output_dir)
        """
        self.rag_system = rag_system or UncertaintyAwareRAG()
        self.output_dir = output_dir or config.evaluation.eval_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric evaluators
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator()
        self.uncertainty_evaluator = UncertaintyCalibrationEvaluator()

        # Store individual results
        self.results: List[EvaluationResult] = []

        logger.info(f"RAG Evaluator initialized. Output dir: {self.output_dir}")

    def evaluate_single_question(
        self,
        eval_question: EvaluationQuestion,
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate RAG system on a single question.

        Args:
            eval_question: Question with ground truth
            verbose: Print detailed logs

        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()

        if verbose:
            logger.info(f"\nEvaluating: {eval_question.query}")

        # 1. Run RAG system
        rag_response = self.rag_system.answer_query(eval_question.query)

        # 2. Evaluate Retrieval
        retrieved_doc_ids = [
            source["doc_id"] for source in rag_response.sources
        ]

        retrieval_metrics = self.retrieval_evaluator.evaluate_query(
            retrieved_docs=retrieved_doc_ids,
            relevant_docs=eval_question.relevant_docs,
            k=5
        )

        # 3. Evaluate Answer Quality
        context_texts = [source["content"] for source in rag_response.sources]

        answer_metrics = self.answer_evaluator.evaluate_answer(
            question=eval_question.query,
            generated_answer=rag_response.answer,
            ground_truth=eval_question.expected_answer,
            context=context_texts,
            abstained=rag_response.abstained
        )

        # Determine if answer is correct (using accuracy threshold from config)
        is_correct = answer_metrics["accuracy"] >= config.evaluation.accuracy_threshold

        # 4. Record Uncertainty
        self.uncertainty_evaluator.evaluate_prediction(
            entropy=rag_response.entropy_score,
            is_correct=is_correct,
            confidence=rag_response.confidence
        )

        latency = time.time() - start_time

        # Create result
        result = EvaluationResult(
            question_id=eval_question.question_id,
            query=eval_question.query,
            rag_response=rag_response.to_dict(),
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            is_correct=is_correct,
            latency=latency
        )

        self.results.append(result)

        if verbose:
            logger.info(f"  Accuracy: {answer_metrics['accuracy']:.2f}")
            logger.info(f"  Precision@5: {retrieval_metrics['precision_at_k']:.2f}")
            logger.info(f"  Entropy: {rag_response.entropy_score:.3f}")
            logger.info(f"  Correct: {is_correct}")

        return result

    def evaluate_test_set(
        self,
        test_questions: List[EvaluationQuestion],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on a full test set.

        Args:
            test_questions: List of test questions
            verbose: Print progress

        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATING RAG SYSTEM ON {len(test_questions)} QUESTIONS")
        logger.info(f"{'='*70}\n")

        # Evaluate each question
        for eval_q in tqdm(test_questions, desc="Evaluating", disable=not verbose):
            self.evaluate_single_question(eval_q, verbose=False)

        # Aggregate metrics
        report = self.generate_report()

        # Save results
        self.save_results(report)

        return report

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Returns:
            Dictionary with all aggregated metrics
        """
        # Aggregate metrics from evaluators
        retrieval_metrics = self.retrieval_evaluator.aggregate_metrics()
        answer_metrics = self.answer_evaluator.aggregate_metrics()
        uncertainty_metrics = self.uncertainty_evaluator.aggregate_metrics()

        # Compute additional metrics
        total_queries = len(self.results)
        correct_count = sum(1 for r in self.results if r.is_correct)
        accuracy = correct_count / total_queries if total_queries > 0 else 0.0

        avg_latency = sum(r.latency for r in self.results) / total_queries if total_queries > 0 else 0.0

        # Expansion and abstention stats
        expansions = sum(1 for r in self.results if r.rag_response["retrieval_expanded"])
        abstentions = sum(1 for r in self.results if r.rag_response["abstained"])

        # Compute F1 score
        f1_retrieval = compute_f1_score(
            retrieval_metrics.precision_at_k,
            retrieval_metrics.recall_at_k
        )

        report = {
            "summary": {
                "total_queries": total_queries,
                "correct_answers": correct_count,
                "overall_accuracy": accuracy,
                "avg_latency": avg_latency,
                "expansions_triggered": expansions,
                "expansion_rate": expansions / total_queries if total_queries > 0 else 0.0,
                "abstentions": abstentions,
                "abstention_rate": abstentions / total_queries if total_queries > 0 else 0.0
            },
            "retrieval": {
                "precision_at_5": retrieval_metrics.precision_at_k,
                "recall_at_5": retrieval_metrics.recall_at_k,
                "f1_score": f1_retrieval,
                "mrr": retrieval_metrics.mrr,
                "ndcg_at_5": retrieval_metrics.ndcg_at_k
            },
            "answer_quality": {
                "accuracy": answer_metrics.accuracy,
                "faithfulness": answer_metrics.faithfulness,
                "citation_precision": answer_metrics.citation_precision,
                "hallucination_rate": answer_metrics.hallucination_rate,
                "abstention_rate": answer_metrics.abstention_rate
            },
            "uncertainty_calibration": {
                "entropy_correct_mean": uncertainty_metrics.entropy_correct_mean,
                "entropy_incorrect_mean": uncertainty_metrics.entropy_incorrect_mean,
                "entropy_gap": uncertainty_metrics.entropy_incorrect_mean - uncertainty_metrics.entropy_correct_mean,
                "correlation": uncertainty_metrics.correlation,
                "auc_roc": uncertainty_metrics.auc_roc,
                "ece": uncertainty_metrics.ece
            },
            "system_stats": self.rag_system.get_statistics()
        }

        return report

    def save_results(self, report: Dict[str, Any], filename: str = None):
        """
        Save evaluation results to JSON.

        Args:
            report: Evaluation report
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"

        output_path = self.output_dir / filename

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Evaluation report saved: {output_path}")

        # Also save detailed results
        detailed_path = self.output_dir / f"detailed_{filename}"
        detailed_results = [
            {
                "question_id": r.question_id,
                "query": r.query,
                "answer": r.rag_response["answer"],
                "is_correct": r.is_correct,
                "confidence": r.rag_response["confidence"],
                "entropy": r.rag_response["entropy_score"],
                "retrieval_expanded": r.rag_response["retrieval_expanded"],
                "abstained": r.rag_response["abstained"],
                "retrieval_metrics": r.retrieval_metrics,
                "answer_metrics": r.answer_metrics,
                "latency": r.latency
            }
            for r in self.results
        ]

        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Detailed results saved: {detailed_path}")

    def print_report(self, report: Dict[str, Any]):
        """Print evaluation report to console"""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        print("\n📊 SUMMARY")
        print("-" * 70)
        for key, value in report["summary"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print("\n🔍 RETRIEVAL METRICS")
        print("-" * 70)
        for key, value in report["retrieval"].items():
            print(f"  {key}: {value:.3f}")

        print("\n📝 ANSWER QUALITY")
        print("-" * 70)
        for key, value in report["answer_quality"].items():
            print(f"  {key}: {value:.3f}")

        print("\n📈 UNCERTAINTY CALIBRATION")
        print("-" * 70)
        for key, value in report["uncertainty_calibration"].items():
            print(f"  {key}: {value:.3f}")

        print("\n" + "=" * 70)


def load_test_questions(test_file: Path) -> List[EvaluationQuestion]:
    """
    Load test questions from JSON file.

    Expected format:
    [
        {
            "question_id": "q001",
            "query": "What was the GDP projection?",
            "expected_answer": "1.6% for 2023",
            "relevant_docs": ["fomcprojtabl20221214"],
            "expected_values": ["1.6%", "2023"],
            "question_type": "factual",
            "difficulty": 2
        },
        ...
    ]
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [EvaluationQuestion.from_dict(q) for q in data]

    logger.info(f"Loaded {len(questions)} test questions from {test_file}")

    return questions


# Example usage
if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("RAG System Evaluator - Example Usage")
    print("=" * 70)

    # Create sample test questions
    sample_questions = [
        EvaluationQuestion(
            question_id="q001",
            query="What was the GDP growth projection for 2023?",
            expected_answer="The GDP growth projection for 2023 was 1.6%",
            relevant_docs=["fomcprojtabl20221214"],
            expected_values=["1.6%", "2023"],
            question_type="factual",
            difficulty=2
        ),
        EvaluationQuestion(
            question_id="q002",
            query="What did Chair Powell say about inflation in December 2023?",
            expected_answer="Chair Powell discussed bringing inflation back to 2% target",
            relevant_docs=["FOMCpresconf20231213"],
            expected_values=["2%", "inflation"],
            question_type="factual",
            difficulty=3
        ),
        EvaluationQuestion(
            question_id="q003",
            query="What was the unemployment rate projection for 2024?",
            expected_answer="The unemployment rate was projected at 4.6% for 2024",
            relevant_docs=["fomcprojtabl20221214"],
            expected_values=["4.6%", "2024"],
            question_type="factual",
            difficulty=2
        )
    ]

    print("\nCreating evaluator...")
    evaluator = RAGEvaluator()

    print(f"\nEvaluating {len(sample_questions)} sample questions...")
    report = evaluator.evaluate_test_set(sample_questions, verbose=True)

    evaluator.print_report(report)

    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")
