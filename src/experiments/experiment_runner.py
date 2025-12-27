"""
Experiment Runner for Ablation Studies

This module runs individual experiments and collects results systematically.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.rag_system import UncertaintyAwareRAG, RAGResponse
from src.evaluation.evaluator import RAGEvaluator, EvaluationQuestion, load_test_questions
from src.evaluation.metrics import RetrievalMetrics, AnswerMetrics, UncertaintyMetrics
from src.utils.logger import logger
from src.utils.config import config

from .ablation_config import (
    ExperimentConfig,
    RetrievalStrategy,
    UncertaintyMethod,
    AbstentionStrategy,
    AlignmentType
)


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    # Metadata
    experiment_id: str
    experiment_name: str
    timestamp: str
    duration_seconds: float

    # Configuration
    config_dict: Dict[str, Any]

    # Metrics
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]

    # Per-question results
    question_results: List[Dict[str, Any]]

    # Statistics
    num_questions: int
    num_expansions: int
    num_abstentions: int
    expansion_rate: float
    abstention_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, output_dir: Path):
        """Save results to JSON"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.experiment_id}_results.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved experiment results to: {output_path}")


class ExperimentRunner:
    """
    Runs ablation experiments with different configurations.

    This class handles:
    1. Setting up RAG system with experiment config
    2. Running evaluation on test set
    3. Collecting metrics
    4. Saving results
    """

    def __init__(
        self,
        test_questions: List[EvaluationQuestion],
        output_dir: Path = None
    ):
        """
        Initialize experiment runner.

        Args:
            test_questions: List of test questions to evaluate
            output_dir: Directory to save results
        """
        self.test_questions = test_questions
        self.output_dir = output_dir or Path("data/experiments/ablation_studies")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment Runner initialized")
        logger.info(f"  Test questions: {len(test_questions)}")
        logger.info(f"  Output dir: {self.output_dir}")

    def run_experiment(
        self,
        exp_config: ExperimentConfig,
        verbose: bool = True
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.

        Args:
            exp_config: Experiment configuration
            verbose: Print progress

        Returns:
            ExperimentResult with all metrics
        """
        logger.info("=" * 70)
        logger.info(f"Running Experiment: {exp_config.experiment_id}")
        logger.info(f"Name: {exp_config.name}")
        logger.info(f"Description: {exp_config.description}")
        logger.info("=" * 70)

        start_time = time.time()

        # 1. Create RAG system with experiment config
        rag_system = self._create_rag_system(exp_config)

        # 2. Create evaluator
        evaluator = RAGEvaluator(
            rag_system=rag_system,
            output_dir=self.output_dir / exp_config.experiment_id
        )

        # 3. Run evaluation on all test questions
        question_results = []
        num_expansions = 0
        num_abstentions = 0

        for question in tqdm(self.test_questions, desc=f"Evaluating {exp_config.experiment_id}", disable=not verbose):
            result = evaluator.evaluate_single_question(question, verbose=False)

            # rag_response is a dict in the result
            rag_resp = result.rag_response

            # Track expansions and abstentions
            if rag_resp["retrieval_expanded"]:
                num_expansions += 1
            if rag_resp["abstained"]:
                num_abstentions += 1

            question_results.append({
                "question_id": question.question_id,
                "query": question.query,
                "answer": rag_resp["answer"],
                "entropy": rag_resp["entropy_score"],
                "confidence": rag_resp["confidence"],
                "expanded": rag_resp["retrieval_expanded"],
                "abstained": rag_resp["abstained"],
                "retrieval_metrics": result.retrieval_metrics,
                "answer_metrics": result.answer_metrics
            })

        # 4. Aggregate metrics
        report = evaluator.generate_report()

        # 5. Create experiment result
        duration = time.time() - start_time
        num_questions = len(self.test_questions)

        result = ExperimentResult(
            experiment_id=exp_config.experiment_id,
            experiment_name=exp_config.name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=duration,
            config_dict=self._config_to_dict(exp_config),
            retrieval_metrics=report["retrieval"],
            answer_metrics=report["answer_quality"],
            uncertainty_metrics=report["uncertainty_calibration"],
            question_results=question_results,
            num_questions=num_questions,
            num_expansions=num_expansions,
            num_abstentions=num_abstentions,
            expansion_rate=num_expansions / num_questions if num_questions > 0 else 0.0,
            abstention_rate=num_abstentions / num_questions if num_questions > 0 else 0.0
        )

        # 6. Save result
        result.save(self.output_dir)

        # 7. Print summary
        if verbose:
            self._print_result_summary(result)

        logger.info(f"✅ Experiment {exp_config.experiment_id} completed in {duration:.2f}s")

        return result

    def run_multiple_experiments(
        self,
        exp_configs: List[ExperimentConfig],
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments sequentially.

        Args:
            exp_configs: List of experiment configurations
            verbose: Print progress

        Returns:
            List of experiment results
        """
        logger.info("=" * 70)
        logger.info(f"Running {len(exp_configs)} experiments")
        logger.info("=" * 70)

        results = []
        for exp_config in exp_configs:
            if not exp_config.enabled:
                logger.info(f"⏭️  Skipping disabled experiment: {exp_config.experiment_id}")
                continue

            try:
                result = self.run_experiment(exp_config, verbose=verbose)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Experiment {exp_config.experiment_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        logger.info("=" * 70)
        logger.info(f"✅ Completed {len(results)}/{len(exp_configs)} experiments")
        logger.info("=" * 70)

        return results

    def _create_rag_system(self, exp_config: ExperimentConfig) -> UncertaintyAwareRAG:
        """
        Create RAG system with experiment configuration.

        This temporarily overrides system config based on experiment settings.
        """
        # Create a modified RAG system based on experiment config
        rag_system = UncertaintyAwareRAG()

        # Override retrieval strategy
        if exp_config.retrieval_strategy == RetrievalStrategy.FIXED_K5:
            rag_system.adaptive_retrieval_enabled = False
            rag_system.initial_k = 5
        elif exp_config.retrieval_strategy == RetrievalStrategy.FIXED_K10:
            rag_system.adaptive_retrieval_enabled = False
            rag_system.initial_k = 10
        elif exp_config.retrieval_strategy == RetrievalStrategy.ADAPTIVE:
            rag_system.adaptive_retrieval_enabled = True
            rag_system.initial_k = exp_config.initial_k
            rag_system.expanded_k = exp_config.expanded_k

        # Override uncertainty method
        if exp_config.uncertainty_method == UncertaintyMethod.NONE:
            rag_system.uncertainty_enabled = False
        elif exp_config.uncertainty_method == UncertaintyMethod.SEMANTIC:
            rag_system.uncertainty_enabled = True
            # Semantic uncertainty uses a different approach (multiple generations + embeddings)
            # It doesn't use entropy aggregation
        else:
            rag_system.uncertainty_enabled = True
            # Extract aggregation method from enum value: "entropy_mean" → "mean"
            rag_system.entropy_aggregation_method = exp_config.uncertainty_method.value.split('_')[1]

        # Override abstention strategy
        if exp_config.abstention_strategy == AbstentionStrategy.NEVER:
            rag_system.abstention_enabled = False
        else:
            rag_system.abstention_enabled = True

        # Override thresholds
        rag_system.entropy_expansion_threshold = exp_config.entropy_expansion_threshold
        rag_system.entropy_abstention_threshold = exp_config.entropy_abstention_threshold

        # Override temperature
        rag_system.llm.temperature = exp_config.temperature

        # Override model (for alignment experiments)
        if exp_config.alignment_type == AlignmentType.DPO:
            # Load aligned model
            # TODO: Implement model swapping
            logger.warning("DPO model loading not yet implemented")

        # Override logprobs usage
        rag_system.use_logprobs = exp_config.use_logprobs

        return rag_system

    def _config_to_dict(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """Convert experiment config to dictionary"""
        return {
            "experiment_id": exp_config.experiment_id,
            "name": exp_config.name,
            "description": exp_config.description,
            "retrieval_strategy": exp_config.retrieval_strategy.value,
            "uncertainty_method": exp_config.uncertainty_method.value,
            "abstention_strategy": exp_config.abstention_strategy.value,
            "alignment_type": exp_config.alignment_type.value,
            "initial_k": exp_config.initial_k,
            "expanded_k": exp_config.expanded_k,
            "entropy_expansion_threshold": exp_config.entropy_expansion_threshold,
            "entropy_abstention_threshold": exp_config.entropy_abstention_threshold,
            "temperature": exp_config.temperature,
            "model_name": exp_config.model_name,
            "use_logprobs": exp_config.use_logprobs
        }

    def _print_result_summary(self, result: ExperimentResult):
        """Print summary of experiment results"""
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nExperiment: {result.experiment_name}")
        logger.info(f"ID: {result.experiment_id}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")

        logger.info(f"\n📊 System Behavior:")
        logger.info(f"  Questions: {result.num_questions}")
        logger.info(f"  Expansions: {result.num_expansions} ({result.expansion_rate:.1%})")
        logger.info(f"  Abstentions: {result.num_abstentions} ({result.abstention_rate:.1%})")

        logger.info(f"\n🔍 Retrieval Metrics:")
        for metric, value in result.retrieval_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")

        logger.info(f"\n💬 Answer Metrics:")
        for metric, value in result.answer_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")

        if result.uncertainty_metrics:
            logger.info(f"\n📈 Uncertainty Metrics:")
            for metric, value in result.uncertainty_metrics.items():
                logger.info(f"  {metric}: {value:.3f}")

        logger.info("\n" + "=" * 70)


def compare_experiments(
    results: List[ExperimentResult],
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Compare multiple experiment results.

    Args:
        results: List of experiment results
        output_dir: Where to save comparison

    Returns:
        Comparison dictionary
    """
    if not results:
        return {}

    output_dir = output_dir or Path("data/experiments/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "num_experiments": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": []
    }

    for result in results:
        comparison["experiments"].append({
            "experiment_id": result.experiment_id,
            "name": result.experiment_name,
            "retrieval_metrics": result.retrieval_metrics,
            "answer_metrics": result.answer_metrics,
            "uncertainty_metrics": result.uncertainty_metrics,
            "expansion_rate": result.expansion_rate,
            "abstention_rate": result.abstention_rate
        })

    # Save comparison
    output_path = output_dir / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Saved comparison to: {output_path}")

    return comparison
