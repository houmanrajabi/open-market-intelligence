"""
Run Ablation Studies

This script runs systematic ablation studies to evaluate the contribution
of each component to the system's performance.

Usage:
    # Run all experiments
    python -m scripts.run_ablation_studies --run-all

    # Run specific experiment group
    python -m scripts.run_ablation_studies --group baseline

    # Run specific research question
    python -m scripts.run_ablation_studies --rq RQ1

    # Run single experiment
    python -m scripts.run_ablation_studies --experiment exp_001_baseline

    # Compare existing results
    python -m scripts.run_ablation_studies --compare
"""

import argparse
import sys
from pathlib import Path
from typing import List

from src.experiments.ablation_config import (
    ALL_EXPERIMENTS,
    BASELINE_EXPERIMENTS,
    UNCERTAINTY_METHOD_EXPERIMENTS,
    ADAPTIVE_RETRIEVAL_EXPERIMENTS,
    THRESHOLD_SENSITIVITY_EXPERIMENTS,
    TEMPERATURE_EXPERIMENTS,
    ALIGNMENT_EXPERIMENTS,
    ENTROPY_AGGREGATION_EXPERIMENTS,
    RESEARCH_QUESTIONS,
    get_experiment_by_id,
    get_experiments_for_rq,
    print_experiment_summary
)
from src.experiments.experiment_runner import ExperimentRunner, ExperimentResult, compare_experiments
from src.evaluation.evaluator import load_test_questions
from src.utils.logger import logger
from src.utils.config import config


def run_experiments_group(group_name: str, test_questions_path: Path):
    """Run a specific group of experiments"""
    # Map group names to experiment lists
    groups = {
        "baseline": BASELINE_EXPERIMENTS,
        "uncertainty": UNCERTAINTY_METHOD_EXPERIMENTS,
        "adaptive": ADAPTIVE_RETRIEVAL_EXPERIMENTS,
        "thresholds": THRESHOLD_SENSITIVITY_EXPERIMENTS,
        "temperature": TEMPERATURE_EXPERIMENTS,
        "alignment": ALIGNMENT_EXPERIMENTS,
        "entropy_agg": ENTROPY_AGGREGATION_EXPERIMENTS,
        "all": ALL_EXPERIMENTS
    }

    if group_name not in groups:
        logger.error(f"Unknown group: {group_name}")
        logger.error(f"Available groups: {list(groups.keys())}")
        sys.exit(1)

    experiments = groups[group_name]

    logger.info("=" * 70)
    logger.info(f"Running Experiment Group: {group_name.upper()}")
    logger.info(f"Number of experiments: {len(experiments)}")
    logger.info("=" * 70)

    # Load test questions
    test_questions = load_test_questions(test_questions_path)

    # Create runner
    runner = ExperimentRunner(test_questions)

    # Run experiments
    results = runner.run_multiple_experiments(experiments, verbose=True)

    # Compare results
    comparison = compare_experiments(results)

    logger.info("=" * 70)
    logger.info(f"✅ Completed {group_name.upper()} experiment group")
    logger.info(f"Results saved to: data/experiments/ablation_studies/")
    logger.info("=" * 70)


def run_research_question(rq_id: str, test_questions_path: Path):
    """Run experiments for a specific research question"""
    if rq_id not in RESEARCH_QUESTIONS:
        logger.error(f"Unknown research question: {rq_id}")
        logger.error(f"Available: {list(RESEARCH_QUESTIONS.keys())}")
        sys.exit(1)

    rq_info = RESEARCH_QUESTIONS[rq_id]

    logger.info("=" * 70)
    logger.info(f"Research Question: {rq_id}")
    logger.info(f"Question: {rq_info['question']}")
    logger.info(f"Hypothesis: {rq_info['hypothesis']}")
    logger.info(f"Experiments: {len(rq_info['experiments'])}")
    logger.info("=" * 70)

    # Get experiments for this RQ
    experiments = get_experiments_for_rq(rq_id)

    # Load test questions
    test_questions = load_test_questions(test_questions_path)

    # Create runner
    runner = ExperimentRunner(test_questions)

    # Run experiments
    results = runner.run_multiple_experiments(experiments, verbose=True)

    # Compare results
    comparison = compare_experiments(results)

    # Analyze results for research question
    logger.info("\n" + "=" * 70)
    logger.info(f"ANALYSIS FOR {rq_id}")
    logger.info("=" * 70)

    logger.info(f"\nResearch Question: {rq_info['question']}")
    logger.info(f"Hypothesis: {rq_info['hypothesis']}")

    logger.info(f"\nKey Metrics to Compare:")
    for metric in rq_info['metrics']:
        logger.info(f"  - {metric}")

    logger.info(f"\nResults Summary:")
    for result in results:
        logger.info(f"\n  {result.experiment_name}:")
        for metric in rq_info['metrics']:
            # Try to find metric in different categories
            value = None
            if metric in result.retrieval_metrics:
                value = result.retrieval_metrics[metric]
            elif metric in result.answer_metrics:
                value = result.answer_metrics[metric]
            elif metric in result.uncertainty_metrics:
                value = result.uncertainty_metrics[metric]
            elif metric == "retrieval_expansion_rate":
                value = result.expansion_rate
            elif metric == "abstention_rate":
                value = result.abstention_rate

            if value is not None:
                logger.info(f"    {metric}: {value:.3f}")
            else:
                logger.info(f"    {metric}: N/A")

    logger.info("\n" + "=" * 70)


def run_single_experiment(exp_id: str, test_questions_path: Path):
    """Run a single experiment"""
    exp_config = get_experiment_by_id(exp_id)

    if exp_config is None:
        logger.error(f"Experiment not found: {exp_id}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"Running Single Experiment: {exp_id}")
    logger.info("=" * 70)

    # Load test questions
    test_questions = load_test_questions(test_questions_path)

    # Create runner
    runner = ExperimentRunner(test_questions)

    # Run experiment
    result = runner.run_experiment(exp_config, verbose=True)

    logger.info("=" * 70)
    logger.info(f"✅ Experiment {exp_id} completed")
    logger.info(f"Results saved to: data/experiments/ablation_studies/{exp_id}/")
    logger.info("=" * 70)


def compare_all_results(output_dir: Path):
    """Compare all existing experiment results"""
    results_dir = output_dir or Path("data/experiments/ablation_studies")

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Loading all experiment results for comparison")
    logger.info("=" * 70)

    # Load all result files
    import json
    results = []

    for result_file in results_dir.glob("*/*_results.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            # Convert back to ExperimentResult
            result = ExperimentResult(**result_data)
            results.append(result)
            logger.info(f"Loaded: {result.experiment_id}")

    if not results:
        logger.warning("No experiment results found")
        return

    logger.info(f"\nLoaded {len(results)} experiment results")

    # Compare
    comparison = compare_experiments(results)

    logger.info("=" * 70)
    logger.info(f"✅ Comparison saved to: data/experiments/comparisons/")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Ablation Studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments"
    )
    group.add_argument(
        "--group",
        type=str,
        choices=["baseline", "uncertainty", "adaptive", "thresholds", "temperature", "alignment", "entropy_agg"],
        help="Run specific experiment group"
    )
    group.add_argument(
        "--rq",
        type=str,
        choices=list(RESEARCH_QUESTIONS.keys()),
        help="Run experiments for research question"
    )
    group.add_argument(
        "--experiment",
        type=str,
        help="Run single experiment by ID"
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare all existing results"
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all experiments"
    )

    # Test set
    parser.add_argument(
        "--test-set",
        type=Path,
        default=config.evaluation.default_test_set,
        help="Path to test questions JSON"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/ablation_studies"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Handle actions
    if args.list:
        print_experiment_summary()
        return

    if args.compare:
        compare_all_results(args.output_dir)
        return

    # Validate test set exists
    if not args.test_set.exists():
        logger.error(f"Test set not found: {args.test_set}")
        sys.exit(1)

    # Run experiments
    try:
        if args.run_all:
            run_experiments_group("all", args.test_set)
        elif args.group:
            run_experiments_group(args.group, args.test_set)
        elif args.rq:
            run_research_question(args.rq, args.test_set)
        elif args.experiment:
            run_single_experiment(args.experiment, args.test_set)
        else:
            parser.print_help()
            logger.info("\nNo action specified. Use --help for options.")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
