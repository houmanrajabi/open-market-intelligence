"""
Red-Team Evaluation Script

Runs adversarial testing on the RAG system to evaluate robustness.

Usage:
    python -m scripts.run_red_team_evaluation
    python -m scripts.run_red_team_evaluation --category out_of_domain
    python -m scripts.run_red_team_evaluation --output-dir data/red_team_results/run1
"""

import argparse
from pathlib import Path

from src.rag_system import UncertaintyAwareRAG
from src.red_team import (
    RedTeamEvaluator,
    load_adversarial_test_set,
    AttackCategory
)
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run red-team adversarial evaluation on RAG system"
    )

    parser.add_argument(
        "--test-set",
        type=str,
        default="data/test_set/adversarial_questions.json",
        help="Path to adversarial test set JSON"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by attack category (e.g., out_of_domain, leading_question)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/red_team_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-question results"
    )

    args = parser.parse_args()

    # Paths
    test_set_path = Path(args.test_set)
    output_dir = Path(args.output_dir)

    logger.info("="*70)
    logger.info("RED-TEAM ADVERSARIAL EVALUATION")
    logger.info("="*70)

    # Load adversarial test set
    logger.info(f"\n📂 Loading adversarial test set from: {test_set_path}")
    all_questions = load_adversarial_test_set(test_set_path)
    logger.info(f"✅ Loaded {len(all_questions)} adversarial questions")

    # Filter by category if specified
    if args.category:
        questions = [q for q in all_questions if q.attack_category == args.category]
        logger.info(f"🔍 Filtered to {len(questions)} questions in category: {args.category}")
    else:
        questions = all_questions

    if not questions:
        logger.error("No questions to evaluate!")
        return

    # Initialize RAG system
    logger.info(f"\n🚀 Initializing RAG system...")
    rag_system = UncertaintyAwareRAG()
    logger.info("✅ RAG system initialized")

    # Create red-team evaluator
    logger.info(f"\n🎯 Creating Red-Team Evaluator...")
    evaluator = RedTeamEvaluator(
        rag_system=rag_system,
        output_dir=output_dir
    )

    # Run evaluation
    logger.info(f"\n🏃 Running red-team evaluation on {len(questions)} questions...")
    results = evaluator.evaluate_all(questions, verbose=True)

    # Generate report
    logger.info(f"\n📊 Generating report...")
    report = evaluator.generate_report(results)

    # Print report
    evaluator.print_report(report)

    # Save results
    filename = f"red_team_{args.category}.json" if args.category else "red_team_all.json"
    evaluator.save_results(results, report, filename=filename)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"📊 Overall Robustness Score: {report.overall_robustness_score:.1%}")
    logger.info(f"✅ Correct Behaviors: {sum(1 for r in results if r.behavior_correct)}/{len(results)}")
    logger.info(f"❌ Critical Failures: {len(report.critical_failures)}")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
