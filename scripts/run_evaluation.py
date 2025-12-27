"""
Run RAG System Evaluation

Usage:
    python -m scripts.run_evaluation --test-set data/test_set/sample_test_questions.json
"""

import argparse
from pathlib import Path
import sys

from src.evaluation.evaluator import RAGEvaluator, load_test_questions
from src.rag_system import UncertaintyAwareRAG
from src.utils.logger import logger


def main():
    """Main entry point for evaluation"""

    parser = argparse.ArgumentParser(
        description="Evaluate RAG system on test questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/test_set/sample_test_questions.json"),
        help="Path to test questions JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--disable-uncertainty",
        action="store_true",
        help="Disable uncertainty features (baseline RAG)"
    )

    parser.add_argument(
        "--disable-expansion",
        action="store_true",
        help="Disable retrieval expansion"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Validate test set exists
    if not args.test_set.exists():
        logger.error(f"Test set not found: {args.test_set}")
        logger.error("Please create a test set JSON file or specify existing one")
        sys.exit(1)

    # Load test questions
    try:
        test_questions = load_test_questions(args.test_set)
    except Exception as e:
        logger.error(f"Failed to load test questions: {e}")
        sys.exit(1)

    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING RAG SYSTEM EVALUATION")
    logger.info(f"{'='*70}")
    logger.info(f"Test Set: {args.test_set}")
    logger.info(f"Questions: {len(test_questions)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Uncertainty Enabled: {not args.disable_uncertainty}")
    logger.info(f"Expansion Enabled: {not args.disable_expansion}")
    logger.info(f"{'='*70}\n")

    # Initialize RAG system
    try:
        rag_system = UncertaintyAwareRAG(
            enable_uncertainty=not args.disable_uncertainty,
            enable_expansion=not args.disable_expansion
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error("\nMake sure:")
        logger.error("  1. Vector database is populated (run scripts/ingest_to_vectordb.py)")
        logger.error("  2. vLLM server is running")
        logger.error("  3. LLM__API_BASE_URL is configured in .env")
        sys.exit(1)

    # Initialize evaluator
    evaluator = RAGEvaluator(
        rag_system=rag_system,
        output_dir=args.output_dir
    )

    # Run evaluation
    try:
        report = evaluator.evaluate_test_set(
            test_questions=test_questions,
            verbose=args.verbose
        )

        # Print report
        evaluator.print_report(report)

        logger.info("\n✅ Evaluation completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
