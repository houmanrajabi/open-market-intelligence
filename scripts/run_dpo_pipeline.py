"""
Complete DPO Training Pipeline

This script runs the full RLAIF pipeline:
1. Generate preference pairs from queries
2. Grade answers with Teacher model (GPT-4)
3. Train Student model (Llama-3) with DPO
4. Evaluate aligned model

Usage:
    # Step 1: Generate preference pairs
    python -m scripts.run_dpo_pipeline generate --queries-file data/test_set/sample_test_questions.json --num-pairs 100

    # Step 2: Train with DPO
    python -m scripts.run_dpo_pipeline train --pairs-file data/alignment/preference_pairs/preference_pairs.json

    # Step 3: Evaluate aligned model
    python -m scripts.run_dpo_pipeline evaluate --model-path models/llama3_dpo_aligned
"""

import argparse
import json
import sys
from pathlib import Path

from src.alignment.preference_generator import PreferenceGenerator
from src.alignment.dpo_trainer import DPOTrainerWrapper, DPOTrainingConfig
from src.rag_system import UncertaintyAwareRAG
from src.evaluation.evaluator import RAGEvaluator, load_test_questions
from src.utils.config import config
from src.utils.logger import logger


def generate_pairs_command(args):
    """Generate preference pairs"""
    logger.info("=" * 70)
    logger.info("STEP 1: GENERATING PREFERENCE PAIRS")
    logger.info("=" * 70)

    # Load queries
    if args.queries_file:
        # Load from test questions file
        test_questions = load_test_questions(args.queries_file)
        queries = [q.query for q in test_questions]
    else:
        # Use default queries
        queries = [
            "What was the GDP growth projection for 2023?",
            "What did Chair Powell say about inflation in December 2023?",
            "What was the unemployment rate projection for 2024?",
            "What is the FOMC's inflation target?",
            "What was discussed about the housing sector in November 2020?"
        ]

    logger.info(f"Loaded {len(queries)} queries")

    # Limit number of pairs
    if args.num_pairs:
        queries = queries[:args.num_pairs]
        logger.info(f"Generating {args.num_pairs} pairs (limited)")

    # Initialize generator
    generator = PreferenceGenerator()

    # Generate pairs
    if args.use_contrastive:
        pairs = generator.generate_contrastive_pairs(queries)
    else:
        pairs = generator.generate_from_queries(queries, num_pairs_per_query=1)

    # Save pairs
    generator.save_pairs(
        filename=args.output_file,
        format="dpo"
    )

    logger.info(f"\n✅ Generated {len(pairs)} preference pairs")
    logger.info(f"Saved to: {generator.output_dir}")


def train_command(args):
    """Train model with DPO"""
    logger.info("=" * 70)
    logger.info("STEP 2: DPO TRAINING")
    logger.info("=" * 70)

    # Validate pairs file
    pairs_file = Path(args.pairs_file)
    if not pairs_file.exists():
        logger.error(f"Preference pairs file not found: {pairs_file}")
        logger.error("Run 'generate' command first to create pairs")
        sys.exit(1)

    # Configure training (override defaults from config.alignment with command-line args)
    training_config = DPOTrainingConfig(
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        use_lora=not args.disable_lora,
        use_4bit=not args.disable_4bit
    )

    logger.info(f"Training configuration:")
    logger.info(f"  Model: {training_config.model_name}")
    logger.info(f"  Beta (KL penalty): {training_config.beta}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Epochs: {training_config.num_train_epochs}")
    logger.info(f"  LoRA: {training_config.use_lora}")
    logger.info(f"  4-bit: {training_config.use_4bit}")

    # Initialize trainer
    trainer = DPOTrainerWrapper(training_config)

    # Load preference pairs
    train_dataset = trainer.load_preference_pairs(pairs_file)

    # Split train/eval if requested
    eval_dataset = None
    if args.eval_split > 0:
        split_idx = int(len(train_dataset) * (1 - args.eval_split))
        eval_dataset = train_dataset.select(range(split_idx, len(train_dataset)))
        train_dataset = train_dataset.select(range(split_idx))
        logger.info(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Prepare model
    logger.info("\nPreparing model (this may take a few minutes)...")
    trainer.prepare_model()

    # Train
    logger.info("\nStarting DPO training...")
    trainer.train(train_dataset, eval_dataset)

    logger.info(f"\n✅ Training complete! Model saved to: {training_config.output_dir}")


def evaluate_command(args):
    """Evaluate aligned model"""
    logger.info("=" * 70)
    logger.info("STEP 3: EVALUATING ALIGNED MODEL")
    logger.info("=" * 70)

    # Load test questions
    test_questions = load_test_questions(args.test_set)

    # TODO: Initialize RAG system with aligned model
    # This requires modifying LlamaInterface to load local models
    logger.warning("Evaluation with custom models not yet implemented")
    logger.info("To evaluate aligned model:")
    logger.info("1. Deploy aligned model with vLLM")
    logger.info("2. Update LLM__API_BASE_URL in .env")
    logger.info("3. Run: python -m scripts.run_evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="DPO Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate pairs command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate preference pairs"
    )
    generate_parser.add_argument(
        "--queries-file",
        type=Path,
        help="JSON file with test questions"
    )
    generate_parser.add_argument(
        "--num-pairs",
        type=int,
        help="Max number of pairs to generate"
    )
    generate_parser.add_argument(
        "--use-contrastive",
        action="store_true",
        help="Use contrastive pair generation (corruption strategies)"
    )
    generate_parser.add_argument(
        "--output-file",
        type=str,
        default="preference_pairs.json",
        help="Output filename"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train model with DPO"
    )
    train_parser.add_argument(
        "--pairs-file",
        type=str,
        required=True,
        help="JSON file with preference pairs"
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default=config.alignment.student_model_name,
        help="Base model name"
    )
    train_parser.add_argument(
        "--beta",
        type=float,
        default=config.alignment.dpo_beta,
        help="DPO beta parameter (KL penalty)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.alignment.dpo_learning_rate,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=config.alignment.dpo_num_epochs,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=config.alignment.dpo_batch_size,
        help="Batch size per device"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.alignment.aligned_model_output_dir),
        help="Output directory for trained model"
    )
    train_parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (0-1)"
    )
    train_parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )
    train_parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate aligned model"
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default=str(config.alignment.aligned_model_output_dir),
        help="Path to aligned model"
    )
    eval_parser.add_argument(
        "--test-set",
        type=Path,
        default=config.evaluation.default_test_set,
        help="Test questions file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        if args.command == "generate":
            generate_pairs_command(args)
        elif args.command == "train":
            train_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
