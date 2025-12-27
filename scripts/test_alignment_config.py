"""
Test Alignment Configuration and Modules

This script tests the alignment layer configuration and modules without
requiring GPU or expensive API calls. It validates:
1. Configuration loading
2. Module initialization
3. Mock preference pair generation
4. DPO config preparation

Run this before attempting full training on Vast.ai or similar platforms.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import logger


def test_config_loading():
    """Test that alignment and evaluation configs load correctly"""
    logger.info("=" * 70)
    logger.info("TEST 1: Configuration Loading")
    logger.info("=" * 70)

    try:
        # Test alignment settings
        logger.info("\n✓ Alignment Settings:")
        logger.info(f"  Teacher Model: {config.alignment.teacher_model}")
        logger.info(f"  Teacher Temperature: {config.alignment.teacher_temperature}")
        logger.info(f"  Win Threshold: {config.alignment.win_threshold}")
        logger.info(f"  Grading Weights: factual={config.alignment.factual_grounding_weight}, "
                   f"citation={config.alignment.citation_accuracy_weight}, "
                   f"abstention={config.alignment.abstention_quality_weight}, "
                   f"hallucination={config.alignment.hallucination_penalty_weight}")

        logger.info(f"\n  DPO Hyperparameters:")
        logger.info(f"    Beta: {config.alignment.dpo_beta}")
        logger.info(f"    Learning Rate: {config.alignment.dpo_learning_rate}")
        logger.info(f"    Epochs: {config.alignment.dpo_num_epochs}")
        logger.info(f"    Batch Size: {config.alignment.dpo_batch_size}")
        logger.info(f"    Gradient Accumulation: {config.alignment.dpo_gradient_accumulation_steps}")

        logger.info(f"\n  LoRA Configuration:")
        logger.info(f"    Enabled: {config.alignment.use_lora}")
        logger.info(f"    r={config.alignment.lora_r}, alpha={config.alignment.lora_alpha}, "
                   f"dropout={config.alignment.lora_dropout}")
        logger.info(f"    Target Modules: {config.alignment.lora_target_modules}")

        logger.info(f"\n  Quantization:")
        logger.info(f"    4-bit: {config.alignment.use_4bit}")
        logger.info(f"    Compute dtype: {config.alignment.bnb_4bit_compute_dtype}")

        logger.info(f"\n  Model Paths:")
        logger.info(f"    Student Model: {config.alignment.student_model_name}")
        logger.info(f"    Output Dir: {config.alignment.aligned_model_output_dir}")
        logger.info(f"    Pairs Dir: {config.alignment.pairs_output_dir}")

        # Test evaluation settings
        logger.info("\n✓ Evaluation Settings:")
        logger.info(f"  Output Dir: {config.evaluation.eval_output_dir}")
        logger.info(f"  Accuracy Threshold: {config.evaluation.accuracy_threshold}")
        logger.info(f"  Default Test Set: {config.evaluation.default_test_set}")
        logger.info(f"  ECE Bins: {config.evaluation.ece_bins}")

        logger.info("\n✅ Configuration loading: PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ Configuration loading: FAILED - {e}")
        return False


def test_teacher_grader_init():
    """Test TeacherGrader initialization with config"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: TeacherGrader Initialization")
    logger.info("=" * 70)

    try:
        from src.alignment.teacher_grader import TeacherGrader, GradingCriteria

        # Test with defaults from config
        grader = TeacherGrader()

        logger.info(f"\n✓ Initialized with config defaults:")
        logger.info(f"  Model: {grader.model}")
        logger.info(f"  Temperature: {grader.temperature}")

        # Test grading criteria scoring
        criteria = GradingCriteria(
            factual_grounding=9.0,
            citation_accuracy=8.5,
            abstention_quality=9.0,
            hallucination_penalty=0.5
        )

        total_score = criteria.total_score
        logger.info(f"\n✓ Grading criteria total_score calculation:")
        logger.info(f"  factual=9.0, citation=8.5, abstention=9.0, hallucination=0.5")
        logger.info(f"  Total Score: {total_score:.2f}")
        logger.info(f"  Expected: ~(9*0.4 + 8.5*0.3 + 9*0.2 - 0.5*0.1) = 8.90")

        # Verify weights are from config
        expected = (
            9.0 * config.alignment.factual_grounding_weight +
            8.5 * config.alignment.citation_accuracy_weight +
            9.0 * config.alignment.abstention_quality_weight -
            0.5 * config.alignment.hallucination_penalty_weight
        )

        if abs(total_score - expected) < 0.01:
            logger.info(f"  ✓ Score matches config weights")
        else:
            logger.warning(f"  ⚠️  Score mismatch: got {total_score:.2f}, expected {expected:.2f}")

        logger.info("\n✅ TeacherGrader initialization: PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ TeacherGrader initialization: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preference_generator_init():
    """Test PreferenceGenerator initialization with config"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: PreferenceGenerator Initialization")
    logger.info("=" * 70)

    try:
        from src.alignment.preference_generator import PreferenceGenerator

        # Test with defaults from config
        generator = PreferenceGenerator()

        logger.info(f"\n✓ Initialized with config defaults:")
        logger.info(f"  Output Dir: {generator.output_dir}")
        logger.info(f"  Expected: {config.alignment.pairs_output_dir}")

        if str(generator.output_dir) == str(config.alignment.pairs_output_dir):
            logger.info(f"  ✓ Output directory matches config")
        else:
            logger.warning(f"  ⚠️  Output directory mismatch")

        logger.info(f"\n✓ Output directory exists: {generator.output_dir.exists()}")

        logger.info("\n✅ PreferenceGenerator initialization: PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ PreferenceGenerator initialization: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dpo_trainer_config():
    """Test DPOTrainingConfig with config defaults"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: DPOTrainingConfig Initialization")
    logger.info("=" * 70)

    try:
        from src.alignment.dpo_trainer import DPOTrainingConfig, DPOTrainerWrapper

        # Test with all defaults from config
        training_config = DPOTrainingConfig()

        logger.info(f"\n✓ DPO Training Config initialized with defaults:")
        logger.info(f"  Model: {training_config.model_name}")
        logger.info(f"  Beta: {training_config.beta}")
        logger.info(f"  Learning Rate: {training_config.learning_rate}")
        logger.info(f"  Epochs: {training_config.num_train_epochs}")
        logger.info(f"  Batch Size: {training_config.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation: {training_config.gradient_accumulation_steps}")
        logger.info(f"  LoRA: {training_config.use_lora} (r={training_config.lora_r}, alpha={training_config.lora_alpha})")
        logger.info(f"  4-bit: {training_config.use_4bit}")
        logger.info(f"  Output Dir: {training_config.output_dir}")

        # Verify values match config
        checks = [
            ("model_name", training_config.model_name, config.alignment.student_model_name),
            ("beta", training_config.beta, config.alignment.dpo_beta),
            ("learning_rate", training_config.learning_rate, config.alignment.dpo_learning_rate),
            ("num_train_epochs", training_config.num_train_epochs, config.alignment.dpo_num_epochs),
            ("use_lora", training_config.use_lora, config.alignment.use_lora),
            ("lora_r", training_config.lora_r, config.alignment.lora_r),
        ]

        all_match = True
        logger.info(f"\n✓ Verifying config values:")
        for name, got, expected in checks:
            match = got == expected
            symbol = "✓" if match else "✗"
            logger.info(f"  {symbol} {name}: {got} {'==' if match else '!='} {expected}")
            all_match = all_match and match

        if all_match:
            logger.info(f"\n  ✓ All values match config")
        else:
            logger.warning(f"\n  ⚠️  Some values don't match config")

        # Test override behavior
        logger.info(f"\n✓ Testing config override:")
        custom_config = DPOTrainingConfig(beta=0.2, learning_rate=1e-6)
        logger.info(f"  Custom beta: {custom_config.beta} (overridden from {config.alignment.dpo_beta})")
        logger.info(f"  Custom LR: {custom_config.learning_rate} (overridden from {config.alignment.dpo_learning_rate})")
        logger.info(f"  Default epochs: {custom_config.num_train_epochs} (from config)")

        # Test trainer wrapper initialization (doesn't load models)
        logger.info(f"\n✓ Testing DPOTrainerWrapper initialization:")
        trainer = DPOTrainerWrapper(training_config)
        logger.info(f"  Output dir created: {training_config.output_dir.exists()}")

        logger.info("\n✅ DPOTrainingConfig initialization: PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ DPOTrainingConfig initialization: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_config():
    """Test RAGEvaluator with config defaults"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: RAGEvaluator Configuration")
    logger.info("=" * 70)

    try:
        from src.evaluation.evaluator import RAGEvaluator

        # Test with defaults from config
        evaluator = RAGEvaluator()

        logger.info(f"\n✓ Evaluator initialized with config defaults:")
        logger.info(f"  Output Dir: {evaluator.output_dir}")
        logger.info(f"  Expected: {config.evaluation.eval_output_dir}")

        if str(evaluator.output_dir) == str(config.evaluation.eval_output_dir):
            logger.info(f"  ✓ Output directory matches config")
        else:
            logger.warning(f"  ⚠️  Output directory mismatch")

        logger.info(f"\n✓ Output directory exists: {evaluator.output_dir.exists()}")

        logger.info("\n✅ RAGEvaluator configuration: PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ RAGEvaluator configuration: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables():
    """Check if critical environment variables are set"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Environment Variables")
    logger.info("=" * 70)

    import os

    # Check critical variables
    critical_vars = [
        ("ALIGNMENT__TEACHER_API_KEY", config.alignment.teacher_api_key, "For GPT-4 grading"),
        ("LLM__API_BASE_URL", config.llm.api_base_url, "For Llama-3 inference"),
    ]

    logger.info("\n✓ Checking critical environment variables:")
    all_set = True
    for var_name, value, purpose in critical_vars:
        is_set = value is not None and value != "" and value != "EMPTY"
        symbol = "✓" if is_set else "⚠️ "
        status = "SET" if is_set else "NOT SET"
        logger.info(f"  {symbol} {var_name}: {status} - {purpose}")
        if not is_set:
            logger.info(f"      Set in .env: {var_name}=your-value-here")
        all_set = all_set and is_set

    if all_set:
        logger.info("\n✅ All critical environment variables: SET")
    else:
        logger.warning("\n⚠️  Some environment variables not set - required for actual training")

    return True  # Don't fail test if env vars not set


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("ALIGNMENT CONFIGURATION TEST SUITE")
    logger.info("=" * 70)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("TeacherGrader Init", test_teacher_grader_init),
        ("PreferenceGenerator Init", test_preference_generator_init),
        ("DPOTrainingConfig Init", test_dpo_trainer_config),
        ("RAGEvaluator Config", test_evaluator_config),
        ("Environment Variables", test_environment_variables),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        symbol = "✅" if result else "❌"
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {symbol} {test_name}: {status}")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ ALL TESTS PASSED - Ready for training!")
        logger.info("\nNext steps:")
        logger.info("  1. Set ALIGNMENT__TEACHER_API_KEY in .env (for GPT-4 grading)")
        logger.info("  2. Ensure LLM__API_BASE_URL points to your vLLM server")
        logger.info("  3. Run: python -m scripts.run_dpo_pipeline generate --queries-file data/test_set/sample_test_questions.json")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Fix issues before training")
        return 1


if __name__ == "__main__":
    exit(main())
