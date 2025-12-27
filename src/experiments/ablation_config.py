"""
Ablation Study Configuration

This module defines all experimental configurations for systematic ablation studies.
Each configuration represents a different system variant to test the contribution
of individual components.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class RetrievalStrategy(Enum):
    """Retrieval strategy variants"""
    FIXED_K5 = "fixed_k5"           # Always retrieve k=5
    FIXED_K10 = "fixed_k10"         # Always retrieve k=10
    ADAPTIVE = "adaptive"            # Adaptive k=5→10 based on entropy


class UncertaintyMethod(Enum):
    """Uncertainty quantification methods"""
    NONE = "none"                    # No uncertainty quantification
    ENTROPY_MEAN = "entropy_mean"    # Mean entropy across tokens
    ENTROPY_MAX = "entropy_max"      # Max entropy across tokens
    ENTROPY_P90 = "entropy_p90"      # 90th percentile entropy
    SEMANTIC = "semantic"            # Semantic uncertainty via multiple generations


class AbstentionStrategy(Enum):
    """Abstention behavior"""
    NEVER = "never"                  # Never abstain
    ENTROPY_THRESHOLD = "entropy_threshold"  # Threshold-based
    CONFIDENCE_THRESHOLD = "confidence_threshold"  # Confidence-based


class AlignmentType(Enum):
    """Alignment method"""
    NONE = "none"                    # No alignment (base model)
    DPO = "dpo"                      # DPO-aligned model
    SFT = "sft"                      # Supervised fine-tuning (future)


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment"""

    # Experiment metadata
    experiment_id: str
    name: str
    description: str

    # Component settings
    retrieval_strategy: RetrievalStrategy
    uncertainty_method: UncertaintyMethod
    abstention_strategy: AbstentionStrategy
    alignment_type: AlignmentType

    # Hyperparameters
    initial_k: int = 5
    expanded_k: int = 10
    entropy_expansion_threshold: float = 1.5
    entropy_abstention_threshold: float = 2.0
    temperature: float = 0.1

    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_logprobs: bool = True

    # Flags
    enabled: bool = True

    def __str__(self) -> str:
        return f"{self.experiment_id}: {self.name}"


# =============================================================================
# ABLATION STUDY CONFIGURATIONS
# =============================================================================

# Baseline experiments
BASELINE_NO_UNCERTAINTY = ExperimentConfig(
    experiment_id="exp_001_baseline",
    name="Baseline (No Uncertainty)",
    description="Standard RAG with fixed k=5, no uncertainty quantification, never abstains",
    retrieval_strategy=RetrievalStrategy.FIXED_K5,
    uncertainty_method=UncertaintyMethod.NONE,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    use_logprobs=False  # Don't need logprobs if no uncertainty
)

BASELINE_FIXED_K10 = ExperimentConfig(
    experiment_id="exp_002_baseline_k10",
    name="Baseline (Fixed k=10)",
    description="Standard RAG with fixed k=10 (upper bound for comparison)",
    retrieval_strategy=RetrievalStrategy.FIXED_K10,
    uncertainty_method=UncertaintyMethod.NONE,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    initial_k=10,
    use_logprobs=False
)

# Uncertainty quantification ablations
ENTROPY_MEAN_ONLY = ExperimentConfig(
    experiment_id="exp_003_entropy_mean",
    name="Entropy (Mean) - No Adaptation",
    description="Calculate mean entropy but don't adapt retrieval or abstain",
    retrieval_strategy=RetrievalStrategy.FIXED_K5,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

ENTROPY_MAX_ONLY = ExperimentConfig(
    experiment_id="exp_004_entropy_max",
    name="Entropy (Max) - No Adaptation",
    description="Calculate max entropy but don't adapt retrieval or abstain",
    retrieval_strategy=RetrievalStrategy.FIXED_K5,
    uncertainty_method=UncertaintyMethod.ENTROPY_MAX,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

ENTROPY_P90_ONLY = ExperimentConfig(
    experiment_id="exp_005_entropy_p90",
    name="Entropy (P90) - No Adaptation",
    description="Calculate 90th percentile entropy but don't adapt",
    retrieval_strategy=RetrievalStrategy.FIXED_K5,
    uncertainty_method=UncertaintyMethod.ENTROPY_P90,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

# Adaptive retrieval ablations
ADAPTIVE_NO_ABSTENTION = ExperimentConfig(
    experiment_id="exp_006_adaptive_no_abstain",
    name="Adaptive Retrieval (No Abstention)",
    description="Adaptive k=5→10 based on entropy, but never abstains",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

# Full uncertainty-aware system
FULL_UNCERTAINTY_SYSTEM = ExperimentConfig(
    experiment_id="exp_007_full_uncertainty",
    name="Full Uncertainty System",
    description="Adaptive retrieval + entropy-based abstention (our proposed system)",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

# Threshold sensitivity experiments
THRESHOLD_LOW_EXPANSION = ExperimentConfig(
    experiment_id="exp_008_thresh_low_expand",
    name="Low Expansion Threshold (1.0)",
    description="More aggressive expansion with lower threshold",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    entropy_expansion_threshold=1.0,  # Lower = expand more often
    use_logprobs=True
)

THRESHOLD_HIGH_EXPANSION = ExperimentConfig(
    experiment_id="exp_009_thresh_high_expand",
    name="High Expansion Threshold (2.0)",
    description="Conservative expansion with higher threshold",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    entropy_expansion_threshold=2.0,  # Higher = expand less often
    use_logprobs=True
)

THRESHOLD_LOW_ABSTENTION = ExperimentConfig(
    experiment_id="exp_010_thresh_low_abstain",
    name="Low Abstention Threshold (1.5)",
    description="More cautious with lower abstention threshold",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    entropy_abstention_threshold=1.5,  # Lower = abstain more often
    use_logprobs=True
)

THRESHOLD_HIGH_ABSTENTION = ExperimentConfig(
    experiment_id="exp_011_thresh_high_abstain",
    name="High Abstention Threshold (2.5)",
    description="Less cautious with higher abstention threshold",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    entropy_abstention_threshold=2.5,  # Higher = abstain less often
    use_logprobs=True
)

# Temperature sensitivity
TEMPERATURE_LOW = ExperimentConfig(
    experiment_id="exp_012_temp_low",
    name="Low Temperature (0.0)",
    description="Greedy decoding (more confident, lower entropy)",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    temperature=0.0,
    use_logprobs=True
)

TEMPERATURE_HIGH = ExperimentConfig(
    experiment_id="exp_013_temp_high",
    name="High Temperature (0.5)",
    description="Higher sampling temperature (less confident, higher entropy)",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    temperature=0.5,
    use_logprobs=True
)

# Alignment experiments
ALIGNED_NO_UNCERTAINTY = ExperimentConfig(
    experiment_id="exp_014_aligned_no_uncertainty",
    name="DPO-Aligned (No Uncertainty)",
    description="DPO-aligned model without uncertainty quantification",
    retrieval_strategy=RetrievalStrategy.FIXED_K5,
    uncertainty_method=UncertaintyMethod.NONE,
    abstention_strategy=AbstentionStrategy.NEVER,
    alignment_type=AlignmentType.DPO,
    use_logprobs=False
)

ALIGNED_WITH_UNCERTAINTY = ExperimentConfig(
    experiment_id="exp_015_aligned_full",
    name="DPO-Aligned + Uncertainty",
    description="Full system: DPO alignment + uncertainty quantification",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MEAN,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.DPO,
    use_logprobs=True
)

# Combined ablations
ENTROPY_MAX_ADAPTIVE = ExperimentConfig(
    experiment_id="exp_016_max_adaptive",
    name="Max Entropy + Adaptive",
    description="Use max entropy instead of mean for adaptive retrieval",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_MAX,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

ENTROPY_P90_ADAPTIVE = ExperimentConfig(
    experiment_id="exp_017_p90_adaptive",
    name="P90 Entropy + Adaptive",
    description="Use 90th percentile entropy for adaptive retrieval",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.ENTROPY_P90,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    use_logprobs=True
)

# Semantic Uncertainty experiment
SEMANTIC_UNCERTAINTY = ExperimentConfig(
    experiment_id="exp_018_semantic_uncertainty",
    name="Semantic Uncertainty",
    description="Use semantic similarity across multiple generations instead of token entropy",
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    uncertainty_method=UncertaintyMethod.SEMANTIC,
    abstention_strategy=AbstentionStrategy.ENTROPY_THRESHOLD,
    alignment_type=AlignmentType.NONE,
    temperature=0.7,  # Higher temperature for diversity in generations
    use_logprobs=False  # Don't need logprobs for semantic approach
)


# =============================================================================
# EXPERIMENT GROUPS
# =============================================================================

# Group experiments by research question
BASELINE_EXPERIMENTS = [
    BASELINE_NO_UNCERTAINTY,
    BASELINE_FIXED_K10
]

UNCERTAINTY_METHOD_EXPERIMENTS = [
    ENTROPY_MEAN_ONLY,
    ENTROPY_MAX_ONLY,
    ENTROPY_P90_ONLY
]

ADAPTIVE_RETRIEVAL_EXPERIMENTS = [
    ADAPTIVE_NO_ABSTENTION,
    FULL_UNCERTAINTY_SYSTEM
]

THRESHOLD_SENSITIVITY_EXPERIMENTS = [
    THRESHOLD_LOW_EXPANSION,
    THRESHOLD_HIGH_EXPANSION,
    THRESHOLD_LOW_ABSTENTION,
    THRESHOLD_HIGH_ABSTENTION
]

TEMPERATURE_EXPERIMENTS = [
    TEMPERATURE_LOW,
    TEMPERATURE_HIGH
]

ALIGNMENT_EXPERIMENTS = [
    ALIGNED_NO_UNCERTAINTY,
    ALIGNED_WITH_UNCERTAINTY
]

ENTROPY_AGGREGATION_EXPERIMENTS = [
    FULL_UNCERTAINTY_SYSTEM,  # mean (default)
    ENTROPY_MAX_ADAPTIVE,     # max
    ENTROPY_P90_ADAPTIVE      # p90
]

SEMANTIC_UNCERTAINTY_EXPERIMENTS = [
    SEMANTIC_UNCERTAINTY
]

# All experiments in execution order
ALL_EXPERIMENTS = [
    # Baselines
    BASELINE_NO_UNCERTAINTY,
    BASELINE_FIXED_K10,

    # Uncertainty methods (no adaptation)
    ENTROPY_MEAN_ONLY,
    ENTROPY_MAX_ONLY,
    ENTROPY_P90_ONLY,

    # Adaptive retrieval
    ADAPTIVE_NO_ABSTENTION,
    FULL_UNCERTAINTY_SYSTEM,

    # Threshold sensitivity
    THRESHOLD_LOW_EXPANSION,
    THRESHOLD_HIGH_EXPANSION,
    THRESHOLD_LOW_ABSTENTION,
    THRESHOLD_HIGH_ABSTENTION,

    # Temperature
    TEMPERATURE_LOW,
    TEMPERATURE_HIGH,

    # Alignment
    ALIGNED_NO_UNCERTAINTY,
    ALIGNED_WITH_UNCERTAINTY,

    # Entropy aggregation
    ENTROPY_MAX_ADAPTIVE,
    ENTROPY_P90_ADAPTIVE,

    # Semantic uncertainty
    SEMANTIC_UNCERTAINTY
]


# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

RESEARCH_QUESTIONS = {
    "RQ1": {
        "question": "Does uncertainty quantification improve answer quality?",
        "experiments": [
            "exp_001_baseline",  # No uncertainty
            "exp_003_entropy_mean",  # Uncertainty but no action
            "exp_007_full_uncertainty"  # Full uncertainty system
        ],
        "metrics": ["accuracy", "faithfulness", "citation_precision", "hallucination_rate"],
        "hypothesis": "Uncertainty quantification improves answer quality by enabling adaptive retrieval and abstention"
    },

    "RQ2": {
        "question": "Does adaptive retrieval improve performance over fixed k?",
        "experiments": [
            "exp_001_baseline",  # Fixed k=5
            "exp_002_baseline_k10",  # Fixed k=10
            "exp_006_adaptive_no_abstain",  # Adaptive k=5→10
            "exp_007_full_uncertainty"  # Adaptive + abstention
        ],
        "metrics": ["precision_at_k", "recall_at_k", "mrr", "accuracy"],
        "hypothesis": "Adaptive retrieval balances efficiency (k=5) with coverage (k=10) based on need"
    },

    "RQ3": {
        "question": "Which entropy aggregation method works best?",
        "experiments": [
            "exp_007_full_uncertainty",  # Mean
            "exp_016_max_adaptive",      # Max
            "exp_017_p90_adaptive"       # P90
        ],
        "metrics": ["entropy_correct_mean", "entropy_incorrect_mean", "correlation", "auc_roc"],
        "hypothesis": "Mean entropy provides the best uncertainty estimate"
    },

    "RQ4": {
        "question": "How sensitive is the system to entropy thresholds?",
        "experiments": [
            "exp_008_thresh_low_expand",   # Expansion 1.0
            "exp_007_full_uncertainty",    # Expansion 1.5 (default)
            "exp_009_thresh_high_expand",  # Expansion 2.0
            "exp_010_thresh_low_abstain",  # Abstention 1.5
            "exp_011_thresh_high_abstain"  # Abstention 2.5
        ],
        "metrics": ["retrieval_expansion_rate", "abstention_rate", "accuracy"],
        "hypothesis": "Optimal thresholds balance coverage and precision"
    },

    "RQ5": {
        "question": "Does RLAIF alignment improve citation and abstention behavior?",
        "experiments": [
            "exp_001_baseline",            # No alignment, no uncertainty
            "exp_007_full_uncertainty",    # No alignment, with uncertainty
            "exp_014_aligned_no_uncertainty",  # DPO, no uncertainty
            "exp_015_aligned_full"         # DPO + uncertainty
        ],
        "metrics": ["citation_precision", "abstention_quality", "hallucination_rate"],
        "hypothesis": "DPO alignment improves citation precision and abstention quality"
    },

    "RQ6": {
        "question": "Is uncertainty calibration improved by alignment?",
        "experiments": [
            "exp_007_full_uncertainty",  # Unaligned
            "exp_015_aligned_full"       # Aligned
        ],
        "metrics": ["correlation", "auc_roc", "ece"],
        "hypothesis": "Aligned models have better calibrated uncertainty estimates"
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_experiment_by_id(experiment_id: str) -> Optional[ExperimentConfig]:
    """Get experiment configuration by ID"""
    for exp in ALL_EXPERIMENTS:
        if exp.experiment_id == experiment_id:
            return exp
    return None


def get_experiments_for_rq(rq_id: str) -> List[ExperimentConfig]:
    """Get all experiments for a research question"""
    if rq_id not in RESEARCH_QUESTIONS:
        return []

    exp_ids = RESEARCH_QUESTIONS[rq_id]["experiments"]
    return [get_experiment_by_id(exp_id) for exp_id in exp_ids]


def print_experiment_summary():
    """Print summary of all experiments"""
    print("=" * 80)
    print("ABLATION STUDY EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal Experiments: {len(ALL_EXPERIMENTS)}")

    print("\n1. BASELINE EXPERIMENTS:")
    for exp in BASELINE_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n2. UNCERTAINTY METHOD EXPERIMENTS:")
    for exp in UNCERTAINTY_METHOD_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n3. ADAPTIVE RETRIEVAL EXPERIMENTS:")
    for exp in ADAPTIVE_RETRIEVAL_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n4. THRESHOLD SENSITIVITY EXPERIMENTS:")
    for exp in THRESHOLD_SENSITIVITY_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n5. TEMPERATURE EXPERIMENTS:")
    for exp in TEMPERATURE_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n6. ALIGNMENT EXPERIMENTS:")
    for exp in ALIGNMENT_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n7. ENTROPY AGGREGATION EXPERIMENTS:")
    for exp in ENTROPY_AGGREGATION_EXPERIMENTS:
        print(f"   - {exp.experiment_id}: {exp.name}")

    print("\n" + "=" * 80)
    print(f"RESEARCH QUESTIONS: {len(RESEARCH_QUESTIONS)}")
    print("=" * 80)
    for rq_id, rq_info in RESEARCH_QUESTIONS.items():
        print(f"\n{rq_id}: {rq_info['question']}")
        print(f"   Experiments: {len(rq_info['experiments'])}")
        print(f"   Hypothesis: {rq_info['hypothesis']}")


if __name__ == "__main__":
    print_experiment_summary()
