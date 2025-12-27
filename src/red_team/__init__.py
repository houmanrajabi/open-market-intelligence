"""
Red-Team Adversarial Testing Module

This module provides tools for adversarial testing and robustness evaluation
of the RAG system.
"""

from .attack_categories import (
    AttackCategory,
    AttackPattern,
    ALL_ATTACK_PATTERNS,
    get_patterns_by_category,
    get_all_categories
)

from .red_team_evaluator import (
    RedTeamEvaluator,
    AdversarialQuestion,
    AdversarialResult,
    RedTeamReport,
    load_adversarial_test_set
)

__all__ = [
    "AttackCategory",
    "AttackPattern",
    "ALL_ATTACK_PATTERNS",
    "get_patterns_by_category",
    "get_all_categories",
    "RedTeamEvaluator",
    "AdversarialQuestion",
    "AdversarialResult",
    "RedTeamReport",
    "load_adversarial_test_set"
]
