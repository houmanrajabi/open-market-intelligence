"""
Ablation Study Analysis and Visualization

This module provides tools to analyze and visualize ablation study results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.utils.logger import logger


def load_experiment_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all experiment results from directory"""
    results = []

    for result_file in results_dir.glob("*/*_results.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            results.append(result_data)

    logger.info(f"Loaded {len(results)} experiment results")
    return results


def create_comparison_table(
    results: List[Dict[str, Any]],
    metrics: List[str],
    output_path: Path = None
) -> str:
    """
    Create markdown comparison table.

    Args:
        results: List of experiment results
        metrics: List of metric names to compare
        output_path: Optional path to save markdown

    Returns:
        Markdown table string
    """
    if not results:
        return "No results to compare"

    # Build table
    lines = []

    # Header
    header = ["Experiment"] + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---" for _ in header]) + "|")

    # Rows
    for result in results:
        row = [result["experiment_name"]]

        for metric in metrics:
            # Find metric value
            value = None
            if metric in result.get("retrieval_metrics", {}):
                value = result["retrieval_metrics"][metric]
            elif metric in result.get("answer_metrics", {}):
                value = result["answer_metrics"][metric]
            elif metric in result.get("uncertainty_metrics", {}):
                value = result["uncertainty_metrics"][metric]
            elif metric == "expansion_rate":
                value = result["expansion_rate"]
            elif metric == "abstention_rate":
                value = result["abstention_rate"]

            if value is not None:
                row.append(f"{value:.3f}")
            else:
                row.append("N/A")

        lines.append("| " + " | ".join(row) + " |")

    table_md = "\n".join(lines)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Ablation Study Results Comparison\n\n")
            f.write(table_md)
        logger.info(f"Saved comparison table to: {output_path}")

    return table_md


def analyze_research_question(
    results: List[Dict[str, Any]],
    rq_id: str,
    rq_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze results for a specific research question.

    Args:
        results: All experiment results
        rq_id: Research question ID
        rq_info: Research question information

    Returns:
        Analysis dictionary
    """
    # Filter results for this RQ
    relevant_exp_ids = rq_info["experiments"]
    rq_results = [r for r in results if r["experiment_id"] in relevant_exp_ids]

    if not rq_results:
        return {"error": "No results found for this research question"}

    # Extract metrics
    metrics = rq_info["metrics"]
    analysis = {
        "research_question": rq_info["question"],
        "hypothesis": rq_info["hypothesis"],
        "num_experiments": len(rq_results),
        "experiments": []
    }

    for result in rq_results:
        exp_metrics = {}
        for metric in metrics:
            value = None
            if metric in result.get("retrieval_metrics", {}):
                value = result["retrieval_metrics"][metric]
            elif metric in result.get("answer_metrics", {}):
                value = result["answer_metrics"][metric]
            elif metric in result.get("uncertainty_metrics", {}):
                value = result["uncertainty_metrics"][metric]
            elif metric == "expansion_rate":
                value = result["expansion_rate"]
            elif metric == "abstention_rate":
                value = result["abstention_rate"]

            exp_metrics[metric] = value

        analysis["experiments"].append({
            "experiment_id": result["experiment_id"],
            "name": result["experiment_name"],
            "metrics": exp_metrics
        })

    return analysis


def compute_improvements(
    baseline_result: Dict[str, Any],
    comparison_result: Dict[str, Any],
    metrics: List[str]
) -> Dict[str, float]:
    """
    Compute percentage improvements over baseline.

    Args:
        baseline_result: Baseline experiment result
        comparison_result: Comparison experiment result
        metrics: List of metrics to compare

    Returns:
        Dictionary of improvements (positive = better)
    """
    improvements = {}

    for metric in metrics:
        # Get baseline value
        baseline_value = None
        if metric in baseline_result.get("retrieval_metrics", {}):
            baseline_value = baseline_result["retrieval_metrics"][metric]
        elif metric in baseline_result.get("answer_metrics", {}):
            baseline_value = baseline_result["answer_metrics"][metric]
        elif metric in baseline_result.get("uncertainty_metrics", {}):
            baseline_value = baseline_result["uncertainty_metrics"][metric]

        # Get comparison value
        comparison_value = None
        if metric in comparison_result.get("retrieval_metrics", {}):
            comparison_value = comparison_result["retrieval_metrics"][metric]
        elif metric in comparison_result.get("answer_metrics", {}):
            comparison_value = comparison_result["answer_metrics"][metric]
        elif metric in comparison_result.get("uncertainty_metrics", {}):
            comparison_value = comparison_result["uncertainty_metrics"][metric]

        # Compute improvement
        if baseline_value is not None and comparison_value is not None and baseline_value != 0:
            improvement = ((comparison_value - baseline_value) / baseline_value) * 100
            improvements[metric] = improvement
        else:
            improvements[metric] = None

    return improvements


def generate_analysis_report(
    results_dir: Path,
    output_path: Path = None
) -> str:
    """
    Generate comprehensive analysis report.

    Args:
        results_dir: Directory with experiment results
        output_path: Optional path to save report

    Returns:
        Markdown report string
    """
    # Load results
    results = load_experiment_results(results_dir)

    if not results:
        return "No results found"

    # Build report
    lines = []
    lines.append("# Ablation Study Analysis Report\n")
    lines.append(f"**Total Experiments**: {len(results)}\n")
    lines.append("---\n")

    # Section 1: Overview
    lines.append("## 1. Experiment Overview\n")
    for result in results:
        lines.append(f"### {result['experiment_name']}")
        lines.append(f"- **ID**: {result['experiment_id']}")
        lines.append(f"- **Description**: {result['config_dict']['description']}")
        lines.append(f"- **Duration**: {result['duration_seconds']:.2f}s")
        lines.append(f"- **Expansion Rate**: {result['expansion_rate']:.1%}")
        lines.append(f"- **Abstention Rate**: {result['abstention_rate']:.1%}\n")

    # Section 2: Metric Comparison
    lines.append("## 2. Metric Comparison\n")

    all_metrics = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "accuracy",
        "faithfulness",
        "citation_precision",
        "hallucination_rate"
    ]

    table = create_comparison_table(results, all_metrics)
    lines.append(table + "\n")

    # Section 3: Key Findings
    lines.append("## 3. Key Findings\n")

    # Find baseline
    baseline = next((r for r in results if r["experiment_id"] == "exp_001_baseline"), None)

    if baseline:
        # Compare with full system
        full_system = next((r for r in results if r["experiment_id"] == "exp_007_full_uncertainty"), None)

        if full_system:
            lines.append("### Baseline vs. Full Uncertainty System\n")
            improvements = compute_improvements(baseline, full_system, all_metrics)

            for metric, improvement in improvements.items():
                if improvement is not None:
                    symbol = "📈" if improvement > 0 else "📉"
                    lines.append(f"- **{metric}**: {symbol} {improvement:+.1f}%")
            lines.append("")

    # Section 4: Expansion Analysis
    lines.append("## 4. Adaptive Retrieval Analysis\n")
    adaptive_results = [r for r in results if r["config_dict"]["retrieval_strategy"] == "adaptive"]

    if adaptive_results:
        lines.append("| Experiment | Expansion Rate | Accuracy | Citation Precision |")
        lines.append("|------------|----------------|----------|-------------------|")

        for result in adaptive_results:
            name = result["experiment_name"]
            exp_rate = result["expansion_rate"]
            acc = result["answer_metrics"].get("accuracy", 0)
            cite = result["answer_metrics"].get("citation_precision", 0)
            lines.append(f"| {name} | {exp_rate:.1%} | {acc:.3f} | {cite:.3f} |")
        lines.append("")

    # Section 5: Abstention Analysis
    lines.append("## 5. Abstention Behavior Analysis\n")
    abstaining_results = [r for r in results if r["abstention_rate"] > 0]

    if abstaining_results:
        lines.append("| Experiment | Abstention Rate | Accuracy (on answered) |")
        lines.append("|------------|-----------------|------------------------|")

        for result in abstaining_results:
            name = result["experiment_name"]
            abs_rate = result["abstention_rate"]
            acc = result["answer_metrics"].get("accuracy", 0)
            lines.append(f"| {name} | {abs_rate:.1%} | {acc:.3f} |")
        lines.append("")

    # Compile report
    report_md = "\n".join(lines)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_md)
        logger.info(f"Saved analysis report to: {output_path}")

    return report_md


if __name__ == "__main__":
    # Example usage
    results_dir = Path("data/experiments/ablation_studies")

    if results_dir.exists():
        report = generate_analysis_report(
            results_dir,
            output_path=Path("data/experiments/ANALYSIS_REPORT.md")
        )
        print(report)
    else:
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first: python -m scripts.run_ablation_studies --run-all")
