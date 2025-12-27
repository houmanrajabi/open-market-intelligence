"""
Preference Pair Generator for DPO Training

This module generates (chosen, rejected) preference pairs by:
1. Running Student model (Llama-3) on queries
2. Grading answers with Teacher model (GPT-4)
3. Creating pairs: Win answer vs Loss answer
4. Saving in DPO training format

Output format for DPO:
{
    "prompt": "Question: ... Context: ...",
    "chosen": "Good answer with citations",
    "rejected": "Bad answer with hallucinations"
}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import time

from src.rag_system import UncertaintyAwareRAG
from src.alignment.teacher_grader import TeacherGrader, GradeLabel
from src.utils.config import config
from src.utils.logger import logger


@dataclass
class PreferencePair:
    """
    A single preference pair for DPO training.

    Attributes:
        prompt: Full prompt (question + context)
        chosen: Preferred answer (win)
        rejected: Dispreferred answer (loss)
        query: Original question
        context: Retrieved context chunks
        chosen_grade: Teacher grade for chosen
        rejected_grade: Teacher grade for rejected
        metadata: Additional info
    """
    prompt: str
    chosen: str
    rejected: str
    query: str
    context: List[str]
    chosen_grade: Dict[str, Any]
    rejected_grade: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dpo_format(self) -> Dict[str, str]:
        """Convert to DPO training format (prompt, chosen, rejected)"""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary with all metadata"""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "query": self.query,
            "context": self.context,
            "chosen_grade": self.chosen_grade,
            "rejected_grade": self.rejected_grade,
            "metadata": self.metadata
        }


class PreferenceGenerator:
    """
    Generates preference pairs for DPO training.

    Strategy:
    1. Sample queries from test set or generate synthetic queries
    2. Run Student RAG system to get answers
    3. Grade answers with Teacher
    4. Create pairs from wins vs losses
    5. Optionally generate contrastive pairs (good vs corrupted versions)
    """

    def __init__(
        self,
        rag_system: Optional[UncertaintyAwareRAG] = None,
        teacher_grader: Optional[TeacherGrader] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize preference generator.

        Args:
            rag_system: Student RAG system
            teacher_grader: Teacher grader
            output_dir: Directory to save preference pairs (defaults to config.alignment.pairs_output_dir)
        """
        self.rag_system = rag_system or UncertaintyAwareRAG()
        self.teacher_grader = teacher_grader or TeacherGrader()
        self.output_dir = output_dir or config.alignment.pairs_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pairs: List[PreferencePair] = []

        logger.info(f"Preference Generator initialized. Output: {self.output_dir}")

    def generate_from_queries(
        self,
        queries: List[str],
        num_pairs_per_query: int = 1,
        use_temperature_sampling: bool = True
    ) -> List[PreferencePair]:
        """
        Generate preference pairs from a list of queries.

        Args:
            queries: List of questions
            num_pairs_per_query: How many pairs to generate per query
            use_temperature_sampling: Sample multiple answers with temperature

        Returns:
            List of preference pairs
        """
        logger.info(f"Generating pairs from {len(queries)} queries...")

        for query in tqdm(queries, desc="Processing queries"):
            for _ in range(num_pairs_per_query):
                try:
                    pair = self._generate_single_pair(
                        query,
                        use_temperature_sampling=use_temperature_sampling
                    )

                    if pair:
                        self.pairs.append(pair)

                except Exception as e:
                    logger.error(f"Failed to generate pair for query '{query[:50]}...': {e}")

        logger.info(f"Generated {len(self.pairs)} preference pairs")

        return self.pairs

    def _generate_single_pair(
        self,
        query: str,
        use_temperature_sampling: bool = True
    ) -> Optional[PreferencePair]:
        """
        Generate a single preference pair.

        Strategy:
        - Generate 2+ answers with different temperatures
        - Grade each answer
        - Pick best (chosen) and worst (rejected)
        """
        # Get response from RAG system
        response = self.rag_system.answer_query(query)

        context = [source["content"] for source in response.sources]
        answer = response.answer

        # Generate alternative answer (if using temperature sampling)
        if use_temperature_sampling:
            # TODO: Implement temperature sampling for diversity
            # For now, we'll use response perturbations
            alternative_answer = self._generate_alternative(query, context)
        else:
            alternative_answer = answer

        # Grade both answers
        grade_1 = self.teacher_grader.grade_answer(query, answer, context)
        grade_2 = self.teacher_grader.grade_answer(query, alternative_answer, context)

        # Determine which is better
        if grade_1.grade_label == GradeLabel.WIN and grade_2.grade_label == GradeLabel.LOSS:
            chosen = answer
            rejected = alternative_answer
            chosen_grade = grade_1
            rejected_grade = grade_2
        elif grade_2.grade_label == GradeLabel.WIN and grade_1.grade_label == GradeLabel.LOSS:
            chosen = alternative_answer
            rejected = answer
            chosen_grade = grade_2
            rejected_grade = grade_1
        else:
            # Both win or both loss - skip this pair
            logger.debug(f"Skipping pair: both {grade_1.grade_label.value}")
            return None

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Create pair
        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            query=query,
            context=context,
            chosen_grade=chosen_grade.to_dict(),
            rejected_grade=rejected_grade.to_dict(),
            metadata={
                "chosen_score": chosen_grade.criteria.total_score,
                "rejected_score": rejected_grade.criteria.total_score,
                "score_gap": chosen_grade.criteria.total_score - rejected_grade.criteria.total_score
            }
        )

        return pair

    def _generate_alternative(self, query: str, context: List[str]) -> str:
        """
        Generate alternative answer for contrast.

        Strategy:
        - Remove citations
        - Add speculation
        - Simplify language
        """
        # For now, simple perturbation: remove citations
        response = self.rag_system.answer_query(query)
        answer = response.answer

        # Remove citations
        import re
        answer_no_citations = re.sub(r'\[Document \d+\]', '', answer)

        return answer_no_citations.strip()

    def _build_prompt(self, query: str, context: List[str]) -> str:
        """Build formatted prompt for DPO training"""
        context_str = "\n\n".join([
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(context)
        ])

        prompt = f"""Context Information:
{context_str}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- Cite specific documents using [Document N] format
- If the context doesn't contain enough information, say "INSUFFICIENT INFORMATION"
- Be precise and concise

Answer:"""

        return prompt

    def generate_contrastive_pairs(
        self,
        queries: List[str],
        corruption_strategies: List[str] = None
    ) -> List[PreferencePair]:
        """
        Generate contrastive pairs by corrupting good answers.

        Corruption strategies:
        - remove_citations: Strip all citations
        - add_hallucination: Add fabricated facts
        - remove_abstention: Force answer when should abstain
        - vague_references: Replace specific citations with "the document says"

        Args:
            queries: List of questions
            corruption_strategies: Which corruptions to apply

        Returns:
            List of preference pairs
        """
        if corruption_strategies is None:
            corruption_strategies = ["remove_citations", "add_hallucination"]

        logger.info(f"Generating contrastive pairs with strategies: {corruption_strategies}")

        for query in tqdm(queries, desc="Generating contrastive pairs"):
            try:
                # Get good answer
                response = self.rag_system.answer_query(query)
                good_answer = response.answer
                context = [source["content"] for source in response.sources]

                # Generate corrupted versions
                for strategy in corruption_strategies:
                    corrupted = self._corrupt_answer(good_answer, strategy)

                    # Grade both
                    grade_good = self.teacher_grader.grade_answer(query, good_answer, context)
                    grade_bad = self.teacher_grader.grade_answer(query, corrupted, context)

                    # Only keep if good is actually better
                    if grade_good.criteria.total_score > grade_bad.criteria.total_score:
                        prompt = self._build_prompt(query, context)

                        pair = PreferencePair(
                            prompt=prompt,
                            chosen=good_answer,
                            rejected=corrupted,
                            query=query,
                            context=context,
                            chosen_grade=grade_good.to_dict(),
                            rejected_grade=grade_bad.to_dict(),
                            metadata={
                                "corruption_strategy": strategy,
                                "chosen_score": grade_good.criteria.total_score,
                                "rejected_score": grade_bad.criteria.total_score
                            }
                        )

                        self.pairs.append(pair)

            except Exception as e:
                logger.error(f"Failed to generate contrastive pair: {e}")

        logger.info(f"Generated {len(self.pairs)} contrastive pairs")

        return self.pairs

    def _corrupt_answer(self, answer: str, strategy: str) -> str:
        """Apply corruption strategy to answer"""
        import re

        if strategy == "remove_citations":
            return re.sub(r'\[Document \d+\]', '', answer).strip()

        elif strategy == "add_hallucination":
            # Add a fabricated sentence
            return answer + " Additionally, recent analysis suggests this trend will continue through 2025."

        elif strategy == "remove_abstention":
            if "INSUFFICIENT INFORMATION" in answer:
                return "Based on the available information, the answer is approximately 2.5%."
            return answer

        elif strategy == "vague_references":
            return re.sub(r'\[Document \d+\]', 'according to the documents', answer)

        return answer

    def save_pairs(
        self,
        filename: str = None,
        format: str = "dpo"
    ):
        """
        Save preference pairs to JSON.

        Args:
            filename: Output filename
            format: "dpo" (minimal) or "full" (with metadata)
        """
        if not self.pairs:
            logger.warning("No pairs to save")
            return

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"preference_pairs_{timestamp}.json"

        output_path = self.output_dir / filename

        if format == "dpo":
            # Minimal format for DPO training
            data = [pair.to_dpo_format() for pair in self.pairs]
        else:
            # Full format with metadata
            data = [pair.to_dict() for pair in self.pairs]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(self.pairs)} preference pairs to: {output_path}")

        # Also save statistics
        stats_path = self.output_dir / f"stats_{filename}"
        stats = self._compute_statistics()

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"✅ Saved statistics to: {stats_path}")

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics about generated pairs"""
        if not self.pairs:
            return {}

        chosen_scores = [p.chosen_grade["criteria"]["total_score"] for p in self.pairs]
        rejected_scores = [p.rejected_grade["criteria"]["total_score"] for p in self.pairs]
        score_gaps = [p.metadata.get("score_gap", 0) for p in self.pairs]

        import numpy as np

        return {
            "total_pairs": len(self.pairs),
            "chosen_score_mean": float(np.mean(chosen_scores)),
            "rejected_score_mean": float(np.mean(rejected_scores)),
            "score_gap_mean": float(np.mean(score_gaps)),
            "score_gap_std": float(np.std(score_gaps)),
            "min_gap": float(np.min(score_gaps)),
            "max_gap": float(np.max(score_gaps))
        }


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("Preference Pair Generator - Example Usage")
    print("=" * 70)

    # Sample queries
    queries = [
        "What was the GDP growth projection for 2023?",
        "What did Chair Powell say about inflation?",
        "What was the unemployment rate in 2024?"
    ]

    print(f"\nGenerating pairs for {len(queries)} queries...")
    print("NOTE: Requires OpenAI API key and running vLLM server")

    # Initialize generator
    generator = PreferenceGenerator()

    # Generate pairs
    pairs = generator.generate_from_queries(queries, num_pairs_per_query=1)

    print(f"\n✅ Generated {len(pairs)} preference pairs")

    if pairs:
        print("\nExample pair:")
        print("-" * 70)
        pair = pairs[0]
        print(f"Query: {pair.query}")
        print(f"\nChosen (score={pair.chosen_grade['criteria']['total_score']:.2f}):")
        print(f"  {pair.chosen[:150]}...")
        print(f"\nRejected (score={pair.rejected_grade['criteria']['total_score']:.2f}):")
        print(f"  {pair.rejected[:150]}...")

    # Save pairs
    generator.save_pairs()

    print("\n" + "=" * 70)
