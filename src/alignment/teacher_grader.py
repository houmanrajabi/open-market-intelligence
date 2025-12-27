"""
Teacher Model Grader (RLAIF)

This module implements the "Teacher" model (GPT-4) that grades Student (Llama-3) answers
on factual grounding and citation precision. These grades are used to create preference
pairs for DPO training.

Architecture:
    Student Answer + Context → Teacher (GPT-4) → Grade (Win/Loss) + Reasoning
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

from openai import OpenAI

from src.utils.config import config
from src.utils.logger import logger


class GradeLabel(Enum):
    """Grade labels for answers"""
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"


@dataclass
class GradingCriteria:
    """Criteria for grading answers"""
    factual_grounding: float  # 0-10: All claims backed by context?
    citation_accuracy: float  # 0-10: Citations match content?
    abstention_quality: float  # 0-10: Says "I don't know" when appropriate?
    hallucination_penalty: float  # 0-10: Unsupported claims?

    @property
    def total_score(self) -> float:
        """Compute weighted total score"""
        # Use weights from config
        return (
            self.factual_grounding * config.alignment.factual_grounding_weight +
            self.citation_accuracy * config.alignment.citation_accuracy_weight +
            self.abstention_quality * config.alignment.abstention_quality_weight -
            self.hallucination_penalty * config.alignment.hallucination_penalty_weight
        )


@dataclass
class TeacherGrade:
    """
    Grade from teacher model.

    Attributes:
        grade_label: WIN, LOSS, or TIE
        criteria: Detailed scoring on each criterion
        reasoning: Explanation of grade
        confidence: Teacher's confidence (0-1)
        flagged_issues: List of specific problems found
    """
    grade_label: GradeLabel
    criteria: GradingCriteria
    reasoning: str
    confidence: float
    flagged_issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "grade_label": self.grade_label.value,
            "criteria": {
                "factual_grounding": self.criteria.factual_grounding,
                "citation_accuracy": self.criteria.citation_accuracy,
                "abstention_quality": self.criteria.abstention_quality,
                "hallucination_penalty": self.criteria.hallucination_penalty,
                "total_score": self.criteria.total_score
            },
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "flagged_issues": self.flagged_issues
        }


class TeacherGrader:
    """
    Teacher model (GPT-4) for grading student answers.

    This implements the RLAIF protocol:
    1. Student generates answer with context
    2. Teacher grades on factual accuracy and citation precision
    3. Grades are used to create (chosen, rejected) pairs for DPO
    """

    # Grading prompt template
    GRADING_PROMPT = """You are an expert evaluator for a question-answering system focused on Federal Reserve FOMC documents.

Your task is to grade the STUDENT's answer against the provided CONTEXT on these criteria:

**GRADING CRITERIA:**

1. **Factual Grounding (0-10)**
   - Score 9-10: Every claim is explicitly supported by context with accurate citations
   - Score 7-8: Most claims supported, minor citation issues
   - Score 4-6: Some claims supported, but significant gaps or vague references
   - Score 1-3: Few claims supported, mostly unsupported statements
   - Score 0: Completely fabricated, no grounding

2. **Citation Accuracy (0-10)**
   - Score 9-10: All citations ([Document N]) correctly reference specific context
   - Score 7-8: Most citations accurate, minor misalignments
   - Score 4-6: Some citations correct, but several are wrong or missing
   - Score 1-3: Few correct citations, mostly wrong references
   - Score 0: No citations or all citations are fabricated

3. **Abstention Quality (0-10)**
   - Score 9-10: Correctly says "INSUFFICIENT INFORMATION" when context lacks answer
   - Score 7-8: Partially abstains but includes some speculation
   - Score 4-6: Attempts answer despite insufficient context
   - Score 1-3: Confidently answers with fabricated information
   - Score 0: Hallucinated answer when should have abstained
   - Note: If context IS sufficient, score based on answer quality

4. **Hallucination Penalty (0-10)**
   - Score 0: No hallucinations detected
   - Score 1-3: Minor unsupported details (e.g., adding "approximately")
   - Score 4-6: Some claims not in context
   - Score 7-8: Multiple fabricated facts
   - Score 9-10: Completely made up answer

**CONTEXT:**
{context}

**QUESTION:**
{question}

**STUDENT ANSWER:**
{student_answer}

**YOUR TASK:**
Grade the STUDENT ANSWER using the criteria above. Respond in JSON format:

{{
  "factual_grounding": <score 0-10>,
  "citation_accuracy": <score 0-10>,
  "abstention_quality": <score 0-10>,
  "hallucination_penalty": <score 0-10>,
  "grade_label": "win" | "loss",
  "reasoning": "Detailed explanation of your grading...",
  "confidence": <0.0-1.0>,
  "flagged_issues": ["issue 1", "issue 2", ...]
}}

**GRADING RULES:**
- Total score = (factual_grounding * {factual_weight}) + (citation_accuracy * {citation_weight}) + (abstention_quality * {abstention_weight}) - (hallucination_penalty * {hallucination_weight})
- If total score >= {win_threshold} → "win"
- If total score < {win_threshold} → "loss"
- Be strict: prefer "loss" when in doubt (we want to train cautious behavior)
- Flag specific issues (e.g., "Claim X not in context", "Citation [Document 2] is incorrect")

Respond with ONLY the JSON object, no additional text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize teacher grader.

        Args:
            api_key: OpenAI API key (defaults to config.alignment.teacher_api_key)
            model: Teacher model (defaults to config.alignment.teacher_model)
            temperature: Sampling temperature (defaults to config.alignment.teacher_temperature)
        """
        # Use config defaults if not provided
        self.api_key = api_key or config.alignment.teacher_api_key
        self.model = model or config.alignment.teacher_model
        self.temperature = temperature if temperature is not None else config.alignment.teacher_temperature

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"Teacher Grader initialized: model={self.model}")

    def grade_answer(
        self,
        question: str,
        student_answer: str,
        context_chunks: List[str],
        max_retries: int = 3
    ) -> TeacherGrade:
        """
        Grade a student answer.

        Args:
            question: User query
            student_answer: Student's generated answer
            context_chunks: Retrieved context
            max_retries: Max retry attempts for API calls

        Returns:
            TeacherGrade with detailed evaluation
        """
        # Format context
        context_str = "\n\n".join([
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Build prompt with config values
        prompt = self.GRADING_PROMPT.format(
            context=context_str,
            question=question,
            student_answer=student_answer,
            factual_weight=config.alignment.factual_grounding_weight,
            citation_weight=config.alignment.citation_accuracy_weight,
            abstention_weight=config.alignment.abstention_quality_weight,
            hallucination_weight=config.alignment.hallucination_penalty_weight,
            win_threshold=config.alignment.win_threshold
        )

        # Call teacher model
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert grader. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}  # Force JSON output
                )

                # Parse response
                grade_data = json.loads(response.choices[0].message.content)

                # Validate and create grade
                grade = self._parse_grade(grade_data)

                logger.debug(
                    f"Graded answer: {grade.grade_label.value} "
                    f"(score: {grade.criteria.total_score:.2f})"
                )

                return grade

            except Exception as e:
                logger.warning(f"Grading attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError("Failed to grade answer after max retries")

    def _parse_grade(self, grade_data: Dict[str, Any]) -> TeacherGrade:
        """Parse grade data from teacher model"""
        criteria = GradingCriteria(
            factual_grounding=float(grade_data["factual_grounding"]),
            citation_accuracy=float(grade_data["citation_accuracy"]),
            abstention_quality=float(grade_data["abstention_quality"]),
            hallucination_penalty=float(grade_data["hallucination_penalty"])
        )

        grade_label = GradeLabel(grade_data["grade_label"])

        return TeacherGrade(
            grade_label=grade_label,
            criteria=criteria,
            reasoning=grade_data["reasoning"],
            confidence=float(grade_data.get("confidence", 0.8)),
            flagged_issues=grade_data.get("flagged_issues", [])
        )

    def compare_answers(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        context_chunks: List[str]
    ) -> Tuple[TeacherGrade, TeacherGrade, str]:
        """
        Compare two answers and determine preference.

        Args:
            question: User query
            answer_a: First answer
            answer_b: Second answer
            context_chunks: Retrieved context

        Returns:
            (grade_a, grade_b, preference) where preference is "a", "b", or "tie"
        """
        # Grade both answers
        grade_a = self.grade_answer(question, answer_a, context_chunks)
        grade_b = self.grade_answer(question, answer_b, context_chunks)

        # Determine preference
        score_a = grade_a.criteria.total_score
        score_b = grade_b.criteria.total_score

        # Use tie threshold from config
        tie_threshold = config.alignment.preference_tie_threshold if hasattr(config.alignment, 'preference_tie_threshold') else 0.5

        if abs(score_a - score_b) < tie_threshold:
            preference = "tie"
        elif score_a > score_b:
            preference = "a"
        else:
            preference = "b"

        logger.info(
            f"Comparison: A={score_a:.2f} vs B={score_b:.2f} → preference={preference}"
        )

        return grade_a, grade_b, preference


# Example usage
if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("Teacher Grader - Example Usage")
    print("=" * 70)

    # Example context and answers
    question = "What was the GDP growth projection for 2023?"

    context = [
        "The FOMC projects GDP growth of 0.5% for 2023 according to December 2022 projections.",
        "The unemployment rate is expected to be 4.6% in 2024."
    ]

    # Good answer (well-cited, grounded)
    answer_good = (
        "According to the December 2022 projections [Document 1], "
        "the GDP growth projection for 2023 was 0.5%."
    )

    # Bad answer (hallucinated)
    answer_bad = (
        "The GDP growth for 2023 was projected to be around 2.5%, "
        "showing strong economic recovery."
    )

    # Abstention (correct when unsure)
    answer_abstain = (
        "INSUFFICIENT INFORMATION: The provided context does not contain "
        "specific GDP projections for 2023."
    )

    print("\nTo run this example, you need an OpenAI API key:")
    print("export OPENAI_API_KEY='your-key-here'")

    if os.getenv("OPENAI_API_KEY"):
        grader = TeacherGrader()

        print("\n1. Grading GOOD answer:")
        print("-" * 70)
        grade = grader.grade_answer(question, answer_good, context)
        print(f"Grade: {grade.grade_label.value}")
        print(f"Score: {grade.criteria.total_score:.2f}")
        print(f"Reasoning: {grade.reasoning}")

        print("\n2. Grading BAD answer:")
        print("-" * 70)
        grade = grader.grade_answer(question, answer_bad, context)
        print(f"Grade: {grade.grade_label.value}")
        print(f"Score: {grade.criteria.total_score:.2f}")
        print(f"Flagged: {grade.flagged_issues}")

        print("\n3. Comparing answers:")
        print("-" * 70)
        grade_a, grade_b, pref = grader.compare_answers(
            question, answer_good, answer_bad, context
        )
        print(f"Preference: {pref} (Good={grade_a.criteria.total_score:.2f}, Bad={grade_b.criteria.total_score:.2f})")

    else:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping live examples.")

    print("\n" + "=" * 70)
