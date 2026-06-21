# =============================================================================
# NemoClaw Virtual Twin Companion — Validator Response Schema
# =============================================================================
# PURPOSE:
#   Defines the data model for the Validator Agent's structured output.
#   The validator produces a verdict (PASS/FAIL), a numeric score, a list of
#   identified issues, actionable suggestions, and a reasoning string.
#
# WHY A DEDICATED MODULE:
#   The validator response is consumed by multiple components:
#     - The Orchestrator reads the verdict for routing decisions
#     - The Design Agent reads issues/suggestions when iterating
#     - The Streamlit UI displays the full response to the user
#   Having a single schema definition ensures all consumers agree on structure.
#
# NVIDIA STACK CONTEXT:
#   The reasoning field captures the LLM's chain-of-thought evaluation,
#   providing NeMo Agent Toolkit-style observability into the validator's
#   decision process.
# =============================================================================

from typing import TypedDict, List


# ---------------------------------------------------------------------------
# Verdict Constants
# ---------------------------------------------------------------------------

VERDICT_PASS: str = "PASS"
"""Design meets all engineering rules and RAG-informed assessment."""

VERDICT_FAIL: str = "FAIL"
"""Design violates one or more engineering rules or has critical RAG issues."""

VALID_VERDICTS = frozenset({VERDICT_PASS, VERDICT_FAIL})
"""Set of all valid verdict values."""

# ---------------------------------------------------------------------------
# Score Bounds
# ---------------------------------------------------------------------------

SCORE_MIN: float = 0.0
"""Minimum possible validator score (worst)."""

SCORE_MAX: float = 1.0
"""Maximum possible validator score (best)."""


# ---------------------------------------------------------------------------
# Validator Response TypedDict
# ---------------------------------------------------------------------------

class ValidatorResponse(TypedDict):
    """
    Structured response from the Validator Agent.

    This is serialized to JSON and stored in the WhiteboardState's
    validator_feedback field. It provides both machine-readable fields
    (verdict, score) and human-readable fields (issues, suggestions,
    reasoning) for the iterative design loop.

    Attributes:
        verdict: "PASS" if design meets all criteria, "FAIL" otherwise.
        score: Numeric confidence between 0.0 (worst) and 1.0 (best).
        issues: List of specific rule violations or concerns identified.
            Each string references the violated rule (e.g., "Structural:
            arm_width < arm_length * 0.08").
        suggestions: List of actionable recommendations to fix issues.
            Each string references a specific parameter with a recommended
            value (e.g., "Increase arm_width to at least 12.0 mm").
        reasoning: Free-text explanation of the evaluation logic, including
            RAG-retrieved context considerations. Provides observability
            into the validator's decision-making process.

    Component: Validator_Agent
    """

    verdict: str
    score: float
    issues: List[str]
    suggestions: List[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def create_validator_response(
    verdict: str,
    score: float,
    issues: List[str] | None = None,
    suggestions: List[str] | None = None,
    reasoning: str = "",
) -> ValidatorResponse:
    """
    Create a ValidatorResponse with defaults for optional fields.

    Args:
        verdict: "PASS" or "FAIL".
        score: Numeric score between 0.0 and 1.0.
        issues: List of identified issues (defaults to empty list).
        suggestions: List of actionable suggestions (defaults to empty list).
        reasoning: Explanation of the evaluation (defaults to empty string).

    Returns:
        ValidatorResponse: A fully populated validator response dict.

    Raises:
        ValueError: If verdict is not in VALID_VERDICTS or score is out of
            [0.0, 1.0] range.

    Component: Validator_Agent
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict '{verdict}'. Must be one of: {VALID_VERDICTS}"
        )
    if not (SCORE_MIN <= score <= SCORE_MAX):
        raise ValueError(
            f"Score {score} out of range [{SCORE_MIN}, {SCORE_MAX}]"
        )

    return ValidatorResponse(
        verdict=verdict,
        score=score,
        issues=issues if issues is not None else [],
        suggestions=suggestions if suggestions is not None else [],
        reasoning=reasoning,
    )
