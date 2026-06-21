# =============================================================================
# NemoClaw Virtual Twin Companion — Models Package
# =============================================================================
# This package defines the shared data models and schemas used across all
# components of the multi-agent system:
#
#   1. WhiteboardState  — The LangGraph shared state TypedDict
#   2. Parameters       — Design parameter ranges, defaults, and constraints
#   3. Validation       — Validator response schema and verdict constants
#
# These models are the "contract" between agents: the Design Agent writes
# parameters conforming to PARAM_RANGES, the Validator Agent produces a
# ValidatorResponse, and the Orchestrator routes based on WhiteboardState.
# =============================================================================

from models.state import WhiteboardState, create_initial_state
from models.parameters import (
    PARAM_RANGES,
    PARAM_DEFAULTS,
    PARAM_UNITS,
    PARAM_DESCRIPTIONS,
    PARAM_NAMES,
    STRUCTURAL_CONSTRAINT_RATIO,
    MANUFACTURABILITY_MIN_THICKNESS,
)
from models.validation import (
    ValidatorResponse,
    create_validator_response,
    VERDICT_PASS,
    VERDICT_FAIL,
    VALID_VERDICTS,
    SCORE_MIN,
    SCORE_MAX,
)

__all__ = [
    # State
    "WhiteboardState",
    "create_initial_state",
    # Parameters
    "PARAM_RANGES",
    "PARAM_DEFAULTS",
    "PARAM_UNITS",
    "PARAM_DESCRIPTIONS",
    "PARAM_NAMES",
    "STRUCTURAL_CONSTRAINT_RATIO",
    "MANUFACTURABILITY_MIN_THICKNESS",
    # Validation
    "ValidatorResponse",
    "create_validator_response",
    "VERDICT_PASS",
    "VERDICT_FAIL",
    "VALID_VERDICTS",
    "SCORE_MIN",
    "SCORE_MAX",
]
