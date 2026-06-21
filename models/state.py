# =============================================================================
# NemoClaw Virtual Twin Companion — Whiteboard State Definition
# =============================================================================
# PURPOSE:
#   Defines the WhiteboardState TypedDict — the shared mutable state dictionary
#   that flows between all LangGraph nodes. This is the backbone of the
#   "whiteboard pattern" where agents communicate through a common state rather
#   than direct message passing.
#
# WHY TypedDict:
#   TypedDict provides static type checking while remaining compatible with
#   LangGraph's StateGraph, which expects plain dict-like objects. This gives
#   us IDE autocompletion and type safety without runtime overhead.
#
# DATA FLOW:
#   User Input → Orchestrator → [Guardrails] → Design Agent → CAD Tool
#   → Validator Agent → (iterate or return to UI)
#   All nodes read from and write to WhiteboardState.
#
# NVIDIA STACK CONTEXT:
#   This state structure follows NeMo Agent Toolkit observability patterns —
#   the agent_trace field enables full audit logging of every node invocation.
# =============================================================================

from typing import TypedDict, Optional, List


class WhiteboardState(TypedDict):
    """
    The LangGraph shared state dictionary for the NemoClaw orchestrator.

    All agents and tools read from and write to this state. The orchestrator
    routes the state between nodes based on conditional logic (e.g., validator
    verdict, iteration count).

    Attributes:
        user_request: Natural-language design goal from the user.
        design_parameters: JSON dict with arm_length, material_thickness,
            arm_width, center_cutout_radius after Design Agent processing.
        validator_feedback: JSON string with verdict, score, issues,
            suggestions, and reasoning from the Validator Agent.
        iteration_count: Number of design→validate cycles completed (max 5).
        cad_output_paths: Paths to generated STEP and STL geometry files.
        agent_trace: Chronological log of node invocations and actions,
            following NeMo Agent Toolkit observability patterns.
        validator_verdict: "PASS" or "FAIL" from the Validator Agent.
        validator_score: 0.0–1.0 confidence score from the Validator Agent.
        error: Error message if any step fails (guardrail rejection, timeout,
            parameter validation failure, etc.).

    Component: Orchestrator
    """

    user_request: str
    design_parameters: Optional[dict]
    validator_feedback: Optional[str]
    iteration_count: int
    cad_output_paths: Optional[List[str]]
    agent_trace: List[dict]
    validator_verdict: Optional[str]
    validator_score: Optional[float]
    error: Optional[str]


def create_initial_state(user_request: str) -> WhiteboardState:
    """
    Create a fresh WhiteboardState with sensible defaults for a new request.

    Args:
        user_request (str): The user's natural-language design goal.

    Returns:
        WhiteboardState: Initialized state ready for graph invocation.

    Component: Orchestrator
    """
    return WhiteboardState(
        user_request=user_request,
        design_parameters=None,
        validator_feedback=None,
        iteration_count=0,
        cad_output_paths=None,
        agent_trace=[],
        validator_verdict=None,
        validator_score=None,
        error=None,
    )
