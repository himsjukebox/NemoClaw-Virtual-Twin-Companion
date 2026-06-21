# =============================================================================
# NemoClaw Virtual Twin Companion — LangGraph Orchestrator
# =============================================================================
# PURPOSE:
#   Defines the LangGraph StateGraph that orchestrates the multi-agent pipeline.
#   This is the main entry point for the NemoClaw system. It wires together:
#     - Guardrails node (NeMo Agent Toolkit safety checks)
#     - Design Agent node (NL → parametric CAD values)
#     - CAD Tool node (parameter injection + sandbox execution)
#     - Validator Agent node (engineering rule + RAG evaluation)
#
# ARCHITECTURE — Whiteboard State Pattern:
#   All agents and tools communicate through a shared WhiteboardState dict.
#   The Orchestrator routes state between nodes using conditional edges based
#   on validator verdict and iteration count. This enables deterministic,
#   observable execution without direct agent-to-agent coupling.
#
# NeMo Agent Toolkit Patterns Used:
#   - Agent Traces: Every node invocation and routing decision is logged to
#     the agent_trace list for full observability.
#   - Guardrails: User input is pre-processed through keyword-based safety
#     filters before any agent receives it.
#   - State Management: The WhiteboardState follows NeMo Agent Toolkit's
#     shared state management pattern for multi-agent coordination.
#
# NVIDIA STACK CONTEXT:
#   - LangGraph provides the state machine framework
#   - ChatNVIDIA (via Design/Validator agents) provides LLM inference
#   - NemoClaw/OpenShell (via CAD Tool) provides sandboxed execution
#   - NVIDIAEmbeddings (via RAG Engine) provides embedding generation
#
# ENTRY POINTS:
#   - run_graph(user_request) → WhiteboardState  (programmatic)
#   - python main.py                              (demo invocation)
# =============================================================================

import logging
import sys
from typing import Dict, Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from models.state import WhiteboardState, create_initial_state
from agents.design_agent import DesignAgent
from agents.validator_agent import ValidatorAgent
from tools.cad_tool import CADTool
from config.loader import load_all_configs, ConfigValidationError

# Load environment variables from .env file for local development.
# This ensures NVIDIA_API_KEY is available before ChatNVIDIA or
# NVIDIAEmbeddings are initialized. In production, env vars should
# be set by the deployment environment directly.
load_dotenv()


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# NeMo Agent Toolkit Guardrails — Keyword-Based Safety Filter
# =============================================================================
# These blocked keywords represent categories of harmful or off-topic content
# that should not be processed by the engineering agents. This is a simplified
# implementation of NeMo Agent Toolkit Guardrails for the hackathon context.
# In production, this would integrate with NVIDIA's full NeMo Guardrails API
# for topic control, jailbreak detection, and output filtering.
# =============================================================================

_BLOCKED_KEYWORDS = [
    "hack",
    "exploit",
    "malware",
    "virus",
    "attack",
    "weapon",
    "bomb",
    "kill",
    "murder",
    "steal",
    "illegal",
    "drugs",
    "porn",
    "nude",
    "sex",
    "violence",
    "terrorism",
    "terrorist",
    "ransomware",
    "phishing",
    "injection",
    "jailbreak",
    "ignore previous",
    "ignore instructions",
    "forget your instructions",
    "system prompt",
]

_OFF_TOPIC_KEYWORDS = [
    "recipe",
    "cooking",
    "weather forecast",
    "stock market",
    "horoscope",
    "movie review",
    "sports score",
]


def check_guardrails(user_input: str) -> tuple:
    """
    Apply NeMo Agent Toolkit safety checks on user input.

    Performs keyword-based filtering to detect harmful content and off-topic
    requests that should not be processed by the engineering agents. This is
    a simplified implementation of NeMo Agent Toolkit Guardrails.

    In production, this would integrate with NVIDIA's NeMo Guardrails API
    for comprehensive topic control, jailbreak detection, and content filtering.

    Args:
        user_input (str): The raw user input string to evaluate.

    Returns:
        tuple: A (is_safe, rejection_reason) tuple where:
            - is_safe (bool): True if input passes all safety checks.
            - rejection_reason (str): Description of why input was blocked,
              or empty string if input is safe.

    Component: Orchestrator
    """
    if not user_input or not user_input.strip():
        return (False, "Empty input is not allowed")

    lower_input = user_input.lower()

    # Check for harmful content keywords
    for keyword in _BLOCKED_KEYWORDS:
        if keyword in lower_input:
            return (
                False,
                f"Input blocked: contains potentially harmful content ('{keyword}')",
            )

    # Check for off-topic content keywords
    for keyword in _OFF_TOPIC_KEYWORDS:
        if keyword in lower_input:
            return (
                False,
                f"Input blocked: off-topic for drone chassis design ('{keyword}')",
            )

    return (True, "")


# =============================================================================
# Startup Configuration Validation
# =============================================================================
# Load and validate all YAML configs at module import time. If any config
# file is missing, has invalid syntax, or fails schema validation, the
# application aborts immediately with a descriptive error message.
# This follows NeMo Agent Toolkit's fail-fast initialization pattern.
# =============================================================================

try:
    _configs = load_all_configs()
    logger.info("All configuration files loaded and validated successfully.")
except ConfigValidationError as e:
    logger.critical(f"Configuration validation failed at startup: {e}")
    sys.exit(1)


# =============================================================================
# Agent and Tool Instances (Lazy Initialization)
# =============================================================================
# Agents are initialized lazily on first graph invocation because ChatNVIDIA
# requires NVIDIA_API_KEY to be set. This allows the module to be imported
# for testing and inspection without requiring the API key at import time.
# The CAD Tool does not require external API access and is initialized eagerly.
# =============================================================================

_design_agent = None
_cad_tool = CADTool()
_validator_agent = None


def _get_design_agent() -> DesignAgent:
    """
    Get or create the Design Agent singleton instance.

    Lazily initializes the DesignAgent on first call. This defers the
    ChatNVIDIA initialization (which requires NVIDIA_API_KEY) until the
    agent is actually needed.

    Returns:
        DesignAgent: The initialized Design Agent instance.

    Component: Orchestrator
    """
    global _design_agent
    if _design_agent is None:
        _design_agent = DesignAgent()
    return _design_agent


def _get_validator_agent() -> ValidatorAgent:
    """
    Get or create the Validator Agent singleton instance.

    Lazily initializes the ValidatorAgent on first call. This defers the
    ChatNVIDIA initialization (which requires NVIDIA_API_KEY) until the
    agent is actually needed.

    Returns:
        ValidatorAgent: The initialized Validator Agent instance.

    Component: Orchestrator
    """
    global _validator_agent
    if _validator_agent is None:
        _validator_agent = ValidatorAgent()
    return _validator_agent


# =============================================================================
# LangGraph Node Functions
# =============================================================================
# Each node function takes a WhiteboardState dict and returns an updated
# WhiteboardState dict. The orchestrator routes between nodes based on
# conditional edge logic.
# =============================================================================


def guardrails_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply NeMo Agent Toolkit Guardrails to user input.

    Node: guardrails
    Reads: user_request
    Writes: error, agent_trace
    Routes to: design_agent (if safe) or END (if blocked)

    This is the entry point of the graph. It pre-processes user input
    through keyword-based safety filters before any agent receives it.
    If the input is blocked, an error is set in state and the graph
    terminates without invoking any agents.

    Args:
        state (Dict[str, Any]): The LangGraph shared state with user_request.

    Returns:
        Dict[str, Any]: Updated state with error populated if input is blocked,
            or unchanged state (with trace appended) if input passes.

    Component: Orchestrator
    """
    user_input = state.get("user_request", "")

    logger.info(f"[guardrails] Processing input: '{user_input[:80]}...'")

    # Apply NeMo Agent Toolkit guardrails check
    is_safe, rejection_reason = check_guardrails(user_input)

    if not is_safe:
        # Input blocked — set error and record in agent trace
        state["error"] = f"Input blocked by safety guardrails: {rejection_reason}"
        state["agent_trace"].append({
            "node": "guardrails",
            "action": "REJECTED",
            "reason": rejection_reason,
        })
        logger.warning(f"[guardrails] Input REJECTED: {rejection_reason}")
    else:
        # Input passed — record in agent trace
        state["agent_trace"].append({
            "node": "guardrails",
            "action": "PASSED",
        })
        logger.info("[guardrails] Input PASSED safety checks")

    return state


def design_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the Design Agent to generate parametric CAD values.

    Node: design_agent
    Reads: user_request, validator_feedback, iteration_count, agent_trace
    Writes: design_parameters, iteration_count, agent_trace
    Routes to: cad_tool (via direct edge)

    The Design Agent translates the user's natural-language design goal
    into concrete parametric values (arm_length, material_thickness,
    arm_width, center_cutout_radius). On subsequent iterations, it
    incorporates validator feedback to refine its output.

    Args:
        state (Dict[str, Any]): The LangGraph shared state.

    Returns:
        Dict[str, Any]: Updated state with design_parameters populated.

    Component: Orchestrator
    """
    logger.info(
        f"[design_agent] Invoked (iteration {state.get('iteration_count', 0) + 1})"
    )
    agent = _get_design_agent()
    state = agent.invoke(state)
    logger.info(
        f"[design_agent] Generated parameters: {state.get('design_parameters')}"
    )
    return state


def cad_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the CAD Tool for sandboxed geometry generation.

    Node: cad_tool
    Reads: design_parameters
    Writes: cad_output_paths, error, agent_trace
    Routes to: validator_agent (via direct edge)

    The CAD Tool validates parameter ranges, injects values into the
    master_drone_template.py script, and dispatches execution to the
    NemoClaw/OpenShell sandboxed runtime. The script is NEVER executed
    locally — only inside the enterprise-grade sandbox.

    Args:
        state (Dict[str, Any]): The LangGraph shared state with design_parameters.

    Returns:
        Dict[str, Any]: Updated state with cad_output_paths or error.

    Component: Orchestrator
    """
    logger.info("[cad_tool] Invoking CAD generation in NemoClaw/OpenShell sandbox")
    state = _cad_tool.invoke(state)
    if state.get("cad_output_paths"):
        logger.info(f"[cad_tool] Generated: {state['cad_output_paths']}")
    elif state.get("error"):
        logger.warning(f"[cad_tool] Error: {state['error']}")
    return state


def validator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the Validator Agent for engineering evaluation.

    Node: validator_agent
    Reads: design_parameters, agent_trace
    Writes: validator_verdict, validator_score, validator_feedback, agent_trace
    Routes to: design_agent (on FAIL + iter < 5) or END (on PASS or iter >= 5)

    The Validator Agent evaluates design parameters against:
      1. Structural rule: arm_width >= arm_length × 0.08
      2. Manufacturability rule: material_thickness >= 2.0 mm
      3. RAG-retrieved engineering context (best-effort)

    It produces a PASS/FAIL verdict with a confidence score and actionable
    feedback for iterative refinement.

    Args:
        state (Dict[str, Any]): The LangGraph shared state with design_parameters.

    Returns:
        Dict[str, Any]: Updated state with validator_verdict, validator_score,
            and validator_feedback populated.

    Component: Orchestrator
    """
    logger.info("[validator_agent] Evaluating design parameters")
    agent = _get_validator_agent()
    state = agent.invoke(state)
    logger.info(
        f"[validator_agent] Verdict: {state.get('validator_verdict')}, "
        f"Score: {state.get('validator_score')}"
    )
    return state


# =============================================================================
# Conditional Edge (Routing) Functions
# =============================================================================


def route_after_guardrails(state: Dict[str, Any]) -> str:
    """
    Route after guardrails: to design_agent if safe, END if blocked.

    If the guardrails node set an error in the state (input was blocked),
    the graph terminates immediately. Otherwise, the request proceeds to
    the Design Agent for parameter generation.

    Args:
        state (Dict[str, Any]): Current graph state after guardrails processing.

    Returns:
        str: "design_agent" if input passed safety checks, or END if blocked.

    Component: Orchestrator
    """
    if state.get("error"):
        logger.info("[routing] Guardrails blocked input → END")
        return END
    logger.info("[routing] Guardrails passed → design_agent")
    return "design_agent"


def route_after_validation(state: Dict[str, Any]) -> str:
    """
    Route after validation: retry on FAIL (if under limit), END otherwise.

    Routing logic:
      - If validator_verdict == "PASS" → END (design accepted)
      - If iteration_count >= 5 → END (max iterations reached)
      - If validator_verdict == "FAIL" and iteration_count < 5 → "design_agent"
        (retry with validator feedback)

    Args:
        state (Dict[str, Any]): Current graph state after validator evaluation.

    Returns:
        str: "design_agent" for iterative retry, or END for termination.

    Component: Orchestrator
    """
    verdict = state.get("validator_verdict")
    iteration_count = state.get("iteration_count", 0)

    if verdict == "PASS":
        logger.info("[routing] Validator PASS → END")
        return END

    if iteration_count >= 5:
        logger.info(
            f"[routing] Max iterations reached ({iteration_count}) → END"
        )
        return END

    logger.info(
        f"[routing] Validator FAIL (iteration {iteration_count}/5) → design_agent"
    )
    return "design_agent"


# =============================================================================
# Graph Construction
# =============================================================================


def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Builds the full orchestration graph with:
      - Entry: guardrails node
      - Conditional edge from guardrails (pass → design_agent, block → END)
      - Direct edge: design_agent → cad_tool → validator_agent
      - Conditional edge from validator (PASS → END, FAIL+iter<5 → design_agent,
        iter>=5 → END)

    Args:
        None.

    Returns:
        CompiledGraph: Compiled LangGraph graph ready for invocation.

    Component: Orchestrator
    """
    graph = StateGraph(WhiteboardState)

    # --- Add nodes ---
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("design_agent", design_agent_node)
    graph.add_node("cad_tool", cad_tool_node)
    graph.add_node("validator_agent", validator_agent_node)

    # --- Set entry point ---
    graph.set_entry_point("guardrails")

    # --- Wire edges ---
    # Conditional: guardrails → design_agent (pass) or END (blocked)
    graph.add_conditional_edges("guardrails", route_after_guardrails)

    # Direct: design_agent → cad_tool → validator_agent
    graph.add_edge("design_agent", "cad_tool")
    graph.add_edge("cad_tool", "validator_agent")

    # Conditional: validator_agent → design_agent (FAIL, iter<5) or END
    graph.add_conditional_edges("validator_agent", route_after_validation)

    return graph.compile()


# =============================================================================
# Public Entry Point
# =============================================================================


def run_graph(user_request: str) -> WhiteboardState:
    """
    Public entry point to invoke the NemoClaw agent pipeline.

    Creates an initial WhiteboardState from the user's natural-language
    design goal, compiles the LangGraph StateGraph, and executes the full
    pipeline (guardrails → design → CAD → validate → iterate/terminate).

    All state transitions and agent invocations are logged to the agent_trace
    field for NeMo Agent Toolkit observability.

    Args:
        user_request (str): The user's natural-language design goal
            (e.g., "optimize for lightweight racing drone").

    Returns:
        WhiteboardState: Final state after graph execution completes,
            containing design_parameters, cad_output_paths, validator_verdict,
            validator_score, validator_feedback, agent_trace, and any error.

    Component: Orchestrator
    """
    logger.info(f"[run_graph] Starting pipeline for: '{user_request[:100]}'")

    # Create initial state using the shared state factory
    initial_state = create_initial_state(user_request)

    # Compile and invoke the graph
    compiled_graph = build_graph()
    final_state = compiled_graph.invoke(initial_state)

    logger.info(
        f"[run_graph] Pipeline complete. "
        f"Verdict: {final_state.get('validator_verdict')}, "
        f"Iterations: {final_state.get('iteration_count')}, "
        f"Error: {final_state.get('error')}"
    )

    return final_state


# =============================================================================
# Demo Invocation
# =============================================================================

if __name__ == "__main__":
    # Simple demo invocation to verify the pipeline works end-to-end.
    # This demonstrates the Orchestrator routing through all nodes.
    print("=" * 70)
    print("NemoClaw Virtual Twin Companion — Demo Invocation")
    print("=" * 70)
    print()

    demo_request = "Design a lightweight racing drone chassis with long arms for 7-inch props"
    print(f"User Request: {demo_request}")
    print("-" * 70)

    result = run_graph(demo_request)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"Verdict:    {result.get('validator_verdict')}")
        print(f"Score:      {result.get('validator_score')}")
        print(f"Iterations: {result.get('iteration_count')}")
        print(f"Parameters: {result.get('design_parameters')}")
        print(f"CAD Files:  {result.get('cad_output_paths')}")

    print()
    print("--- Agent Trace ---")
    for i, entry in enumerate(result.get("agent_trace", []), 1):
        print(f"  [{i}] {entry}")

    print()
    print("=" * 70)
    print("Demo complete.")
