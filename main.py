# =============================================================================
# NemoClaw Virtual Twin Companion — LangGraph Orchestrator
# =============================================================================
import json
import logging
import re
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from tools.cad_tool import CADTool

# Load environment variables (NVIDIA_API_KEY) from .env for local development.
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy, cached RAG Engine singleton.
# The RAG engine ingests PDFs from data/ (text + multimodal image captions)
# and is queried by the Validator Agent for grounding context. Initialization
# is deferred and tolerant: if it fails (no API key, no PDFs), the validator
# simply proceeds without RAG context.
# ---------------------------------------------------------------------------
_rag_engine = None
_rag_init_attempted = False


def _get_rag_engine():
    """Return a cached RAGEngine instance, or None if it cannot be built."""
    global _rag_engine, _rag_init_attempted
    if not _rag_init_attempted:
        _rag_init_attempted = True
        try:
            from tools.rag_engine import RAGEngine
            _rag_engine = RAGEngine()
            logger.info("RAG engine initialized for validator grounding.")
        except Exception as e:
            logger.warning("RAG engine unavailable (%s). Validator will run rules-only.", e)
            _rag_engine = None
    return _rag_engine


class WhiteboardState(TypedDict):
    user_prompt: str
    component_type: str
    design_parameters: Dict[str, Any]
    cad_output_paths: List[str]
    validator_verdict: str
    validator_score: float
    validator_feedback: str
    iteration_count: int
    agent_trace: List[Dict[str, Any]]
    error: Optional[str]


llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.1)


def design_agent_node(state: WhiteboardState) -> WhiteboardState:
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    system_prompt = """You are an expert Drone CAD Systems Engineer. Your job is to convert natural language requests into specific CAD configuration parameters.

You support three distinct components:
1. "chassis" (Drone frame base plates)
2. "propeller" (Aero rotor profiles)
3. "motor_mount" (Circular motor adapter plates)

CRITICAL: You must choose the correct component type and only output parameters matching that specific schema.

SCHEMA SPECIFICATIONS AND RANGES:
- If component_type is "chassis":
    * "arm_count" (int, allowed: 3 to 8) — number of arms/motors. Map the requested airframe: tricopter=3, quadcopter=4, pentacopter=5, hexacopter=6, octocopter=8.
    * "arm_length" (float, allowed: 80.0 to 200.0)
    * "material_thickness" (float, allowed: 2.0 to 10.0)
    * "arm_width" (float, allowed: 8.0 to 25.0)
    * "center_cutout_radius" (float, allowed: 10.0 to 30.0)

- If component_type is "propeller":
    * "blade_count" (float/int, allowed: 2.0 to 4.0)
    * "diameter_mm" (float, allowed: 50.0 to 400.0)
    * "pitch_inches" (float, allowed: 2.0 to 12.0)
    * "hub_radius_mm" (float, allowed: 3.0 to 15.0)
    * "hub_thickness_mm" (float, allowed: 4.0 to 20.0)

- If component_type is "motor_mount":
    * "outer_diameter" (float, allowed: 15.0 to 60.0)
    * "mount_thickness" (float, allowed: 1.5 to 8.0)
    * "center_hole_diameter" (float, allowed: 3.0 to 15.0)
    * "bolt_spacing" (float, allowed: 8.0 to 30.0)

You MUST respond with a single, valid JSON block. Do not write markdown text outside the JSON block.

RESPONSE FORMAT EXAMPLE:
{
  "component_type": "motor_mount",
  "design_parameters": {
    "outer_diameter": 32.0,
    "mount_thickness": 4.5,
    "center_hole_diameter": 8.0,
    "bolt_spacing": 16.0
  }
}
"""

    user_content = f"User Request: {state['user_prompt']}"
    if state.get("validator_verdict") == "FAIL":
        user_content += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED REJECTED CRITERIA:\n{state['validator_feedback']}\nPlease adjust parameters to satisfy these rules."

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        raw_text = response.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        parsed_output = json.loads(raw_text)
        state["component_type"] = parsed_output.get("component_type", "chassis")
        state["design_parameters"] = parsed_output.get("design_parameters", {})
        state["error"] = None

        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].append({
            "node": "design_agent", "action": "GENERATED_PARAMS",
            "parameters": state["design_parameters"], "component_type": state["component_type"], "iteration": iteration
        })
    except Exception as e:
        state["error"] = f"Design Agent Parameter Parsing Crash: {str(e)}"
    return state


def cad_node(state: WhiteboardState) -> WhiteboardState:
    if state.get("error"):
        return state
    tool_executor = CADTool()
    return tool_executor(state)


def validator_agent_node(state: WhiteboardState) -> WhiteboardState:
    if state.get("error"):
        state["validator_verdict"] = "FAIL"
        return state

    comp_type = state.get("component_type", "chassis")
    params = state.get("design_parameters", {})
    iteration = state.get("iteration_count", 1)

    system_prompt = f"""You are an Engineering Inspector Agent checking compliance rules for a drone: "{comp_type}".

CRITICAL CHECKLIST LIMITS:
- For "chassis":
    * arm_count: 3 to 8 | arm_length: 80.0 to 200.0 | material_thickness: 2.0 to 10.0 | arm_width: 8.0 to 25.0 | center_cutout_radius: 10.0 to 30.0
- For "propeller":
    * blade_count: 2.0 to 4.0 | diameter_mm: 50.0 to 400.0 | pitch_inches: 2.0 to 12.0 | hub_radius_mm: 3.0 to 15.0 | hub_thickness_mm: 4.0 to 20.0
- For "motor_mount":
    * outer_diameter: 15.0 to 60.0 | mount_thickness: 1.5 to 8.0 | center_hole_diameter: 3.0 to 15.0 | bolt_spacing: 8.0 to 30.0

Analyze if the parameters fit perfectly within range.
Provide your final verdict in a strict JSON format string:
{{
  "verdict": "PASS",
  "score": 1.0,
  "issues": [],
  "reasoning": "Brief explanation"
}}
"""
    user_content = f"Parameters under evaluation: {json.dumps(params)}"

    # --- RAG grounding: retrieve engineering context from ingested PDFs ---
    rag_used = False
    engine = _get_rag_engine()
    if engine is not None:
        try:
            rag_query = (
                f"{comp_type} drone design engineering standards, structural and "
                f"manufacturability requirements for parameters: {json.dumps(params)}"
            )
            chunks = engine.query(rag_query)
            if chunks:
                rag_used = True
                context_text = "\n\n".join(
                    f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
                    for c in chunks
                )
                user_content += (
                    "\n\n--- RETRIEVED ENGINEERING CONTEXT (from knowledge base) ---\n"
                    f"{context_text}\n"
                    "Consider the above engineering context alongside the numeric "
                    "range checks when forming your verdict and reasoning."
                )
                if "agent_trace" not in state:
                    state["agent_trace"] = []
                state["agent_trace"].append({
                    "node": "validator_agent",
                    "action": "RAG_RETRIEVED",
                    "chunks": len(chunks),
                    "sources": sorted({c.get("source", "unknown") for c in chunks}),
                    "iteration": iteration,
                })
        except Exception as rag_err:
            logger.warning("RAG query failed (%s). Proceeding rules-only.", rag_err)

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])
        raw_text = response.content.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        evaluation = json.loads(raw_text)
        state["validator_verdict"] = evaluation.get("verdict", "FAIL")
        state["validator_score"] = evaluation.get("score", 0.0)
        state["validator_feedback"] = json.dumps({
            "issues": evaluation.get("issues", []), "reasoning": evaluation.get("reasoning", "")
        })

        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].append({
            "node": "validator_agent", "action": "EVALUATED", "verdict": state["validator_verdict"], "score": state["validator_score"], "rag_grounded": rag_used, "iteration": iteration
        })
    except Exception as e:
        state["validator_verdict"] = "FAIL"
        state["validator_feedback"] = str(e)
    return state


def routing_verdict_edge(state: WhiteboardState):
    if state.get("error") or state.get("iteration_count", 1) >= 3 or state.get("validator_verdict") == "PASS":
        return END
    return "design_agent"


workflow = StateGraph(WhiteboardState)
workflow.add_node("design_agent", design_agent_node)
workflow.add_node("cad_tool", cad_node)
workflow.add_node("validator_agent", validator_agent_node)

workflow.set_entry_point("design_agent")
workflow.add_edge("design_agent", "cad_tool")
workflow.add_edge("cad_tool", "validator_agent")
workflow.add_conditional_edges("validator_agent", routing_verdict_edge, {"design_agent": "design_agent", END: END})

compiled_graph = workflow.compile()


def run_graph(prompt: str) -> dict:
    initial_state = {
        "user_prompt": prompt, "component_type": "chassis", "design_parameters": {}, "cad_output_paths": [],
        "validator_verdict": "PENDING", "validator_score": 0.0, "validator_feedback": "", "iteration_count": 0, "agent_trace": [], "error": None
    }
    return compiled_graph.invoke(initial_state)
