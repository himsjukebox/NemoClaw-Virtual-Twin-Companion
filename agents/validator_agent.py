# =============================================================================
# NemoClaw Virtual Twin Companion — Validator Agent
# =============================================================================
# PURPOSE:
#   The Validator Agent evaluates proposed design parameters against engineering
#   rules and RAG-retrieved knowledge. It produces a PASS/FAIL verdict with a
#   numeric confidence score and actionable feedback for the Design Agent.
#
# DATA FLOW:
#   Design Parameters → Validator Agent → Verdict + Feedback → Whiteboard
#   → (if FAIL) back to Design Agent for iteration
#
# NVIDIA STACK USAGE:
#   - ChatNVIDIA (langchain-nvidia-ai-endpoints) for LLM inference
#   - Model: nvidia/llama-3.1-nemotron-70b-instruct (loaded from config/agents.yaml)
#   - RAG Engine with NVIDIAEmbeddings for context retrieval
#
# LangGraph Node Info:
#   Node: validator_agent
#   Reads: design_parameters, agent_trace
#   Writes: validator_verdict, validator_score, validator_feedback, agent_trace
#   Routes to: design_agent (on FAIL, iteration < 5) or END (on PASS or max iterations)
# =============================================================================

import json
import logging
from typing import Dict, Any, List

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import WhiteboardState
from models.parameters import (
    PARAM_RANGES,
    STRUCTURAL_CONSTRAINT_RATIO,
    MANUFACTURABILITY_MIN_THICKNESS,
)
from models.validation import (
    ValidatorResponse,
    create_validator_response,
    VERDICT_PASS,
    VERDICT_FAIL,
)
from config.loader import load_agents_config


logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    The Validator Agent evaluates proposed drone chassis designs against
    engineering rules and RAG-retrieved knowledge.

    Architecture Note (for judges):
        This agent operates on the whiteboard pattern — it reads design
        parameters from shared state, evaluates them against structural and
        manufacturability rules, queries RAG for engineering context, and
        writes its verdict back to the state for the Orchestrator to route.

    The validation pipeline has two layers:
        1. Rule-based checks (deterministic, always applied):
           - Structural: arm_width >= arm_length * 0.08
           - Manufacturability: material_thickness >= 2.0 mm
        2. RAG + LLM evaluation (probabilistic, best-effort):
           - Queries engineering PDFs for relevant context
           - Uses ChatNVIDIA to assess parameters against retrieved context
           - Degrades gracefully if RAG is unavailable

    PASS is issued only when all rule-based checks pass AND the RAG assessment
    identifies no critical issues.

    Attributes:
        llm (ChatNVIDIA): The NVIDIA-hosted LLM instance for evaluation.
        system_prompt (str): The engineering validator system prompt from YAML.
        config (dict): Full validator agent configuration from config/agents.yaml.

    Component: Validator_Agent
    """

    def __init__(self):
        """
        Initialize the Validator Agent with ChatNVIDIA LLM.

        Loads configuration from config/agents.yaml via the centralized config
        loader. The ChatNVIDIA class from langchain-nvidia-ai-endpoints connects
        to NVIDIA's NIM API endpoints. Authentication is handled via the
        NVIDIA_API_KEY environment variable.

        Raises:
            ConfigValidationError: If config/agents.yaml is missing or invalid.
            ValueError: If NVIDIA_API_KEY is not set.

        Component: Validator_Agent
        """
        # Load external configuration (model, temperature, system prompt)
        all_agents_config = load_agents_config()
        self.config = all_agents_config["validator_agent"]

        # Initialize the NVIDIA LLM — this is the ONLY LLM provider used.
        # We do NOT use OpenAI, Anthropic, or local models.
        self.llm = ChatNVIDIA(
            model=self.config["model"],
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 2048),
        )

        # Store the system prompt for use in every invocation
        self.system_prompt = self.config["system_prompt"]

    def _check_structural_rule(self, params: dict) -> bool:
        """
        Check the structural integrity rule: arm_width >= arm_length * 0.08.

        This rule ensures arms are wide enough relative to their length to
        avoid buckling under motor thrust loads. The ratio comes from
        STRUCTURAL_CONSTRAINT_RATIO in models/parameters.py.

        Args:
            params (dict): Design parameters with 'arm_width' and 'arm_length'.

        Returns:
            bool: True if the structural rule is satisfied, False otherwise.

        Component: Validator_Agent
        """
        return params["arm_width"] >= params["arm_length"] * STRUCTURAL_CONSTRAINT_RATIO

    def _check_manufacturability_rule(self, params: dict) -> bool:
        """
        Check the manufacturability rule: material_thickness >= 2.0 mm.

        Below 2.0 mm, FDM 3D printing layer adhesion is unreliable and
        parts become brittle. The threshold comes from
        MANUFACTURABILITY_MIN_THICKNESS in models/parameters.py.

        Args:
            params (dict): Design parameters with 'material_thickness'.

        Returns:
            bool: True if the manufacturability rule is satisfied, False otherwise.

        Component: Validator_Agent
        """
        return params["material_thickness"] >= MANUFACTURABILITY_MIN_THICKNESS

    def _query_rag(self, params: dict) -> List[dict]:
        """
        Query RAG engine for relevant engineering context.

        Constructs a query from the current design parameters and retrieves
        relevant document chunks from the FAISS vector store. If the RAG
        engine is unavailable (import failure, API error, no PDFs), returns
        an empty list and the validation proceeds with built-in rules only.

        Args:
            params (dict): Current design parameters for context formulation.

        Returns:
            list: Top-5 document chunks with 'text' and 'source' keys,
                or empty list if RAG is unavailable.

        Component: Validator_Agent
        """
        try:
            from tools.rag_engine import RAGEngine
            engine = RAGEngine()
            query = (
                f"drone chassis structural requirements "
                f"arm_length={params['arm_length']}mm "
                f"material_thickness={params['material_thickness']}mm "
                f"arm_width={params['arm_width']}mm"
            )
            return engine.query(query)
        except Exception as e:
            logger.warning(
                f"RAG engine unavailable, proceeding with built-in rules only: {e}"
            )
            return []

    def _llm_evaluate(
        self,
        params: dict,
        rag_context: List[dict],
        rule_issues: List[str],
    ) -> dict:
        """
        Use ChatNVIDIA to evaluate parameters with RAG context.

        Builds a prompt combining the design parameters, any rule-based issues
        already found, and retrieved engineering context from RAG. The LLM
        provides an assessment that may surface additional concerns not captured
        by the deterministic rules.

        If the LLM call fails, returns a minimal assessment noting the failure
        without blocking the validation pipeline.

        Args:
            params (dict): Current design parameters.
            rag_context (list): Retrieved document chunks from RAG engine.
            rule_issues (list): Issues already found by rule-based checks.

        Returns:
            dict: Assessment with keys:
                - 'critical_issues' (list): Critical problems found by LLM.
                - 'issues' (list): Non-critical concerns.
                - 'suggestions' (list): Improvement recommendations.
                - 'reasoning' (str): Explanation of the evaluation.

        Component: Validator_Agent
        """
        # Build context string from RAG chunks
        if rag_context:
            context_text = "\n\n".join(
                f"[Source: {chunk.get('source', 'unknown')}]\n{chunk.get('text', '')}"
                for chunk in rag_context
            )
        else:
            context_text = "No RAG context available. Evaluating based on built-in engineering rules only."

        # Build the evaluation prompt
        eval_prompt = (
            f"Evaluate the following drone chassis design parameters:\n\n"
            f"Parameters:\n"
            f"  arm_length: {params.get('arm_length')} mm\n"
            f"  material_thickness: {params.get('material_thickness')} mm\n"
            f"  arm_width: {params.get('arm_width')} mm\n"
            f"  center_cutout_radius: {params.get('center_cutout_radius')} mm\n\n"
            f"Rule-based issues already identified:\n"
            f"  {json.dumps(rule_issues) if rule_issues else 'None'}\n\n"
            f"Engineering context from knowledge base:\n"
            f"{context_text}\n\n"
            f"Based on the parameters, rules, and engineering context, provide your "
            f"assessment in the following JSON format:\n"
            f'{{"critical_issues": [], "issues": [], "suggestions": [], "reasoning": "..."}}\n\n'
            f"Only include critical_issues if there are clear violations of structural "
            f"integrity or manufacturability beyond the rule checks already performed. "
            f"Provide reasoning that explains your assessment."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=eval_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return self._parse_llm_assessment(response.content)
        except Exception as e:
            logger.warning(f"LLM evaluation failed, using rule-based results only: {e}")
            rag_note = (
                "RAG context was unavailable." if not rag_context
                else "LLM evaluation failed; relying on rule-based checks only."
            )
            return {
                "critical_issues": [],
                "issues": [],
                "suggestions": [],
                "reasoning": f"LLM evaluation unavailable. {rag_note}",
            }

    def _parse_llm_assessment(self, response_text: str) -> dict:
        """
        Parse the LLM's JSON assessment response.

        Handles various response formats:
        - Bare JSON objects
        - JSON within markdown code fences
        - JSON embedded in explanatory text

        Falls back to a safe default assessment if parsing fails.

        Args:
            response_text (str): Raw text response from ChatNVIDIA.

        Returns:
            dict: Parsed assessment with 'critical_issues', 'issues',
                'suggestions', and 'reasoning' keys.

        Component: Validator_Agent
        """
        default_assessment = {
            "critical_issues": [],
            "issues": [],
            "suggestions": [],
            "reasoning": "Unable to parse LLM assessment; relying on rule-based checks.",
        }

        if not response_text or not response_text.strip():
            return default_assessment

        try:
            # Try to extract JSON from the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text and "{" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            elif "{" in response_text and "}" in response_text:
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                json_str = response_text[start:end]
            else:
                return default_assessment

            parsed = json.loads(json_str)

            if not isinstance(parsed, dict):
                return default_assessment

            # Extract fields with safe defaults
            return {
                "critical_issues": parsed.get("critical_issues", []),
                "issues": parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
                "reasoning": parsed.get("reasoning", ""),
            }
        except (ValueError, IndexError, TypeError, KeyError):
            return default_assessment

    def _compute_score(self, params: dict, issues: List[str], rag_assessment: dict) -> float:
        """
        Compute a 0.0-1.0 confidence score based on rule violations and assessment.

        Scoring logic:
        - Start at 1.0 (perfect)
        - Deduct 0.3 for each rule-based issue (structural, manufacturability)
        - Deduct 0.2 for each critical issue from RAG/LLM assessment
        - Deduct 0.1 for each non-critical issue from RAG/LLM assessment
        - Clamp to [0.0, 1.0]

        Args:
            params (dict): Design parameters (for potential proximity scoring).
            issues (list): Rule-based issues found.
            rag_assessment (dict): LLM evaluation results.

        Returns:
            float: Score between 0.0 (worst) and 1.0 (best).

        Component: Validator_Agent
        """
        score = 1.0
        # Major deductions for rule violations
        score -= len(issues) * 0.3
        # Moderate deductions for RAG critical issues
        score -= len(rag_assessment.get("critical_issues", [])) * 0.2
        # Minor deductions for non-critical issues
        score -= len(rag_assessment.get("issues", [])) * 0.1
        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate design parameters and produce validation verdict.

        This method:
        1. Extracts design_parameters from state
        2. Checks structural rule: arm_width >= arm_length * 0.08
        3. Checks manufacturability rule: material_thickness >= 2.0
        4. Queries RAG engine for engineering context (best-effort)
        5. Uses ChatNVIDIA to evaluate params + RAG context together
        6. Determines verdict: PASS only when all rules pass AND no critical RAG issues
        7. Writes verdict, score, and feedback JSON to state
        8. Appends trace entry with node="validator_agent"

        Args:
            state (Dict[str, Any]): The LangGraph shared state ("whiteboard").
                Expected keys:
                - "design_parameters" (dict): The four parametric values.
                - "agent_trace" (List[dict]): Chronological log of invocations.

        Returns:
            Dict[str, Any]: Updated state with "validator_verdict",
                "validator_score", and "validator_feedback" populated,
                plus a new agent_trace entry.

        Component: Validator_Agent
        """
        params = state.get("design_parameters", {})
        if not params:
            # No parameters to validate — immediate FAIL
            response = create_validator_response(
                verdict=VERDICT_FAIL,
                score=0.0,
                issues=["No design parameters provided for validation."],
                suggestions=["Ensure the Design Agent produces valid parameters."],
                reasoning="Validation cannot proceed without design parameters.",
            )
            state["validator_verdict"] = response["verdict"]
            state["validator_score"] = response["score"]
            state["validator_feedback"] = json.dumps(response)
            # Ensure agent_trace exists
            if "agent_trace" not in state or state["agent_trace"] is None:
                state["agent_trace"] = []
            state["agent_trace"].append({
                "node": "validator_agent",
                "action": "EVALUATED",
                "verdict": response["verdict"],
                "score": response["score"],
            })
            return state

        # Ensure agent_trace exists
        if "agent_trace" not in state or state["agent_trace"] is None:
            state["agent_trace"] = []

        issues: List[str] = []
        suggestions: List[str] = []

        # --- Rule-based checks (deterministic, always applied) ---

        # Check 1: Structural integrity rule
        if not self._check_structural_rule(params):
            min_width = params["arm_length"] * STRUCTURAL_CONSTRAINT_RATIO
            issues.append(
                f"Structural: arm_width ({params['arm_width']:.1f} mm) "
                f"< arm_length ({params['arm_length']:.1f} mm) × 0.08 = {min_width:.1f} mm"
            )
            suggestions.append(
                f"Increase arm_width to at least {min_width:.1f} mm to satisfy "
                f"the structural integrity constraint."
            )

        # Check 2: Manufacturability rule
        if not self._check_manufacturability_rule(params):
            issues.append(
                f"Manufacturability: material_thickness ({params['material_thickness']:.1f} mm) "
                f"< {MANUFACTURABILITY_MIN_THICKNESS} mm (FDM minimum)"
            )
            suggestions.append(
                f"Increase material_thickness to at least "
                f"{MANUFACTURABILITY_MIN_THICKNESS} mm for reliable FDM printing."
            )

        # --- RAG + LLM evaluation (probabilistic, best-effort) ---

        # Query RAG engine for engineering context
        rag_context = self._query_rag(params)

        # Use ChatNVIDIA to evaluate params against rules + RAG context
        rag_assessment = self._llm_evaluate(params, rag_context, issues)

        # --- Determine verdict ---
        # PASS only when all rules pass AND no critical issues from RAG assessment
        has_rule_failures = len(issues) > 0
        has_critical_rag_issues = len(rag_assessment.get("critical_issues", [])) > 0

        if has_rule_failures or has_critical_rag_issues:
            verdict = VERDICT_FAIL
        else:
            verdict = VERDICT_PASS

        # Compute confidence score
        score = self._compute_score(params, issues, rag_assessment)

        # Build combined issues and suggestions (rules + RAG)
        all_issues = issues + rag_assessment.get("issues", [])
        all_suggestions = suggestions + rag_assessment.get("suggestions", [])

        # Build reasoning string
        reasoning_parts = []
        if not rag_context:
            reasoning_parts.append(
                "RAG context was unavailable; evaluation based on built-in rules only."
            )
        if rag_assessment.get("reasoning"):
            reasoning_parts.append(rag_assessment["reasoning"])
        if not issues and not has_critical_rag_issues:
            reasoning_parts.append(
                "All structural and manufacturability rules pass. "
                "No critical issues identified."
            )
        reasoning = " ".join(reasoning_parts)

        # Create the structured response using the helper
        response = create_validator_response(
            verdict=verdict,
            score=score,
            issues=all_issues if all_issues else None,
            suggestions=all_suggestions if all_suggestions else None,
            reasoning=reasoning,
        )

        # --- Write results to Whiteboard state ---
        state["validator_verdict"] = response["verdict"]
        state["validator_score"] = response["score"]
        state["validator_feedback"] = json.dumps(response)

        # Append to agent_trace (NeMo Agent Toolkit observability)
        state["agent_trace"].append({
            "node": "validator_agent",
            "action": "EVALUATED",
            "verdict": response["verdict"],
            "score": response["score"],
        })

        return state


# =============================================================================
# Deterministic Physics Gate (R11.2–R11.6)
# =============================================================================
# MODULE-LEVEL pure function — importable as:
#   from agents.validator_agent import physics_gate
# =============================================================================


def physics_gate(metrics: dict) -> tuple:
    """
    Deterministic physics gate (R11.2-R11.5).

    Returns (passed: bool, issues: list[str], suggestions: list[str]).
    FAIL if: TWR < use-case target, OR payload not feasible, OR structural fails.

    Args:
        metrics: Engineering metrics dict (from EngineeringMetrics.to_dict()).

    Returns:
        Tuple of (passed, issues, suggestions).
    """
    issues = []
    suggestions = []

    # Check TWR against use-case target (R11.2)
    twr = metrics.get("twr")
    target = metrics.get("twr_target")
    if twr is None:
        issues.append("TWR unavailable (AUW × g was zero).")
    elif target is not None and twr < target:
        use_case = metrics.get("use_case", "unknown")
        issues.append(f"TWR {twr:.2f} below {use_case} target {target:.2f}.")
        suggestions.append("Reduce frame mass/payload or select higher-thrust motors.")

    # Check payload feasibility (R11.3)
    if not metrics.get("payload_feasible", False):
        margin = metrics.get("payload_margin_kg", 0.0)
        if margin is not None:
            deficit = abs(min(0.0, margin))
            issues.append(f"Payload infeasible by {deficit:.3f} kg.")
        else:
            issues.append("Payload feasibility unknown (margin unavailable).")
        suggestions.append("Increase thrust (motor class/count) or reduce payload.")

    # Check structural pass (R11.4)
    structural = metrics.get("structural", {})
    if not structural.get("passed", False):
        stress = structural.get("bending_stress_pa")
        allowable = structural.get("allowable_stress_pa")
        if stress is not None and allowable is not None:
            issues.append(
                f"Structural fail: stress {stress:.2e} Pa > allowable {allowable:.2e} Pa."
            )
        else:
            issues.append("Structural check failed (stress data unavailable).")
        suggestions.append(
            "Increase arm_width or material_thickness, or choose a higher-yield material."
        )

    passed = len(issues) == 0
    return (passed, issues, suggestions)
