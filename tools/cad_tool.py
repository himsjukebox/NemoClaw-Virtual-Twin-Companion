# =============================================================================
# NemoClaw Virtual Twin Companion — CAD Tool
# =============================================================================
# PURPOSE:
#   Writes design parameters into master_drone_template.py and executes it
#   inside the NemoClaw/OpenShell sandboxed environment.
#
# ARCHITECTURE — PARAMETRIC BREP VS MESH-BASED GEOMETRY:
#   This tool generates geometry via CadQuery which uses the OpenCASCADE BREP
#   (Boundary Representation) kernel — the same mathematical foundation as
#   Dassault CATIA and 3DEXPERIENCE. BREP represents geometry as exact
#   mathematical curves and surfaces (NURBS, conics, planes), preserving
#   manufacturing-precision topology.
#
#   In contrast, mesh-based engines (Blender, PyVista, Three.js) approximate
#   surfaces with discrete triangular facets. While mesh is sufficient for
#   visualization and FEA pre-processing, it LOSES:
#     - Exact edge/face topology needed for CAM toolpath generation
#     - Parametric history (changing a parameter regenerates exact geometry)
#     - GD&T (Geometric Dimensioning & Tolerancing) reference datums
#
#   Our pipeline exports BOTH:
#     - .STEP (ISO 10303) — BREP exchange for manufacturing/3DEXPERIENCE import
#     - .STL (tessellated mesh) — for Streamlit/PyVista web viewer & FEA
#
# SECURITY MODEL:
#   The CAD script is NEVER executed on the local filesystem. Parameters are
#   injected into a copy of master_drone_template.py, then the modified script
#   is dispatched to NemoClaw/OpenShell for sandboxed execution. This prevents
#   LLM-generated values from compromising the host system.
#
# NVIDIA STACK CONTEXT:
#   NemoClaw/OpenShell provides enterprise-grade sandboxing for AI-generated
#   code execution, ensuring safety policies are enforced even when the LLM
#   produces adversarial or malformed parameter values.
# =============================================================================

import os
import re
from typing import Dict, Any, List, Tuple

from models.state import WhiteboardState
from models.parameters import PARAM_RANGES, PARAM_NAMES


# ---------------------------------------------------------------------------
# Template file location (relative to project root)
# ---------------------------------------------------------------------------
_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "master_drone_template.py",
)

# ---------------------------------------------------------------------------
# Sandbox timeout (seconds)
# ---------------------------------------------------------------------------
_SANDBOX_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Output file names produced by the CadQuery script
# ---------------------------------------------------------------------------
_OUTPUT_FILES = [
    "optimized_drone_chassis.step",
    "optimized_drone_chassis.stl",
]


class CADTool:
    """
    CAD Tool that writes parametric values into master_drone_template.py and
    dispatches execution to the NemoClaw/OpenShell sandboxed runtime.

    Architecture Note — Parametric BREP vs Mesh:
        This tool operates on parametric BREP geometry via CadQuery, which uses
        the same OpenCASCADE kernel as Dassault CATIA/3DEXPERIENCE. BREP
        preserves exact mathematical curves and surfaces, enabling:
          - Manufacturing-precision STEP export (ISO 10303)
          - Parametric regeneration when design variables change
          - GD&T datum preservation for downstream CAM toolpaths

        Mesh-based approaches (Blender, PyVista) approximate geometry with
        triangular facets — suitable for visualization but insufficient for
        manufacturing tolerances and parametric design iteration.

    Component: CAD_Tool
    """

    def __init__(self) -> None:
        """
        Initialize the CAD Tool with template path and sandbox configuration.

        Component: CAD_Tool
        """
        self.template_path: str = _TEMPLATE_PATH
        self.timeout_seconds: int = _SANDBOX_TIMEOUT_SECONDS
        self.output_files: List[str] = _OUTPUT_FILES

    def invoke(self, state: WhiteboardState) -> WhiteboardState:
        """
        Execute CAD generation in NemoClaw/OpenShell sandbox.

        This is the LangGraph node entry point. It validates parameters,
        injects them into the template, and dispatches to the sandbox.
        The script is NEVER executed locally.

        Node: cad_tool
        Reads: design_parameters
        Writes: cad_output_paths, error, agent_trace
        Routes to: validator_agent (via orchestrator edge)

        Args:
            state (WhiteboardState): Shared graph state with design_parameters.

        Returns:
            WhiteboardState: Updated state with cad_output_paths or error.

        Component: CAD_Tool
        """
        params = state.get("design_parameters")

        # Guard: missing parameters entirely
        if params is None:
            state["error"] = "CAD Tool received None design_parameters"
            state["agent_trace"].append({
                "node": "cad_tool",
                "action": "ERROR",
                "reason": "design_parameters is None",
            })
            return state

        # --- Step 1: Validate parameter ranges ---
        validation_errors = self._validate_ranges(params)
        if validation_errors:
            error_msg = (
                f"Parameter validation failed: {'; '.join(validation_errors)}"
            )
            state["error"] = error_msg
            state["agent_trace"].append({
                "node": "cad_tool",
                "action": "VALIDATION_FAILED",
                "errors": validation_errors,
            })
            return state

        # --- Step 2: Inject parameters into template script ---
        try:
            script_content = self._inject_parameters(params)
        except Exception as exc:
            error_msg = f"Parameter injection failed: {str(exc)}"
            state["error"] = error_msg
            state["agent_trace"].append({
                "node": "cad_tool",
                "action": "INJECTION_FAILED",
                "reason": str(exc),
            })
            return state

        # --- Step 3: Execute in NemoClaw/OpenShell sandbox ---
        # --- NEMOCLAW_OPENSHELL EXECUTION START ---
        # The modified script is dispatched to NemoClaw/OpenShell for
        # sandboxed execution. The script is NOT run on the local filesystem.
        # NemoClaw enforces memory limits, timeout policies, and filesystem
        # isolation so that LLM-generated parameter values cannot compromise
        # the host system.
        success, output = self._execute_in_sandbox(script_content)
        # --- NEMOCLAW_OPENSHELL EXECUTION END ---

        # --- Step 4: Update state based on execution result ---
        if success:
            state["cad_output_paths"] = list(self.output_files)
            state["agent_trace"].append({
                "node": "cad_tool",
                "action": "GENERATED",
                "outputs": state["cad_output_paths"],
                "parameters": params,
            })
        else:
            error_msg = f"CAD execution failed: {output}"
            state["error"] = error_msg
            state["agent_trace"].append({
                "node": "cad_tool",
                "action": "EXECUTION_FAILED",
                "reason": output,
            })

        return state

    def _validate_ranges(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate all four design parameters against their defined bounds.

        Checks each parameter in PARAM_NAMES against the canonical ranges
        defined in models.parameters.PARAM_RANGES. Returns a list of human-
        readable error strings identifying which parameters violated bounds.

        Args:
            params (Dict[str, Any]): Design parameters to validate.

        Returns:
            List[str]: List of validation error strings. Empty if all valid.
                Each error string identifies the parameter name, its value,
                and the acceptable range it violated.

        Component: CAD_Tool
        """
        errors: List[str] = []

        for param_name in PARAM_NAMES:
            lo, hi = PARAM_RANGES[param_name]
            value = params.get(param_name)

            if value is None:
                errors.append(
                    f"{param_name} is missing (expected value in [{lo}, {hi}])"
                )
            elif not isinstance(value, (int, float)):
                errors.append(
                    f"{param_name}={value!r} is not a numeric type "
                    f"(expected float in [{lo}, {hi}])"
                )
            elif value < lo or value > hi:
                errors.append(
                    f"{param_name}={value} outside [{lo}, {hi}]"
                )

        return errors

    def _inject_parameters(self, params: Dict[str, float]) -> str:
        """
        Write parameter values into master_drone_template.py's parametric
        variable section by replacing their numeric assignments.

        Reads the template file, finds lines matching the pattern
        `<param_name> = <numeric_value>` in the parametric variables section,
        and replaces the numeric literal with the new parameter value.

        The resulting script content is returned as a string — it is NOT
        written back to disk. The string is passed to the sandbox for
        execution.

        Args:
            params (Dict[str, float]): Validated design parameters with
                keys: arm_length, material_thickness, arm_width,
                center_cutout_radius.

        Returns:
            str: Modified script content with injected parameter values,
                ready for sandbox execution.

        Raises:
            FileNotFoundError: If master_drone_template.py does not exist.
            ValueError: If a parameter assignment line cannot be found in
                the template.

        Component: CAD_Tool
        """
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f"CAD template not found: {self.template_path}"
            )

        with open(self.template_path, "r", encoding="utf-8") as f:
            script_content = f.read()

        # Replace each parametric variable's numeric assignment.
        # Pattern matches lines like: `arm_length = 120.0`
        # where the value can be int or float (with optional decimal).
        for param_name in PARAM_NAMES:
            new_value = params[param_name]

            # Regex: captures the variable name and equals sign, replaces
            # the numeric literal (int or float, possibly negative).
            pattern = re.compile(
                rf"^({re.escape(param_name)}\s*=\s*)-?[\d]+(?:\.[\d]+)?",
                re.MULTILINE,
            )

            replacement = rf"\g<1>{new_value}"
            new_content, count = pattern.subn(replacement, script_content)

            if count == 0:
                raise ValueError(
                    f"Could not find assignment for '{param_name}' in "
                    f"template: {self.template_path}"
                )

            script_content = new_content

        return script_content

    def _execute_in_sandbox(self, script_content: str) -> Tuple[bool, str]:
        """
        Execute the modified CadQuery script in NemoClaw/OpenShell sandbox.

        This is a STUB implementation. In production, this method would:
          1. Connect to the NemoClaw/OpenShell API endpoint
          2. Submit the script_content for sandboxed execution
          3. Enforce a 60-second timeout
          4. Retrieve output files (STEP, STL) from the sandbox filesystem
          5. Return success/failure with output messages

        The sandbox provides:
          - Memory isolation (512 MB limit per config/tools.yaml)
          - Filesystem isolation (no host filesystem access)
          - Timeout enforcement (60 seconds)
          - Python environment with CadQuery pre-installed

        IMPORTANT: This method does NOT execute the script locally.

        Args:
            script_content (str): Complete Python/CadQuery script content
                with injected parameter values, ready for execution.

        Returns:
            Tuple[bool, str]: A tuple of (success, message) where:
                - success: True if execution completed and produced output
                  files, False on timeout or execution error.
                - message: On success, a summary of output files produced.
                  On failure, the error description or timeout notice.

        Component: CAD_Tool
        """
        # =====================================================================
        # NemoClaw/OpenShell Sandbox Execution Stub
        # =====================================================================
        # In a production deployment, this would call the NVIDIA NemoClaw API:
        #
        #   from nemoclaw import OpenShellClient
        #   client = OpenShellClient()
        #   result = client.execute(
        #       script=script_content,
        #       timeout=self.timeout_seconds,
        #       memory_limit_mb=512,
        #       runtime="cadquery",
        #   )
        #   if result.timed_out:
        #       return (False, f"Execution timed out after {self.timeout_seconds}s")
        #   if result.exit_code != 0:
        #       return (False, f"Execution error: {result.stderr}")
        #   return (True, f"Generated: {', '.join(self.output_files)}")
        #
        # For now, we return a simulated success to allow the orchestrator
        # pipeline to proceed during development and testing.
        # =====================================================================

        # Stub: simulate successful execution
        return (True, f"Generated: {', '.join(self.output_files)}")
