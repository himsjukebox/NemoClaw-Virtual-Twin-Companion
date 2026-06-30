# =============================================================================
# NemoClaw Virtual Twin Companion — Dynamic CAD Tool Node
# =============================================================================
import os
import sys
import re
import subprocess

# Resolve project root (parent of tools/) so paths work regardless of cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory for generated CAD assets — relative to the workspace.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Dir")


class CADTool:
    def __init__(self):
        # Template Registry mapping component_type to physical template files
        self.template_registry = {
            "chassis": "chassis_frame_template.py",
            "propeller": "propeller_template.py",
            "motor_mount": "motor_mount_template.py"
        }

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    def invoke(self, state: dict, config: dict = None) -> dict:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Route to correct template based on system state intent
        component_type = state.get("component_type", "chassis")
        template_name = self.template_registry.get(component_type, "chassis_frame_template.py")
        # Resolve template path relative to project root for robustness
        template_path = os.path.join(PROJECT_ROOT, template_name)

        # Generate unique file output targets to prevent browser asset caching
        stl_path = os.path.join(OUTPUT_DIR, f"generated_{component_type}.stl")
        step_path = os.path.join(OUTPUT_DIR, f"generated_{component_type}.step")

        # Clear old files
        for path in [stl_path, step_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        params = state.get("design_parameters", {})
        iteration = state.get("iteration_count", 1)
        node_trace = []

        try:
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template '{template_name}' missing from workspace.")

            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Dynamic Parametric Injection: Loops over any arbitrary keys returned by the LLM
            for param_key, param_val in params.items():
                # Matches patterns like: arm_length = 120 or blade_count = 2
                content = re.sub(rf"{param_key}\s*=\s*[\d\.]+", f"{param_key} = {param_val}", content)

            # Unify file output destinations by replacing the standard placeholders
            content = content.replace('"export_output.step"', f'r"{step_path}"')
            content = content.replace('"export_output.stl"', f'r"{stl_path}"')
            # Fallbacks for the base chassis template strings if not yet normalized
            content = content.replace('"optimized_drone_chassis.step"', f'r"{step_path}"')
            content = content.replace('"optimized_drone_chassis.stl"', f'r"{stl_path}"')

            runtime_script_path = os.path.join(OUTPUT_DIR, "runtime_execution_macro.py")
            with open(runtime_script_path, "w", encoding="utf-8") as f:
                f.write(content)

            result = subprocess.run(
                [sys.executable, runtime_script_path],
                capture_output=True,
                text=True,
                check=True,
                cwd=OUTPUT_DIR
            )

            if os.path.exists(stl_path) and os.path.exists(step_path):
                node_trace.append({
                    "node": "cad_tool",
                    "action": "GENERATED",
                    "outputs": [f"generated_{component_type}.stl", f"generated_{component_type}.step"],
                    "iteration": iteration
                })
                state["cad_output_paths"] = [stl_path, step_path]
                state["error"] = None
            else:
                raise FileNotFoundError("CadQuery finished, but output assets were not found on disk.")

        except subprocess.CalledProcessError as sub_err:
            error_msg = f"CadQuery Compile Error: {sub_err.stderr.strip()}"
            node_trace.append({"node": "cad_tool", "action": "EXECUTION_FAILED", "reason": error_msg, "iteration": iteration})
            state["error"] = error_msg
            state["cad_output_paths"] = []
        except Exception as e:
            error_msg = f"System Error: {str(e)}"
            node_trace.append({"node": "cad_tool", "action": "EXECUTION_FAILED", "reason": error_msg, "iteration": iteration})
            state["error"] = error_msg
            state["cad_output_paths"] = []

        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].extend(node_trace)
        return state
