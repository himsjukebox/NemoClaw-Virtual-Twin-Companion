# =============================================================================
# NemoClaw Virtual Twin Companion — Dynamic Polymorphic Streamlit UI
# =============================================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import os
import io
import tempfile
from datetime import datetime

import pyvista as pv

from main import run_graph, stream_graph

# Output directory for generated CAD assets — relative to the workspace.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output_Dir")

st.set_page_config(page_title="NemoClaw Virtual Twin Companion", page_icon="🚁", layout="wide")

# Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "cad_output_paths" not in st.session_state:
    st.session_state.cad_output_paths = []
if "active_component_type" not in st.session_state:
    st.session_state.active_component_type = "chassis"

# Dynamic Parameter Constraint Reference Dictionary for Table Formats
COMPONENT_METADATA = {
    "chassis": {
        "arm_count": (3.0, 8.0), "arm_length": (80.0, 200.0), "material_thickness": (2.0, 10.0),
        "arm_width": (8.0, 25.0), "center_cutout_radius": (10.0, 30.0)
    },
    "propeller": {
        "blade_count": (2.0, 4.0), "diameter_mm": (50.0, 400.0),
        "pitch_inches": (2.0, 12.0), "hub_radius_mm": (3.0, 15.0),
        "hub_thickness_mm": (4.0, 20.0)
    },
    "motor_mount": {
        "outer_diameter": (15.0, 60.0), "mount_thickness": (1.5, 8.0),
        "center_hole_diameter": (3.0, 15.0), "bolt_spacing": (8.0, 30.0)
    }
}

def render_engineering_metrics(metrics):
    """
    Display engineering metrics with targets and pass/fail indicators (R13.1-R13.4).

    Called in the same result view as parameters and verdict.
    """
    if not metrics:
        st.info("Engineering analysis was not computed for this cycle.")
        return

    def _fmt(val, fmt=".2f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    def _pass_icon(flag):
        if flag is None:
            return "—"
        return "✅" if flag else "❌"

    rows = [
        {"Metric": "AUW (kg)", "Value": _fmt(metrics.get("auw_kg")), "Target": "—", "Status": "—"},
        {"Metric": "TWR", "Value": _fmt(metrics.get("twr")), "Target": _fmt(metrics.get("twr_target")), "Status": _pass_icon(metrics.get("twr_pass"))},
        {"Metric": "Payload Margin (kg)", "Value": _fmt(metrics.get("payload_margin_kg"), ".3f"), "Target": "≥ 0.0", "Status": _pass_icon(metrics.get("payload_feasible"))},
        {"Metric": "Flight Time (min)", "Value": _fmt(metrics.get("flight_time_min"), ".1f"), "Target": _fmt(metrics.get("flight_time_target_min"), ".1f"), "Status": _pass_icon(metrics.get("flight_time_pass"))},
        {"Metric": "Disk Loading (N/m²)", "Value": _fmt(metrics.get("disk_loading_nm2"), ".1f"), "Target": "—", "Status": "—"},
    ]

    st.subheader("📊 Engineering Metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def generate_physics_narrative(metrics):
    """
    Generate a plain-English summary of the engineering analysis using ChatNVIDIA.
    Returns a short 2-3 sentence narrative explaining the key results.
    """
    if not metrics:
        return None

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.3, max_tokens=200)

        prompt = f"""Summarize this drone engineering analysis in 2-3 sentences for a non-engineer. Be concise and specific with numbers:
- All-Up Weight: {metrics.get('auw_kg', 'N/A')} kg
- Thrust-to-Weight Ratio: {metrics.get('twr', 'N/A')} (target: {metrics.get('twr_target', 'N/A')}, {'PASS' if metrics.get('twr_pass') else 'FAIL'})
- Payload Margin: {metrics.get('payload_margin_kg', 'N/A')} kg ({'feasible' if metrics.get('payload_feasible') else 'infeasible'})
- Flight Time: {metrics.get('flight_time_min', 'N/A')} min (target: {metrics.get('flight_time_target_min', 'N/A')} min, {'PASS' if metrics.get('flight_time_pass') else 'FAIL'})
- Structural: {'PASS' if metrics.get('structural', {}).get('passed') else 'FAIL'}
- Use Case: {metrics.get('use_case', 'unknown')}
- Material: {metrics.get('structural', {}).get('material', 'unknown')}"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception:
        # Fallback to a template-based narrative if LLM fails
        twr = metrics.get("twr")
        auw = metrics.get("auw_kg")
        ft = metrics.get("flight_time_min")
        material = metrics.get("structural", {}).get("material", "PLA")
        use_case = metrics.get("use_case", "general")

        parts = []
        if auw and twr:
            parts.append(f"This {material} drone weighs {auw:.2f} kg with a TWR of {twr:.1f}.")
        if ft:
            parts.append(f"Estimated hover time is {ft:.0f} minutes for {use_case} use.")
        if metrics.get("structural", {}).get("passed"):
            margin = metrics.get("structural", {}).get("safety_margin", 0)
            parts.append(f"Arms pass structural checks with {margin:.1f}x safety margin.")
        return " ".join(parts) if parts else None


def capture_model_screenshot(stl_path):
    """Capture a PNG screenshot of the 3D model using PyVista offscreen rendering."""
    if not stl_path or not os.path.exists(stl_path):
        return None

    try:
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        mesh = pv.read(stl_path)
        plotter.add_mesh(mesh, color="#76B900", show_edges=True, edge_color="#1A1A1A", smooth_shading=True)
        plotter.view_isometric()
        plotter.background_color = "#FFFFFF"

        # Save to a temp file and read back
        screenshot_path = os.path.join(OUTPUT_DIR, "model_screenshot.png")
        plotter.screenshot(screenshot_path)
        plotter.close()

        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                return f.read()
    except Exception:
        pass
    return None


def generate_design_report_pdf(result, stl_path=None):
    """
    Generate a PDF design report with parameters, metrics, verdict, and 3D model screenshot.
    Returns PDF bytes or None if fpdf2 is not available.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "NemoClaw Design Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Verdict
    verdict = result.get("validator_verdict", "N/A")
    score = result.get("validator_score", 0.0)
    pdf.set_font("Helvetica", "B", 14)
    status = "PASS" if verdict == "PASS" else "FAIL"
    pdf.cell(0, 10, f"Verdict: {status} (Score: {score:.2f})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Design Parameters
    params = result.get("design_parameters", {})
    if params:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Design Parameters", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        comp_type = result.get("component_type", "chassis")
        pdf.cell(0, 6, f"Component Type: {comp_type}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Material: {result.get('material', 'PLA')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for key, val in params.items():
            val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            pdf.cell(0, 5, f"  {key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Mission Profile
    mission = result.get("mission_profile")
    if mission:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Mission Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, f"  Use Case: {mission.get('use_case', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"  Payload: {mission.get('payload_mass_kg', 0):.2f} kg", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"  Target Flight Time: {mission.get('target_flight_time_min', 'N/A')} min", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Engineering Metrics
    metrics = result.get("engineering_metrics")
    if metrics:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Engineering Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        def _v(val, fmt=".2f"):
            return f"{val:{fmt}}" if val is not None else "N/A"

        def _status(flag):
            return "PASS" if flag else "FAIL" if flag is not None else "N/A"

        metric_rows = [
            ("All-Up Weight (AUW)", f"{_v(metrics.get('auw_kg'))} kg", ""),
            ("Thrust-to-Weight Ratio", _v(metrics.get("twr")), f"Target: {_v(metrics.get('twr_target'))} [{_status(metrics.get('twr_pass'))}]"),
            ("Payload Margin", f"{_v(metrics.get('payload_margin_kg'), '.3f')} kg", f"[{_status(metrics.get('payload_feasible'))}]"),
            ("Flight Time", f"{_v(metrics.get('flight_time_min'), '.1f')} min", f"Target: {_v(metrics.get('flight_time_target_min'), '.1f')} min [{_status(metrics.get('flight_time_pass'))}]"),
            ("Disk Loading", f"{_v(metrics.get('disk_loading_nm2'), '.1f')} N/m2", ""),
        ]

        for name, value, note in metric_rows:
            line = f"  {name}: {value}"
            if note:
                line += f"  ({note})"
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

        # Structural detail
        structural = metrics.get("structural", {})
        pdf.ln(2)
        pdf.cell(0, 5, f"  Structural Check: {_status(structural.get('passed'))}", new_x="LMARGIN", new_y="NEXT")
        if structural.get("bending_stress_pa") is not None:
            pdf.cell(0, 5, f"    Bending Stress: {structural['bending_stress_pa']:.2e} Pa", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, f"    Allowable Stress: {structural.get('allowable_stress_pa', 0):.2e} Pa", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, f"    Safety Margin: {structural.get('safety_margin', 0):.2f}x", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Notes/Issues
        notes = metrics.get("notes", [])
        issues = metrics.get("issues", [])
        if notes:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 5, "Notes:", new_x="LMARGIN", new_y="NEXT")
            for note in notes:
                pdf.cell(0, 4, f"  - {note}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        if issues:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 5, "Issues:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for issue in issues:
                pdf.cell(0, 4, f"  - {issue}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # 3D Model Screenshot
    screenshot_data = capture_model_screenshot(stl_path)
    if screenshot_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "3D Model Preview", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        # Write screenshot to temp file for fpdf
        tmp_img = os.path.join(OUTPUT_DIR, "_report_screenshot.png")
        with open(tmp_img, "wb") as f:
            f.write(screenshot_data)
        pdf.image(tmp_img, x=15, w=180)
        # Clean up
        if os.path.exists(tmp_img):
            os.remove(tmp_img)

    # Return PDF bytes
    return pdf.output()


st.title("🚁 NemoClaw Virtual Twin Companion")
st.caption("AI-Powered Parametric Drone Component Designer — Generate chassis frames, propellers, and motor mounts with real-time physics analysis. Powered by NVIDIA NIM.")

# -------------------------------------------------------------------------
# Preset Demo Buttons — let judges try instantly without typing
# -------------------------------------------------------------------------
st.markdown("**🎯 Quick Start — Generate a drone component:**")

# Row 1: Chassis presets (with physics)
st.caption("🔧 **Chassis / Frame** (with full physics analysis)")
chassis_cols = st.columns(4)
preset_prompt = None

with chassis_cols[0]:
    if st.button("✅ Racing Quad", use_container_width=True):
        preset_prompt = "Design a chassis for a racing quadcopter with 4 arms, arm_length 100mm, arm_width 12mm, material_thickness 4mm, center_cutout_radius 15mm, using carbon_fiber material"

with chassis_cols[1]:
    if st.button("✅ Delivery Hex", use_container_width=True):
        preset_prompt = "Design a chassis for a delivery hexacopter with 6 arms, arm_length 140mm, arm_width 18mm, material_thickness 6mm, center_cutout_radius 20mm, using aluminum material, payload 1.5kg"

with chassis_cols[2]:
    if st.button("✅ Cinema Quad", use_container_width=True):
        preset_prompt = "Design a chassis for a cinematography quadcopter with 4 arms, arm_length 130mm, arm_width 15mm, material_thickness 5mm, center_cutout_radius 18mm, using PLA material, payload 0.5kg"

with chassis_cols[3]:
    if st.button("❌ Stress Fail Octo", use_container_width=True, help="This design will FAIL structural checks — arms too thin for their length"):
        preset_prompt = "Design a chassis for an octocopter with 8 arms, arm_length 200mm, arm_width 8mm, material_thickness 2mm, center_cutout_radius 28mm, using PLA material, payload 3kg"

# Row 2: Propeller and Motor Mount presets
st.caption("✈️ **Propellers & Motor Mounts**")
other_cols = st.columns(4)

with other_cols[0]:
    if st.button("✅ Racing Prop", use_container_width=True):
        preset_prompt = "Design a propeller with 2 blades, 127mm diameter, 5.5 inches pitch, hub_radius 6mm, hub_thickness 8mm"

with other_cols[1]:
    if st.button("❌ Oversized Prop", use_container_width=True, help="This propeller exceeds diameter limits"):
        preset_prompt = "Design a propeller with 4 blades, 500mm diameter, 14 inches pitch, hub_radius 20mm, hub_thickness 25mm"

with other_cols[2]:
    if st.button("✅ Standard Mount", use_container_width=True):
        preset_prompt = "Design a motor mount with outer_diameter 28mm, mount_thickness 4mm, center_hole_diameter 8mm, bolt_spacing 16mm"

with other_cols[3]:
    if st.button("❌ Invalid Mount", use_container_width=True, help="This mount has parameters outside valid ranges"):
        preset_prompt = "Design a motor mount with outer_diameter 70mm, mount_thickness 10mm, center_hole_diameter 20mm, bolt_spacing 40mm"

st.divider()

col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Design Chat")

    # Chat message history in a scrollable container
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("is_error"):
                    st.error(msg["content"])
                elif msg.get("is_result"):
                    st.markdown(msg["content"])
                else:
                    st.write(msg["content"])

    # Chat input at the bottom of the chat column
    if prompt := st.chat_input("Describe a component (e.g., 'Design a 6-arm chassis with carbon fiber' or 'Make a 3-blade 5-inch propeller')..."):
        preset_prompt = prompt  # Manual input takes over

    # Handle both preset and manual prompts
    if preset_prompt:
        st.session_state.messages.append({"role": "user", "content": preset_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.write(preset_prompt)

        with chat_container:
            with st.chat_message("assistant"):
                # Real-time progress display using graph streaming
                progress_placeholder = st.empty()
                result = None

                NODE_LABELS = {
                    "design_agent": "🎨 Design Agent — generating parameters...",
                    "cad_tool": "🔧 CAD Generation Agent — generating 3D geometry...",
                    "physics_analysis": "⚡ Physics Analysis Agent — computing engineering metrics...",
                    "validator_agent": "✅ Validator Agent — evaluating design...",
                }

                try:
                    completed_nodes = []
                    for node_name, state in stream_graph(preset_prompt):
                        result = state
                        completed_nodes.append(node_name)

                        # Build progress display
                        progress_lines = []
                        for completed in completed_nodes[:-1]:
                            label = NODE_LABELS.get(completed, completed)
                            progress_lines.append(f"✅ {label} Done!")
                        # Current node — show as in-progress until next one starts
                        current_label = NODE_LABELS.get(node_name, node_name)
                        progress_lines.append(f"🔄 {current_label}")

                        progress_placeholder.markdown("\n\n".join(progress_lines))

                    # Final update — mark last node as done before showing results
                    if completed_nodes:
                        progress_lines = []
                        for completed in completed_nodes:
                            label = NODE_LABELS.get(completed, completed)
                            progress_lines.append(f"✅ {label} Done!")
                        progress_placeholder.markdown("\n\n".join(progress_lines))

                    # Clear progress and show final results
                    progress_placeholder.empty()

                    if result is None:
                        st.error("⚠️ Pipeline returned no results.")
                        st.session_state.messages.append({"role": "assistant", "content": "Pipeline returned no results.", "is_error": True})
                    else:
                        st.session_state.last_trace = result.get("agent_trace", [])
                        st.session_state["last_result"] = result

                        comp_type = result.get("component_type", "chassis")
                        st.session_state.active_component_type = comp_type

                        if result.get("error"):
                            error_msg = result["error"]
                            st.error(f"⚠️ {error_msg}")
                            st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
                        else:
                            verdict = result.get("validator_verdict", "N/A")
                            score = result.get("validator_score", 0.0)
                            iterations = result.get("iteration_count", 1)

                            # Show iteration info if pipeline looped
                            if iterations > 1:
                                st.warning(f"🔄 **Design iterated {iterations} times.** The validator sent feedback to the Design Agent to improve the design.")

                            # Verdict header
                            st.markdown(f"### {'✅' if verdict == 'PASS' else '❌'} Validator Verdict: **{verdict}** (Score: {score:.2f})")

                            # POLYMORPHIC PARAMETER GENERATOR TABLE
                            params = result.get("design_parameters")
                            if params:
                                st.markdown(f"#### 📐 Generated {comp_type.title()} Parameters")
                                param_data = {"Parameter": [], "Value": [], "Min Constraint": [], "Max Constraint": []}

                                ranges = COMPONENT_METADATA.get(comp_type, {})
                                for key, val in params.items():
                                    param_data["Parameter"].append(key)
                                    param_data["Value"].append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                                    if key in ranges:
                                        param_data["Min Constraint"].append(f"{ranges[key][0]:.1f}")
                                        param_data["Max Constraint"].append(f"{ranges[key][1]:.1f}")
                                    else:
                                        param_data["Min Constraint"].append("N/A")
                                        param_data["Max Constraint"].append("N/A")

                                st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

                            # Display engineering metrics (R13.1-R13.3)
                            render_engineering_metrics(result.get("engineering_metrics"))

                            # Physics narrative — LLM-generated plain-English summary
                            eng_metrics = result.get("engineering_metrics")
                            if eng_metrics and eng_metrics.get("available"):
                                with st.spinner("💡 Generating engineering summary..."):
                                    narrative = generate_physics_narrative(eng_metrics)
                                st.session_state["cached_narrative"] = narrative
                                if narrative:
                                    st.info(f"**🧠 Engineering Summary:** {narrative}")
                            else:
                                st.session_state["cached_narrative"] = None

                            # Show iteration feedback if FAIL
                            if verdict == "FAIL":
                                feedback_raw = result.get("validator_feedback", "")
                                try:
                                    fb = json.loads(feedback_raw) if feedback_raw else {}
                                except (json.JSONDecodeError, TypeError):
                                    fb = {}
                                issues = fb.get("issues", [])
                                suggestions = fb.get("suggestions", [])
                                if issues:
                                    st.error("**Issues found:**\n" + "\n".join(f"- {i}" for i in issues))
                                if suggestions:
                                    st.warning("**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions))

                            cad_paths = result.get("cad_output_paths")
                            if cad_paths:
                                st.session_state.cad_output_paths = cad_paths

                            st.session_state.messages.append({"role": "assistant", "content": f"Verdict: {verdict} (Score: {score:.2f})", "is_result": True})

                except Exception as e:
                    st.error(f"🔥 Unexpected error: {str(e)}")

    # -------------------------------------------------------------------------
    # Persistent display of last result (survives page refresh)
    # -------------------------------------------------------------------------
    elif st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        if not result.get("error"):
            with chat_container:
                comp_type = result.get("component_type", "chassis")
                verdict = result.get("validator_verdict", "N/A")
                score = result.get("validator_score", 0.0)
                iterations = result.get("iteration_count", 1)

                if iterations > 1:
                    st.warning(f"🔄 **Design iterated {iterations} times.** The validator sent feedback to the Design Agent to improve the design.")

                st.markdown(f"### {'✅' if verdict == 'PASS' else '❌'} Last Verdict: **{verdict}** (Score: {score:.2f})")

                params = result.get("design_parameters")
                if params:
                    st.markdown(f"#### 📐 {comp_type.title()} Parameters")
                    param_data = {"Parameter": [], "Value": [], "Min": [], "Max": []}
                    ranges = COMPONENT_METADATA.get(comp_type, {})
                    for key, val in params.items():
                        param_data["Parameter"].append(key)
                        param_data["Value"].append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                        if key in ranges:
                            param_data["Min"].append(f"{ranges[key][0]:.1f}")
                            param_data["Max"].append(f"{ranges[key][1]:.1f}")
                        else:
                            param_data["Min"].append("N/A")
                            param_data["Max"].append("N/A")
                    st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

                render_engineering_metrics(result.get("engineering_metrics"))

                # Show cached engineering narrative (don't re-call LLM on rerun)
                eng_metrics = result.get("engineering_metrics")
                if eng_metrics and eng_metrics.get("available"):
                    if "cached_narrative" not in st.session_state:
                        try:
                            st.session_state["cached_narrative"] = generate_physics_narrative(eng_metrics)
                        except Exception:
                            st.session_state["cached_narrative"] = None
                    narrative = st.session_state.get("cached_narrative")
                    if narrative:
                        st.info(f"**🧠 Engineering Summary:** {narrative}")

                # Show issues and suggestions on FAIL
                if verdict == "FAIL":
                    feedback_raw = result.get("validator_feedback", "")
                    try:
                        fb = json.loads(feedback_raw) if feedback_raw else {}
                    except (json.JSONDecodeError, TypeError):
                        fb = {}
                    issues = fb.get("issues", [])
                    suggestions = fb.get("suggestions", [])
                    if issues:
                        st.error("**Issues found:**\n" + "\n".join(f"- {i}" for i in issues))
                    if suggestions:
                        st.warning("**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions))

with col_sidebar:
    with st.expander("🔍 Agent Trace", expanded=True):
        # Human-friendly node names
        NODE_DISPLAY_NAMES = {
            "design_agent": "🎨 Design Agent",
            "cad_tool": "🔧 CAD Generation Agent",
            "physics_analysis": "⚡ Physics Analysis Agent",
            "validator_agent": "✅ Validator Agent",
        }

        trace = st.session_state.last_trace
        if trace:
            for i, entry in enumerate(trace, 1):
                node = entry.get("node", "unknown")
                display_name = NODE_DISPLAY_NAMES.get(node, node)
                action = entry.get("action", "")
                st.markdown(f"**{i}.** {display_name} → {action}")
        else:
            st.caption("No agent trace paths logged.")

    # POLYMORPHIC 3D VIEWPORT ENGINE
    st.subheader("🖥️ 3D Model Viewer")
    stl_path, step_path = None, None

    # Only show 3D model if the last verdict was PASS
    last_result_for_3d = st.session_state.get("last_result")
    show_3d = last_result_for_3d and last_result_for_3d.get("validator_verdict") == "PASS"

    if show_3d:
        for path in st.session_state.get("cad_output_paths", []):
            if path.endswith(".stl"):
                stl_path = path
            elif path.endswith(".step") or path.endswith(".stp"):
                step_path = path

    # Local fallback scanner logic — only use if a model was generated this session
    if show_3d and (not stl_path or not os.path.exists(stl_path)) and st.session_state.get("last_result"):
        c_type = st.session_state.active_component_type
        disk_stl = os.path.join(OUTPUT_DIR, f"generated_{c_type}.stl")
        disk_step = os.path.join(OUTPUT_DIR, f"generated_{c_type}.step")
        if os.path.exists(disk_stl):
            stl_path = disk_stl
        if os.path.exists(disk_step):
            step_path = disk_step

    if show_3d and stl_path and os.path.exists(stl_path):
        try:
            pv.OFF_SCREEN = True
            plotter = pv.Plotter(notebook=True, window_size=[600, 400])
            mesh = pv.read(stl_path)
            plotter.add_mesh(mesh, color="#76B900", show_edges=True, edge_color="#1A1A1A", smooth_shading=True)
            plotter.view_isometric()
            plotter.background_color = "#FFFFFF"

            parent_dir = os.path.dirname(stl_path) if os.path.dirname(stl_path) else "."
            temp_html_output = os.path.join(parent_dir, "active_twin_viewport.html")
            plotter.export_html(temp_html_output)

            with open(temp_html_output, "r", encoding="utf-8") as html_file:
                components.html(html_file.read(), height=420, scrolling=False)
            if os.path.exists(temp_html_output):
                os.remove(temp_html_output)

            # Download buttons for both manufacturing formats (STEP + STL)
            comp = st.session_state.active_component_type.upper()
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if step_path and os.path.exists(step_path):
                    with open(step_path, "rb") as step_binary:
                        st.download_button(
                            label=f"📥 STEP ({comp})",
                            data=step_binary.read(),
                            file_name=os.path.basename(step_path),
                            mime="application/step",
                            use_container_width=True,
                            help="BREP solid for manufacturing / CAD import (CATIA, SolidWorks, Fusion).",
                        )
                else:
                    st.caption("STEP unavailable")
            with dl_col2:
                with open(stl_path, "rb") as stl_binary:
                    st.download_button(
                        label=f"📥 STL ({comp})",
                        data=stl_binary.read(),
                        file_name=os.path.basename(stl_path),
                        mime="application/octet-stream",
                        use_container_width=True,
                        help="Tessellated mesh for 3D printing / slicers (PrusaSlicer, Cura).",
                    )
        except Exception as view_err:
            st.error(f"Failed to compile 3D Viewport engine: {view_err}")
    else:
        if last_result_for_3d and last_result_for_3d.get("validator_verdict") == "FAIL":
            st.warning("⚠️ **Design validation failed.** 3D model not displayed for failed designs. Fix the issues and try again.")
        else:
            st.info("💡 **No 3D Model Found.** Run a conversational part generation task to update the workspace viewport.")

    # -------------------------------------------------------------------------
    # PDF Design Report Export
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("📄 Export Design Report")

    last_result = st.session_state.get("last_result")
    if last_result and not last_result.get("error"):
        # Generate PDF once and cache in session state
        if "cached_pdf" not in st.session_state or st.session_state.get("cached_pdf_for") != id(last_result):
            pdf_bytes = generate_design_report_pdf(last_result, stl_path=stl_path)
            if pdf_bytes:
                st.session_state["cached_pdf"] = bytes(pdf_bytes)
                st.session_state["cached_pdf_for"] = id(last_result)
            else:
                st.session_state["cached_pdf"] = None

        cached = st.session_state.get("cached_pdf")
        if cached:
            st.download_button(
                label="📥 Download Design Report (PDF)",
                data=cached,
                file_name=f"NemoClaw_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("PDF generation requires `fpdf2`. Install with: `pip install fpdf2`")
    else:
        st.caption("Run a design to enable report export.")
