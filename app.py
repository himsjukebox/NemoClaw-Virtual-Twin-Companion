# =============================================================================
# NemoClaw Virtual Twin Companion — Dynamic Polymorphic Streamlit UI
# =============================================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import os
import pyvista as pv

from main import run_graph

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

st.title("🚁 NemoClaw Virtual Twin Companion")
st.caption("Polymorphic Multi-Part Design Engine powered by NVIDIA NIM and CadQuery Library Archetypes")

col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Design Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_error"):
                st.error(msg["content"])
            elif msg.get("is_result"):
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("Describe your design goal (e.g., 'Design a 3-blade propeller with 100mm diameter')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔄 Processing through multi-part template factory pipeline..."):
                try:
                    result = run_graph(prompt)
                    st.session_state.last_trace = result.get("agent_trace", [])

                    # Track component type returned by the Graph
                    comp_type = result.get("component_type", "chassis")
                    st.session_state.active_component_type = comp_type

                    if result.get("error"):
                        error_msg = result["error"]
                        st.error(f"⚠️ {error_msg}")
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
                    else:
                        response_parts = []
                        verdict = result.get("validator_verdict", "N/A")
                        score = result.get("validator_score", 0.0)

                        response_parts.append(f"### {'✅' if verdict == 'PASS' else '❌'} Validator Verdict: **{verdict}** (Score: {score:.2f})")
                        response_parts.append("")

                        # POLYMORPHIC PARAMETER GENERATOR TABLE
                        params = result.get("design_parameters")
                        if params:
                            response_parts.append(f"#### 📐 Generated {comp_type.title()} Parameters")
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

                        cad_paths = result.get("cad_output_paths")
                        if cad_paths:
                            st.session_state.cad_output_paths = cad_paths

                        full_response = "\n".join(response_parts)
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_result": True})

                except Exception as e:
                    st.error(f"🔥 Unexpected error: {str(e)}")

with col_sidebar:
    with st.expander("🔍 Agent Trace", expanded=False):
        trace = st.session_state.last_trace
        if trace:
            for i, entry in enumerate(trace, 1):
                st.markdown(f"**{i}.** `{entry.get('node')}` → {entry.get('action')}")
        else:
            st.caption("No agent trace paths logged.")

    # POLYMORPHIC 3D VIEWPORT ENGINE
    st.subheader("🖥️ 3D Model Viewer")
    stl_path, step_path = None, None

    for path in st.session_state.get("cad_output_paths", []):
        if path.endswith(".stl"):
            stl_path = path
        elif path.endswith(".step") or path.endswith(".stp"):
            step_path = path

    # Local fallback scanner logic
    if not stl_path or not os.path.exists(stl_path):
        c_type = st.session_state.active_component_type
        disk_stl = os.path.join(OUTPUT_DIR, f"generated_{c_type}.stl")
        disk_step = os.path.join(OUTPUT_DIR, f"generated_{c_type}.step")
        if os.path.exists(disk_stl):
            stl_path = disk_stl
        if os.path.exists(disk_step):
            step_path = disk_step

    if stl_path and os.path.exists(stl_path):
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
        st.info("💡 **No 3D Model Found.** Run a conversational part generation task to update the workspace viewport.")
