# =============================================================================
# NemoClaw Virtual Twin Companion — Streamlit UI
# =============================================================================
# PURPOSE:
#   Provides the web-based chat interface for the NemoClaw Virtual Twin
#   Companion system. Users type natural-language drone design goals, the
#   Orchestrator (main.py) processes them through the multi-agent pipeline,
#   and results are displayed with full agent trace visibility.
#
# NVIDIA STACK CONTEXT:
#   - The chat pipeline invokes LangGraph → ChatNVIDIA → NemoClaw/OpenShell
#   - Agent Trace section demonstrates NeMo Agent Toolkit observability
#   - NVIDIA Riva ASR/TTS stubs show the Speech-to-CAD workflow path
#   - 3D model viewer stub shows PyVista/stpyvista integration path
#
# ENTRY POINT:
#   streamlit run app.py
#
# Component: Streamlit_UI
# =============================================================================

import streamlit as st
import pandas as pd
import json

from main import run_graph


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NemoClaw Virtual Twin Companion",
    page_icon="🚁",
    layout="wide",
)


# =============================================================================
# Session State Initialization
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = []


# =============================================================================
# Header
# =============================================================================

st.title("🚁 NemoClaw Virtual Twin Companion")
st.caption(
    "Conversational parametric CAD design powered by NVIDIA NIM, "
    "NeMo Agent Toolkit, and NemoClaw/OpenShell"
)


# =============================================================================
# Main Layout — Two Columns
# =============================================================================

col_chat, col_sidebar = st.columns([2, 1])

# =============================================================================
# Chat Column — User Input and Response Display
# =============================================================================

with col_chat:
    st.subheader("💬 Design Chat")

    # Display chat history preserving all messages within the session
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_error"):
                st.error(msg["content"])
            elif msg.get("is_result"):
                # Display design parameters as a formatted table
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])

    # Chat input where users type natural-language design goals
    if prompt := st.chat_input("Describe your drone design goal..."):
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Show loading indicator (st.spinner) while Orchestrator processes
        with st.chat_message("assistant"):
            with st.spinner("🔄 Processing design through multi-agent pipeline..."):
                try:
                    # Invoke the LangGraph orchestrator pipeline
                    result = run_graph(prompt)

                    # Store agent trace for display
                    st.session_state.last_trace = result.get("agent_trace", [])

                    # Check for errors (guardrail rejection, timeout, max iterations)
                    if result.get("error"):
                        error_msg = result["error"]
                        st.error(f"⚠️ {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "is_error": True,
                        })
                    else:
                        # Build response content with design parameters and verdict
                        response_parts = []

                        # Display validator verdict with score prominently
                        verdict = result.get("validator_verdict", "N/A")
                        score = result.get("validator_score", 0.0)
                        iterations = result.get("iteration_count", 0)

                        verdict_emoji = "✅" if verdict == "PASS" else "❌"
                        response_parts.append(
                            f"### {verdict_emoji} Validator Verdict: **{verdict}** "
                            f"(Score: {score:.2f})"
                        )
                        response_parts.append(
                            f"*Completed in {iterations} iteration(s)*"
                        )
                        response_parts.append("")

                        # Display Design_Parameters as a formatted table
                        params = result.get("design_parameters")
                        if params:
                            response_parts.append("#### 📐 Design Parameters")
                            # Create a DataFrame for clean table display
                            param_data = {
                                "Parameter": [],
                                "Value (mm)": [],
                                "Min": [],
                                "Max": [],
                            }
                            ranges = {
                                "arm_length": (80.0, 200.0),
                                "material_thickness": (2.0, 10.0),
                                "arm_width": (8.0, 25.0),
                                "center_cutout_radius": (10.0, 30.0),
                            }
                            for key, (lo, hi) in ranges.items():
                                param_data["Parameter"].append(key)
                                param_data["Value (mm)"].append(
                                    f"{params.get(key, 'N/A'):.2f}"
                                    if isinstance(params.get(key), (int, float))
                                    else "N/A"
                                )
                                param_data["Min"].append(f"{lo:.1f}")
                                param_data["Max"].append(f"{hi:.1f}")

                            df = pd.DataFrame(param_data)
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True,
                            )

                        # Display CAD output paths if available
                        cad_paths = result.get("cad_output_paths")
                        if cad_paths:
                            response_parts.append("#### 📁 CAD Output Files")
                            for path in cad_paths:
                                response_parts.append(f"- `{path}`")
                            response_parts.append("")

                        # Display validator feedback summary
                        feedback_raw = result.get("validator_feedback")
                        if feedback_raw:
                            try:
                                feedback = json.loads(feedback_raw)
                                if feedback.get("issues"):
                                    response_parts.append("#### ⚠️ Issues")
                                    for issue in feedback["issues"]:
                                        response_parts.append(f"- {issue}")
                                if feedback.get("suggestions"):
                                    response_parts.append("#### 💡 Suggestions")
                                    for suggestion in feedback["suggestions"]:
                                        response_parts.append(f"- {suggestion}")
                                if feedback.get("reasoning"):
                                    response_parts.append("#### 🧠 Reasoning")
                                    response_parts.append(feedback["reasoning"])
                            except (json.JSONDecodeError, TypeError):
                                # Feedback is plain text, not JSON
                                response_parts.append(f"**Feedback:** {feedback_raw}")

                        full_response = "\n".join(response_parts)
                        st.markdown(full_response)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "is_result": True,
                        })

                except Exception as e:
                    # Display unexpected errors with visual differentiation
                    error_detail = f"Unexpected error: {str(e)}"
                    st.error(f"🔥 {error_detail}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_detail,
                        "is_error": True,
                    })


# =============================================================================
# Sidebar Column — Agent Trace, 3D Viewer, and Riva Stubs
# =============================================================================

with col_sidebar:

    # =========================================================================
    # Agent Trace — Expandable Section (NeMo Agent Toolkit Observability)
    # =========================================================================
    # The Agent Trace shows chronological agent invocations, tool calls, and
    # state transitions as logged by NeMo Agent Toolkit observability patterns.
    # Each entry records the node name, action taken, and any relevant metadata.
    # =========================================================================

    with st.expander("🔍 Agent Trace", expanded=True):
        trace = st.session_state.last_trace
        if trace:
            for i, entry in enumerate(trace, 1):
                node = entry.get("node", "unknown")
                action = entry.get("action", "")

                # Color-code by action type for visual differentiation
                if action == "REJECTED":
                    st.markdown(
                        f"**{i}.** 🚫 `{node}` → {action}  \n"
                        f"&nbsp;&nbsp;&nbsp;Reason: {entry.get('reason', 'N/A')}"
                    )
                elif action == "PASSED":
                    st.markdown(f"**{i}.** ✅ `{node}` → {action}")
                elif action == "GENERATED_PARAMS":
                    params = entry.get("parameters", {})
                    iteration = entry.get("iteration", "?")
                    st.markdown(
                        f"**{i}.** 🎯 `{node}` → {action} "
                        f"(iter {iteration})"
                    )
                elif action == "GENERATED":
                    outputs = entry.get("outputs", [])
                    st.markdown(
                        f"**{i}.** 🏭 `{node}` → {action}  \n"
                        f"&nbsp;&nbsp;&nbsp;Files: {', '.join(outputs)}"
                    )
                elif action == "EVALUATED":
                    verdict = entry.get("verdict", "?")
                    score = entry.get("score", 0)
                    v_emoji = "✅" if verdict == "PASS" else "❌"
                    st.markdown(
                        f"**{i}.** {v_emoji} `{node}` → {action}  \n"
                        f"&nbsp;&nbsp;&nbsp;Verdict: {verdict} | Score: {score:.2f}"
                    )
                elif action in ("VALIDATION_FAILED", "EXECUTION_FAILED"):
                    st.markdown(
                        f"**{i}.** 🔥 `{node}` → {action}  \n"
                        f"&nbsp;&nbsp;&nbsp;{entry.get('reason', entry.get('errors', ''))}"
                    )
                else:
                    st.markdown(f"**{i}.** ⚙️ `{node}` → {action}")
        else:
            st.caption("No agent trace yet. Submit a design goal to see the pipeline in action.")

    # =========================================================================
    # 3D Model Viewer Placeholder — PyVista / stpyvista Integration Stub
    # =========================================================================
    # In production, this section would render the generated STEP/STL geometry
    # using PyVista (VTK-based 3D rendering) with stpyvista for Streamlit
    # embedding. The CAD Tool outputs (optimized_drone_chassis.step/.stl)
    # would be loaded and displayed interactively here.
    #
    # Integration path:
    #   1. Install: pip install pyvista stpyvista
    #   2. Load STL: mesh = pyvista.read("optimized_drone_chassis.stl")
    #   3. Render: stpyvista.stpyvista(plotter) inside this section
    # =========================================================================

    st.subheader("🖥️ 3D Model Viewer")
    st.info(
        "**PyVista/stpyvista 3D viewer** — integration pending.  \n"
        "Once connected, this section will render the generated drone chassis "
        "geometry (STEP/STL) with interactive rotation, zoom, and cross-section views."
    )

    # =========================================================================
    # NVIDIA Riva ASR Stub — Speech-to-CAD Workflow
    # =========================================================================
    # NVIDIA Riva ASR (Automatic Speech Recognition) enables the Speech-to-CAD
    # workflow where engineers can speak design goals hands-free:
    #
    # Workflow:
    #   1. User speaks into microphone → Riva ASR captures audio stream
    #   2. Riva ASR transcribes speech to text in real-time (streaming gRPC)
    #   3. Transcribed text is fed into the st.chat_input pipeline
    #   4. The Orchestrator processes the voice-originated request identically
    #      to typed input (same LangGraph pipeline)
    #   5. This enables workshop/lab environments where hands are occupied
    #
    # Integration path:
    #   1. Deploy Riva ASR server (NGC container or Riva ServiceMaker)
    #   2. Use riva.client.ASRService for streaming recognition
    #   3. Feed recognized text into st.session_state for chat processing
    #
    # NVIDIA Riva API: https://docs.nvidia.com/deeplearning/riva/
    # =========================================================================

    st.subheader("🎤 Voice Input (NVIDIA Riva ASR)")
    st.info(
        "**NVIDIA Riva Speech-to-Text** — integration pending.  \n"
        "Enables hands-free drone design: speak your design goals and "
        "Riva ASR transcribes them into the chat pipeline for processing."
    )

    # =========================================================================
    # NVIDIA Riva TTS Stub — Voice Feedback for Design Results
    # =========================================================================
    # NVIDIA Riva TTS (Text-to-Speech) provides voice feedback so engineers
    # can hear design results and validation verdicts without reading the screen:
    #
    # Workflow:
    #   1. Orchestrator completes design cycle → produces verdict + parameters
    #   2. Riva TTS synthesizes a spoken summary of the result
    #   3. Audio is played back through the browser (st.audio or WebRTC)
    #   4. Example output: "Design passed validation with score 0.85.
    #      Arm length 150mm, material thickness 4mm."
    #
    # Integration path:
    #   1. Deploy Riva TTS server (NGC container or Riva ServiceMaker)
    #   2. Use riva.client.TTSService to synthesize response text
    #   3. Stream audio to Streamlit via st.audio component
    #
    # NVIDIA Riva API: https://docs.nvidia.com/deeplearning/riva/
    # =========================================================================

    st.subheader("🔊 Voice Output (NVIDIA Riva TTS)")
    st.info(
        "**NVIDIA Riva Text-to-Speech** — integration pending.  \n"
        "Provides audible feedback of design results and validator verdicts "
        "for multimodal, hands-free interaction in workshop environments."
    )


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption(
    "Built with NVIDIA NIM • NeMo Agent Toolkit • NemoClaw/OpenShell • "
    "LangGraph • Streamlit"
)
