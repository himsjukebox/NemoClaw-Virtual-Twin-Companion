# Feature: drone-physics-engineering-layer, Property 11
"""
Property-based and integration tests for the Physics Analysis Node.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from main import physics_analysis_node


# =============================================================================
# Property 11: Physics node pass-through on upstream error
# =============================================================================


@settings(max_examples=100)
@given(error_msg=st.text(min_size=1, max_size=100))
def test_physics_node_passthrough_on_error(error_msg):
    """
    Property 11: Physics node pass-through on upstream error.

    An error state is returned unchanged with no engineering_metrics.

    **Validates: Requirements 10.6**
    """
    state = {
        "user_prompt": "test",
        "component_type": "chassis",
        "design_parameters": {
            "arm_count": 4,
            "arm_length": 120.0,
            "arm_width": 15.0,
            "material_thickness": 5.0,
            "center_cutout_radius": 20.0,
        },
        "cad_output_paths": [],
        "validator_verdict": "PENDING",
        "validator_score": 0.0,
        "validator_feedback": "",
        "iteration_count": 1,
        "agent_trace": [],
        "error": error_msg,  # Error is set — node should pass through
        "material": "PLA",
        "mission_profile": {
            "payload_mass_kg": 0.0,
            "use_case": "cinematography",
            "target_flight_time_min": 12.0,
        },
        "frame_volume_m3": None,
        "engineering_metrics": None,
    }

    result = physics_analysis_node(state)

    # State is returned unchanged — no metrics computed
    assert result["error"] == error_msg
    assert result.get("engineering_metrics") is None
    # No physics trace entry added
    physics_traces = [
        t for t in result.get("agent_trace", []) if t.get("node") == "physics_analysis"
    ]
    assert len(physics_traces) == 0


# =============================================================================
# Integration: physics_analysis_node writes metrics and trace
# =============================================================================


class TestPhysicsNodeIntegration:
    """Integration tests for the physics analysis node pipeline order."""

    def test_physics_node_writes_metrics(self):
        """Physics node writes engineering_metrics to state when no error (R10.3)."""
        state = {
            "user_prompt": "build a racing quadcopter",
            "component_type": "chassis",
            "design_parameters": {
                "arm_count": 4,
                "arm_length": 120.0,
                "arm_width": 15.0,
                "material_thickness": 5.0,
                "center_cutout_radius": 20.0,
            },
            "cad_output_paths": ["output.step"],
            "validator_verdict": "PENDING",
            "validator_score": 0.0,
            "validator_feedback": "",
            "iteration_count": 1,
            "agent_trace": [],
            "error": None,
            "material": "PLA",
            "mission_profile": {
                "payload_mass_kg": 0.5,
                "use_case": "cinematography",
                "target_flight_time_min": 12.0,
            },
            "frame_volume_m3": None,
            "engineering_metrics": None,
        }

        result = physics_analysis_node(state)

        # Engineering metrics are written
        assert result["engineering_metrics"] is not None
        metrics = result["engineering_metrics"]
        assert "auw_kg" in metrics
        assert "twr" in metrics
        assert "structural" in metrics
        assert metrics["available"] is True

    def test_physics_node_appends_trace(self):
        """Physics node appends trace entry with expected keys (R10.5)."""
        state = {
            "user_prompt": "test",
            "component_type": "chassis",
            "design_parameters": {
                "arm_count": 4,
                "arm_length": 120.0,
                "arm_width": 15.0,
                "material_thickness": 5.0,
                "center_cutout_radius": 20.0,
            },
            "cad_output_paths": [],
            "validator_verdict": "PENDING",
            "validator_score": 0.0,
            "validator_feedback": "",
            "iteration_count": 1,
            "agent_trace": [{"node": "design_agent", "action": "GENERATED_PARAMS"}],
            "error": None,
            "material": "PLA",
            "mission_profile": {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0},
            "frame_volume_m3": None,
            "engineering_metrics": None,
        }

        result = physics_analysis_node(state)

        # Trace has a physics_analysis entry after the design_agent entry
        traces = result["agent_trace"]
        physics_traces = [t for t in traces if t["node"] == "physics_analysis"]
        assert len(physics_traces) == 1
        trace = physics_traces[0]
        assert trace["action"] == "COMPUTED_METRICS"
        assert "auw_kg" in trace
        assert "twr" in trace
        assert "structural_pass" in trace

    def test_physics_node_preserves_order(self):
        """Physics trace entry appears after any prior trace entries (R10.5)."""
        state = {
            "user_prompt": "test",
            "component_type": "chassis",
            "design_parameters": {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0, "material_thickness": 5.0, "center_cutout_radius": 20.0},
            "cad_output_paths": [],
            "validator_verdict": "PENDING",
            "validator_score": 0.0,
            "validator_feedback": "",
            "iteration_count": 1,
            "agent_trace": [
                {"node": "design_agent", "action": "GENERATED_PARAMS", "iteration": 1},
                {"node": "cad_tool", "action": "GENERATED_FILES", "iteration": 1},
            ],
            "error": None,
            "material": "carbon_fiber",
            "mission_profile": {"payload_mass_kg": 1.0, "use_case": "delivery", "target_flight_time_min": 15.0},
            "frame_volume_m3": 0.00005,
            "engineering_metrics": None,
        }

        result = physics_analysis_node(state)

        traces = result["agent_trace"]
        assert traces[0]["node"] == "design_agent"
        assert traces[1]["node"] == "cad_tool"
        assert traces[2]["node"] == "physics_analysis"
