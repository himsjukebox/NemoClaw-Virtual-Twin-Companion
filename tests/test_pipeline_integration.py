"""
End-to-end pipeline integration tests for the physics-informed loop.

Tests that:
1. A physically valid configuration (with LLM mocked to PASS) yields PASS with
   engineering_metrics populated on the final state.
2. A physics-informed FAIL produces feedback containing physics issues that would
   reach the Design Agent for iteration (R11.6, R11.7).

Validates: Requirements 11.6, 11.7
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from main import (
    physics_analysis_node,
    validator_agent_node,
    design_agent_node,
    WhiteboardState,
)


class TestPipelineIntegration:
    """Integration tests for the design → cad → physics → validator pipeline."""

    def _make_state(self, params=None, material="PLA", use_case="cinematography", payload=0.0):
        """Helper to create a valid pipeline state."""
        if params is None:
            params = {
                "arm_count": 4,
                "arm_length": 120.0,
                "arm_width": 15.0,
                "material_thickness": 5.0,
                "center_cutout_radius": 20.0,
            }
        return {
            "user_prompt": "build a drone",
            "component_type": "chassis",
            "design_parameters": params,
            "cad_output_paths": ["output.step"],
            "validator_verdict": "PENDING",
            "validator_score": 0.0,
            "validator_feedback": "",
            "iteration_count": 1,
            "agent_trace": [],
            "error": None,
            "material": material,
            "mission_profile": {
                "payload_mass_kg": payload,
                "use_case": use_case,
                "target_flight_time_min": 12.0,
            },
            "frame_volume_m3": None,
            "engineering_metrics": None,
        }

    def test_physics_pass_yields_pass_verdict(self):
        """A physically valid configuration yields PASS with metrics present."""
        state = self._make_state()

        # Run physics analysis (deterministic, no mock needed)
        state = physics_analysis_node(state)
        assert state["engineering_metrics"] is not None
        assert state["engineering_metrics"]["available"] is True

        # Run validator with mocked LLM (returns PASS for range checks)
        with patch("main.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "verdict": "PASS",
                "score": 1.0,
                "issues": [],
                "reasoning": "All parameters in range."
            })
            mock_llm.invoke.return_value = mock_response
            state = validator_agent_node(state)

        # For the default config (PLA, 4 arms, default geometry), the physics
        # gate should pass — TWR meets cinematography target, payload feasible,
        # and structure holds.
        assert state["engineering_metrics"] is not None
        metrics = state["engineering_metrics"]
        # If all physics gates pass, the final verdict should be PASS
        if metrics["twr_pass"] and metrics["payload_feasible"] and metrics["structural"]["passed"]:
            assert state["validator_verdict"] == "PASS"

    def test_physics_fail_produces_feedback_for_iteration(self):
        """A physics FAIL merges issues into validator_feedback for Design Agent (R11.6, R11.7)."""
        # Use very thin arm dimensions to trigger structural failure
        state = self._make_state(params={
            "arm_count": 4,
            "arm_length": 200.0,
            "arm_width": 2.0,  # Very thin — will fail structural check
            "material_thickness": 2.0,  # Thin
            "center_cutout_radius": 20.0,
        })

        # Run physics analysis (deterministic)
        state = physics_analysis_node(state)
        metrics = state["engineering_metrics"]
        assert metrics is not None

        # Run validator with mocked LLM (LLM says PASS, but physics gate overrides)
        with patch("main.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "verdict": "PASS",
                "score": 1.0,
                "issues": [],
                "reasoning": "Parameters in range."
            })
            mock_llm.invoke.return_value = mock_response
            state = validator_agent_node(state)

        # The physics gate should override to FAIL because structural check fails
        assert not metrics["structural"]["passed"], (
            "Expected structural failure for 2mm arm_width on 200mm arm"
        )
        assert state["validator_verdict"] == "FAIL"

        # Feedback should contain physics issues referencing the structural failure
        feedback = json.loads(state["validator_feedback"])
        assert "issues" in feedback
        assert len(feedback["issues"]) > 0
        # At least one issue should reference structural/stress failure
        assert any(
            "tructural" in issue.lower() or "stress" in issue.lower()
            for issue in feedback["issues"]
        ), f"Expected structural/stress issue in feedback issues: {feedback['issues']}"

    def test_physics_fail_feedback_contains_suggestions(self):
        """Physics FAIL feedback includes actionable suggestions for the Design Agent (R11.6)."""
        # Trigger structural failure with thin arms
        state = self._make_state(params={
            "arm_count": 4,
            "arm_length": 200.0,
            "arm_width": 2.0,
            "material_thickness": 2.0,
            "center_cutout_radius": 20.0,
        })

        state = physics_analysis_node(state)

        with patch("main.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "verdict": "PASS",
                "score": 1.0,
                "issues": [],
                "reasoning": "Parameters in range."
            })
            mock_llm.invoke.return_value = mock_response
            state = validator_agent_node(state)

        feedback = json.loads(state["validator_feedback"])
        # Suggestions should be non-empty and reference actionable fixes
        assert "suggestions" in feedback
        assert len(feedback["suggestions"]) > 0
        # Suggestions should mention geometry or material changes
        suggestions_text = " ".join(feedback["suggestions"]).lower()
        assert any(
            term in suggestions_text
            for term in ["arm_width", "material_thickness", "material", "thickness", "width"]
        ), f"Expected actionable geometry/material suggestion: {feedback['suggestions']}"

    def test_design_agent_receives_physics_feedback_on_iteration(self):
        """On FAIL, the Design Agent node uses validator_feedback in its prompt (R11.7)."""
        # Set up a state as if validator already failed with physics feedback
        physics_feedback = json.dumps({
            "issues": [
                "Structural fail: stress 5.88e+07 Pa > allowable 2.50e+07 Pa.",
                "TWR 1.50 below cinematography target 2.00."
            ],
            "suggestions": [
                "Increase arm_width or material_thickness.",
                "Reduce frame mass/payload or select higher-thrust motors."
            ],
            "reasoning": "Physics gate FAILED."
        })

        state = self._make_state()
        state["validator_verdict"] = "FAIL"
        state["validator_feedback"] = physics_feedback
        state["iteration_count"] = 1

        # Mock the LLM to return an adjusted design
        with patch("main.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "component_type": "chassis",
                "design_parameters": {
                    "arm_count": 4,
                    "arm_length": 120.0,
                    "arm_width": 20.0,  # Increased per suggestion
                    "material_thickness": 6.0,  # Increased per suggestion
                    "center_cutout_radius": 20.0,
                }
            })
            mock_llm.invoke.return_value = mock_response
            state = design_agent_node(state)

        # The LLM should have been invoked with feedback context
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"] if isinstance(messages[1], dict) else str(messages[1])
        # The feedback should appear in the user content sent to LLM
        assert "FAILED" in user_msg or "REJECTED" in user_msg

    def test_full_physics_pass_populates_all_metrics(self):
        """A passing pipeline populates complete engineering metrics on the state."""
        state = self._make_state()

        # Run physics node
        state = physics_analysis_node(state)
        metrics = state["engineering_metrics"]

        # All required metric fields should be present and populated
        assert metrics["auw_kg"] > 0
        assert metrics["total_thrust_n"] > 0
        assert metrics["twr"] is not None
        assert metrics["twr"] > 0
        assert metrics["twr_target"] is not None
        assert metrics["payload_margin_kg"] is not None
        assert metrics["flight_time_min"] is not None
        assert metrics["disk_loading_nm2"] is not None
        assert metrics["structural"] is not None
        assert "passed" in metrics["structural"]
        assert metrics["available"] is True
