# =============================================================================
# NemoClaw Virtual Twin Companion — Design Agent Unit Tests
# =============================================================================
# Tests for the DesignAgent class verifying parameter clamping, structural
# constraint enforcement, parse failure handling, and LangGraph state management.
# =============================================================================

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.parameters import PARAM_DEFAULTS, PARAM_RANGES, STRUCTURAL_CONSTRAINT_RATIO


# =============================================================================
# Test clamp_parameters
# =============================================================================


class TestClampParameters:
    """Tests for DesignAgent.clamp_parameters method."""

    @pytest.fixture
    def agent(self):
        """Create a DesignAgent with mocked LLM."""
        with patch("agents.design_agent.load_agents_config") as mock_config, \
             patch("agents.design_agent.ChatNVIDIA") as mock_llm:
            mock_config.return_value = {
                "design_agent": {
                    "name": "Test Agent",
                    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "system_prompt": "You are a test agent.",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
            }
            from agents.design_agent import DesignAgent
            agent = DesignAgent()
            return agent

    def test_values_within_range_unchanged(self, agent):
        """Parameters already within bounds should not change."""
        params = {
            "arm_length": 150.0,
            "material_thickness": 5.0,
            "arm_width": 15.0,
            "center_cutout_radius": 20.0,
        }
        result = agent.clamp_parameters(params)
        assert result == params

    def test_values_below_minimum_clamped_up(self, agent):
        """Parameters below minimum should be raised to minimum."""
        params = {
            "arm_length": 10.0,
            "material_thickness": 0.5,
            "arm_width": 2.0,
            "center_cutout_radius": 1.0,
        }
        result = agent.clamp_parameters(params)
        assert result["arm_length"] == 80.0
        assert result["material_thickness"] == 2.0
        assert result["arm_width"] == 8.0
        assert result["center_cutout_radius"] == 10.0

    def test_values_above_maximum_clamped_down(self, agent):
        """Parameters above maximum should be lowered to maximum."""
        params = {
            "arm_length": 999.0,
            "material_thickness": 50.0,
            "arm_width": 100.0,
            "center_cutout_radius": 500.0,
        }
        result = agent.clamp_parameters(params)
        assert result["arm_length"] == 200.0
        assert result["material_thickness"] == 10.0
        assert result["arm_width"] == 25.0
        assert result["center_cutout_radius"] == 30.0

    def test_missing_keys_use_defaults(self, agent):
        """Missing keys should use default values."""
        params = {"arm_length": 150.0}
        result = agent.clamp_parameters(params)
        assert result["arm_length"] == 150.0
        assert result["material_thickness"] == PARAM_DEFAULTS["material_thickness"]
        assert result["arm_width"] == PARAM_DEFAULTS["arm_width"]
        assert result["center_cutout_radius"] == PARAM_DEFAULTS["center_cutout_radius"]


# =============================================================================
# Test enforce_structural_constraint
# =============================================================================


class TestEnforceStructuralConstraint:
    """Tests for DesignAgent.enforce_structural_constraint method."""

    @pytest.fixture
    def agent(self):
        """Create a DesignAgent with mocked LLM."""
        with patch("agents.design_agent.load_agents_config") as mock_config, \
             patch("agents.design_agent.ChatNVIDIA") as mock_llm:
            mock_config.return_value = {
                "design_agent": {
                    "name": "Test Agent",
                    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "system_prompt": "You are a test agent.",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
            }
            from agents.design_agent import DesignAgent
            agent = DesignAgent()
            return agent

    def test_constraint_satisfied_no_change(self, agent):
        """If arm_width >= arm_length * 0.08, no change occurs."""
        params = {"arm_length": 100.0, "arm_width": 10.0}  # 10 >= 100*0.08=8
        result = agent.enforce_structural_constraint(params)
        assert result["arm_width"] == 10.0

    def test_constraint_violated_width_increased(self, agent):
        """If arm_width < arm_length * 0.08, width is increased."""
        params = {"arm_length": 200.0, "arm_width": 8.0}  # 8 < 200*0.08=16
        result = agent.enforce_structural_constraint(params)
        assert result["arm_width"] >= 200.0 * STRUCTURAL_CONSTRAINT_RATIO

    def test_constraint_respects_upper_bound(self, agent):
        """Enforced arm_width should not exceed the upper bound (25.0)."""
        # arm_length=200 → min_width=16.0, which is within [8, 25]
        params = {"arm_length": 200.0, "arm_width": 8.0}
        result = agent.enforce_structural_constraint(params)
        assert result["arm_width"] <= PARAM_RANGES["arm_width"][1]


# =============================================================================
# Test _parse_parameters
# =============================================================================


class TestParseParameters:
    """Tests for DesignAgent._parse_parameters method."""

    @pytest.fixture
    def agent(self):
        """Create a DesignAgent with mocked LLM."""
        with patch("agents.design_agent.load_agents_config") as mock_config, \
             patch("agents.design_agent.ChatNVIDIA") as mock_llm:
            mock_config.return_value = {
                "design_agent": {
                    "name": "Test Agent",
                    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "system_prompt": "You are a test agent.",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
            }
            from agents.design_agent import DesignAgent
            agent = DesignAgent()
            return agent

    def test_valid_json_parsed(self, agent):
        """Valid JSON with all four keys should be parsed correctly."""
        text = '{"arm_length": 150.0, "material_thickness": 4.0, "arm_width": 18.0, "center_cutout_radius": 22.0}'
        result = agent._parse_parameters(text)
        assert result["arm_length"] == 150.0
        assert result["material_thickness"] == 4.0
        assert result["arm_width"] == 18.0
        assert result["center_cutout_radius"] == 22.0

    def test_json_in_code_fence(self, agent):
        """JSON inside ```json fence should be extracted."""
        text = 'Here are the params:\n```json\n{"arm_length": 130.0, "material_thickness": 3.0, "arm_width": 12.0, "center_cutout_radius": 18.0}\n```\nDone.'
        result = agent._parse_parameters(text)
        assert result["arm_length"] == 130.0

    def test_invalid_json_returns_defaults(self, agent):
        """Non-JSON text should return defaults."""
        result = agent._parse_parameters("This is just text with no JSON")
        assert result == PARAM_DEFAULTS

    def test_empty_string_returns_defaults(self, agent):
        """Empty string should return defaults."""
        result = agent._parse_parameters("")
        assert result == PARAM_DEFAULTS

    def test_partial_keys_returns_defaults(self, agent):
        """JSON missing some keys should return defaults."""
        text = '{"arm_length": 150.0, "material_thickness": 4.0}'
        result = agent._parse_parameters(text)
        assert result == PARAM_DEFAULTS

    def test_malformed_json_returns_defaults(self, agent):
        """Malformed JSON should return defaults."""
        text = '{"arm_length": 150.0, "material_thickness": }'
        result = agent._parse_parameters(text)
        assert result == PARAM_DEFAULTS


# =============================================================================
# Test invoke method
# =============================================================================


class TestInvoke:
    """Tests for DesignAgent.invoke method."""

    @pytest.fixture
    def agent(self):
        """Create a DesignAgent with mocked LLM that returns valid JSON."""
        with patch("agents.design_agent.load_agents_config") as mock_config, \
             patch("agents.design_agent.ChatNVIDIA") as mock_llm_cls:
            mock_config.return_value = {
                "design_agent": {
                    "name": "Test Agent",
                    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "system_prompt": "You are a test agent.",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
            }
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"arm_length": 150.0, "material_thickness": 4.0, "arm_width": 18.0, "center_cutout_radius": 22.0}'
            mock_instance.invoke.return_value = mock_response
            mock_llm_cls.return_value = mock_instance

            from agents.design_agent import DesignAgent
            agent = DesignAgent()
            return agent

    def test_empty_request_returns_defaults(self, agent):
        """Empty user_request returns defaults without calling LLM."""
        state = {
            "user_request": "",
            "validator_feedback": None,
            "iteration_count": 0,
            "agent_trace": [],
        }
        result = agent.invoke(state)
        assert result["design_parameters"] == PARAM_DEFAULTS
        assert result["iteration_count"] == 1
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0]["action"] == "RETURNED_DEFAULTS"

    def test_whitespace_request_returns_defaults(self, agent):
        """Whitespace-only user_request returns defaults without calling LLM."""
        state = {
            "user_request": "   ",
            "validator_feedback": None,
            "iteration_count": 0,
            "agent_trace": [],
        }
        result = agent.invoke(state)
        assert result["design_parameters"] == PARAM_DEFAULTS

    def test_valid_request_increments_iteration(self, agent):
        """A valid request should increment iteration_count."""
        state = {
            "user_request": "Make a lightweight drone",
            "validator_feedback": None,
            "iteration_count": 0,
            "agent_trace": [],
        }
        result = agent.invoke(state)
        assert result["iteration_count"] == 1

    def test_valid_request_appends_trace(self, agent):
        """A valid request should append to agent_trace."""
        state = {
            "user_request": "Make a lightweight drone",
            "validator_feedback": None,
            "iteration_count": 0,
            "agent_trace": [],
        }
        result = agent.invoke(state)
        assert len(result["agent_trace"]) == 1
        assert result["agent_trace"][0]["node"] == "design_agent"
        assert result["agent_trace"][0]["action"] == "GENERATED_PARAMS"

    def test_llm_failure_returns_defaults(self):
        """LLM exception should result in default parameters."""
        with patch("agents.design_agent.load_agents_config") as mock_config, \
             patch("agents.design_agent.ChatNVIDIA") as mock_llm_cls:
            mock_config.return_value = {
                "design_agent": {
                    "name": "Test Agent",
                    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "system_prompt": "You are a test agent.",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
            }
            mock_instance = MagicMock()
            mock_instance.invoke.side_effect = Exception("API timeout")
            mock_llm_cls.return_value = mock_instance

            from agents.design_agent import DesignAgent
            agent = DesignAgent()

            state = {
                "user_request": "Make a drone",
                "validator_feedback": None,
                "iteration_count": 0,
                "agent_trace": [],
            }
            result = agent.invoke(state)
            # Defaults are clamped and constraint-enforced — which equals defaults
            assert result["design_parameters"] == PARAM_DEFAULTS

    def test_validator_feedback_included_in_state(self, agent):
        """When validator_feedback is present, iteration_count is still incremented."""
        state = {
            "user_request": "Make a drone",
            "validator_feedback": '{"verdict": "FAIL", "issues": ["arm too thin"]}',
            "iteration_count": 2,
            "agent_trace": [],
        }
        result = agent.invoke(state)
        assert result["iteration_count"] == 3
