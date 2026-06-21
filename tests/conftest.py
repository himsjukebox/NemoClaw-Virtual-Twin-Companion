# =============================================================================
# NemoClaw Virtual Twin Companion — Test Configuration & Shared Fixtures
# =============================================================================
# This conftest.py provides shared pytest fixtures and mocked NVIDIA client
# stubs used across all test modules. Mocking the NVIDIA API endpoints allows
# tests to run without a live API key or network access.
# =============================================================================

import sys
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Whiteboard State Fixtures
# =============================================================================


@pytest.fixture
def empty_state() -> Dict[str, Any]:
    """
    Provide a clean WhiteboardState with all keys at their initial values.

    Component: Test Infrastructure
    """
    return {
        "user_request": "",
        "design_parameters": None,
        "validator_feedback": None,
        "iteration_count": 0,
        "cad_output_paths": None,
        "agent_trace": [],
        "validator_verdict": None,
        "validator_score": None,
        "error": None,
    }


@pytest.fixture
def sample_design_parameters() -> Dict[str, float]:
    """
    Provide a valid set of design parameters within all defined ranges.

    Component: Test Infrastructure
    """
    return {
        "arm_length": 120.0,
        "material_thickness": 5.0,
        "arm_width": 15.0,
        "center_cutout_radius": 20.0,
    }


@pytest.fixture
def state_with_parameters(empty_state, sample_design_parameters) -> Dict[str, Any]:
    """
    Provide a WhiteboardState populated with valid design parameters.

    Component: Test Infrastructure
    """
    empty_state["user_request"] = "Design a lightweight racing drone"
    empty_state["design_parameters"] = sample_design_parameters
    return empty_state


# =============================================================================
# Mocked NVIDIA Client Stubs
# =============================================================================


@pytest.fixture
def mock_chat_nvidia():
    """
    Mock the ChatNVIDIA class from langchain-nvidia-ai-endpoints.

    Returns a MagicMock that simulates LLM responses without requiring
    a live NVIDIA API key or network access.

    Component: Test Infrastructure
    """
    with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA") as mock_cls:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"arm_length": 150.0, "material_thickness": 4.0, "arm_width": 18.0, "center_cutout_radius": 22.0}'
        mock_instance.invoke.return_value = mock_response
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_nvidia_embeddings():
    """
    Mock the NVIDIAEmbeddings class from langchain-nvidia-ai-endpoints.

    Returns a MagicMock that simulates embedding generation without
    requiring a live NVIDIA API key or network access.

    Component: Test Infrastructure
    """
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        # Simulate a 1024-dim embedding vector
        mock_instance.embed_query.return_value = [0.1] * 1024
        mock_instance.embed_documents.return_value = [[0.1] * 1024]
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_nvidia_api_key(monkeypatch):
    """
    Set a fake NVIDIA_API_KEY environment variable for tests.

    Component: Test Infrastructure
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-for-unit-tests")
