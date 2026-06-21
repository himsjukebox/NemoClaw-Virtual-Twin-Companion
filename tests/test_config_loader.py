# =============================================================================
# NemoClaw Virtual Twin Companion — Configuration Loader Tests
# =============================================================================
# Unit tests for config/loader.py validating:
#   - Successful loading of valid YAML configs
#   - Descriptive errors for missing files
#   - Descriptive errors for invalid YAML syntax
#   - Descriptive errors for missing required keys
#   - Startup abort behavior (exception propagation)
# =============================================================================

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from config.loader import (
    ConfigValidationError,
    load_agents_config,
    load_all_configs,
    load_rag_config,
    load_tools_config,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_config_dir(tmp_path):
    """Create a temporary config directory with valid YAML files."""
    # agents.yaml
    agents = {
        "design_agent": {
            "name": "Test Design Agent",
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "system_prompt": "You are a test agent.",
        },
        "validator_agent": {
            "name": "Test Validator Agent",
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "system_prompt": "You are a test validator.",
        },
    }
    (tmp_path / "agents.yaml").write_text(yaml.dump(agents), encoding="utf-8")

    # tools.yaml
    tools = {
        "cad_generator": {
            "name": "Parametric CAD Generator",
            "script_path": "master_drone_template.py",
        },
        "rag_retriever": {
            "name": "Engineering Knowledge Retriever",
            "embedding_model": "NV-Embed-QA",
        },
    }
    (tmp_path / "tools.yaml").write_text(yaml.dump(tools), encoding="utf-8")

    # rag.yaml
    rag = {
        "rag_pipeline": {
            "embedding": {"model": "NV-Embed-QA", "truncate": "END"},
            "vector_store": {"type": "faiss", "persist_directory": "data/vectorstore"},
            "retrieval": {"top_k": 5, "search_type": "similarity"},
        }
    }
    (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")

    return tmp_path


@pytest.fixture
def empty_config_dir(tmp_path):
    """Create an empty temporary config directory (no YAML files)."""
    return tmp_path


# =============================================================================
# Tests — Successful Loading
# =============================================================================


class TestSuccessfulLoading:
    """Tests for successfully loading valid configuration files."""

    def test_load_agents_config_returns_dict(self, valid_config_dir):
        """Agents config loads as a dictionary with agent keys."""
        result = load_agents_config(valid_config_dir)
        assert isinstance(result, dict)
        assert "design_agent" in result
        assert "validator_agent" in result

    def test_load_agents_config_has_required_keys(self, valid_config_dir):
        """Each agent entry contains name, model, and system_prompt."""
        result = load_agents_config(valid_config_dir)
        for agent_name, agent_config in result.items():
            assert "name" in agent_config
            assert "model" in agent_config
            assert "system_prompt" in agent_config

    def test_load_tools_config_returns_dict(self, valid_config_dir):
        """Tools config loads as a dictionary with tool keys."""
        result = load_tools_config(valid_config_dir)
        assert isinstance(result, dict)
        assert "cad_generator" in result
        assert "rag_retriever" in result

    def test_load_tools_config_has_execution_target(self, valid_config_dir):
        """Each tool entry has either script_path or embedding_model."""
        result = load_tools_config(valid_config_dir)
        assert "script_path" in result["cad_generator"]
        assert "embedding_model" in result["rag_retriever"]

    def test_load_rag_config_returns_dict(self, valid_config_dir):
        """RAG config loads with rag_pipeline key and sub-sections."""
        result = load_rag_config(valid_config_dir)
        assert isinstance(result, dict)
        assert "rag_pipeline" in result
        pipeline = result["rag_pipeline"]
        assert "embedding" in pipeline
        assert "vector_store" in pipeline
        assert "retrieval" in pipeline

    def test_load_all_configs_returns_all_three(self, valid_config_dir):
        """load_all_configs returns dict with agents, tools, and rag keys."""
        result = load_all_configs(valid_config_dir)
        assert "agents" in result
        assert "tools" in result
        assert "rag" in result

    def test_load_real_project_configs(self):
        """Validates that the actual project config files load successfully."""
        config_dir = Path(__file__).parent.parent / "config"
        result = load_all_configs(config_dir)
        assert "agents" in result
        assert "tools" in result
        assert "rag" in result


# =============================================================================
# Tests — Missing Files
# =============================================================================


class TestMissingFiles:
    """Tests for descriptive errors when config files are missing."""

    def test_missing_agents_yaml_raises_error(self, empty_config_dir):
        """Missing agents.yaml raises ConfigValidationError with file path."""
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(empty_config_dir)
        error_msg = str(exc_info.value)
        assert "agents.yaml" in error_msg
        assert "not found" in error_msg.lower()

    def test_missing_tools_yaml_raises_error(self, empty_config_dir):
        """Missing tools.yaml raises ConfigValidationError with file path."""
        with pytest.raises(ConfigValidationError) as exc_info:
            load_tools_config(empty_config_dir)
        error_msg = str(exc_info.value)
        assert "tools.yaml" in error_msg
        assert "not found" in error_msg.lower()

    def test_missing_rag_yaml_raises_error(self, empty_config_dir):
        """Missing rag.yaml raises ConfigValidationError with file path."""
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(empty_config_dir)
        error_msg = str(exc_info.value)
        assert "rag.yaml" in error_msg
        assert "not found" in error_msg.lower()

    def test_load_all_aborts_on_first_missing_file(self, empty_config_dir):
        """load_all_configs raises on the first missing file (aborts startup)."""
        with pytest.raises(ConfigValidationError):
            load_all_configs(empty_config_dir)


# =============================================================================
# Tests — Invalid YAML Syntax
# =============================================================================


class TestInvalidYAMLSyntax:
    """Tests for descriptive errors when YAML syntax is invalid."""

    def test_invalid_yaml_agents(self, tmp_path):
        """Invalid YAML in agents.yaml raises error with file path."""
        (tmp_path / "agents.yaml").write_text(
            "design_agent:\n  name: [unclosed bracket", encoding="utf-8"
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "agents.yaml" in error_msg
        assert "syntax" in error_msg.lower() or "YAML" in error_msg

    def test_invalid_yaml_tools(self, tmp_path):
        """Invalid YAML in tools.yaml raises error with file path."""
        (tmp_path / "tools.yaml").write_text(
            "cad_generator:\n  name: {bad: yaml: here", encoding="utf-8"
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            load_tools_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "tools.yaml" in error_msg

    def test_invalid_yaml_rag(self, tmp_path):
        """Invalid YAML in rag.yaml raises error with file path."""
        (tmp_path / "rag.yaml").write_text(
            "rag_pipeline:\n  - this\n  is: [broken", encoding="utf-8"
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "rag.yaml" in error_msg

    def test_empty_file_raises_error(self, tmp_path):
        """An empty YAML file raises a descriptive error."""
        (tmp_path / "agents.yaml").write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "empty" in error_msg.lower()
        assert "agents.yaml" in error_msg


# =============================================================================
# Tests — Missing Required Keys
# =============================================================================


class TestMissingRequiredKeys:
    """Tests for descriptive errors when required keys are missing."""

    def test_agent_missing_name(self, tmp_path):
        """Agent missing 'name' key raises error identifying the key."""
        agents = {
            "design_agent": {
                "model": "nvidia/test-model",
                "system_prompt": "test",
            }
        }
        (tmp_path / "agents.yaml").write_text(yaml.dump(agents), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "name" in error_msg
        assert "design_agent" in error_msg
        assert "agents.yaml" in error_msg

    def test_agent_missing_model(self, tmp_path):
        """Agent missing 'model' key raises error identifying the key."""
        agents = {
            "design_agent": {
                "name": "Test Agent",
                "system_prompt": "test",
            }
        }
        (tmp_path / "agents.yaml").write_text(yaml.dump(agents), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "model" in error_msg
        assert "agents.yaml" in error_msg

    def test_agent_missing_system_prompt(self, tmp_path):
        """Agent missing 'system_prompt' key raises error identifying the key."""
        agents = {
            "validator_agent": {
                "name": "Test Validator",
                "model": "nvidia/test-model",
            }
        }
        (tmp_path / "agents.yaml").write_text(yaml.dump(agents), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_agents_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "system_prompt" in error_msg
        assert "agents.yaml" in error_msg

    def test_tool_missing_name(self, tmp_path):
        """Tool missing 'name' key raises error identifying the key."""
        tools = {
            "cad_generator": {
                "script_path": "master_drone_template.py",
            }
        }
        (tmp_path / "tools.yaml").write_text(yaml.dump(tools), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_tools_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "name" in error_msg
        assert "tools.yaml" in error_msg

    def test_tool_missing_script_path_and_embedding_model(self, tmp_path):
        """Tool missing both script_path and embedding_model raises error."""
        tools = {
            "cad_generator": {
                "name": "Parametric CAD Generator",
            }
        }
        (tmp_path / "tools.yaml").write_text(yaml.dump(tools), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_tools_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "script_path" in error_msg or "embedding_model" in error_msg
        assert "tools.yaml" in error_msg

    def test_rag_missing_rag_pipeline(self, tmp_path):
        """RAG config missing 'rag_pipeline' key raises error."""
        rag = {"some_other_key": {"model": "test"}}
        (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "rag_pipeline" in error_msg
        assert "rag.yaml" in error_msg

    def test_rag_missing_embedding(self, tmp_path):
        """RAG config missing 'embedding' sub-key raises error."""
        rag = {
            "rag_pipeline": {
                "vector_store": {"type": "faiss"},
                "retrieval": {"top_k": 5},
            }
        }
        (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "embedding" in error_msg
        assert "rag.yaml" in error_msg

    def test_rag_missing_vector_store(self, tmp_path):
        """RAG config missing 'vector_store' sub-key raises error."""
        rag = {
            "rag_pipeline": {
                "embedding": {"model": "NV-Embed-QA"},
                "retrieval": {"top_k": 5},
            }
        }
        (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "vector_store" in error_msg
        assert "rag.yaml" in error_msg

    def test_rag_missing_retrieval(self, tmp_path):
        """RAG config missing 'retrieval' sub-key raises error."""
        rag = {
            "rag_pipeline": {
                "embedding": {"model": "NV-Embed-QA"},
                "vector_store": {"type": "faiss"},
            }
        }
        (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_rag_config(tmp_path)
        error_msg = str(exc_info.value)
        assert "retrieval" in error_msg
        assert "rag.yaml" in error_msg


# =============================================================================
# Tests — Startup Abort Behavior
# =============================================================================


class TestStartupAbort:
    """Tests verifying that config failures abort startup (exceptions propagate)."""

    def test_load_all_configs_does_not_catch_errors(self, empty_config_dir):
        """ConfigValidationError propagates without being caught."""
        with pytest.raises(ConfigValidationError):
            load_all_configs(empty_config_dir)

    def test_invalid_agents_aborts_all(self, tmp_path):
        """Invalid agents.yaml prevents tools and rag from loading."""
        # Write invalid agents but valid tools and rag
        (tmp_path / "agents.yaml").write_text("not: valid: yaml: [", encoding="utf-8")
        tools = {
            "cad_generator": {
                "name": "Test",
                "script_path": "test.py",
            }
        }
        (tmp_path / "tools.yaml").write_text(yaml.dump(tools), encoding="utf-8")
        rag = {
            "rag_pipeline": {
                "embedding": {"model": "test"},
                "vector_store": {"type": "faiss"},
                "retrieval": {"top_k": 5},
            }
        }
        (tmp_path / "rag.yaml").write_text(yaml.dump(rag), encoding="utf-8")

        with pytest.raises(ConfigValidationError):
            load_all_configs(tmp_path)
