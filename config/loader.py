# =============================================================================
# NemoClaw Virtual Twin Companion — Configuration Loader & Validator
# =============================================================================
# PURPOSE:
#   Provides functions to load and validate all YAML configuration files
#   (agents.yaml, tools.yaml, rag.yaml) at startup. If any file is missing,
#   has invalid YAML syntax, or is missing required keys, the loader raises
#   a descriptive error and aborts startup.
#
# DESIGN RATIONALE:
#   All configuration is externalized to YAML so the system is tunable without
#   code changes. Strict validation at startup catches misconfigurations early
#   rather than at runtime deep within the agent pipeline.
#
# NVIDIA STACK CONTEXT:
#   These configs drive which NVIDIA NIM models are used by ChatNVIDIA and
#   NVIDIAEmbeddings, ensuring the project is fully NVIDIA-stack compliant.
# =============================================================================

import os
from pathlib import Path
from typing import Any, Dict

import yaml


# =============================================================================
# Custom Exception
# =============================================================================


class ConfigValidationError(Exception):
    """
    Raised when a configuration file is missing, malformed, or fails validation.

    The error message always includes:
      - The file path that caused the failure
      - A description of what was expected
      - The specific key or structure that is missing/invalid

    Component: Orchestrator
    """

    pass


# =============================================================================
# Internal Helpers
# =============================================================================

# Default base directory for config files (project root's config/ folder)
_CONFIG_DIR = Path(__file__).parent


def _resolve_config_path(filename: str, config_dir: Path | None = None) -> Path:
    """
    Resolve the absolute path for a configuration file.

    Args:
        filename (str): Name of the YAML file (e.g., 'agents.yaml').
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Path: Absolute path to the configuration file.

    Component: Orchestrator
    """
    base = config_dir if config_dir is not None else _CONFIG_DIR
    return Path(base) / filename


def _load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML file, raising descriptive errors on failure.

    Args:
        filepath (Path): Absolute path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content as a dictionary.

    Raises:
        ConfigValidationError: If the file is missing or has invalid YAML syntax.

    Component: Orchestrator
    """
    if not filepath.exists():
        raise ConfigValidationError(
            f"Configuration file not found: '{filepath}'. "
            f"Expected a valid YAML file at this path. "
            f"Please ensure the file exists in the config/ directory."
        )

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            f"Invalid YAML syntax in '{filepath}': {e}. "
            f"Please fix the YAML formatting and try again."
        )

    if content is None:
        raise ConfigValidationError(
            f"Configuration file is empty: '{filepath}'. "
            f"Expected valid YAML content with required keys."
        )

    if not isinstance(content, dict):
        raise ConfigValidationError(
            f"Configuration file '{filepath}' must contain a YAML mapping "
            f"(key-value pairs) at the top level, but got {type(content).__name__}."
        )

    return content


def _validate_required_keys(
    data: Dict[str, Any],
    required_keys: list,
    context: str,
    filepath: Path,
) -> None:
    """
    Validate that all required keys are present in a dictionary.

    Args:
        data (Dict[str, Any]): Dictionary to validate.
        required_keys (list): List of keys that must be present.
        context (str): Description of where validation is happening (e.g., agent name).
        filepath (Path): Path to the config file (for error messages).

    Returns:
        None. Raises on validation failure instead.

    Raises:
        ConfigValidationError: If any required key is missing.

    Component: Orchestrator
    """
    for key in required_keys:
        if key not in data:
            raise ConfigValidationError(
                f"Missing required key '{key}' in {context} "
                f"in configuration file '{filepath}'. "
                f"Expected keys: {required_keys}."
            )


# =============================================================================
# Public Loaders
# =============================================================================


def load_agents_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and validate config/agents.yaml.

    Each top-level key represents an agent and must contain:
      - name (str): Display name of the agent
      - model (str): NVIDIA NIM model identifier
      - system_prompt (str): The agent's system prompt

    Args:
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Dict[str, Any]: Validated agents configuration dictionary.

    Raises:
        ConfigValidationError: If the file is missing, has invalid YAML,
            or is missing required keys.

    Component: Orchestrator
    """
    filepath = _resolve_config_path("agents.yaml", config_dir)
    data = _load_yaml_file(filepath)

    # Validate each agent entry has the required keys
    agent_required_keys = ["name", "model", "system_prompt"]

    if not data:
        raise ConfigValidationError(
            f"Configuration file '{filepath}' contains no agent definitions. "
            f"Expected at least one agent with keys: {agent_required_keys}."
        )

    for agent_name, agent_config in data.items():
        if not isinstance(agent_config, dict):
            raise ConfigValidationError(
                f"Agent '{agent_name}' in '{filepath}' must be a mapping "
                f"with keys: {agent_required_keys}, but got {type(agent_config).__name__}."
            )
        _validate_required_keys(
            agent_config,
            agent_required_keys,
            context=f"agent '{agent_name}'",
            filepath=filepath,
        )

    return data


def load_tools_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and validate config/tools.yaml.

    Each top-level key represents a tool and must contain:
      - name (str): Display name of the tool
      - script_path (str) OR embedding_model (str): Execution target

    Args:
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Dict[str, Any]: Validated tools configuration dictionary.

    Raises:
        ConfigValidationError: If the file is missing, has invalid YAML,
            or is missing required keys.

    Component: Orchestrator
    """
    filepath = _resolve_config_path("tools.yaml", config_dir)
    data = _load_yaml_file(filepath)

    if not data:
        raise ConfigValidationError(
            f"Configuration file '{filepath}' contains no tool definitions. "
            f"Expected at least one tool with keys: ['name', 'script_path' or 'embedding_model']."
        )

    for tool_name, tool_config in data.items():
        if not isinstance(tool_config, dict):
            raise ConfigValidationError(
                f"Tool '{tool_name}' in '{filepath}' must be a mapping "
                f"with keys: ['name', 'script_path' or 'embedding_model'], "
                f"but got {type(tool_config).__name__}."
            )

        # 'name' is always required
        _validate_required_keys(
            tool_config,
            ["name"],
            context=f"tool '{tool_name}'",
            filepath=filepath,
        )

        # Must have either 'script_path' or 'embedding_model'
        if "script_path" not in tool_config and "embedding_model" not in tool_config:
            raise ConfigValidationError(
                f"Tool '{tool_name}' in '{filepath}' is missing required key "
                f"'script_path' or 'embedding_model'. "
                f"Each tool must specify at least one of these execution targets."
            )

    return data


def load_rag_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and validate config/rag.yaml.

    Required structure:
      rag_pipeline:
        embedding: (dict with model config)
        vector_store: (dict with persistence config)
        retrieval: (dict with search parameters)

    Args:
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Dict[str, Any]: Validated RAG pipeline configuration dictionary.

    Raises:
        ConfigValidationError: If the file is missing, has invalid YAML,
            or is missing required keys.

    Component: Orchestrator
    """
    filepath = _resolve_config_path("rag.yaml", config_dir)
    data = _load_yaml_file(filepath)

    # Require top-level 'rag_pipeline' key
    if "rag_pipeline" not in data:
        raise ConfigValidationError(
            f"Missing required top-level key 'rag_pipeline' in '{filepath}'. "
            f"Expected structure: rag_pipeline -> embedding, vector_store, retrieval."
        )

    pipeline = data["rag_pipeline"]
    if not isinstance(pipeline, dict):
        raise ConfigValidationError(
            f"'rag_pipeline' in '{filepath}' must be a mapping, "
            f"but got {type(pipeline).__name__}."
        )

    # Validate required sub-keys within rag_pipeline
    pipeline_required_keys = ["embedding", "vector_store", "retrieval"]
    _validate_required_keys(
        pipeline,
        pipeline_required_keys,
        context="'rag_pipeline'",
        filepath=filepath,
    )

    # Validate each sub-section is a dict
    for section_key in pipeline_required_keys:
        section = pipeline[section_key]
        if not isinstance(section, dict):
            raise ConfigValidationError(
                f"'rag_pipeline.{section_key}' in '{filepath}' must be a mapping, "
                f"but got {type(section).__name__}."
            )

    return data


def load_physics_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load and validate config/physics.yaml.

    Required top-level keys:
      constants, materials, motors, batteries, use_cases, factors, components

    Sub-table validation:
      - Each material entry must have: density, yield_strength
      - Each use_case entry must have: target_twr, default_flight_time_min
      - Each motor entry must have: mass_kg, max_thrust_n
      - Each battery entry must have: capacity_mah, cells_s, mass_kg
      - factors must have: structural_safety_factor, usable_capacity_fraction,
        nominal_cell_voltage, propulsion_efficiency_factor
      - components must have: esc_mass_kg, propeller_mass_kg,
        propeller_diameter_mm, default_motor_class, default_battery_option

    Args:
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Dict[str, Any]: Validated physics configuration dictionary.

    Raises:
        ConfigValidationError: If the file is missing, has invalid YAML,
            or is missing required keys.

    Component: Orchestrator
    """
    filepath = _resolve_config_path("physics.yaml", config_dir)
    data = _load_yaml_file(filepath)

    # Validate top-level required keys
    top_level_keys = [
        "constants", "materials", "motors", "batteries",
        "use_cases", "factors", "components",
    ]
    _validate_required_keys(data, top_level_keys, "physics config", filepath)

    # Validate materials sub-table: each entry needs density + yield_strength
    materials = data["materials"]
    if not isinstance(materials, dict):
        raise ConfigValidationError(
            f"'materials' in '{filepath}' must be a mapping, "
            f"but got {type(materials).__name__}."
        )
    material_required_keys = ["density", "yield_strength"]
    for mat_name, mat_config in materials.items():
        if not isinstance(mat_config, dict):
            raise ConfigValidationError(
                f"Material '{mat_name}' in '{filepath}' must be a mapping "
                f"with keys: {material_required_keys}, "
                f"but got {type(mat_config).__name__}."
            )
        _validate_required_keys(
            mat_config, material_required_keys,
            context=f"material '{mat_name}'", filepath=filepath,
        )

    # Validate use_cases sub-table: each entry needs target_twr + default_flight_time_min
    use_cases = data["use_cases"]
    if not isinstance(use_cases, dict):
        raise ConfigValidationError(
            f"'use_cases' in '{filepath}' must be a mapping, "
            f"but got {type(use_cases).__name__}."
        )
    use_case_required_keys = ["target_twr", "default_flight_time_min"]
    for uc_name, uc_config in use_cases.items():
        if not isinstance(uc_config, dict):
            raise ConfigValidationError(
                f"Use case '{uc_name}' in '{filepath}' must be a mapping "
                f"with keys: {use_case_required_keys}, "
                f"but got {type(uc_config).__name__}."
            )
        _validate_required_keys(
            uc_config, use_case_required_keys,
            context=f"use case '{uc_name}'", filepath=filepath,
        )

    # Validate motors sub-table: each entry needs mass_kg + max_thrust_n
    motors = data["motors"]
    if not isinstance(motors, dict):
        raise ConfigValidationError(
            f"'motors' in '{filepath}' must be a mapping, "
            f"but got {type(motors).__name__}."
        )
    motor_required_keys = ["mass_kg", "max_thrust_n"]
    for motor_name, motor_config in motors.items():
        if not isinstance(motor_config, dict):
            raise ConfigValidationError(
                f"Motor '{motor_name}' in '{filepath}' must be a mapping "
                f"with keys: {motor_required_keys}, "
                f"but got {type(motor_config).__name__}."
            )
        _validate_required_keys(
            motor_config, motor_required_keys,
            context=f"motor '{motor_name}'", filepath=filepath,
        )

    # Validate batteries sub-table: each entry needs capacity_mah + cells_s + mass_kg
    batteries = data["batteries"]
    if not isinstance(batteries, dict):
        raise ConfigValidationError(
            f"'batteries' in '{filepath}' must be a mapping, "
            f"but got {type(batteries).__name__}."
        )
    battery_required_keys = ["capacity_mah", "cells_s", "mass_kg"]
    for batt_name, batt_config in batteries.items():
        if not isinstance(batt_config, dict):
            raise ConfigValidationError(
                f"Battery '{batt_name}' in '{filepath}' must be a mapping "
                f"with keys: {battery_required_keys}, "
                f"but got {type(batt_config).__name__}."
            )
        _validate_required_keys(
            batt_config, battery_required_keys,
            context=f"battery '{batt_name}'", filepath=filepath,
        )

    # Validate factors sub-table
    factors = data["factors"]
    if not isinstance(factors, dict):
        raise ConfigValidationError(
            f"'factors' in '{filepath}' must be a mapping, "
            f"but got {type(factors).__name__}."
        )
    factors_required_keys = [
        "structural_safety_factor", "usable_capacity_fraction",
        "nominal_cell_voltage", "propulsion_efficiency_factor",
    ]
    _validate_required_keys(
        factors, factors_required_keys,
        context="'factors'", filepath=filepath,
    )

    # Validate components sub-table
    components = data["components"]
    if not isinstance(components, dict):
        raise ConfigValidationError(
            f"'components' in '{filepath}' must be a mapping, "
            f"but got {type(components).__name__}."
        )
    components_required_keys = [
        "esc_mass_kg", "propeller_mass_kg", "propeller_diameter_mm",
        "default_motor_class", "default_battery_option",
    ]
    _validate_required_keys(
        components, components_required_keys,
        context="'components'", filepath=filepath,
    )

    return data


def load_all_configs(config_dir: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Load and validate all configuration files. Aborts startup on any failure.

    This is the primary entry point for the Orchestrator at startup. If any
    configuration file is missing, has invalid syntax, or fails validation,
    a ConfigValidationError is raised and startup is aborted.

    Args:
        config_dir (Path | None): Optional override for the config directory path.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with keys 'agents', 'tools', 'rag',
            'physics', each containing the validated configuration for that component.

    Raises:
        ConfigValidationError: If any configuration file fails to load or validate.
            The error message includes the file path and specific failure reason.

    Component: Orchestrator
    """
    return {
        "agents": load_agents_config(config_dir),
        "tools": load_tools_config(config_dir),
        "rag": load_rag_config(config_dir),
        "physics": load_physics_config(config_dir),
    }
