# =============================================================================
# NemoClaw Virtual Twin Companion — CAD Tool Integration Tests
# =============================================================================
# These tests exercise the REAL CadQuery geometry generation pipeline through
# CADTool.invoke(). They verify that:
#   - Each component type (chassis, propeller, motor_mount) produces STL + STEP
#   - The chassis arm_count parameter actually changes the generated geometry
#     (regression test for the "tricopter still builds a quadcopter" bug)
#   - Out-of-template component types fall back to the chassis template
#
# NOTE: These are integration tests that run CadQuery in a subprocess. They are
# skipped automatically if cadquery is not installed in the environment.
# =============================================================================

import os

import pytest

# Skip the whole module if cadquery isn't available (keeps unit suite green
# in minimal environments).
cadquery = pytest.importorskip("cadquery")

from tools.cad_tool import CADTool, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CHASSIS_PARAMS = {
    "arm_count": 4,
    "arm_length": 120.0,
    "material_thickness": 5.0,
    "arm_width": 15.0,
    "center_cutout_radius": 20.0,
}

PROPELLER_PARAMS = {
    "blade_count": 3,
    "diameter_mm": 127.0,
    "pitch_inches": 4.5,
    "hub_radius_mm": 6.0,
    "hub_thickness_mm": 7.0,
}

MOTOR_MOUNT_PARAMS = {
    "outer_diameter": 32.0,
    "mount_thickness": 4.5,
    "center_hole_diameter": 8.0,
    "bolt_spacing": 16.0,
}


def _run(component_type, params):
    """Invoke the CADTool for a component and return the resulting state."""
    tool = CADTool()
    state = {
        "component_type": component_type,
        "design_parameters": params,
        "iteration_count": 1,
        "agent_trace": [],
    }
    return tool.invoke(state)


def _output_files_exist(state):
    """True if all reported cad_output_paths exist and are non-empty."""
    paths = state.get("cad_output_paths", [])
    return len(paths) == 2 and all(
        os.path.exists(p) and os.path.getsize(p) > 0 for p in paths
    )


# ---------------------------------------------------------------------------
# Per-component generation tests
# ---------------------------------------------------------------------------

class TestComponentGeneration:
    """Each supported component type must produce STL + STEP output."""

    def test_chassis_generates_outputs(self):
        state = _run("chassis", dict(CHASSIS_PARAMS))
        assert state.get("error") is None
        assert _output_files_exist(state)

    def test_propeller_generates_outputs(self):
        state = _run("propeller", dict(PROPELLER_PARAMS))
        assert state.get("error") is None
        assert _output_files_exist(state)

    def test_motor_mount_generates_outputs(self):
        state = _run("motor_mount", dict(MOTOR_MOUNT_PARAMS))
        assert state.get("error") is None
        assert _output_files_exist(state)

    def test_unknown_component_falls_back_to_chassis(self):
        # An unregistered component type should fall back to the chassis template.
        state = _run("unknown_widget", dict(CHASSIS_PARAMS))
        assert state.get("error") is None
        assert _output_files_exist(state)


# ---------------------------------------------------------------------------
# Regression test: arm_count must actually change the geometry
# ---------------------------------------------------------------------------

class TestArmCountAffectsGeometry:
    """
    Regression test for the bug where requesting a tricopter still produced a
    quadcopter. Different arm_count values must produce measurably different
    geometry (different STL byte size / content).
    """

    def _generate_stl_bytes(self, arm_count):
        params = dict(CHASSIS_PARAMS)
        params["arm_count"] = arm_count
        state = _run("chassis", params)
        assert state.get("error") is None, f"generation failed: {state.get('error')}"
        stl_path = next(p for p in state["cad_output_paths"] if p.endswith(".stl"))
        with open(stl_path, "rb") as f:
            return f.read()

    def test_tricopter_generates(self):
        params = dict(CHASSIS_PARAMS)
        params["arm_count"] = 3
        state = _run("chassis", params)
        assert state.get("error") is None
        assert _output_files_exist(state)

    def test_tricopter_differs_from_octocopter(self):
        tri = self._generate_stl_bytes(3)
        octo = self._generate_stl_bytes(8)
        # More arms => meaningfully different (and larger) mesh.
        assert tri != octo
        assert len(octo) > len(tri)

    def test_quadcopter_differs_from_hexacopter(self):
        quad = self._generate_stl_bytes(4)
        hexa = self._generate_stl_bytes(6)
        assert quad != hexa


# ---------------------------------------------------------------------------
# Parameter injection test
# ---------------------------------------------------------------------------

class TestParameterInjection:
    """The injected runtime macro must contain the requested parameter values."""

    def test_runtime_macro_contains_injected_arm_count(self):
        params = dict(CHASSIS_PARAMS)
        params["arm_count"] = 6
        state = _run("chassis", params)
        assert state.get("error") is None

        macro_path = os.path.join(OUTPUT_DIR, "runtime_execution_macro.py")
        assert os.path.exists(macro_path)
        with open(macro_path, "r", encoding="utf-8") as f:
            content = f.read()
        # The injected macro should reflect the requested arm_count.
        assert "arm_count = 6" in content
