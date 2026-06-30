# =============================================================================
# NemoClaw Virtual Twin Companion — Physics Config Smoke Test
# =============================================================================
# Smoke test that validates the actual production config/physics.yaml file
# loads correctly and contains all required tables, entries, and factors.
#
# Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
# =============================================================================

import pytest

from config.loader import load_physics_config


class TestPhysicsConfigPresence:
    """Smoke tests verifying config/physics.yaml exists and has all required schema elements."""

    def test_physics_config_loads(self):
        """The actual project physics.yaml loads without error."""
        config = load_physics_config()
        assert config is not None

    def test_has_all_top_level_sections(self):
        """All 7 required top-level sections are present."""
        config = load_physics_config()
        for key in ("constants", "materials", "motors", "batteries", "use_cases", "factors", "components"):
            assert key in config, f"Missing top-level key: {key}"

    def test_has_required_materials(self):
        """Required materials (PLA, ABS, carbon_fiber, aluminum) present with density and yield_strength."""
        config = load_physics_config()
        for mat in ("PLA", "ABS", "carbon_fiber", "aluminum"):
            assert mat in config["materials"], f"Missing material: {mat}"
            assert "density" in config["materials"][mat], f"{mat} missing 'density'"
            assert "yield_strength" in config["materials"][mat], f"{mat} missing 'yield_strength'"

    def test_has_required_use_cases(self):
        """Required use cases (racing, cinematography, delivery, mapping) present with target_twr and flight time."""
        config = load_physics_config()
        for uc in ("racing", "cinematography", "delivery", "mapping"):
            assert uc in config["use_cases"], f"Missing use case: {uc}"
            assert "target_twr" in config["use_cases"][uc], f"{uc} missing 'target_twr'"
            assert "default_flight_time_min" in config["use_cases"][uc], f"{uc} missing 'default_flight_time_min'"

    def test_has_required_factors(self):
        """All required engineering factors are present."""
        config = load_physics_config()
        for key in ("structural_safety_factor", "usable_capacity_fraction", "nominal_cell_voltage", "propulsion_efficiency_factor"):
            assert key in config["factors"], f"Missing factor: {key}"

    def test_has_required_components(self):
        """All required component defaults are present."""
        config = load_physics_config()
        for key in ("esc_mass_kg", "propeller_mass_kg", "propeller_diameter_mm", "default_motor_class", "default_battery_option"):
            assert key in config["components"], f"Missing component: {key}"

    def test_constants_have_g_and_air_density(self):
        """Constants section includes g and air_density."""
        config = load_physics_config()
        assert "g" in config["constants"]
        assert "air_density" in config["constants"]
