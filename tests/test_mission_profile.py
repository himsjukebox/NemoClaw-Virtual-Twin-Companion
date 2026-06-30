import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.design_agent import build_mission_profile, select_material

# Feature: drone-physics-engineering-layer, Property 12


# A minimal valid physics_config for testing
MOCK_PHYSICS_CONFIG = {
    "use_cases": {
        "racing": {"target_twr": 4.0, "default_flight_time_min": 5.0},
        "cinematography": {"target_twr": 2.0, "default_flight_time_min": 12.0},
        "delivery": {"target_twr": 2.0, "default_flight_time_min": 15.0},
        "mapping": {"target_twr": 2.0, "default_flight_time_min": 20.0},
    }
}


@settings(max_examples=100)
@given(payload_val=st.floats(min_value=-100, max_value=100))
def test_payload_clamp_property(payload_val):
    """
    Property 12: Payload clamp — assembled payload mass is always within
    [0.0, 10.0] and equals the nearest bound when out of range.

    Validates: Requirements 1.7
    """
    parsed = {"payload_mass_kg": payload_val}
    result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)

    # The returned payload_mass_kg is always in [0.0, 10.0]
    assert 0.0 <= result["payload_mass_kg"] <= 10.0

    # Values below 0 get clamped to 0.0
    if payload_val < 0.0:
        assert result["payload_mass_kg"] == 0.0
    # Values above 10.0 get clamped to 10.0
    elif payload_val > 10.0:
        assert result["payload_mass_kg"] == 10.0
    # Values within [0.0, 10.0] remain unchanged
    else:
        assert result["payload_mass_kg"] == payload_val


# ---------------------------------------------------------------------------
# Edge-case unit tests for boundary values
# ---------------------------------------------------------------------------


class TestPayloadClampBoundary:
    """Unit tests for payload clamping at exact boundary values."""

    def test_payload_exactly_zero(self):
        """Payload at lower bound (0.0) stays at 0.0."""
        parsed = {"payload_mass_kg": 0.0, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 0.0

    def test_payload_exactly_ten(self):
        """Payload at upper bound (10.0) stays at 10.0."""
        parsed = {"payload_mass_kg": 10.0, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 10.0

    def test_payload_negative_one(self):
        """Payload below lower bound (-1.0) is clamped to 0.0."""
        parsed = {"payload_mass_kg": -1.0, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 0.0

    def test_payload_slightly_above_upper(self):
        """Payload slightly above upper bound (10.1) is clamped to 10.0."""
        parsed = {"payload_mass_kg": 10.1, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 10.0

    def test_payload_large_negative(self):
        """Large negative payload (-100.0) is clamped to 0.0."""
        parsed = {"payload_mass_kg": -100.0, "use_case": "delivery"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 0.0

    def test_payload_large_positive(self):
        """Large positive payload (999.0) is clamped to 10.0."""
        parsed = {"payload_mass_kg": 999.0, "use_case": "racing"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 10.0

    def test_payload_just_inside_lower(self):
        """Payload just above lower bound (0.001) stays unchanged."""
        parsed = {"payload_mass_kg": 0.001, "use_case": "mapping"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 0.001

    def test_payload_just_inside_upper(self):
        """Payload just below upper bound (9.999) stays unchanged."""
        parsed = {"payload_mass_kg": 9.999, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 9.999

    def test_payload_midrange(self):
        """Payload in the middle of range (5.0) stays unchanged."""
        parsed = {"payload_mass_kg": 5.0, "use_case": "cinematography"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 5.0


# Feature: drone-physics-engineering-layer, Property 13

VALID_USE_CASES = ("racing", "cinematography", "delivery", "mapping")
VALID_MATERIALS = ("PLA", "ABS", "carbon_fiber", "aluminum")


@settings(max_examples=100)
@given(random_str=st.text(min_size=1, max_size=20))
def test_default_selection_property(random_str):
    """
    Property 13: Mission and material default selection.
    absent/unknown use case → cinematography; absent payload → 0.0;
    absent/unsupported material → PLA.

    Validates: Requirements 1.4, 1.6, 2.3
    """
    # --- Unknown use_case defaults to cinematography ---
    result = build_mission_profile({"use_case": random_str}, MOCK_PHYSICS_CONFIG)
    if random_str not in VALID_USE_CASES:
        assert result["use_case"] == "cinematography"
    else:
        # Valid use_case is preserved
        assert result["use_case"] == random_str

    # --- Absent payload defaults to 0.0 ---
    result2 = build_mission_profile({}, MOCK_PHYSICS_CONFIG)
    assert result2["payload_mass_kg"] == 0.0

    # --- Unknown material defaults to PLA ---
    mat = select_material({"material": random_str})
    if random_str not in VALID_MATERIALS:
        assert mat == "PLA"
    else:
        # Valid material is preserved
        assert mat == random_str


# =============================================================================
# Unit tests for mission/material extraction (Task 4.5)
# =============================================================================


class TestMissionProfileExtraction:
    """Unit tests verifying correct extraction from parsed LLM output."""

    def test_extracts_payload_mass(self):
        """Payload mass is extracted from parsed dict (R1.1)."""
        parsed = {"payload_mass_kg": 2.5, "use_case": "delivery"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["payload_mass_kg"] == 2.5

    def test_extracts_flight_time(self):
        """Target flight time is extracted when present (R1.2)."""
        parsed = {"target_flight_time_min": 8.0, "use_case": "racing"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["target_flight_time_min"] == 8.0

    def test_extracts_use_case(self):
        """Use case is extracted when valid (R1.3)."""
        for uc in ("racing", "cinematography", "delivery", "mapping"):
            parsed = {"use_case": uc}
            result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
            assert result["use_case"] == uc

    def test_flight_time_defaults_to_use_case(self):
        """Target flight time defaults to use-case default when absent (R1.5)."""
        parsed = {"use_case": "mapping"}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert result["target_flight_time_min"] == 20.0  # mapping default

    def test_mission_profile_keys(self):
        """Mission profile always has the three required keys (R1.8)."""
        parsed = {"payload_mass_kg": 1.0, "use_case": "delivery", "target_flight_time_min": 10.0}
        result = build_mission_profile(parsed, MOCK_PHYSICS_CONFIG)
        assert "payload_mass_kg" in result
        assert "use_case" in result
        assert "target_flight_time_min" in result


class TestMaterialSelection:
    """Unit tests verifying correct material selection."""

    def test_selects_valid_material(self):
        """Valid material is preserved (R2.1)."""
        for mat in ("PLA", "ABS", "carbon_fiber", "aluminum"):
            assert select_material({"material": mat}) == mat

    def test_unknown_material_defaults_pla(self):
        """Unknown material defaults to PLA (R2.2, R2.3)."""
        assert select_material({"material": "titanium"}) == "PLA"
        assert select_material({"material": ""}) == "PLA"
        assert select_material({}) == "PLA"
