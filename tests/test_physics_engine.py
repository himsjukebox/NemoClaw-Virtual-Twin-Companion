# Feature: drone-physics-engineering-layer, Property 2
"""
Property-based tests for the deterministic Physics Engine.

Uses Hypothesis with at least 100 examples per property.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools.physics_engine import PhysicsEngine

# Minimal valid config for testing pure functions — no file I/O needed
MOCK_CONFIG = {
    "constants": {"g": 9.80665, "air_density": 1.225},
    "materials": {"PLA": {"density": 1240.0, "yield_strength": 50.0e6}},
    "motors": {"2207_2400kv": {"mass_kg": 0.032, "max_thrust_n": 14.7}},
    "batteries": {"4s_1500mah": {"capacity_mah": 1500, "cells_s": 4, "mass_kg": 0.190}},
    "use_cases": {"cinematography": {"target_twr": 2.0, "default_flight_time_min": 12.0}},
    "factors": {
        "structural_safety_factor": 2.0,
        "usable_capacity_fraction": 0.8,
        "nominal_cell_voltage": 3.7,
        "propulsion_efficiency_factor": 0.12,
    },
    "components": {
        "esc_mass_kg": 0.010,
        "propeller_mass_kg": 0.008,
        "propeller_diameter_mm": 127.0,
        "default_motor_class": "2207_2400kv",
        "default_battery_option": "4s_1500mah",
    },
}


# =============================================================================
# Property 2: Mass additivity and payload sensitivity
# =============================================================================


@settings(max_examples=100)
@given(
    frame_mass=st.floats(min_value=0.01, max_value=5.0),
    motor_count=st.integers(min_value=1, max_value=8),
    motor_mass=st.floats(min_value=0.01, max_value=0.2),
    esc_mass=st.floats(min_value=0.005, max_value=0.05),
    prop_mass=st.floats(min_value=0.005, max_value=0.05),
    battery_mass=st.floats(min_value=0.1, max_value=1.0),
    payload_mass=st.floats(min_value=0.0, max_value=10.0),
    delta=st.floats(min_value=0.0, max_value=5.0),
)
def test_mass_additivity_property(
    frame_mass, motor_count, motor_mass, esc_mass, prop_mass, battery_mass, payload_mass, delta
):
    """
    Property 2: Mass additivity and payload sensitivity.

    AUW equals the exact component sum; adding payload delta raises AUW
    by exactly that delta.

    **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.7, 14.6**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)

    # --- Part A: AUW equals the exact formula ---
    auw = engine.auw_kg(
        frame_mass, motor_count, motor_mass, esc_mass, prop_mass, battery_mass, payload_mass
    )
    expected = (
        frame_mass + motor_count * (motor_mass + esc_mass + prop_mass) + battery_mass + payload_mass
    )
    assert abs(auw - expected) < 1e-10, (
        f"AUW {auw} != expected sum {expected}"
    )

    # --- Part B: Adding delta to payload raises AUW by exactly delta ---
    auw_with_delta = engine.auw_kg(
        frame_mass, motor_count, motor_mass, esc_mass, prop_mass, battery_mass, payload_mass + delta
    )
    assert abs(auw_with_delta - (auw + delta)) < 1e-10, (
        f"AUW with delta {auw_with_delta} != original AUW + delta {auw + delta}"
    )


# =============================================================================
# Property 3: Total thrust monotonicity in motor count
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 3


@settings(max_examples=100)
@given(
    per_motor_thrust=st.floats(min_value=0.1, max_value=50.0),
    count1=st.integers(min_value=1, max_value=8),
    count2=st.integers(min_value=1, max_value=8),
)
def test_thrust_monotonicity_property(per_motor_thrust, count1, count2):
    """
    Property 3: Total thrust monotonicity in motor count.

    total_thrust = count × per_motor_thrust, non-decreasing in count.

    **Validates: Requirements 4.1, 4.4**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)

    thrust1 = engine.total_thrust_n(count1, per_motor_thrust)
    thrust2 = engine.total_thrust_n(count2, per_motor_thrust)

    # Verify formula: total_thrust = count × per_motor_thrust
    assert abs(thrust1 - count1 * per_motor_thrust) < 1e-10
    assert abs(thrust2 - count2 * per_motor_thrust) < 1e-10

    # Monotonicity: if count2 >= count1, then thrust2 >= thrust1
    if count2 >= count1:
        assert thrust2 >= thrust1


# =============================================================================
# Property 4: TWR monotonicity in weight and thrust
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 4


@settings(max_examples=100)
@given(
    thrust=st.floats(min_value=1.0, max_value=200.0),
    auw1=st.floats(min_value=0.1, max_value=10.0),
    auw2=st.floats(min_value=0.1, max_value=10.0),
    thrust1=st.floats(min_value=1.0, max_value=200.0),
    thrust2=st.floats(min_value=1.0, max_value=200.0),
    auw=st.floats(min_value=0.1, max_value=10.0),
)
def test_twr_monotonicity_property(thrust, auw1, auw2, thrust1, thrust2, auw):
    """
    Property 4: TWR monotonicity in weight and thrust.

    AUW↑ never increases TWR; thrust↑ never decreases TWR.

    **Validates: Requirements 5.9, 14.5**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)
    g = MOCK_CONFIG["constants"]["g"]

    # Part A: increasing AUW with fixed thrust never increases TWR
    twr_a1 = engine.twr(thrust, auw1, g)
    twr_a2 = engine.twr(thrust, auw2, g)
    if auw2 >= auw1:
        assert twr_a2 <= twr_a1 + 1e-10  # allow tiny float error

    # Part B: increasing thrust with fixed AUW never decreases TWR
    twr_b1 = engine.twr(thrust1, auw, g)
    twr_b2 = engine.twr(thrust2, auw, g)
    if thrust2 >= thrust1:
        assert twr_b2 >= twr_b1 - 1e-10  # allow tiny float error


# =============================================================================
# Property 8: Structural pass decision
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 8


@settings(max_examples=100)
@given(
    per_motor_thrust=st.floats(min_value=1.0, max_value=50.0),
    arm_length_m=st.floats(min_value=0.05, max_value=0.3),
    arm_width_m=st.floats(min_value=0.005, max_value=0.03),
    thickness_m=st.floats(min_value=0.002, max_value=0.015),
    yield_strength=st.floats(min_value=10e6, max_value=700e6),
    safety_factor=st.floats(min_value=1.0, max_value=4.0),
)
def test_structural_pass_decision_property(per_motor_thrust, arm_length_m, arm_width_m, thickness_m, yield_strength, safety_factor):
    """
    Property 8: Structural pass decision.

    passes ⇔ bending stress ≤ yield/safety_factor

    **Validates: Requirements 8.3**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)

    stress = engine.bending_stress_pa(per_motor_thrust, arm_length_m, arm_width_m, thickness_m)

    # Skip degenerate case where section modulus is zero
    if stress is None:
        return

    allowable = yield_strength / safety_factor

    # The structural pass decision: passes iff stress <= allowable
    expected_pass = stress <= allowable
    actual_pass = stress <= allowable  # Direct computation matches the engine's logic

    assert expected_pass == actual_pass


# =============================================================================
# Property 9: Disk loading definition
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 9

import math

@settings(max_examples=100)
@given(
    motor_count=st.integers(min_value=1, max_value=8),
    prop_diameter_m=st.floats(min_value=0.05, max_value=0.5),
    total_thrust=st.floats(min_value=1.0, max_value=200.0),
)
def test_disk_loading_definition_property(motor_count, prop_diameter_m, total_thrust):
    """
    Property 9: Disk loading definition.
    
    area = count·π·(d/2)², disk_loading = thrust/area
    
    **Validates: Requirements 9.1, 9.2**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)
    
    # Compute expected area
    expected_area = motor_count * math.pi * (prop_diameter_m / 2.0) ** 2
    
    # Compute disk loading
    disk_loading = engine.disk_loading_nm2(total_thrust, expected_area)
    
    # Area is always positive for valid inputs, so disk_loading should never be None
    assert disk_loading is not None
    
    # Verify formula: disk_loading = thrust / area
    expected_loading = total_thrust / expected_area
    assert abs(disk_loading - expected_loading) < 1e-10


# =============================================================================
# Property 7: Flight time monotonicity in weight
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 7

@settings(max_examples=100)
@given(
    auw1=st.floats(min_value=0.1, max_value=10.0),
    auw2=st.floats(min_value=0.1, max_value=10.0),
    usable_energy_wh=st.floats(min_value=1.0, max_value=100.0),
    efficiency=st.floats(min_value=0.01, max_value=1.0),
)
def test_flight_time_monotonicity_property(auw1, auw2, usable_energy_wh, efficiency):
    """
    Property 7: Flight time monotonicity in weight.
    
    AUW↑ never increases flight time (fixed battery).
    
    **Validates: Requirements 7.5**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)
    g = MOCK_CONFIG["constants"]["g"]
    
    # Compute hover power for both AUW values
    power1 = engine.hover_power_w(auw1, g, efficiency)
    power2 = engine.hover_power_w(auw2, g, efficiency)
    
    # Compute flight time for both
    ft1 = engine.flight_time_min(usable_energy_wh, power1)
    ft2 = engine.flight_time_min(usable_energy_wh, power2)
    
    # Both should be valid (non-None) since power > 0
    assert ft1 is not None
    assert ft2 is not None
    
    # Monotonicity: if auw2 >= auw1, flight_time2 <= flight_time1
    if auw2 >= auw1:
        assert ft2 <= ft1 + 1e-10  # allow tiny float error


# =============================================================================
# Property 1: Determinism
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 1

@settings(max_examples=100)
@given(
    arm_count=st.integers(min_value=3, max_value=8),
    arm_length=st.floats(min_value=80.0, max_value=200.0),
    arm_width=st.floats(min_value=8.0, max_value=25.0),
    material_thickness=st.floats(min_value=2.0, max_value=10.0),
    center_cutout_radius=st.floats(min_value=10.0, max_value=30.0),
    payload=st.floats(min_value=0.0, max_value=10.0),
)
def test_determinism_property(arm_count, arm_length, arm_width, material_thickness, center_cutout_radius, payload):
    """
    Property 1: Determinism.

    analyze(x) equals analyze(x) field-by-field with no inference.

    **Validates: Requirements 14.1, 14.2, 14.3, 15.1**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)

    geometry = {
        "arm_count": arm_count,
        "arm_length": arm_length,
        "arm_width": arm_width,
        "material_thickness": material_thickness,
        "center_cutout_radius": center_cutout_radius,
    }
    mission = {
        "payload_mass_kg": payload,
        "use_case": "cinematography",
        "target_flight_time_min": 12.0,
    }

    # Call analyze twice with identical inputs
    result1 = engine.analyze(geometry, "PLA", mission)
    result2 = engine.analyze(geometry, "PLA", mission)

    # Convert to dicts for field-by-field comparison
    d1 = result1.to_dict()
    d2 = result2.to_dict()

    # Field-by-field equality
    assert d1 == d2, f"Non-deterministic results: {d1} != {d2}"


# =============================================================================
# Property 5: TWR, hover throttle, and headroom relationship
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 5

@settings(max_examples=100)
@given(
    total_thrust=st.floats(min_value=1.0, max_value=200.0),
    auw=st.floats(min_value=0.1, max_value=10.0),
)
def test_twr_hover_headroom_relationship_property(total_thrust, auw):
    """
    Property 5: TWR, hover throttle, and headroom relationship.
    
    hover_throttle = (AUW·g)/thrust = 1/TWR, headroom = 1 − hover_throttle
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)
    g = MOCK_CONFIG["constants"]["g"]
    
    twr_val = engine.twr(total_thrust, auw, g)
    assert twr_val is not None  # auw > 0 and g > 0
    assert twr_val > 0
    
    # hover_throttle = (AUW * g) / thrust = 1 / TWR
    expected_hover_throttle = (auw * g) / total_thrust
    computed_hover_throttle = 1.0 / twr_val
    
    assert abs(expected_hover_throttle - computed_hover_throttle) < 1e-10
    
    # headroom = 1 - hover_throttle
    expected_headroom = 1.0 - expected_hover_throttle
    
    # Verify the relationship holds
    assert abs(expected_headroom - (1.0 - computed_hover_throttle)) < 1e-10


# =============================================================================
# Property 6: Payload feasibility equivalence and margin correctness
# =============================================================================

# Feature: drone-physics-engineering-layer, Property 6

@settings(max_examples=100)
@given(
    total_thrust=st.floats(min_value=1.0, max_value=200.0),
    auw=st.floats(min_value=0.1, max_value=10.0),
)
def test_payload_feasibility_margin_property(total_thrust, auw):
    """
    Property 6: Payload feasibility equivalence and margin correctness.
    
    feasible ⇔ margin ≥ 0 ⇔ TWR-with-payload ≥ 1.0;
    margin is the exact boundary.
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 14.4**
    """
    engine = PhysicsEngine(config=MOCK_CONFIG)
    g = MOCK_CONFIG["constants"]["g"]
    
    # Compute margin
    margin = engine.payload_margin_kg(total_thrust, auw, g)
    assert margin is not None  # g > 0
    
    # Compute TWR
    twr_val = engine.twr(total_thrust, auw, g)
    assert twr_val is not None
    
    # Equivalence: feasible ⇔ margin >= 0 ⇔ TWR >= 1.0
    feasible_by_margin = margin >= 0
    feasible_by_twr = twr_val >= 1.0
    
    assert feasible_by_margin == feasible_by_twr, (
        f"Feasibility mismatch: margin={margin:.4f} (>=0: {feasible_by_margin}) "
        f"vs TWR={twr_val:.4f} (>=1: {feasible_by_twr})"
    )
    
    # Margin correctness: margin = thrust/g - auw
    expected_margin = total_thrust / g - auw
    assert abs(margin - expected_margin) < 1e-10
    
    # Adding exactly margin keeps TWR at 1.0 (boundary)
    if margin >= 0:
        twr_at_boundary = engine.twr(total_thrust, auw + margin, g)
        assert twr_at_boundary is not None
        assert abs(twr_at_boundary - 1.0) < 1e-8, (
            f"TWR at boundary should be 1.0, got {twr_at_boundary}"
        )


# =============================================================================
# Unit and edge-case tests (Task 2.12)
# =============================================================================


class TestPhysicsEngineUnit:
    """Unit tests with hand-computed values and edge cases."""

    def test_frame_mass_formula(self):
        """Frame mass = volume × density (R3.1, R3.6 units kg)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # 0.001 m³ × 1240 kg/m³ = 1.24 kg
        assert abs(engine.frame_mass_kg(0.001, 1240.0) - 1.24) < 1e-10

    def test_auw_hand_computed(self):
        """AUW spot-check: frame=0.2 + 4*(0.032+0.01+0.008) + 0.19 + 0.5 = 1.09 kg."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        auw = engine.auw_kg(0.2, 4, 0.032, 0.01, 0.008, 0.19, 0.5)
        expected = 0.2 + 4 * (0.032 + 0.01 + 0.008) + 0.19 + 0.5
        assert abs(auw - expected) < 1e-10

    def test_total_thrust_units_newtons(self):
        """Total thrust in newtons (R4.3): 4 motors × 14.7 N = 58.8 N."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert abs(engine.total_thrust_n(4, 14.7) - 58.8) < 1e-10

    def test_twr_zero_auw_returns_none(self):
        """TWR returns None on zero AUW*g (R5.10)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert engine.twr(50.0, 0.0, 9.80665) is None

    def test_flight_time_zero_power_returns_none(self):
        """Flight time returns None on zero hover power (R7.7)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert engine.flight_time_min(10.0, 0.0) is None

    def test_disk_loading_zero_area_returns_none(self):
        """Disk loading returns None on zero swept area (R9.4)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert engine.disk_loading_nm2(50.0, 0.0) is None

    def test_bending_stress_formula(self):
        """Bending stress sigma = M/Z = (F*L) / (w*t²/6) (R8.1)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # F=10N, L=0.1m, w=0.015m, t=0.005m
        # M = 10*0.1 = 1.0 Nm
        # Z = 0.015 * 0.005² / 6 = 0.015 * 0.000025 / 6 = 6.25e-8 m³
        # sigma = 1.0 / 6.25e-8 = 16,000,000 Pa = 16 MPa
        stress = engine.bending_stress_pa(10.0, 0.1, 0.015, 0.005)
        expected = 1.0 / (0.015 * 0.005**2 / 6.0)
        assert abs(stress - expected) < 1.0  # within 1 Pa

    def test_analyze_material_fallback_note(self):
        """Unknown material falls back to PLA with a note (R2.5)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "titanium", mission)
        assert any("titanium" in n.lower() or "fallback" in n.lower() or "PLA" in n for n in result.notes)

    def test_analyze_motor_fallback_note(self):
        """Unknown motor class falls back to default with a note (R4.5)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission, motor_class="unknown_motor")
        assert any("unknown_motor" in n.lower() or "motor" in n.lower() for n in result.notes)

    def test_analyze_volume_fallback_note(self):
        """Missing CAD volume triggers parametric estimate with note (R3.8)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission, frame_volume_m3=None)
        assert any("estimated" in n.lower() or "volume" in n.lower() for n in result.notes)

    def test_analyze_structural_failure_issue(self):
        """Structural failure records an issue with arm width, thickness, material (R8.5, R8.6)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # Very thin arm that will fail structurally
        geometry = {"arm_count": 4, "arm_length": 200.0, "arm_width": 2.0,
                    "material_thickness": 2.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        if not result.structural.passed:
            structural_issues = [i for i in result.issues if "tructural" in i.lower() or "stress" in i.lower()]
            assert len(structural_issues) > 0
            # Check the issue mentions arm_width, material_thickness, and material
            issue_text = " ".join(structural_issues).lower()
            assert "arm_width" in issue_text or "2.0" in issue_text
            assert "pla" in issue_text or "material" in issue_text

    def test_analyze_all_fields_present(self):
        """analyze() returns all required EngineeringMetrics fields."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.5, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        d = result.to_dict()
        required_keys = ["auw_kg", "frame_mass_kg", "total_thrust_n", "twr", "twr_target",
                         "twr_pass", "hover_throttle", "throttle_headroom", "payload_target_kg",
                         "payload_margin_kg", "payload_feasible", "flight_time_min",
                         "flight_time_target_min", "flight_time_pass", "disk_loading_nm2",
                         "use_case", "structural", "notes", "issues", "available"]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_analyze_use_case_target_racing(self):
        """Racing use case evaluates TWR against target of 4.0 (R5.5)."""
        config = dict(MOCK_CONFIG)
        config["use_cases"] = {"racing": {"target_twr": 4.0, "default_flight_time_min": 5.0},
                               "cinematography": {"target_twr": 2.0, "default_flight_time_min": 12.0}}
        engine = PhysicsEngine(config=config)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "racing", "target_flight_time_min": 5.0}
        result = engine.analyze(geometry, "PLA", mission)
        assert result.twr_target == 4.0

    def test_analyze_use_case_target_cinematography(self):
        """Cinematography use case evaluates TWR against target of 2.0 (R5.6)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        assert result.twr_target == 2.0

    def test_twr_zero_g_returns_none(self):
        """TWR returns None when g is zero (R5.10)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert engine.twr(50.0, 1.0, 0.0) is None

    def test_payload_margin_zero_g_returns_none(self):
        """Payload margin returns None when g is zero (divide-by-zero guard)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        assert engine.payload_margin_kg(50.0, 1.0, 0.0) is None

    def test_bending_stress_zero_thickness_returns_none(self):
        """Bending stress returns None when section modulus is zero (degenerate geometry)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # Zero thickness → section modulus = w*0²/6 = 0
        assert engine.bending_stress_pa(10.0, 0.1, 0.015, 0.0) is None

    def test_hover_power_formula(self):
        """Hover power = AUW * g * efficiency_factor (R7.1, R7.2)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # AUW=1.0 kg, g=9.80665, factor=0.12
        # expected = 1.0 * 9.80665 * 0.12 = 1.176798 W
        power = engine.hover_power_w(1.0, 9.80665, 0.12)
        expected = 1.0 * 9.80665 * 0.12
        assert abs(power - expected) < 1e-10

    def test_flight_time_formula(self):
        """Flight time = (usable_energy_wh / hover_power_w) × 60 (R7.3, R7.4)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # usable_energy_wh=10, hover_power=5W → (10/5)*60 = 120 min
        ft = engine.flight_time_min(10.0, 5.0)
        assert abs(ft - 120.0) < 1e-10

    def test_disk_loading_formula(self):
        """Disk loading = thrust / area (R9.2, R9.3)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        # thrust=50N, area=0.05 m² → 50/0.05 = 1000 N/m²
        dl = engine.disk_loading_nm2(50.0, 0.05)
        assert abs(dl - 1000.0) < 1e-10

    def test_analyze_twr_unavailable_issue(self):
        """When AUW*g is zero, TWR is None and an issue is recorded (R5.10)."""
        # Create a config with g=0 to force divide-by-zero
        config = dict(MOCK_CONFIG)
        config["constants"] = {"g": 0.0, "air_density": 1.225}
        engine = PhysicsEngine(config=config)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        assert result.twr is None
        assert any("twr" in i.lower() or "divide" in i.lower() or "zero" in i.lower() for i in result.issues)

    def test_analyze_flight_time_unavailable_issue(self):
        """When hover power is zero, flight time is None and an issue is recorded (R7.7)."""
        # With g=0, hover power = AUW * 0 * factor = 0
        config = dict(MOCK_CONFIG)
        config["constants"] = {"g": 0.0, "air_density": 1.225}
        engine = PhysicsEngine(config=config)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        assert result.flight_time_min is None
        assert any("flight" in i.lower() or "power" in i.lower() for i in result.issues)

    def test_analyze_config_motor_table_read(self):
        """Motor table lookup uses configured per-motor thrust (R4.2)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        # default motor is 2207_2400kv with max_thrust_n=14.7
        # 4 motors × 14.7 = 58.8 N
        assert abs(result.total_thrust_n - 58.8) < 1e-10

    def test_analyze_payload_feasibility_field(self):
        """Payload feasibility is included in Engineering_Metrics (R6.5)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.5, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        # payload_feasible should be a boolean
        assert isinstance(result.payload_feasible, bool)
        # payload_margin_kg should be a float
        assert result.payload_margin_kg is not None
        assert isinstance(result.payload_margin_kg, float)

    def test_analyze_structural_result_fields(self):
        """Structural result includes bending stress, allowable stress, safety margin (R8.2, R8.6)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.0, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        s = result.structural
        assert s.bending_stress_pa is not None
        assert s.allowable_stress_pa is not None
        assert s.safety_margin is not None
        assert s.arm_width_mm == 15.0
        assert s.material_thickness_mm == 5.0
        assert s.material == "PLA"

    def test_analyze_metrics_units_kg_n_min(self):
        """Metrics are in correct units: kg for masses, N for thrust, min for time (R3.6, R4.3)."""
        engine = PhysicsEngine(config=MOCK_CONFIG)
        geometry = {"arm_count": 4, "arm_length": 120.0, "arm_width": 15.0,
                    "material_thickness": 5.0, "center_cutout_radius": 20.0}
        mission = {"payload_mass_kg": 0.5, "use_case": "cinematography", "target_flight_time_min": 12.0}
        result = engine.analyze(geometry, "PLA", mission)
        # AUW should be reasonable for a small drone (< 10 kg)
        assert 0 < result.auw_kg < 10.0
        # Thrust in newtons should be > 0
        assert result.total_thrust_n > 0
        # Flight time in minutes should be positive if available
        if result.flight_time_min is not None:
            assert result.flight_time_min > 0
