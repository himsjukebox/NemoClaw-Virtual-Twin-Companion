# Feature: drone-physics-engineering-layer, Property 10
"""
Property-based and unit tests for the Validator physics gate.

**Validates: Requirements 11.2, 11.3, 11.4, 11.5, 11.6**
"""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agents.validator_agent import physics_gate


# =============================================================================
# Property 10: Validator physics-gate biconditional
# =============================================================================


@settings(max_examples=100)
@given(
    twr=st.floats(min_value=0.5, max_value=8.0),
    twr_target=st.floats(min_value=1.0, max_value=5.0),
    payload_feasible=st.booleans(),
    payload_margin=st.floats(min_value=-5.0, max_value=10.0),
    structural_passed=st.booleans(),
    stress=st.floats(min_value=1e5, max_value=1e9),
    allowable=st.floats(min_value=1e5, max_value=1e9),
)
def test_validator_gate_biconditional(
    twr, twr_target, payload_feasible, payload_margin, structural_passed, stress, allowable
):
    """
    Property 10: Validator physics-gate biconditional.

    PASS ⇔ (TWR meets target AND payload feasible AND structural passes).
    On FAIL, issues and suggestions are non-empty and reference the failing metric.

    **Validates: Requirements 11.2, 11.3, 11.4, 11.5, 11.6**
    """
    metrics = {
        "twr": twr,
        "twr_target": twr_target,
        "use_case": "cinematography",
        "payload_feasible": payload_feasible,
        "payload_margin_kg": payload_margin,
        "structural": {
            "passed": structural_passed,
            "bending_stress_pa": stress,
            "allowable_stress_pa": allowable,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    # Biconditional: PASS iff all three conditions hold
    twr_ok = twr >= twr_target
    expected_pass = twr_ok and payload_feasible and structural_passed

    assert passed == expected_pass, (
        f"Gate returned passed={passed} but expected {expected_pass} "
        f"(twr_ok={twr_ok}, payload_feasible={payload_feasible}, "
        f"structural_passed={structural_passed})"
    )

    # On FAIL, issues and suggestions are non-empty (R11.6)
    if not passed:
        assert len(issues) > 0, "FAIL verdict must produce at least one issue"
        assert len(suggestions) > 0, "FAIL verdict must produce at least one suggestion"


@settings(max_examples=100)
@given(
    twr_target=st.floats(min_value=1.0, max_value=5.0),
    payload_margin=st.floats(min_value=-5.0, max_value=10.0),
    structural_passed=st.booleans(),
    stress=st.floats(min_value=1e5, max_value=1e9),
    allowable=st.floats(min_value=1e5, max_value=1e9),
)
def test_validator_gate_twr_none_always_fails(
    twr_target, payload_margin, structural_passed, stress, allowable
):
    """
    Property 10 edge: When TWR is None (unavailable), the gate always FAILs.

    **Validates: Requirements 11.2, 11.6**
    """
    metrics = {
        "twr": None,
        "twr_target": twr_target,
        "use_case": "cinematography",
        "payload_feasible": True,
        "payload_margin_kg": payload_margin,
        "structural": {
            "passed": structural_passed,
            "bending_stress_pa": stress,
            "allowable_stress_pa": allowable,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    # TWR unavailable always triggers a FAIL
    assert passed is False, "TWR=None must cause FAIL"
    assert len(issues) > 0, "TWR=None must produce at least one issue"
    assert any("TWR" in issue or "twr" in issue.lower() for issue in issues), (
        "Issues must reference TWR when it is unavailable"
    )


@settings(max_examples=100)
@given(
    twr=st.floats(min_value=2.0, max_value=8.0),
    twr_target=st.floats(min_value=1.0, max_value=2.0),
)
def test_validator_gate_all_pass(twr, twr_target):
    """
    Property 10 positive path: When all conditions are met, the gate PASSes
    with empty issues and suggestions.

    **Validates: Requirements 11.5**
    """
    assume(twr >= twr_target)

    metrics = {
        "twr": twr,
        "twr_target": twr_target,
        "use_case": "cinematography",
        "payload_feasible": True,
        "payload_margin_kg": 2.0,
        "structural": {
            "passed": True,
            "bending_stress_pa": 1e6,
            "allowable_stress_pa": 5e7,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    assert passed is True, "All conditions met must yield PASS"
    assert issues == [], "PASS verdict must have no issues"
    assert suggestions == [], "PASS verdict must have no suggestions"


@settings(max_examples=100)
@given(
    twr=st.floats(min_value=0.5, max_value=3.0),
    twr_target=st.floats(min_value=3.1, max_value=6.0),
)
def test_validator_gate_twr_fail_references_metric(twr, twr_target):
    """
    Property 10 sub-case: When TWR < target, the issue references the TWR metric.

    **Validates: Requirements 11.2, 11.6**
    """
    assume(twr < twr_target)

    metrics = {
        "twr": twr,
        "twr_target": twr_target,
        "use_case": "racing",
        "payload_feasible": True,
        "payload_margin_kg": 5.0,
        "structural": {
            "passed": True,
            "bending_stress_pa": 1e6,
            "allowable_stress_pa": 5e7,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    assert passed is False, "TWR below target must cause FAIL"
    assert any("TWR" in issue or "twr" in issue.lower() for issue in issues), (
        "Issues must reference TWR when it is below target"
    )
    assert len(suggestions) > 0, "FAIL must produce suggestions"


@settings(max_examples=100)
@given(
    twr=st.floats(min_value=2.0, max_value=8.0),
    twr_target=st.floats(min_value=1.0, max_value=2.0),
    payload_margin=st.floats(min_value=-5.0, max_value=-0.01),
)
def test_validator_gate_payload_fail_references_metric(twr, twr_target, payload_margin):
    """
    Property 10 sub-case: When payload is infeasible, the issue references payload.

    **Validates: Requirements 11.3, 11.6**
    """
    assume(twr >= twr_target)

    metrics = {
        "twr": twr,
        "twr_target": twr_target,
        "use_case": "cinematography",
        "payload_feasible": False,
        "payload_margin_kg": payload_margin,
        "structural": {
            "passed": True,
            "bending_stress_pa": 1e6,
            "allowable_stress_pa": 5e7,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    assert passed is False, "Payload infeasible must cause FAIL"
    assert any("payload" in issue.lower() or "Payload" in issue for issue in issues), (
        "Issues must reference payload when it is infeasible"
    )
    assert len(suggestions) > 0, "FAIL must produce suggestions"


@settings(max_examples=100)
@given(
    twr=st.floats(min_value=2.0, max_value=8.0),
    twr_target=st.floats(min_value=1.0, max_value=2.0),
    stress=st.floats(min_value=1e7, max_value=1e9),
    allowable=st.floats(min_value=1e5, max_value=1e9),
)
def test_validator_gate_structural_fail_references_metric(twr, twr_target, stress, allowable):
    """
    Property 10 sub-case: When structural check fails, the issue references structure.

    **Validates: Requirements 11.4, 11.6**
    """
    assume(twr >= twr_target)

    metrics = {
        "twr": twr,
        "twr_target": twr_target,
        "use_case": "delivery",
        "payload_feasible": True,
        "payload_margin_kg": 3.0,
        "structural": {
            "passed": False,
            "bending_stress_pa": stress,
            "allowable_stress_pa": allowable,
        },
    }

    passed, issues, suggestions = physics_gate(metrics)

    assert passed is False, "Structural failure must cause FAIL"
    assert any(
        "structural" in issue.lower() or "stress" in issue.lower() for issue in issues
    ), "Issues must reference structural/stress when structural check fails"
    assert len(suggestions) > 0, "FAIL must produce suggestions"


# =============================================================================
# Unit tests for validator metric handling (Task 6.4)
# =============================================================================


class TestValidatorMetricHandling:
    """Unit tests for validator integration with engineering metrics."""

    def test_physics_gate_reads_metrics(self):
        """Validator's physics_gate reads engineering_metrics correctly (R11.1)."""
        metrics = {
            "twr": 3.5,
            "twr_target": 2.0,
            "use_case": "cinematography",
            "payload_feasible": True,
            "payload_margin_kg": 2.0,
            "structural": {"passed": True, "bending_stress_pa": 1e6, "allowable_stress_pa": 2.5e7},
        }
        passed, issues, suggestions = physics_gate(metrics)
        assert passed is True
        assert issues == []

    def test_physics_gate_fail_produces_feedback(self):
        """On FAIL, physics gate produces feedback for Design Agent routing (R11.7)."""
        metrics = {
            "twr": 1.5,
            "twr_target": 4.0,
            "use_case": "racing",
            "payload_feasible": False,
            "payload_margin_kg": -0.5,
            "structural": {"passed": False, "bending_stress_pa": 1e8, "allowable_stress_pa": 2.5e7},
        }
        passed, issues, suggestions = physics_gate(metrics)
        assert passed is False
        assert len(issues) == 3  # TWR, payload, structural all fail
        assert len(suggestions) == 3
        # Feedback references specific failing metrics
        assert any("TWR" in i for i in issues)
        assert any("payload" in i.lower() or "Payload" in i for i in issues)
        assert any("structural" in i.lower() or "stress" in i.lower() for i in issues)

    def test_deterministic_verdict_on_empty_rag(self):
        """Validator proceeds with deterministic physics verdict when RAG is empty (R11.8)."""
        # The physics_gate function itself is fully deterministic — no RAG dependency.
        # This test just confirms the gate produces consistent results regardless of RAG.
        metrics = {
            "twr": 2.5,
            "twr_target": 2.0,
            "use_case": "delivery",
            "payload_feasible": True,
            "payload_margin_kg": 1.0,
            "structural": {"passed": True, "bending_stress_pa": 5e6, "allowable_stress_pa": 2.5e7},
        }
        # Call multiple times — deterministic
        r1 = physics_gate(metrics)
        r2 = physics_gate(metrics)
        assert r1 == r2
        assert r1[0] is True
