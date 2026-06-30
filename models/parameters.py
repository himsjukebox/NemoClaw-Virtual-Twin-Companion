# =============================================================================
# NemoClaw Virtual Twin Companion — Design Parameters Schema
# =============================================================================
# PURPOSE:
#   Defines the canonical parameter ranges, defaults, and schema constants for
#   the four parametric CAD variables that control drone chassis geometry.
#   This module is the SINGLE SOURCE OF TRUTH for parameter bounds — all other
#   components (Design Agent clamping, CAD Tool validation, Validator rules)
#   reference these constants rather than hardcoding their own ranges.
#
# WHY CENTRALIZED:
#   Avoids drift between the Design Agent's clamping ranges, the CAD Tool's
#   validation ranges, and the Validator's rule checks. A single change here
#   propagates everywhere.
#
# PARAMETERS (from chassis_frame_template.py):
#   - arm_length: Distance from center hub to motor mount (mm)
#   - material_thickness: Plate thickness for FDM/SLS printing (mm)
#   - arm_width: Cross-section width of each arm (mm)
#   - center_cutout_radius: Radius of the center weight-reduction cutout (mm)
# =============================================================================

from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Parameter Range Definitions
# ---------------------------------------------------------------------------
# Each tuple is (min_value, max_value) in millimeters.
# These ranges are derived from drone engineering constraints:
#   - arm_length: <80mm is too short for quad stability; >200mm exceeds
#     typical hobby/racing frame envelope
#   - material_thickness: <2mm fails FDM print reliability; >10mm is
#     excessively heavy for flight
#   - arm_width: <8mm is structurally insufficient; >25mm adds unnecessary
#     drag and weight
#   - center_cutout_radius: <10mm provides negligible weight savings;
#     >30mm compromises hub structural integrity
# ---------------------------------------------------------------------------

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "arm_length": (80.0, 200.0),
    "material_thickness": (2.0, 10.0),
    "arm_width": (8.0, 25.0),
    "center_cutout_radius": (10.0, 30.0),
}
"""
Mapping of parameter names to their (min, max) bounds in mm.
Used by Design Agent for clamping and CAD Tool for validation.
"""

# ---------------------------------------------------------------------------
# Default Parameter Values
# ---------------------------------------------------------------------------
# Safe defaults that produce a well-balanced, manufacturable chassis.
# Used when:
#   - User provides an empty request
#   - LLM output cannot be parsed
#   - Any parameter is missing from LLM response
# ---------------------------------------------------------------------------

PARAM_DEFAULTS: Dict[str, float] = {
    "arm_length": 120.0,
    "material_thickness": 5.0,
    "arm_width": 15.0,
    "center_cutout_radius": 20.0,
}
"""
Default parameter values (mm) representing a balanced, manufacturable design.
These satisfy all structural constraints and are within all defined ranges.
"""

# ---------------------------------------------------------------------------
# Parameter Units
# ---------------------------------------------------------------------------

PARAM_UNITS: Dict[str, str] = {
    "arm_length": "mm",
    "material_thickness": "mm",
    "arm_width": "mm",
    "center_cutout_radius": "mm",
}
"""Units for each parameter (all millimeters for this chassis design)."""

# ---------------------------------------------------------------------------
# Parameter Descriptions (for UI display and LLM prompting)
# ---------------------------------------------------------------------------

PARAM_DESCRIPTIONS: Dict[str, str] = {
    "arm_length": "Distance from center hub to motor mount",
    "material_thickness": "Plate thickness for FDM/SLS manufacturing",
    "arm_width": "Cross-section width of each arm",
    "center_cutout_radius": "Radius of the center weight-reduction cutout",
}
"""Human-readable descriptions for each parameter."""

# ---------------------------------------------------------------------------
# Structural Constraints
# ---------------------------------------------------------------------------

STRUCTURAL_CONSTRAINT_RATIO: float = 0.08
"""
Minimum ratio of arm_width to arm_length.
Constraint: arm_width >= arm_length * STRUCTURAL_CONSTRAINT_RATIO
This ensures arms are wide enough relative to their length to avoid
buckling under motor thrust loads.
"""

MANUFACTURABILITY_MIN_THICKNESS: float = 2.0
"""
Minimum material thickness (mm) for FDM 3D printing processes.
Below this value, layer adhesion is unreliable and parts are brittle.
"""

# ---------------------------------------------------------------------------
# All Parameter Names (ordered)
# ---------------------------------------------------------------------------

PARAM_NAMES: Tuple[str, ...] = (
    "arm_length",
    "material_thickness",
    "arm_width",
    "center_cutout_radius",
)
"""Canonical ordered tuple of all parameter names."""
