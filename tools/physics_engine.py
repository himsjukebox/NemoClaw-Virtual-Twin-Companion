# =============================================================================
# NemoClaw Virtual Twin Companion — Deterministic Physics Engine
# =============================================================================
# PURPOSE:
#   Compute all Engineering_Metrics deterministically from geometry, material,
#   components, and mission inputs. Performs NO LLM/network inference.
#   Each metric is a pure function; analyze() composes them (task 2.2).
#
# DESIGN RATIONALE:
#   All engineering math is deterministic Python with no network calls (R14.1,
#   R15.1). Each metric function's output depends only on its explicit inputs
#   (R14.3). Ratio functions return None on zero denominator (R5.10, R7.7, R9.4).
#
# Component: Physics_Engine
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import List, Optional

from config.loader import load_physics_config


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class StructuralResult:
    """
    Result of the structural bending-stress check on a drone arm.

    Fields:
        bending_stress_pa: Computed bending stress at the arm root (Pa).
        allowable_stress_pa: Material yield / safety factor (Pa).
        safety_margin: allowable / bending_stress (dimensionless); >1 means passing.
        arm_width_mm: Arm cross-section width used in the check (mm).
        material_thickness_mm: Arm cross-section height used in the check (mm).
        material: Name of the material used.
        passed: True if bending_stress <= allowable_stress.
        heuristic_passed: True if arm_width >= arm_length * 0.08 (legacy check).

    Component: Physics_Engine
    """

    bending_stress_pa: Optional[float] = None
    allowable_stress_pa: Optional[float] = None
    safety_margin: Optional[float] = None
    arm_width_mm: Optional[float] = None
    material_thickness_mm: Optional[float] = None
    material: Optional[str] = None
    passed: bool = False
    heuristic_passed: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary for Whiteboard storage."""
        return {
            "bending_stress_pa": self.bending_stress_pa,
            "allowable_stress_pa": self.allowable_stress_pa,
            "safety_margin": self.safety_margin,
            "arm_width_mm": self.arm_width_mm,
            "material_thickness_mm": self.material_thickness_mm,
            "material": self.material,
            "passed": self.passed,
            "heuristic_passed": self.heuristic_passed,
        }


@dataclass
class EngineeringMetrics:
    """
    Structured result object produced by PhysicsEngine.analyze().

    Contains all computed engineering metrics, target values, pass/fail flags,
    structural sub-result, notes for informational fallbacks, issues for
    infeasible/unavailable metrics, and an availability flag.

    Unavailable ratio metrics (TWR, flight time, disk loading) are represented
    as None with an explanatory entry in `issues` (R5.10, R7.7, R9.4).

    Component: Physics_Engine
    """

    auw_kg: float = 0.0
    frame_mass_kg: float = 0.0
    total_thrust_n: float = 0.0
    twr: Optional[float] = None
    twr_target: Optional[float] = None
    twr_pass: bool = False
    hover_throttle: Optional[float] = None
    throttle_headroom: Optional[float] = None
    payload_target_kg: float = 0.0
    payload_margin_kg: Optional[float] = None
    payload_feasible: bool = False
    flight_time_min: Optional[float] = None
    flight_time_target_min: Optional[float] = None
    flight_time_pass: bool = False
    disk_loading_nm2: Optional[float] = None
    use_case: str = "cinematography"
    structural: StructuralResult = field(default_factory=StructuralResult)
    notes: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    available: bool = True

    def to_dict(self) -> dict:
        """
        Serialize the EngineeringMetrics to a plain dict matching the
        Engineering_Metrics JSON schema. The `structural` field becomes
        a nested dict. Notes and issues are copied (not aliased).
        """
        return {
            "auw_kg": self.auw_kg,
            "frame_mass_kg": self.frame_mass_kg,
            "total_thrust_n": self.total_thrust_n,
            "twr": self.twr,
            "twr_target": self.twr_target,
            "twr_pass": self.twr_pass,
            "hover_throttle": self.hover_throttle,
            "throttle_headroom": self.throttle_headroom,
            "payload_target_kg": self.payload_target_kg,
            "payload_margin_kg": self.payload_margin_kg,
            "payload_feasible": self.payload_feasible,
            "flight_time_min": self.flight_time_min,
            "flight_time_target_min": self.flight_time_target_min,
            "flight_time_pass": self.flight_time_pass,
            "disk_loading_nm2": self.disk_loading_nm2,
            "use_case": self.use_case,
            "structural": self.structural.to_dict(),
            "notes": list(self.notes),
            "issues": list(self.issues),
            "available": self.available,
        }


# =============================================================================
# Physics Engine
# =============================================================================


class PhysicsEngine:
    """
    Deterministic physics engine for drone engineering analysis.

    Computes all Engineering_Metrics from geometry, material, components,
    and mission inputs. Performs NO LLM/network inference (R14.1, R15.1).

    Each metric is a pure function whose output depends only on explicit
    arguments (R14.3). The analyze() method (task 2.2) orchestrates them.

    Component: Physics_Engine
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize with a validated Physics_Config dict.

        If config is None, loads via load_physics_config(). Raises
        ConfigValidationError if the config file is missing or invalid (R12.9).

        Args:
            config: Optional pre-loaded and validated physics config dict.
        """
        self.config = config if config is not None else load_physics_config()

    # =========================================================================
    # Pure metric functions (output depends only on explicit args)
    # =========================================================================

    def frame_mass_kg(self, volume_m3: float, material_density: float) -> float:
        """
        Compute frame mass from chassis solid volume and material density.

        Frame_Mass = volume × density (R3.1).

        Args:
            volume_m3: Chassis solid volume in cubic meters.
            material_density: Material density in kg/m³.

        Returns:
            Frame mass in kilograms.
        """
        return volume_m3 * material_density

    def estimate_frame_volume_m3(self, geometry: dict) -> float:
        """
        Parametric volume fallback when CAD volume is unavailable (R3.8).

        Estimates frame volume from: central hub cylinder + arm volumes
        minus center cutout cylinder. All geometry inputs in mm, output in m³.

        Args:
            geometry: Dict with keys arm_count, arm_length, arm_width,
                      material_thickness, center_cutout_radius (all in mm).

        Returns:
            Estimated frame volume in cubic meters (always >= 0).
        """
        arm_count = int(geometry.get("arm_count", 4))
        arm_length = float(geometry.get("arm_length", 120.0))
        arm_width = float(geometry.get("arm_width", 15.0))
        thickness = float(geometry.get("material_thickness", 5.0))
        cutout_radius = float(geometry.get("center_cutout_radius", 20.0))

        # Fixed parametric hub radius (mm)
        hub_radius = 35.0

        # Hub cylinder volume (mm³)
        hub_vol_mm3 = math.pi * hub_radius**2 * thickness

        # Arms: rectangular cross-section beams (mm³)
        arms_vol_mm3 = arm_count * arm_length * arm_width * thickness

        # Center cutout subtracted (mm³)
        cutout_vol_mm3 = math.pi * cutout_radius**2 * thickness

        total_mm3 = hub_vol_mm3 + arms_vol_mm3 - cutout_vol_mm3
        return max(0.0, total_mm3 * 1e-9)  # convert mm³ to m³

    def auw_kg(
        self,
        frame_mass: float,
        motor_count: int,
        motor_mass: float,
        esc_mass: float,
        prop_mass: float,
        battery_mass: float,
        payload_mass: float,
    ) -> float:
        """
        Compute All-Up Weight as strict sum of component masses.

        AUW = frame + motor_count × (motor + ESC + prop) + battery + payload
        (R3.2–R3.6). Mass additivity: AUW increases by exactly the delta of
        any component mass increase (R3.7, R14.6).

        Args:
            frame_mass: Frame mass in kg.
            motor_count: Number of motors (= arm_count).
            motor_mass: Per-motor mass in kg.
            esc_mass: Per-ESC mass in kg.
            prop_mass: Per-propeller mass in kg.
            battery_mass: Battery mass in kg.
            payload_mass: Payload mass in kg.

        Returns:
            All-up weight in kilograms.
        """
        return (
            frame_mass
            + motor_count * (motor_mass + esc_mass + prop_mass)
            + battery_mass
            + payload_mass
        )

    def total_thrust_n(self, motor_count: int, per_motor_thrust_n: float) -> float:
        """
        Compute total available thrust from all motors.

        Total_Thrust = motor_count × per_motor_thrust (R4.1, R4.4).
        Monotone in motor count: more motors → equal or greater thrust.

        Args:
            motor_count: Number of motors.
            per_motor_thrust_n: Maximum thrust per motor in newtons.

        Returns:
            Total thrust in newtons.
        """
        return motor_count * per_motor_thrust_n

    def twr(self, total_thrust: float, auw: float, g: float) -> Optional[float]:
        """
        Compute Thrust-to-Weight Ratio.

        TWR = Total_Thrust / (AUW × g) (R5.1, R5.10).
        Returns None if AUW × g == 0 to avoid division by zero.

        Args:
            total_thrust: Total thrust in newtons.
            auw: All-up weight in kilograms.
            g: Gravitational acceleration in m/s².

        Returns:
            TWR (dimensionless) or None if denominator is zero.
        """
        denom = auw * g
        return None if denom == 0 else total_thrust / denom

    def payload_margin_kg(
        self, total_thrust: float, auw: float, g: float
    ) -> Optional[float]:
        """
        Compute payload margin: max additional payload keeping TWR >= 1.0.

        Payload_Margin = (Total_Thrust / g) − AUW (R6.2).
        Non-negative iff configuration is payload-feasible (R6.1, R6.3, R6.4, R14.4).
        Returns None if g == 0 to avoid division by zero.

        Args:
            total_thrust: Total thrust in newtons.
            auw: All-up weight in kilograms.
            g: Gravitational acceleration in m/s².

        Returns:
            Payload margin in kg, or None if g is zero.
        """
        return None if g == 0 else (total_thrust / g) - auw

    def hover_power_w(
        self, auw: float, g: float, power_per_newton: float
    ) -> float:
        """
        Compute hover power draw from weight and efficiency factor.

        Hover_Power = AUW × g × power_per_newton (R7.2).
        Monotone in AUW: heavier drone → more power required.

        Args:
            auw: All-up weight in kg.
            g: Gravitational acceleration in m/s².
            power_per_newton: Propulsion efficiency factor (W per N of thrust).

        Returns:
            Hover power in watts.
        """
        return auw * g * power_per_newton

    def flight_time_min(
        self, usable_energy_wh: float, hover_power_w: float
    ) -> Optional[float]:
        """
        Compute estimated hover flight time.

        Flight_Time = (usable_energy_wh / hover_power_w) × 60 (R7.3, R7.7).
        Returns None if hover_power_w == 0 to avoid division by zero.

        Args:
            usable_energy_wh: Usable battery energy in watt-hours.
            hover_power_w: Hover power draw in watts.

        Returns:
            Flight time in minutes, or None if power is zero.
        """
        return None if hover_power_w == 0 else (usable_energy_wh / hover_power_w) * 60.0

    def bending_stress_pa(
        self,
        per_motor_thrust_n: float,
        arm_length_m: float,
        arm_width_m: float,
        thickness_m: float,
    ) -> Optional[float]:
        """
        Compute cantilever bending stress on a drone arm.

        σ = M / Z, where M = F × L (bending moment) and
        Z = width × thickness² / 6 (section modulus for rectangular beam) (R8.1).

        Returns None if the section modulus is zero (degenerate geometry).

        Args:
            per_motor_thrust_n: Thrust force at the arm tip in newtons.
            arm_length_m: Arm length (moment arm) in meters.
            arm_width_m: Arm cross-section width in meters.
            thickness_m: Arm cross-section height (thickness) in meters.

        Returns:
            Bending stress in pascals, or None if section modulus is zero.
        """
        section_modulus = (arm_width_m * thickness_m**2) / 6.0
        if section_modulus == 0:
            return None
        moment = per_motor_thrust_n * arm_length_m
        return moment / section_modulus

    def disk_loading_nm2(
        self, total_thrust: float, total_swept_area_m2: float
    ) -> Optional[float]:
        """
        Compute disk loading: thrust per unit rotor swept area.

        Disk_Loading = Total_Thrust / total_swept_area (R9.2, R9.4).
        Returns None if total_swept_area_m2 == 0 to avoid division by zero.

        Args:
            total_thrust: Total thrust in newtons.
            total_swept_area_m2: Total rotor swept area in m²
                (motor_count × π × (diameter/2)²).

        Returns:
            Disk loading in N/m², or None if area is zero.
        """
        return None if total_swept_area_m2 == 0 else total_thrust / total_swept_area_m2

    # =========================================================================
    # Orchestration — analyze()
    # =========================================================================

    def analyze(
        self,
        geometry: dict,
        material: str,
        mission: dict,
        frame_volume_m3: float | None = None,
        motor_class: str | None = None,
        battery_option: str | None = None,
    ) -> EngineeringMetrics:
        """
        Compose all pure functions into an EngineeringMetrics result.

        Deterministic: identical inputs → identical output (R14.2).
        Records notes for fallbacks and issues for infeasible/unavailable metrics.
        Reads constants/tables/targets exclusively from self.config (R12.8).

        Args:
            geometry: Dict with arm_count, arm_length, arm_width,
                      material_thickness, center_cutout_radius (mm).
            material: Material name string (e.g. "PLA", "carbon_fiber").
            mission: Dict with payload_mass_kg, target_flight_time_min, use_case.
            frame_volume_m3: Optional CAD-computed chassis solid volume in m³.
            motor_class: Optional motor class key for the motors table.
            battery_option: Optional battery option key for the batteries table.

        Returns:
            Populated EngineeringMetrics dataclass instance.
        """
        notes: List[str] = []
        issues: List[str] = []

        # --- Config references ---
        config = self.config
        g = config["constants"]["g"]
        materials_table = config["materials"]
        motors_table = config["motors"]
        batteries_table = config["batteries"]
        use_cases_table = config["use_cases"]
        factors = config["factors"]
        components = config["components"]

        # =====================================================================
        # 1. Material lookup with PLA fallback (R2.4, R2.5)
        # =====================================================================
        if material in materials_table:
            mat_entry = materials_table[material]
        else:
            mat_entry = materials_table["PLA"]
            notes.append(
                f"Material '{material}' not found in config; falling back to PLA."
            )
            material = "PLA"

        material_density = float(mat_entry["density"])
        material_yield = float(mat_entry["yield_strength"])

        # =====================================================================
        # 2. Motor lookup with default fallback (R4.2, R4.5)
        # =====================================================================
        default_motor = components["default_motor_class"]
        motor_key = motor_class if motor_class is not None else default_motor
        if motor_key in motors_table:
            motor_entry = motors_table[motor_key]
        else:
            motor_entry = motors_table[default_motor]
            notes.append(
                f"Motor class '{motor_key}' not found in config; "
                f"falling back to default '{default_motor}'."
            )

        motor_mass = motor_entry["mass_kg"]
        per_motor_thrust = motor_entry["max_thrust_n"]

        # =====================================================================
        # 3. Battery lookup with default fallback
        # =====================================================================
        default_battery = components["default_battery_option"]
        battery_key = battery_option if battery_option is not None else default_battery
        if battery_key in batteries_table:
            battery_entry = batteries_table[battery_key]
        else:
            battery_entry = batteries_table[default_battery]
            notes.append(
                f"Battery option '{battery_key}' not found in config; "
                f"falling back to default '{default_battery}'."
            )

        battery_capacity_mah = battery_entry["capacity_mah"]
        battery_cells_s = battery_entry["cells_s"]
        battery_mass = battery_entry["mass_kg"]

        # =====================================================================
        # 4. Frame volume: use provided or estimate with note (R3.8)
        # =====================================================================
        if frame_volume_m3 is not None:
            volume = frame_volume_m3
        else:
            volume = self.estimate_frame_volume_m3(geometry)
            notes.append(
                "Frame volume estimated from parametric geometry (CAD volume unavailable)."
            )

        # =====================================================================
        # 5. Frame mass (R3.1)
        # =====================================================================
        frame_mass = self.frame_mass_kg(volume, material_density)

        # =====================================================================
        # 6. AUW (R3.2–R3.7)
        # =====================================================================
        motor_count = int(geometry.get("arm_count", 4))
        esc_mass = components["esc_mass_kg"]
        prop_mass = components["propeller_mass_kg"]
        payload_mass = float(mission.get("payload_mass_kg", 0.0))

        auw = self.auw_kg(
            frame_mass, motor_count, motor_mass, esc_mass, prop_mass,
            battery_mass, payload_mass
        )

        # =====================================================================
        # 7. Total thrust (R4.1, R4.4)
        # =====================================================================
        total_thrust = self.total_thrust_n(motor_count, per_motor_thrust)

        # =====================================================================
        # 8. TWR with divide-by-zero guard (R5.1, R5.10)
        # =====================================================================
        twr_val = self.twr(total_thrust, auw, g)
        if twr_val is None:
            issues.append("TWR unavailable: AUW × g is zero (divide-by-zero guard).")

        # =====================================================================
        # 9. Hover throttle and headroom (R5.2, R5.3)
        # =====================================================================
        hover_throttle: Optional[float] = None
        throttle_headroom: Optional[float] = None
        if twr_val is not None and twr_val != 0:
            hover_throttle = 1.0 / twr_val
            throttle_headroom = 1.0 - hover_throttle

        # =====================================================================
        # 10. Use-case target TWR and pass flag (R5.4–R5.8)
        # =====================================================================
        use_case = mission.get("use_case", "cinematography")
        if use_case not in use_cases_table:
            use_case = "cinematography"

        uc_entry = use_cases_table[use_case]
        twr_target = uc_entry["target_twr"]
        twr_pass = (twr_val is not None) and (twr_val >= twr_target)

        # =====================================================================
        # 11. Payload margin and feasibility (R6.1–R6.5)
        # =====================================================================
        margin = self.payload_margin_kg(total_thrust, auw, g)
        payload_feasible = (margin is not None) and (margin >= 0)
        if not payload_feasible:
            if margin is not None:
                issues.append(
                    f"Payload infeasible: margin is {margin:.3f} kg "
                    f"(deficit of {abs(margin):.3f} kg)."
                )
            else:
                issues.append(
                    "Payload margin unavailable: g is zero (divide-by-zero guard)."
                )

        # =====================================================================
        # 12. Flight time estimate (R7.1–R7.7)
        # =====================================================================
        nominal_cell_voltage = factors["nominal_cell_voltage"]
        usable_fraction = factors["usable_capacity_fraction"]
        efficiency_factor = factors["propulsion_efficiency_factor"]

        usable_energy_wh = (
            battery_capacity_mah * battery_cells_s * nominal_cell_voltage
            / 1000.0 * usable_fraction
        )

        hover_power = self.hover_power_w(auw, g, efficiency_factor)
        flight_time = self.flight_time_min(usable_energy_wh, hover_power)

        if flight_time is None:
            issues.append(
                "Flight time unavailable: hover power is zero (divide-by-zero guard)."
            )

        # Flight time target and pass flag (R7.6)
        flight_time_target = float(mission.get(
            "target_flight_time_min",
            uc_entry["default_flight_time_min"]
        ))
        flight_time_pass = (flight_time is not None) and (flight_time >= flight_time_target)

        if (flight_time is not None) and (flight_time < flight_time_target):
            shortfall = flight_time_target - flight_time
            issues.append(
                f"Flight time shortfall: estimated {flight_time:.1f} min "
                f"vs target {flight_time_target:.1f} min "
                f"(shortfall of {shortfall:.1f} min)."
            )

        # =====================================================================
        # 13. Structural load check (R8.1–R8.6)
        # =====================================================================
        arm_length_mm = float(geometry.get("arm_length", 120.0))
        arm_width_mm = float(geometry.get("arm_width", 15.0))
        thickness_mm = float(geometry.get("material_thickness", 5.0))

        # Convert to meters for stress calculation
        arm_length_m = arm_length_mm / 1000.0
        arm_width_m = arm_width_mm / 1000.0
        thickness_m = thickness_mm / 1000.0

        # Per-motor thrust for structural loading
        per_motor_thrust_structural = total_thrust / motor_count if motor_count > 0 else 0.0

        stress = self.bending_stress_pa(
            per_motor_thrust_structural, arm_length_m, arm_width_m, thickness_m
        )

        safety_factor = factors["structural_safety_factor"]
        allowable_stress = material_yield / safety_factor

        # Determine structural pass
        if stress is not None:
            structural_passed = stress <= allowable_stress
            safety_margin = allowable_stress / stress if stress > 0 else float("inf")
        else:
            structural_passed = False
            safety_margin = None

        # Legacy heuristic: arm_width >= 0.08 * arm_length (R8.4)
        heuristic_passed = arm_width_mm >= arm_length_mm * 0.08

        structural_result = StructuralResult(
            bending_stress_pa=stress,
            allowable_stress_pa=allowable_stress,
            safety_margin=safety_margin,
            arm_width_mm=arm_width_mm,
            material_thickness_mm=thickness_mm,
            material=material,
            passed=structural_passed,
            heuristic_passed=heuristic_passed,
        )

        if not structural_passed:
            issues.append(
                f"Structural failure: bending stress "
                f"{stress if stress is not None else 'N/A':.2e} Pa exceeds "
                f"allowable {allowable_stress:.2e} Pa "
                f"(arm_width={arm_width_mm} mm, "
                f"material_thickness={thickness_mm} mm, material={material})."
                if stress is not None else
                f"Structural check indeterminate: section modulus is zero "
                f"(arm_width={arm_width_mm} mm, "
                f"material_thickness={thickness_mm} mm, material={material})."
            )

        # =====================================================================
        # 14. Disk loading (R9.1–R9.4)
        # =====================================================================
        prop_diameter_mm = float(components["propeller_diameter_mm"])
        prop_diameter_m = prop_diameter_mm / 1000.0
        total_swept_area = motor_count * math.pi * (prop_diameter_m / 2.0) ** 2

        disk_loading = self.disk_loading_nm2(total_thrust, total_swept_area)
        if disk_loading is None:
            issues.append(
                "Disk loading unavailable: total swept area is zero "
                "(divide-by-zero guard)."
            )

        # =====================================================================
        # 15. Assemble EngineeringMetrics
        # =====================================================================
        return EngineeringMetrics(
            auw_kg=auw,
            frame_mass_kg=frame_mass,
            total_thrust_n=total_thrust,
            twr=twr_val,
            twr_target=twr_target,
            twr_pass=twr_pass,
            hover_throttle=hover_throttle,
            throttle_headroom=throttle_headroom,
            payload_target_kg=payload_mass,
            payload_margin_kg=margin,
            payload_feasible=payload_feasible,
            flight_time_min=flight_time,
            flight_time_target_min=flight_time_target,
            flight_time_pass=flight_time_pass,
            disk_loading_nm2=disk_loading,
            use_case=use_case,
            structural=structural_result,
            notes=notes,
            issues=issues,
            available=True,
        )
