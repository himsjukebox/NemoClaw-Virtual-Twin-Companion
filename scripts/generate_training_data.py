#!/usr/bin/env python3
"""
NemoClaw Virtual Twin Companion — Synthetic Training Data Generator
====================================================================

Generates physics-validated instruction-output pairs for fine-tuning an LLM
to reliably translate natural-language drone design goals into correct
parametric values.

Every generated pair is validated through the Physics Engine to guarantee:
- All parameters are within valid ranges
- Structural checks pass (bending stress ≤ yield / safety factor)
- TWR meets the use-case target
- Payload is feasible

Output: JSONL file in standard fine-tuning format (instruction + output fields)
compatible with NeMo Framework, NVIDIA NeMo Curator, and PEFT/LoRA pipelines.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --count 5000 --output data/training_data.jsonl

Requirements:
    - The physics engine must be importable (run from project root)
    - No NVIDIA API key needed (physics is local, deterministic)
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.physics_engine import PhysicsEngine

# =============================================================================
# Configuration
# =============================================================================

MATERIALS = ["PLA", "ABS", "carbon_fiber", "aluminum"]
USE_CASES = ["racing", "cinematography", "delivery", "mapping"]
ARM_COUNTS = [3, 4, 5, 6, 8]

# Parameter ranges (from models/parameters.py)
ARM_LENGTH_RANGE = (80.0, 200.0)
ARM_WIDTH_RANGE = (8.0, 25.0)
THICKNESS_RANGE = (2.0, 10.0)
CUTOUT_RANGE = (10.0, 30.0)

# Payload ranges per use case
PAYLOAD_RANGES = {
    "racing": (0.0, 0.3),
    "cinematography": (0.2, 1.5),
    "delivery": (0.5, 5.0),
    "mapping": (0.3, 2.0),
}

# Airframe names for natural language
AIRFRAME_NAMES = {
    3: ["tricopter", "tri-rotor", "3-arm drone"],
    4: ["quadcopter", "quad", "4-arm drone", "quadrotor"],
    5: ["pentacopter", "5-arm drone"],
    6: ["hexacopter", "hex", "6-arm drone", "hexarotor"],
    8: ["octocopter", "octo", "8-arm drone", "octorotor"],
}

# Material descriptions for natural language
MATERIAL_DESCRIPTIONS = {
    "PLA": ["PLA", "PLA plastic", "standard PLA", "biodegradable PLA"],
    "ABS": ["ABS", "ABS plastic", "impact-resistant ABS"],
    "carbon_fiber": ["carbon fiber", "carbon fibre", "CF", "carbon composite"],
    "aluminum": ["aluminum", "aluminium", "alloy aluminum", "metal frame"],
}

# Use case descriptions
USE_CASE_DESCRIPTIONS = {
    "racing": ["racing", "FPV racing", "speed-optimized", "agile racing", "fast drone"],
    "cinematography": ["cinematography", "filming", "camera drone", "aerial photography", "video drone", "cinema"],
    "delivery": ["delivery", "cargo", "package delivery", "payload carrying", "logistics"],
    "mapping": ["mapping", "aerial survey", "surveying", "photogrammetry", "terrain mapping"],
}

# Design goal templates
PROMPT_TEMPLATES = [
    "Design a {airframe} chassis for {use_case} using {material}",
    "Create a {use_case} {airframe} frame with {material} material",
    "Build a {material} {airframe} chassis optimized for {use_case}",
    "I need a {airframe} frame for {use_case} made from {material}",
    "Design a {arm_count}-arm drone chassis using {material} for {use_case} missions",
    "Make a {use_case} drone frame with {arm_count} arms in {material}",
    "Generate a {material} {airframe} for {use_case} with {payload}kg payload",
    "Design a {airframe} that can carry {payload}kg using {material} for {use_case}",
    "Create a {use_case} {airframe} chassis, {material} material, {payload}kg payload capacity",
    "Build a {arm_count}-arm {material} frame for {use_case}, needs to carry {payload}kg",
    "I want a {airframe} drone for {use_case} applications, using {material}, carrying {payload}kg",
    "Design a lightweight {airframe} for {use_case} in {material} with arm length around {arm_length}mm",
    "Create a sturdy {airframe} chassis with thick {thickness}mm plates using {material} for {use_case}",
    "Make a {use_case} {airframe} with wide {arm_width}mm arms for stability, {material} material",
]


# =============================================================================
# Physics Config (embedded for standalone execution)
# =============================================================================

PHYSICS_CONFIG = {
    "constants": {"g": 9.80665, "air_density": 1.225},
    "materials": {
        "PLA": {"density": 1240.0, "yield_strength": 50.0e6},
        "ABS": {"density": 1040.0, "yield_strength": 40.0e6},
        "carbon_fiber": {"density": 1600.0, "yield_strength": 600.0e6},
        "aluminum": {"density": 2700.0, "yield_strength": 270.0e6},
    },
    "motors": {
        "2207_2400kv": {"mass_kg": 0.032, "max_thrust_n": 14.7},
        "2806_1300kv": {"mass_kg": 0.045, "max_thrust_n": 19.6},
    },
    "batteries": {
        "4s_1500mah": {"capacity_mah": 1500, "cells_s": 4, "mass_kg": 0.190},
        "6s_5000mah": {"capacity_mah": 5000, "cells_s": 6, "mass_kg": 0.700},
    },
    "use_cases": {
        "racing": {"target_twr": 4.0, "default_flight_time_min": 5.0},
        "cinematography": {"target_twr": 2.0, "default_flight_time_min": 12.0},
        "delivery": {"target_twr": 2.0, "default_flight_time_min": 15.0},
        "mapping": {"target_twr": 2.0, "default_flight_time_min": 20.0},
    },
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
# Data Generation
# =============================================================================


def generate_valid_geometry(arm_count: int, material: str, use_case: str) -> dict:
    """
    Generate random but physically valid geometry parameters.
    
    Ensures:
    - arm_width >= arm_length * 0.10 (above the 0.08 structural constraint with margin)
    - All values within defined ranges
    - Structural stress check will pass
    """
    # For racing: shorter arms, thinner. For delivery: longer, thicker
    if use_case == "racing":
        arm_length = random.uniform(80, 130)
        thickness = random.uniform(2.5, 5.0)
    elif use_case == "delivery":
        arm_length = random.uniform(120, 200)
        thickness = random.uniform(4.0, 8.0)
    elif use_case == "mapping":
        arm_length = random.uniform(130, 200)
        thickness = random.uniform(4.0, 7.0)
    else:  # cinematography
        arm_length = random.uniform(100, 160)
        thickness = random.uniform(3.0, 6.0)

    # Ensure structural constraint: arm_width >= arm_length * 0.08
    min_width = arm_length * 0.10  # Use 0.10 for comfortable margin above 0.08
    arm_width = max(min_width, random.uniform(max(8.0, min_width), 25.0))
    arm_width = min(arm_width, 25.0)  # Clamp to max

    # If arm_width can't satisfy both constraints, adjust arm_length down
    if arm_width < arm_length * 0.08:
        arm_length = arm_width / 0.08 - 5.0  # Reduce with margin

    cutout_radius = random.uniform(10, min(30, arm_length * 0.2))

    return {
        "arm_count": arm_count,
        "arm_length": round(arm_length, 1),
        "arm_width": round(arm_width, 1),
        "material_thickness": round(thickness, 1),
        "center_cutout_radius": round(cutout_radius, 1),
    }


def generate_natural_language_prompt(
    arm_count: int, material: str, use_case: str, payload: float,
    geometry: dict
) -> str:
    """Generate a varied natural-language design prompt."""
    template = random.choice(PROMPT_TEMPLATES)
    airframe = random.choice(AIRFRAME_NAMES[arm_count])
    mat_desc = random.choice(MATERIAL_DESCRIPTIONS[material])
    uc_desc = random.choice(USE_CASE_DESCRIPTIONS[use_case])

    prompt = template.format(
        airframe=airframe,
        use_case=uc_desc,
        material=mat_desc,
        arm_count=arm_count,
        payload=payload,
        arm_length=geometry.get("arm_length", 120),
        arm_width=geometry.get("arm_width", 15),
        thickness=geometry.get("material_thickness", 5),
    )
    return prompt


def validate_with_physics(geometry: dict, material: str, use_case: str, payload: float) -> bool:
    """
    Run the physics engine and check if the design passes all checks.
    Returns True only if TWR meets target, payload is feasible, and structural passes.
    """
    engine = PhysicsEngine(config=PHYSICS_CONFIG)
    mission = {
        "payload_mass_kg": payload,
        "use_case": use_case,
        "target_flight_time_min": PHYSICS_CONFIG["use_cases"][use_case]["default_flight_time_min"],
    }

    try:
        metrics = engine.analyze(geometry, material, mission)
        return (
            metrics.twr_pass
            and metrics.payload_feasible
            and metrics.structural.passed
            and metrics.structural.heuristic_passed
        )
    except Exception:
        return False


def generate_dataset(count: int, seed: int = 42) -> list:
    """
    Generate `count` physics-validated training pairs.
    
    Each pair consists of:
    - instruction: Natural-language design goal
    - output: JSON with component_type, design_parameters, material, payload, use_case
    
    All outputs are guaranteed to pass the physics engine validation.
    """
    random.seed(seed)
    dataset = []
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops

    while len(dataset) < count and attempts < max_attempts:
        attempts += 1

        # Random configuration
        arm_count = random.choice(ARM_COUNTS)
        material = random.choice(MATERIALS)
        use_case = random.choice(USE_CASES)
        payload_lo, payload_hi = PAYLOAD_RANGES[use_case]
        payload = round(random.uniform(payload_lo, payload_hi), 2)

        # Generate geometry
        geometry = generate_valid_geometry(arm_count, material, use_case)

        # Validate with physics engine
        if not validate_with_physics(geometry, material, use_case, payload):
            continue

        # Generate natural language prompt
        prompt = generate_natural_language_prompt(
            arm_count, material, use_case, payload, geometry
        )

        # Build the expected output (what the fine-tuned model should produce)
        output = {
            "component_type": "chassis",
            "design_parameters": geometry,
            "material": material,
            "payload_mass_kg": payload,
            "use_case": use_case,
            "target_flight_time_min": PHYSICS_CONFIG["use_cases"][use_case]["default_flight_time_min"],
        }

        dataset.append({
            "instruction": prompt,
            "output": json.dumps(output),
        })

    return dataset


def write_jsonl(dataset: list, output_path: str):
    """Write dataset to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate physics-validated training data for NemoClaw fine-tuning"
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Number of training pairs to generate (default: 1000)"
    )
    parser.add_argument(
        "--output", type=str, default="data/training_data.jsonl",
        help="Output JSONL file path (default: data/training_data.jsonl)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    print(f"🔧 Generating {args.count} physics-validated training pairs...")
    print(f"   Seed: {args.seed}")
    print(f"   Output: {args.output}")
    print()

    dataset = generate_dataset(args.count, seed=args.seed)

    print(f"✅ Generated {len(dataset)} valid pairs (from {args.count} target)")
    print(f"   Materials: {MATERIALS}")
    print(f"   Use cases: {USE_CASES}")
    print(f"   Arm counts: {ARM_COUNTS}")
    print()

    # Stats
    material_counts = {}
    use_case_counts = {}
    for entry in dataset:
        output = json.loads(entry["output"])
        material_counts[output["material"]] = material_counts.get(output["material"], 0) + 1
        use_case_counts[output["use_case"]] = use_case_counts.get(output["use_case"], 0) + 1

    print("   Distribution:")
    print(f"   Materials: {material_counts}")
    print(f"   Use cases: {use_case_counts}")
    print()

    write_jsonl(dataset, args.output)
    print(f"📄 Written to: {args.output}")
    print(f"   File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print()
    print("   Format: JSONL (one JSON object per line)")
    print("   Fields: instruction (str), output (str/JSON)")
    print()
    print("   Compatible with:")
    print("   - NVIDIA NeMo Framework (PEFT/LoRA)")
    print("   - NeMo Curator for data curation")
    print("   - Any instruction-tuning pipeline (Alpaca format)")


if __name__ == "__main__":
    main()
