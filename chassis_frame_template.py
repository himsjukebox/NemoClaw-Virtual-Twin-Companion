import cadquery as cq
import math

# =====================================================================
# 1. PARAMETRIC VARIABLES (AI CONTROLLED)
# =====================================================================
# These variables represent the "Design Space".
# The NemoClaw Design Agent will iteratively modify these exact numbers
# based on the Validator Agent's feedback.
arm_count = 4               # Number of arms/motors (3=tri, 4=quad, 6=hexa, 8=octo)
arm_length = 120.0          # Distance from center origin to motor shaft (mm)
material_thickness = 5.0    # Z-axis extrusion height of the entire frame (mm)
arm_width = 15.0            # Y-axis width of the structural arms (mm)
center_cutout_radius = 20.0 # Radius of the center hole to reduce mass (mm)

# =====================================================================
# 2. STATIC VARIABLES (FIXED ENGINEERING CONSTRAINTS)
# =====================================================================
# These represent standard off-the-shelf components that cannot change.
motor_mount_radius = 18.0   # Matches a standard 2205 brushless motor base
center_body_radius = 35.0   # Minimum space required to mount the flight controller
fillet_radius = 3.0         # Stress relief for internal corners

# =====================================================================
# 3. GEOMETRY GENERATION (The CAD Macro)
# =====================================================================

def build_drone_chassis():
    """
    Generates the 3D BREP model of a multirotor chassis with a configurable
    number of arms (tricopter, quadcopter, hexacopter, octocopter, etc.).

    The arms are distributed radially at evenly-spaced angles (360 / arm_count),
    so the same macro produces any rotor configuration from the arm_count
    parameter alone.
    """

    # Defensive: arm_count must be a whole number >= 2 for a valid frame.
    n_arms = int(round(arm_count))
    if n_arms < 2:
        n_arms = 2

    # --- STEP A: The Center Hub (CATIA: Pad a Circle) ---
    # Draw a circle at the origin and extrude it to material_thickness.
    frame = cq.Workplane("XY").circle(center_body_radius).extrude(material_thickness)

    # --- STEP B: Radial Arms + Motor Mounts (CATIA: Multi-Pad on points) ---
    # For each arm, build a rectangular arm pointing along +X from the center,
    # then rotate it about the Z axis to its evenly-spaced angle. A circular
    # motor mount is placed at the tip of each arm.
    angle_step = 360.0 / n_arms
    for i in range(n_arms):
        angle_deg = i * angle_step
        angle_rad = math.radians(angle_deg)

        # Arm body: a rectangle centered at half the arm length along +X,
        # then rotated about the global Z axis to the target angle.
        arm = (
            cq.Workplane("XY")
            .center(arm_length / 2.0, 0)
            .rect(arm_length, arm_width)
            .extrude(material_thickness)
            .rotate((0, 0, 0), (0, 0, 1), angle_deg)
        )
        frame = frame.union(arm)

        # Motor mount disc at the arm tip.
        mount_x = arm_length * math.cos(angle_rad)
        mount_y = arm_length * math.sin(angle_rad)
        mount = (
            cq.Workplane("XY")
            .center(mount_x, mount_y)
            .circle(motor_mount_radius)
            .extrude(material_thickness)
        )
        frame = frame.union(mount)

    # --- STEP C: Weight Relief Cutout (CATIA: Pocket) ---
    # Select the top face, draw a circle in the center, and pocket downwards.
    frame = (
        frame.faces(">Z")                   # Select top-most face
        .workplane()                        # Create a new sketch on this face
        .circle(center_cutout_radius)       # Draw the weight reduction circle
        .cutBlind(-material_thickness)      # Pocket downwards through the part
    )

    # --- STEP D: Stress Relief Fillets (CATIA: Edge Fillet) ---
    # Sharp internal corners cause stress fractures. Select all vertical edges
    # ("|Z") and apply a fillet. Wrapped in try/except because aggressive
    # parameter combinations can make the fillet mathematically impossible.
    try:
        frame = frame.edges("|Z").fillet(fillet_radius)
    except Exception as e:
        print(f"Geometry error during fillet. Check parameters: {e}")
        pass

    return frame

# =====================================================================
# 4. EXECUTION & EXPORT
# =====================================================================
if __name__ == "__main__":
    # Generate the 3D solid
    drone_part = build_drone_chassis()

    # Export as a dumb solid (.STEP) for manufacturing/3DEXPERIENCE
    cq.exporters.export(drone_part, "optimized_drone_chassis.step")

    # Export as a mesh (.STL) for your Streamlit/PyVista Web UI visualization
    cq.exporters.export(drone_part, "optimized_drone_chassis.stl")

    print("Design successfully generated and exported.")
