# =============================================================================
# NemoClaw Virtual Twin Companion — Drone Chassis Geometry Generator
# =============================================================================
# PURPOSE:
#   Generates a 3D mesh representation of the quadcopter drone chassis from
#   parametric design values. Uses PyVista to create geometry programmatically
#   without requiring CadQuery/OpenCASCADE (which has complex dependencies).
#
#   This produces a visual approximation of the drone chassis for the Streamlit
#   3D viewer. In production, the actual CadQuery BREP would be generated
#   inside NemoClaw/OpenShell and the resulting STL loaded for display.
#
# GEOMETRY:
#   The drone chassis consists of:
#     - A central hub (circular plate with cutout)
#     - Four arms extending at 45°, 135°, 225°, 315° from center
#     - Motor mounts (cylinders) at each arm tip
#
# Component: CAD_Tool (visualization helper)
# =============================================================================

import numpy as np
import pyvista as pv
from typing import Dict, Optional


def generate_drone_mesh(params: Dict[str, float]) -> pv.PolyData:
    """
    Generate a PyVista mesh of the quadcopter drone chassis.

    Creates a simplified but recognizable drone frame geometry from the
    four parametric design values. The geometry includes a central hub,
    four arms, and motor mount indicators.

    Args:
        params (dict): Design parameters with keys:
            - arm_length (float): Distance from center to motor mount (mm)
            - material_thickness (float): Z-axis plate height (mm)
            - arm_width (float): Y-axis arm width (mm)
            - center_cutout_radius (float): Center hole radius (mm)

    Returns:
        pv.PolyData: Combined mesh of the complete drone chassis.

    Component: CAD_Tool
    """
    arm_length = params.get("arm_length", 120.0)
    thickness = params.get("material_thickness", 5.0)
    arm_width = params.get("arm_width", 15.0)
    cutout_radius = params.get("center_cutout_radius", 20.0)

    # Scale to reasonable visualization size (mm → display units)
    scale = 1.0

    meshes = []

    # --- Central Hub ---
    # Create a disc (cylinder) for the center body
    hub_radius = cutout_radius * 2.0  # Hub is 2x the cutout radius
    hub = pv.Cylinder(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        radius=hub_radius * scale,
        height=thickness * scale,
        resolution=64,
    )
    meshes.append(hub)

    # Central cutout (represented as a thinner cylinder to show the hole visually)
    # We'll use boolean subtraction approximation by adding a darker-colored inner disc
    # For simplicity, just leave the hub solid — the cutout is parametric info

    # --- Four Arms ---
    # Arms extend at 45°, 135°, 225°, 315° (X-configuration)
    arm_angles = [45, 135, 225, 315]

    for angle_deg in arm_angles:
        angle_rad = np.radians(angle_deg)

        # Arm center point (halfway between hub edge and motor mount)
        arm_center_dist = (hub_radius + arm_length) / 2.0
        cx = arm_center_dist * np.cos(angle_rad) * scale
        cy = arm_center_dist * np.sin(angle_rad) * scale

        # Create arm as a box
        arm_box_length = (arm_length - hub_radius + 10) * scale  # slight overlap with hub
        arm_box = pv.Box(
            bounds=(
                -arm_box_length / 2, arm_box_length / 2,
                -arm_width * scale / 2, arm_width * scale / 2,
                -thickness * scale / 2, thickness * scale / 2,
            )
        )

        # Rotate arm to correct angle
        arm_box = arm_box.rotate_z(angle_deg, inplace=False)

        # Translate arm to correct position
        arm_box = arm_box.translate(
            (cx, cy, 0), inplace=False
        )

        meshes.append(arm_box)

        # --- Motor Mount (cylinder at arm tip) ---
        motor_x = arm_length * np.cos(angle_rad) * scale
        motor_y = arm_length * np.sin(angle_rad) * scale
        motor_radius = arm_width * 0.8 * scale  # Motor mount slightly smaller than arm

        motor_mount = pv.Cylinder(
            center=(motor_x, motor_y, thickness * scale * 0.5),
            direction=(0, 0, 1),
            radius=motor_radius,
            height=thickness * scale * 1.5,
            resolution=32,
        )
        meshes.append(motor_mount)

    # Combine all meshes
    combined = meshes[0]
    for mesh in meshes[1:]:
        combined = combined.merge(mesh)

    return combined


def create_plotter(mesh: pv.PolyData, params: Optional[Dict[str, float]] = None) -> pv.Plotter:
    """
    Create a PyVista plotter configured for Streamlit display.

    Sets up lighting, camera angle, and coloring for an attractive
    visualization of the drone chassis mesh.

    Args:
        mesh (pv.PolyData): The drone chassis mesh to display.
        params (dict, optional): Design parameters for title annotation.

    Returns:
        pv.Plotter: Configured plotter ready for stpyvista rendering.

    Component: Streamlit_UI
    """
    plotter = pv.Plotter(window_size=(600, 400))

    # Add mesh with drone-appropriate coloring
    plotter.add_mesh(
        mesh,
        color="#1a1a2e",  # Dark blue-black (carbon fiber look)
        specular=0.5,
        specular_power=15,
        smooth_shading=True,
        show_edges=False,
    )

    # Set camera for a nice isometric-ish view
    plotter.camera_position = "iso"
    plotter.camera.zoom(0.8)

    # Add subtle background
    plotter.set_background("white")

    # Add axes for orientation
    plotter.add_axes()

    return plotter
