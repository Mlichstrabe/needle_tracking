"""Generate the parametric IMU + STP-23L blunt training probe.

Run from the repository root:
    uv run --with cadquery python cad/integrated_probe.py

The model is for non-clinical phantom demonstrations only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import cadquery as cq


ROOT = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT / "cad" / "probe_parameters.json"
VENDOR_STEP = ROOT / "cad" / "vendor" / "STP-23L模块.stp"
OUTPUT_DIR = ROOT / "output" / "cad_v2b_rounded_tip"


def load_parameters() -> dict[str, Any]:
    return json.loads(PARAM_PATH.read_text(encoding="utf-8"))


def as_shape(obj: cq.Workplane | cq.Shape) -> cq.Shape:
    return obj.val() if isinstance(obj, cq.Workplane) else obj


def rounded_prism_x(
    x0: float,
    length: float,
    width: float,
    z0: float,
    height: float,
    radius: float,
) -> cq.Workplane:
    body = cq.Workplane("XY", origin=(x0, 0, z0)).box(
        length, width, height, centered=(False, True, False)
    )
    return body.edges("|X").fillet(radius)


def rounded_tunnel_x(
    x0: float,
    length: float,
    width: float,
    height: float,
    center_z: float,
    radius: float,
) -> cq.Workplane:
    tunnel = cq.Workplane("XY", origin=(x0, 0, center_z - height / 2)).box(
        length, width, height, centered=(False, True, False)
    )
    return tunnel.edges("|X").fillet(radius)


def box_xyz(
    x0: float,
    length: float,
    y_center: float,
    width: float,
    z0: float,
    height: float,
) -> cq.Workplane:
    return cq.Workplane("XY", origin=(x0, y_center, z0)).box(
        length, width, height, centered=(False, True, False)
    )


def cylinder_x(x0: float, length: float, radius: float, y: float, z: float) -> cq.Workplane:
    return cq.Workplane("YZ", origin=(x0, y, z)).circle(radius).extrude(length)


def build_shells(p: dict[str, Any], installed_stp: cq.Shape) -> tuple[cq.Workplane, cq.Workplane]:
    handle = p["handle"]
    needle = p["needle"]
    imu = p["imu"]
    stp = p["stp23l"]
    snap = p["snap"]

    length = handle["length"]
    width = handle["width"]
    height = handle["height"]
    half_h = height / 2

    full_outer = rounded_prism_x(
        0,
        length,
        width,
        -half_h,
        height,
        handle["cross_section_radius"],
    )
    lower_clip = box_xyz(-1, length + 2, 0, width + 2, -half_h - 1, half_h + 1)
    upper_clip = box_xyz(-1, length + 2, 0, width + 2, 0, half_h + 1)
    lower = full_outer.intersect(lower_clip)
    upper = full_outer.intersect(upper_clip)

    # STP-23L is rotated so the vendor +Z optical axis points along probe -X.
    stp_box = installed_stp.BoundingBox()
    stp_tunnel = rounded_tunnel_x(
        stp["optical_face_x"] - stp["axial_clearance"],
        length - stp["optical_face_x"] + stp["axial_clearance"] + 1,
        stp_box.ylen + 2 * stp["side_clearance"],
        stp_box.zlen + 2 * stp["side_clearance"],
        stp["center_z"],
        1.2,
    )
    lower = lower.cut(stp_tunnel)

    # Only the optical window needs to see through the front nose.
    optical_aperture = rounded_tunnel_x(
        -1,
        stp["optical_face_x"] + 1.5,
        stp["optical_aperture_width"],
        stp["optical_aperture_height"],
        stp["center_z"],
        4.0,
    )
    lower = lower.cut(optical_aperture)

    # The upper shell is open at the rear; the removable cable clamp closes it.
    upper_cavity = box_xyz(
        25,
        length - 24,
        0,
        width - 2 * handle["wall"],
        0,
        half_h - handle["wall"],
    )
    upper = upper.cut(upper_cavity)

    # Bayonet socket: a running bore, two insertion keyways, and a larger lock chamber.
    plug_r = needle["plug_diameter"] / 2
    bore_r = plug_r + needle["socket_clearance_radial"]
    axis_z = needle["axis_z"]
    bore = cylinder_x(-0.5, needle["plug_length"] + 0.5, bore_r, 0, axis_z)
    chamber = cylinder_x(
        13.2,
        needle["plug_length"] - 13.0,
        needle["chamber_radius"],
        0,
        axis_z,
    )
    keyway_offset = (plug_r + needle["lug_radius"]) / 2
    keyways = None
    for sign in (-1, 1):
        key = box_xyz(
            -0.5,
            14.3,
            sign * keyway_offset,
            needle["lug_radius"] - plug_r + 0.7,
            axis_z - needle["lug_width"] / 2 - 0.25,
            needle["lug_width"] + 0.5,
        )
        keyways = key if keyways is None else keyways.union(key)

    socket_cut = bore.union(chamber).union(keyways)
    lower = lower.cut(socket_cut)
    upper = upper.cut(socket_cut)

    # Snap pockets in the lower shell.
    snap_xs = (22.0, 73.0)
    for x_center in snap_xs:
        for sign in (-1, 1):
            pocket = box_xyz(
                x_center - snap["arm_width"] / 2 - 0.6,
                snap["arm_width"] + 1.2,
                sign * (width / 2 - 1.2),
                3.2,
                -5.2,
                5.5,
            )
            lower = lower.cut(pocket)

    # Four compliant upper-cover arms. Small hooks point toward the centerline.
    for x_center in snap_xs:
        for sign in (-1, 1):
            arm_y = sign * (width / 2 - snap["arm_thickness"] / 2 - 0.35)
            arm = box_xyz(
                x_center - snap["arm_width"] / 2,
                snap["arm_width"],
                arm_y,
                snap["arm_thickness"],
                -6.0,
                10.0,
            )
            hook_y = sign * (width / 2 - snap["arm_thickness"] - snap["undercut"] / 2 - 0.35)
            hook = box_xyz(
                x_center - snap["arm_width"] / 2,
                snap["arm_width"],
                hook_y,
                snap["arm_thickness"] + snap["undercut"],
                -6.0,
                1.6,
            )
            upper = upper.union(arm).union(hook)

    # IMU hard datums: four pads, two true locator pins, and two clearance pins.
    board_x0 = imu["front_x"]
    board_xc = board_x0 + imu["axial_length"] / 2
    dx = imu["hole_spacing_axial"] / 2
    dy = imu["hole_spacing_transverse"] / 2
    locator_positions = [
        (board_xc - dx, -dy, True),
        (board_xc - dx, dy, False),
        (board_xc + dx, -dy, False),
        (board_xc + dx, dy, True),
    ]
    for x, y, is_locator in locator_positions:
        pad = cq.Workplane("XY", origin=(x, y, 0)).circle(2.3).extrude(imu["base_z"])
        pin_d = imu["locator_pin_diameter"] if is_locator else imu["locator_pin_diameter"] - 0.25
        pin = cq.Workplane("XY", origin=(x, y, 0)).circle(pin_d / 2).extrude(3.0)
        lower = lower.union(pad).union(pin)

    # Two low clips retain only the PCB edge; the 16 mm module body remains unloaded.
    pcb_top = imu["base_z"] + imu["pcb_thickness"]
    for sign in (-1, 1):
        clip_y = sign * (imu["width"] / 2 + 1.1)
        arm = box_xyz(board_xc - 4, 8, clip_y, 1.8, 0, pcb_top + 1.5)
        hook_y = sign * (imu["width"] / 2 + 0.15)
        hook = box_xyz(
            board_xc - 4,
            8,
            hook_y,
            1.9,
            pcb_top,
            1.0,
        )
        lower = lower.union(arm).union(hook)

    return lower, upper


def build_needle(p: dict[str, Any]) -> cq.Workplane:
    n = p["needle"]
    axis_z = n["axis_z"]
    radius = n["diameter"] / 2
    transition_start_x = -n["root_transition_length"]
    tip_front_x = -n["exposed_length"]
    tip_taper_start_x = tip_front_x + n["tip_transition_length"]
    ball_radius = n["tip_ball_diameter"] / 2
    neck_radius = n["tip_neck_diameter"] / 2
    ball_center_x = tip_front_x + ball_radius
    ball_rear_x = ball_center_x + ball_radius
    neck_rear_x = ball_rear_x + 1.6

    # Ø6 middle shaft: 16 mm tip + 154 mm straight section + 30 mm root.
    shaft = cylinder_x(
        tip_taper_start_x,
        transition_start_x + 2.0 - tip_taper_start_x,
        radius,
        0,
        axis_z,
    )
    needle_shape = shaft.val()

    # B tip: a smooth axisymmetric taper, short Ø2.4 neck, and Ø3.6 bead.
    taper_span = tip_taper_start_x - neck_rear_x
    taper = (
        cq.Workplane("YZ", origin=(neck_rear_x, 0, axis_z))
        .circle(neck_radius)
        .workplane(offset=taper_span * 0.32)
        .circle(neck_radius + (radius - neck_radius) * 0.18)
        .workplane(offset=taper_span * 0.38)
        .circle(neck_radius + (radius - neck_radius) * 0.70)
        .workplane(offset=taper_span * 0.30)
        .circle(radius)
        .loft(combine=True)
    )
    neck = cylinder_x(ball_rear_x - 0.25, neck_rear_x - ball_rear_x + 0.45, neck_radius, 0, axis_z)
    # Revolve a near-spherical bulb with a Ø0.4 front micro-flat. Avoiding an
    # exact mathematical pole also prevents degenerate STL triangles.
    bead_profile = (
        cq.Workplane("XY", origin=(0, 0, axis_z))
        .moveTo(tip_front_x, 0)
        .lineTo(tip_front_x, 0.2)
        .threePointArc(
            (ball_center_x - ball_radius * 0.68, ball_radius * 0.82),
            (ball_center_x, ball_radius),
        )
        .threePointArc(
            (ball_center_x + ball_radius * 0.58, ball_radius * 0.92),
            (ball_rear_x + 0.12, neck_radius),
        )
        .lineTo(ball_rear_x + 0.35, neck_radius)
        .lineTo(ball_rear_x + 0.35, 0)
        .lineTo(tip_front_x, 0)
        .close()
    )
    bead = bead_profile.revolve(360, (0, 0), (1, 0))
    needle_shape = needle_shape.fuse(taper.val()).fuse(neck.val()).fuse(bead.val())

    # Photo-inspired leaf/spindle reinforcement: wide in Y, comparatively
    # thin in Z, and smoothly lofted into the unchanged bayonet plug.
    max_y = n["root_transition_max_width"] / 2
    max_z = n["root_transition_max_thickness"] / 2
    fairing = (
        cq.Workplane("YZ", origin=(transition_start_x - 1.0, 0, axis_z))
        .circle(radius)
        .workplane(offset=9.0)
        .ellipse(radius + 2.2, radius + 1.0)
        .workplane(offset=10.0)
        .ellipse(max_y, max_z)
        .workplane(offset=8.0)
        .ellipse(5.5, 4.5)
        .workplane(offset=4.0)
        .circle(5.0)
        .loft(combine=True)
    )
    needle_shape = needle_shape.fuse(fairing.val())

    # Preserve the V1 18 mm detachable bayonet interface.
    plug = cylinder_x(
        -0.5,
        n["plug_length"] + 0.5,
        n["plug_diameter"] / 2,
        0,
        axis_z,
    )
    needle_shape = needle_shape.fuse(plug.val())

    lug_inner_radius = n["plug_diameter"] / 2 - 0.45
    lug_radial_thickness = n["lug_radius"] - lug_inner_radius
    lug_center = lug_inner_radius + lug_radial_thickness / 2
    for sign in (-1, 1):
        lug = box_xyz(
            n["lug_axial_start"],
            n["lug_axial_length"],
            sign * lug_center,
            lug_radial_thickness,
            axis_z - n["lug_width"] / 2,
            n["lug_width"],
        )
        needle_shape = needle_shape.fuse(lug.val())

    # One continuous underside flat lets both the long shaft and the wider root
    # reinforcement contact the print bed without support.
    flat_plane_z = axis_z - radius + n["shaft_print_flat_depth"]
    underside_cut = box_xyz(
        -n["exposed_length"] - 1,
        n["exposed_length"] + n["plug_length"] + 2.0,
        0,
        20,
        -10,
        flat_plane_z + 10,
    )
    needle_shape = needle_shape.cut(underside_cut.val())
    return cq.Workplane(obj=needle_shape.clean())


def add_tip_support_web(needle: cq.Shape, p: dict[str, Any]) -> cq.Shape:
    """Add a one-wall sacrificial web only to the print-ready needle STL."""
    n = p["needle"]
    radius = n["diameter"] / 2
    flat_plane_z = n["axis_z"] - radius + n["shaft_print_flat_depth"]
    neck_bottom_z = n["axis_z"] - n["tip_neck_diameter"] / 2
    web = box_xyz(
        -n["exposed_length"] + 0.5,
        n["tip_transition_length"] + 0.8,
        0,
        n["tip_support_web_thickness"],
        flat_plane_z,
        neck_bottom_z - flat_plane_z + 0.12,
    )
    return needle.fuse(web.val()).clean()


def build_imu_proxy(p: dict[str, Any]) -> cq.Workplane:
    imu = p["imu"]
    x0 = imu["front_x"]
    z0 = imu["base_z"]
    board = box_xyz(x0, imu["axial_length"], 0, imu["width"], z0, imu["pcb_thickness"])

    x_center = x0 + imu["axial_length"] / 2
    dx = imu["hole_spacing_axial"] / 2
    dy = imu["hole_spacing_transverse"] / 2
    for x in (x_center - dx, x_center + dx):
        for y in (-dy, dy):
            hole = cq.Workplane("XY", origin=(x, y, z0 - 0.2)).circle(
                imu["hole_diameter"] / 2
            ).extrude(imu["pcb_thickness"] + 0.4)
            board = board.cut(hole)

    body_height = imu["total_height"] - imu["pcb_thickness"]
    body = box_xyz(
        x0 + 1.0,
        imu["module_body_axial"],
        0,
        imu["module_body_width"],
        z0 + imu["pcb_thickness"],
        body_height,
    )
    usb = box_xyz(
        x0 + imu["axial_length"],
        imu["usb_protrusion"],
        0,
        11.0,
        z0,
        4.5,
    )
    return board.union(body).union(usb)


def orient_vendor_stp(p: dict[str, Any]) -> cq.Shape:
    if not VENDOR_STEP.exists():
        raise FileNotFoundError(f"Vendor STEP missing: {VENDOR_STEP}")

    raw = cq.importers.importStep(str(VENDOR_STEP)).val()
    bb = raw.BoundingBox()
    centered = raw.translate(
        cq.Vector(
            -(bb.xmin + bb.xmax) / 2,
            -(bb.ymin + bb.ymax) / 2,
            -(bb.zmin + bb.zmax) / 2,
        )
    )
    # Vendor +Z -> assembly -X, vendor +X -> assembly -Y, vendor +Y -> assembly +Z.
    oriented = centered.rotate((0, 0, 0), (0, 1, 0), -90)
    oriented = oriented.rotate((0, 0, 0), (1, 0, 0), 90)
    ob = oriented.BoundingBox()
    stp = p["stp23l"]
    return oriented.translate(
        cq.Vector(
            stp["optical_face_x"] - ob.xmin,
            -(ob.ymin + ob.ymax) / 2,
            stp["center_z"] - (ob.zmin + ob.zmax) / 2,
        )
    )


def build_cable_clamp(p: dict[str, Any]) -> cq.Workplane:
    h = p["handle"]
    c = p["cable"]
    plate_x = h["length"]
    plate_height = h["height"] - 3.0
    plate_width = h["width"] - 2.0
    clamp = rounded_prism_x(
        plate_x,
        c["clamp_thickness"],
        plate_width,
        -plate_height / 2,
        plate_height,
        4.0,
    )

    usb_open = rounded_tunnel_x(
        plate_x - 0.5,
        c["clamp_thickness"] + 1.0,
        c["usb_opening_width"],
        c["usb_opening_height"],
        9.0,
        c["opening_radius"],
    )
    uart_open = rounded_tunnel_x(
        plate_x - 0.5,
        c["clamp_thickness"] + 1.0,
        c["uart_opening_width"],
        c["uart_opening_height"],
        -11.0,
        c["opening_radius"],
    )
    clamp = clamp.cut(usb_open).cut(uart_open)

    # Four inward legs enter the open rear of the two shell halves.
    for sign_y in (-1, 1):
        for zc in (-11.0, 11.0):
            leg_y = sign_y * (h["width"] / 2 - h["wall"] - 0.8)
            leg = box_xyz(plate_x - 7.0, 7.0, leg_y, 1.6, zc - 3.5, 7.0)
            bump_y = sign_y * (h["width"] / 2 - h["wall"] - 0.2)
            bump = box_xyz(plate_x - 6.5, 1.5, bump_y, 1.2, zc - 2.5, 5.0)
            clamp = clamp.union(leg).union(bump)
    return clamp


def build_fit_coupon(p: dict[str, Any]) -> cq.Shape:
    clearances = p["snap"]["fit_coupon_clearances"]
    female = box_xyz(0, 78, -10, 14, 0, 5)
    male = box_xyz(0, 78, 10, 14, 0, 3)
    tab_xs = (13.0, 39.0, 65.0)
    tab_w = 8.0
    tab_d = 4.0

    for x, clearance in zip(tab_xs, clearances):
        pocket = box_xyz(
            x - tab_w / 2 - clearance,
            tab_w + 2 * clearance,
            -10,
            tab_d + 2 * clearance,
            1.5,
            4.0,
        )
        female = female.cut(pocket)
        tab = box_xyz(x - tab_w / 2, tab_w, 10, tab_d, 3, 4.0)
        lead = box_xyz(x - tab_w / 2, tab_w, 10, tab_d + 0.8, 6.2, 0.8)
        male = male.union(tab).union(lead)

    bridge = box_xyz(0, 5, 0, 10, 0, 1.0)
    return female.union(male).union(bridge).clean().val()


def locked_needle(needle: cq.Shape, p: dict[str, Any]) -> cq.Shape:
    z = p["needle"]["axis_z"]
    return needle.rotate((0, 0, z), (1, 0, z), p["needle"]["lock_rotation_deg"])


def export_part(shape: cq.Shape, stem: str) -> None:
    cq.exporters.export(shape, str(OUTPUT_DIR / f"{stem}.step"))
    cq.exporters.export(
        shape,
        str(OUTPUT_DIR / f"{stem}.stl"),
        tolerance=0.05,
        angularTolerance=0.12,
    )


def drop_to_bed(shape: cq.Shape, margin_x: float = 0.0, margin_y: float = 0.0) -> cq.Shape:
    bb = shape.BoundingBox()
    return shape.translate(cq.Vector(margin_x - bb.xmin, margin_y - bb.ymin, -bb.zmin))


def export_print_ready_stls(
    lower: cq.Shape,
    upper: cq.Shape,
    needle: cq.Shape,
    clamp: cq.Shape,
    p: dict[str, Any],
) -> dict[str, cq.Shape]:
    lower_print = drop_to_bed(lower)
    upper_flipped = upper.rotate((0, 0, 0), (1, 0, 0), 180)
    upper_print = drop_to_bed(upper_flipped)

    needle_with_tip_support = add_tip_support_web(needle, p)
    needle_diagonal = needle_with_tip_support.rotate((0, 0, 0), (0, 0, 1), 45)
    needle_print = drop_to_bed(needle_diagonal, 5.0, 5.0)

    clamp_flat = clamp.rotate((0, 0, 0), (0, 1, 0), -90)
    clamp_print = drop_to_bed(clamp_flat)

    oriented = {
        "lower_chassis_print_ready": lower_print,
        "upper_cover_print_ready": upper_print,
        "printed_needle_a1mini_diagonal": needle_print,
        "cable_clamp_print_ready": clamp_print,
    }
    for stem, shape in oriented.items():
        cq.exporters.export(
            shape,
            str(OUTPUT_DIR / f"{stem}.stl"),
            tolerance=0.05,
            angularTolerance=0.12,
        )
    return oriented


def collision_volume(a: cq.Shape, b: cq.Shape) -> float:
    try:
        result = cq.Workplane(obj=a).intersect(cq.Workplane(obj=b))
        if not result.vals():
            return 0.0
        return sum(s.Volume() for s in result.vals())
    except Exception:
        return float("nan")


def render_preview(
    entries: Iterable[tuple[cq.Shape, str, float]],
    output_path: Path,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    elevation: float,
    azimuth: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for shape, color, alpha in entries:
        vertices, triangles = shape.tessellate(0.6)
        vertex_tuples = [v.toTuple() for v in vertices]
        faces = [[vertex_tuples[i] for i in triangle] for triangle in triangles]
        mesh = Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor=(0.12, 0.15, 0.18, min(alpha + 0.1, 1.0)),
            linewidth=0.08,
            alpha=alpha,
        )
        ax.add_collection3d(mesh)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("X / probe axis (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_parameter_report(
    p: dict[str, Any],
    parts: dict[str, cq.Shape],
    checks: dict[str, float],
) -> None:
    lines = [
        "# Generated CAD parameter report",
        "",
        "All dimensions are in millimetres. This model is for non-clinical phantom demonstrations only.",
        "",
        "## Core parameters",
        "",
        f"- Handle target envelope: {p['handle']['length']} x {p['handle']['width']} x {p['handle']['height']}",
        f"- Printed needle: diameter {p['needle']['diameter']}, exposed length {p['needle']['exposed_length']}",
        f"- Needle root transition: length {p['needle']['root_transition_length']}, max width {p['needle']['root_transition_max_width']}, modeled max thickness {p['needle']['root_transition_max_thickness']}",
        f"- Needle tip B: transition {p['needle']['tip_transition_length']}, neck diameter {p['needle']['tip_neck_diameter']}, bead diameter {p['needle']['tip_ball_diameter']}",
        f"- IMU envelope: {p['imu']['axial_length']} x {p['imu']['width']} x {p['imu']['total_height']}",
        f"- STP conservative minimum range: {p['stp23l']['conservative_min_range']}",
        f"- Initial/end target distance: {p['stp23l']['initial_target_distance']} / {p['stp23l']['end_target_distance']}",
        "",
        "## Generated solids",
        "",
        "| Part | Valid | Solids | X size | Y size | Z size | Volume (mm^3) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, shape in parts.items():
        bb = shape.BoundingBox()
        lines.append(
            f"| {name} | {shape.isValid()} | {len(shape.Solids())} | {bb.xlen:.3f} | {bb.ylen:.3f} | {bb.zlen:.3f} | {shape.Volume():.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interference checks",
            "",
            "Zero is expected for module and needle checks. The lower/upper shell check includes the intentionally interfering snap undercuts.",
            "",
        ]
    )
    for label, value in checks.items():
        rendered = "unavailable" if math.isnan(value) else f"{value:.6f} mm^3"
        lines.append(f"- {label}: {rendered}")

    lines.extend(
        [
            "",
            "## Fit coupon",
            "",
            "The three stations are ordered from front/left to rear/right as 0.20, 0.25, and 0.30 mm single-side clearance.",
            "Print the coupon before the complete housing and regenerate the model with the selected clearance.",
            "",
            "## SolidWorks",
            "",
            "Open `integrated_probe_assembly.step` for the full assembly and each individual STEP for editable imported bodies. Use Import Diagnostics, then save the parts as SLDPRT if desired.",
        ]
    )
    (OUTPUT_DIR / "parameter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = load_parameters()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stp = orient_vendor_stp(p)
    lower_wp, upper_wp = build_shells(p, stp)
    needle_wp = build_needle(p)
    imu_wp = build_imu_proxy(p)
    clamp_wp = build_cable_clamp(p)
    coupon = build_fit_coupon(p)

    lower = as_shape(lower_wp)
    upper = as_shape(upper_wp)
    needle = as_shape(needle_wp)
    needle_assembled = locked_needle(needle, p)
    imu = as_shape(imu_wp)
    clamp = as_shape(clamp_wp)

    required_single_solids = {
        "lower_chassis": lower,
        "upper_cover": upper,
        "printed_needle": needle,
        "cable_clamp": clamp,
        "fit_coupon": coupon,
        "imu_reference_envelope": imu,
        "stp23l_oriented_vendor": stp,
    }
    for name, shape in required_single_solids.items():
        solids = shape.Solids()
        if len(solids) != 1:
            raise RuntimeError(f"{name} must be one connected solid, got {len(solids)}")
        required_single_solids[name] = solids[0]

    lower = required_single_solids["lower_chassis"]
    upper = required_single_solids["upper_cover"]
    needle = required_single_solids["printed_needle"]
    clamp = required_single_solids["cable_clamp"]
    coupon = required_single_solids["fit_coupon"]
    imu = required_single_solids["imu_reference_envelope"]
    stp = required_single_solids["stp23l_oriented_vendor"]

    parts = {
        "lower_chassis": lower,
        "upper_cover": upper,
        "printed_needle": needle,
        "cable_clamp": clamp,
        "fit_coupon": coupon,
        "imu_reference_envelope": imu,
        "stp23l_oriented_vendor": stp,
    }
    for name, shape in parts.items():
        if not shape.isValid():
            raise RuntimeError(f"Invalid solid generated: {name}")

    export_part(lower, "lower_chassis")
    export_part(upper, "upper_cover")
    export_part(needle, "printed_needle")
    export_part(clamp, "cable_clamp")
    print_ready = export_print_ready_stls(lower, upper, needle, clamp, p)
    cq.exporters.export(
        coupon,
        str(OUTPUT_DIR / "fit_coupon.stl"),
        tolerance=0.05,
        angularTolerance=0.12,
    )
    cq.exporters.export(imu, str(OUTPUT_DIR / "imu_reference_envelope.step"))
    cq.exporters.export(stp, str(OUTPUT_DIR / "stp23l_oriented_vendor.step"))

    assembly_compound = cq.Compound.makeCompound(
        [lower, upper, needle_assembled, clamp, imu, stp]
    )
    cq.exporters.export(
        assembly_compound,
        str(OUTPUT_DIR / "integrated_probe_multibody.step"),
    )
    assembly = cq.Assembly(name="integrated_probe")
    assembly.add(lower, name="lower_chassis", color=cq.Color(0.45, 0.67, 0.82))
    assembly.add(upper, name="upper_cover", color=cq.Color(0.78, 0.86, 0.91, 0.45))
    assembly.add(needle_assembled, name="printed_needle", color=cq.Color(0.42, 0.46, 0.50))
    assembly.add(clamp, name="cable_clamp", color=cq.Color(0.88, 0.63, 0.28))
    assembly.add(imu, name="imu_reference_envelope", color=cq.Color(0.12, 0.13, 0.15))
    assembly.add(stp, name="stp23l_vendor", color=cq.Color(0.28, 0.55, 0.75))
    assembly.save(str(OUTPUT_DIR / "integrated_probe_assembly.step"), mode="default")
    cq.exporters.export(
        assembly_compound,
        str(OUTPUT_DIR / "integrated_probe_preview.stl"),
        tolerance=0.08,
        angularTolerance=0.15,
    )

    checks = {
        "lower chassis vs upper cover": collision_volume(lower, upper),
        "STP-23L vs lower chassis": collision_volume(stp, lower),
        "STP-23L vs upper cover": collision_volume(stp, upper),
        "IMU envelope vs upper cover": collision_volume(imu, upper),
        "locked needle vs shell assembly": collision_volume(
            needle_assembled,
            cq.Compound.makeCompound([lower, upper]),
        ),
    }
    beam_clearance = as_shape(
        rounded_tunnel_x(
            -10.0,
            p["stp23l"]["optical_face_x"] + 10.1,
            24.0,
            14.0,
            p["stp23l"]["center_z"],
            3.0,
        )
    )
    checks["optical beam envelope vs shell assembly"] = collision_volume(
        beam_clearance,
        cq.Compound.makeCompound([lower, upper]),
    )
    write_parameter_report(p, parts, checks)

    bed = p["printer"]["model"]
    needle_bed_box = print_ready["printed_needle_a1mini_diagonal"].BoundingBox()
    if needle_bed_box.xlen > 180 or needle_bed_box.ylen > 180:
        raise RuntimeError(
            f"Diagonal needle does not fit {bed}: {needle_bed_box.xlen:.2f} x {needle_bed_box.ylen:.2f}"
        )

    render_preview(
        [
            (lower, "#8fb3cc", 0.92),
            (upper, "#c9dbe7", 0.28),
            (needle_assembled, "#737b84", 1.0),
            (clamp, "#e2ad55", 0.9),
            (imu, "#25282c", 0.95),
            (stp, "#4e91bd", 0.95),
        ],
        OUTPUT_DIR / "assembly_preview.png",
        ((-205, 101), (-32, 32), (-28, 28)),
        24,
        -132,
    )
    render_preview(
        [
            (lower, "#8fb3cc", 0.78),
            (upper, "#c9dbe7", 0.12),
            (clamp, "#e2ad55", 0.75),
            (imu, "#25282c", 1.0),
            (stp, "#4e91bd", 1.0),
        ],
        OUTPUT_DIR / "handle_cutaway_preview.png",
        ((-2, 101), (-30, 30), (-27, 27)),
        22,
        -135,
    )
    render_preview(
        [
            (needle_assembled, "#59636d", 1.0),
            (lower, "#8fb3cc", 0.30),
            (upper, "#c9dbe7", 0.12),
        ],
        OUTPUT_DIR / "needle_root_closeup.png",
        ((-42, 12), (-18, 18), (-10, 20)),
        24,
        -128,
    )
    render_preview(
        [(needle, "#343c45", 1.0)],
        OUTPUT_DIR / "needle_tip_closeup.png",
        ((-203, -180), (-8, 8), (-2, 12)),
        18,
        -118,
    )

    print(f"Generated CAD outputs in: {OUTPUT_DIR}")
    for name, shape in parts.items():
        bb = shape.BoundingBox()
        print(
            f"{name}: valid={shape.isValid()} bbox="
            f"({bb.xlen:.3f}, {bb.ylen:.3f}, {bb.zlen:.3f}) volume={shape.Volume():.1f}"
        )
    for label, volume in checks.items():
        print(f"interference {label}: {volume:.6f} mm^3")


if __name__ == "__main__":
    main()
