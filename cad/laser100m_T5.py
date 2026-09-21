"""Build the PLA-friendly T5 housing and tapered ballpoint needle STEP package."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import zipfile

import cadquery as cq

ROOT = Path(__file__).resolve().parents[1]
CAD = Path(__file__).resolve().parent
OUT = ROOT / "output" / "cad_laser100m_T5_widened_clearance"
PARAM = CAD / "laser100m_T5_parameters.json"


def load_base_module():
    spec = importlib.util.spec_from_file_location("laser100m_T2_base", CAD / "laser100m_T2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_base_module()


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def solid(shape):
    value = shape.val() if isinstance(shape, cq.Workplane) else shape
    ss = value.Solids()
    if len(ss) != 1 or not value.isValid():
        raise ValueError(f"Expected one valid solid; solids={len(ss)}, valid={value.isValid()}")
    return ss[0]


def overlap(a, b):
    return sum(s.Volume() for s in a.intersect(b).Solids())


def build_parts(p):
    h, n, l, im = (p[k] for k in ("housing", "needle", "laser_reference", "imu_reference"))
    L, W, wall = h["length"], h["width"], h["wall"]
    bottom, shoulder, seam = h["bottom_z"], h["shoulder_z"], h["seam_z"]
    floor = bottom + wall
    outer = base.box(0, L, -W / 2, W / 2, bottom, shoulder).edges("|Z").fillet(h["corner_radius"])
    c = p["crown"]
    outer = outer.union(base.loft_rect((c["base_x"][0], c["base_x"][1], -c["base_width"] / 2, c["base_width"] / 2, c["base_z"]),
                                        (c["top_x"][0], c["top_x"][1], -c["top_width"] / 2, c["top_width"] / 2, c["top_z"]), c["radius"]))
    inner = base.box(wall, L - wall, -W / 2 + wall, W / 2 - wall, floor, shoulder - wall).edges("|Z").fillet(h["corner_radius"] - wall)
    inner = inner.union(base.loft_rect((18.6, 48.4, -14.4, 14.4, 6.7), (21.6, 45.4, -11.4, 11.4, 18.4), .9))
    shell = outer.cut(inner)
    lower = shell.intersect(base.box(-1, L + 1, -W, W, bottom - 1, seam))
    upper = shell.intersect(base.box(-1, L + 1, -W, W, seam + h["seam_clearance"], h["top_z"] + 1))
    lower = lower.cut(base.optical_tunnel(p, -1, l["front_x"] + 1))
    lower = lower.cut(base.box(-1, 2, -3.9, 3.9, .3, seam + 1))
    lower = lower.cut(base.box(L - 3, L + 1, -7, 7, -22, -11))
    usb_clear = base.box(L - 3, L + 1, -8, 8, 2.3, 7.8)
    lower = lower.cut(usb_clear)
    upper = upper.cut(usb_clear)
    upper = upper.cut(base.cyl_x(-1, 3, n["socket_radius"] + .25, n["axis_z"]))

    # Laser seats, rigid cover guides and broad adhesive landings. No spring hooks.
    for sign in (-1, 1):
        lower = lower.union(base.mirrored_box(4, 51, 18.8, 20.6, floor, -6.5, sign))
        for x in (8, 40):
            lower = lower.union(base.mirrored_box(x, x + 9, 13, 17, floor, l["bottom_z"], sign))
        lower = lower.union(base.mirrored_box(2.2, 3.7, 13, 20.6, floor, -22.8, sign))
        lower = lower.union(base.mirrored_box(50.77, 52.37, 13, 20.6, floor, -4, sign))
        lower = lower.union(base.mirrored_box(3.5, 6.5, 19.8, 21.5, seam - .2, 7.7, sign))
        lower = lower.union(base.mirrored_box(51.5, 54.5, 19.8, 21.5, seam - .2, 7.7, sign))
        for x in (8, 25, 42):
            lower = lower.union(base.mirrored_box(x, x + 8, 19.8, 21.4, seam - .5, seam, sign))
        upper = upper.cut(base.mirrored_box(3.1, 6.9, 19.45, 21.8, seam + h["seam_clearance"], 7.9, sign))
        upper = upper.cut(base.mirrored_box(51.1, 54.9, 19.45, 21.8, seam + h["seam_clearance"], 7.9, sign))
        upper = upper.union(base.mirrored_box(8, 16, 19.8, 21.4, seam + h["seam_clearance"], 6.8, sign))
        upper = upper.union(base.mirrored_box(42, 50, 19.8, 21.4, seam + h["seam_clearance"], 6.8, sign))

    # Carrier keeps the sensor supports, but removes all cantilever latch teeth.
    carrier = base.box(8.5, 55, -19, 19, .2, 1.8)
    carrier = carrier.cut(base.box(17, 50, -12, 12, -1, 3))
    carrier = carrier.union(base.box(6, 14, -4, 4, .2, 1.8))
    carrier = carrier.union(base.cyl_x(n["exit_x"], n["insertion"], n["socket_radius"], n["axis_z"]))
    carrier = carrier.cut(base.cyl_x(n["exit_x"] - 1, n["insertion"] + 1.2, n["socket_diameter"] / 2, n["axis_z"]))
    laser_top = l["bottom_z"] + l["height"]
    for sign in (-1, 1):
        for x in (14, 43):
            carrier = carrier.union(base.mirrored_box(x, x + 6, 14, 17, laser_top, .3, sign))
        carrier = carrier.union(base.mirrored_box(10, 11.7, 0, 16.7, 1.8, 3.8, sign))
        carrier = carrier.union(base.mirrored_box(54.3, 55.5, 11, 17, 1.8, 3.8, sign))
        carrier = carrier.union(base.mirrored_box(15, 18, 15.3, 16.7, 1.8, 3.6, sign))
        carrier = carrier.union(base.mirrored_box(17, 31, 15.3, 16.7, 2.05, 3.6, sign))
        carrier = carrier.union(base.mirrored_box(28, 31, 14.5, 16.7, 3.45, 4.5, sign))
        carrier = carrier.union(base.mirrored_box(14, 22, 16.5, 18.5, 1.8, 3.4, sign))
        carrier = carrier.union(base.mirrored_box(40, 48, 16.5, 18.5, 1.8, 3.4, sign))

    ix, iy, iz = im["front_x"], im["width"] / 2, im["bottom_z"]
    pcb_top = iz + im["pcb_thickness_assumed"]
    imu = base.box(ix, ix + im["length"], -iy, iy, iz, pcb_top)
    imu = imu.union(base.box(ix + 10, ix + 32, -10, 10, pcb_top, iz + im["height"]))
    imu = solid(imu)
    laser_full = solid(base.box(l["front_x"], l["front_x"] + l["length"], -l["width"] / 2, l["width"] / 2, l["bottom_z"], laser_top))
    laser = cq.Workplane(obj=laser_full)
    for y, r in ((0, 7), (-11.35, 2.2)):
        laser = laser.cut(cq.Workplane("YZ", origin=(l["front_x"] - .1, y, -9.35)).circle(r).extrude(2.1))
    return {
        "01_lower_chassis": solid(lower),
        "02_upper_cover": solid(upper),
        "03_sensor_carrier": solid(carrier),
        "REF_05_HWT906P_approximate": imu,
        "REF_06_laser100m_envelope": solid(laser),
    }, {
        "laser_full_envelope": laser_full,
        "imu_full_envelope": solid(base.box(ix, ix + im["length"], -iy, iy, iz, iz + im["height"])),
        "optical_corridor": solid(base.optical_tunnel(p, -6, l["front_x"] - .01)),
        "laser_top_insertion_sweep": solid(base.box(4, 50.47, -18.5, 18.5, -25.1, 34)),
    }


def build_needle(p):
    n = p["needle"]
    tip, axis_z = n["exit_x"] - n["exposed_length"], n["axis_z"]
    shaft_r, insert_r, ball_r = 2.5, 2.0, 2.2
    ball_center, transition, join = tip + ball_r, tip + n["exposed_length"] / 2, .2
    def cone_x(x, length, r1, r2):
        return cq.Workplane(obj=cq.Solid.makeCone(r1, r2, length, cq.Vector(x, 0, axis_z), cq.Vector(1, 0, 0)))
    rod = cq.Workplane("XY").sphere(ball_r).translate((ball_center, 0, axis_z))
    rod = rod.union(cone_x(ball_center - join, transition - (ball_center - join), ball_r * .995, shaft_r))
    rod = rod.union(base.cyl_x(transition - join, -24 - (transition - join), shaft_r, axis_z))
    rod = rod.union(cone_x(-24 - join, 14.2, shaft_r, 4.25))
    rod = rod.union(base.cyl_x(-10, 4.2, 4.25, axis_z))
    rod = rod.union(cone_x(-6, 3.0, 4.25, insert_r))
    rod = rod.union(base.cyl_x(-3.2, n["insertion"] + join, insert_r, axis_z))
    return solid(rod)


def exact_bounds(shape):
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    b = Bnd_Box(); BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    return b.Get()


def dims(shape):
    b = exact_bounds(shape)
    return [round(b[i + 3] - b[i], 6) for i in range(3)]


def roundtrip(path, source, expected_solids):
    imported = cq.importers.importStep(str(path)).val()
    delta = abs(imported.Volume() - source.Volume())
    bd = max(abs(a - b) for a, b in zip(exact_bounds(imported), exact_bounds(source)))
    if not imported.isValid() or len(imported.Solids()) != expected_solids or delta > max(1e-4, source.Volume() * 1e-8) or bd > 1e-5:
        raise RuntimeError(f"STEP reimport failed: {path.name}; valid={imported.isValid()}; solids={len(imported.Solids())}; volume_delta={delta}; bbox_delta={bd}")
    return imported, {"valid": True, "solid_count": expected_solids, "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def check_mesh(path, shape):
    import trimesh
    cq.exporters.export(shape, str(path), tolerance=.003 if "rod" in path.name else .03, angularTolerance=.03 if "rod" in path.name else .08)
    mesh = trimesh.load_mesh(str(path))
    if "rod" in path.name:
        mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices(); mesh.export(str(path)); mesh = trimesh.load_mesh(str(path))
    result = {"watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent), "components": len(mesh.split())}
    if not result["watertight"] or not result["winding_consistent"] or result["components"] != 1:
        raise RuntimeError(f"Invalid STL: {path.name}: {result}")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    overrides = json.loads(PARAM.read_text(encoding="utf-8"))
    p = merge(base_params, overrides)
    parts, env = build_parts(p)
    parts["REF_04_rod_d5_L212_ballpoint"] = build_needle(p)
    ordered = {
        "01_lower_chassis": parts["01_lower_chassis"], "02_upper_cover": parts["02_upper_cover"],
        "03_sensor_carrier": parts["03_sensor_carrier"], "REF_04_rod_d5_L212_ballpoint": parts["REF_04_rod_d5_L212_ballpoint"],
        "REF_05_HWT906P_approximate": parts["REF_05_HWT906P_approximate"], "REF_06_laser100m_envelope": parts["REF_06_laser100m_envelope"],
    }
    parts = ordered
    structural = {k: v for k, v in parts.items() if k.startswith(("01_", "02_", "03_"))}
    checks = {}
    for (an, a), (bn, b) in itertools.combinations(parts.items(), 2):
        ov = overlap(a, b)
        if ov > 1e-5: raise RuntimeError(f"Unexpected overlap {an}/{bn}: {ov}")
        checks[f"{an} / {bn}"] = round(ov, 8)
    for name, shape in structural.items():
        for label, ref in (("laser_full", env["laser_full_envelope"]), ("optical", env["optical_corridor"])):
            ov = overlap(ref, shape)
            if ov > 1e-5: raise RuntimeError(f"{label}/{name}: {ov}")
            checks[f"{label} / {name}"] = round(ov, 8)
    for name in ("01_lower_chassis", "02_upper_cover", "03_sensor_carrier"):
        ov = overlap(parts["REF_04_rod_d5_L212_ballpoint"], parts[name])
        if ov > 1e-5: raise RuntimeError(f"needle/{name}: {ov}")
    checks["laser_vertical_insertion_without_carrier"] = round(overlap(env["laser_top_insertion_sweep"], parts["01_lower_chassis"]), 8)

    rows, roundtrips, meshes = {}, {}, {}
    for name, shape in parts.items():
        rows[name] = {"valid": shape.isValid(), "solid_count": len(shape.Solids()), "dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 6)}
        path = OUT / f"{name}.step"; cq.exporters.export(shape, str(path)); _, roundtrips[path.name] = roundtrip(path, shape, 1)
        meshes[name] = check_mesh(OUT / f"{name}.stl", shape)
    coupon = base.coupon(p); cq.exporters.export(coupon, str(OUT / "needle_socket_fit_coupon.step")); _, roundtrips["needle_socket_fit_coupon.step"] = roundtrip(OUT / "needle_socket_fit_coupon.step", coupon, 1)
    meshes["needle_socket_fit_coupon"] = check_mesh(OUT / "needle_socket_fit_coupon.stl", coupon)

    compound = cq.Compound.makeCompound(list(parts.values())); printed = cq.Compound.makeCompound(list(structural.values()))
    palette = [(.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.60, .67, .70), (.19, .59, .43), (.38, .40, .42)]
    assembly = cq.Assembly(name="T5_PLA_FRIENDLY_WIDENED_CLEARANCE")
    shell_assembly = cq.Assembly(name="T5_THREE_PRINTED_COMPONENTS")
    exploded = cq.Assembly(name="T5_EXPLODED_REFERENCE")
    exploded_shapes = []
    for (name, shape), color in zip(parts.items(), palette):
        assembly.add(shape, name=name, color=cq.Color(*color))
        if name in structural: shell_assembly.add(shape, name=name, color=cq.Color(*color))
        dz = 62 if name == "02_upper_cover" else (32 if name in ("03_sensor_carrier", "REF_04_rod_d5_L212_ballpoint") else 0)
        shifted = shape.translate((0, 0, dz)); exploded_shapes.append(shifted); exploded.add(shifted, name=name, color=cq.Color(*color))
    assembly.export(str(OUT / "laser100m_HWT906P_T5_assembly.step")); shell_assembly.export(str(OUT / "T5_shell_only.step")); exploded.export(str(OUT / "T5_exploded_assembly.step"))
    cq.exporters.export(compound, str(OUT / "laser100m_HWT906P_T5_multibody.step"))
    _, roundtrips["laser100m_HWT906P_T5_assembly.step"] = roundtrip(OUT / "laser100m_HWT906P_T5_assembly.step", compound, 6)
    _, roundtrips["T5_shell_only.step"] = roundtrip(OUT / "T5_shell_only.step", printed, 3)
    _, roundtrips["laser100m_HWT906P_T5_multibody.step"] = roundtrip(OUT / "laser100m_HWT906P_T5_multibody.step", compound, 6)
    exploded_compound = cq.Compound.makeCompound(exploded_shapes)
    _, roundtrips["T5_exploded_assembly.step"] = roundtrip(OUT / "T5_exploded_assembly.step", exploded_compound, 6)

    distances = {"setback_from_rod_exit_mm": 7.0, "initial_geometric_distance_mm": 207.0, "distance_after_150mm_stroke_mm": 57.0, "near_range_verified": False}
    report = {
        "revision": p["revision"], "housing_envelope_mm": [58.0, 46.0, 47.0], "structure_including_socket_envelope_mm": dims(printed),
        "parts": rows, "zero_overlap_checks_mm3": checks, "step_roundtrip": roundtrips, "stl_checks": meshes,
        "pla_clip_design": {"elastic_snap_hooks": False, "rigid_locator_posts": True, "adhesive_landings": True, "inner_clearance_added_each_side_mm": 1.0, "wall_nominal_mm": 1.6},
        "needle_profile": {"root_body_diameter_mm": 5.0, "root_reinforcement_max_diameter_mm": 8.5, "ball_tip_diameter_mm": 4.4, "exposed_length_mm": 200.0},
        "geometric_distances": distances,
        "limitations": ["Inner clearance and connector orientation still require physical fit check.", "PLA guide posts and adhesive landings have no mechanical retention test.", "The 57 mm near-range distance is geometric only and hardware near-range validity is unverified."],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# T5 PLA 友好加宽内腔 STEP

本版按用户确认的方案导出：外壳宽度 46 mm，内腔左右各向外放宽 1 mm，名义壁厚保持 1.6 mm。外盖和中间载架取消易碎弹性卡扣，改为刚性导向柱、封闭定位袋和宽胶粘定位面；中间载架可直接落座，必要时点胶。

针体外露 200 mm，主体 Ø5 mm；根部渐变加固，最大约 Ø8.5 mm；针尖为约 Ø4.4 mm 的圆珠笔头式圆球。针体插入段仍为 Ø4 mm × 12 mm。

主要文件：`laser100m_HWT906P_T5_assembly.step`、`T5_shell_only.step`、`T5_exploded_assembly.step`、`laser100m_HWT906P_T5_multibody.step`，以及各单件 STEP/STL。`validation.json` 记录重导入、干涉和网格检查，`parameters.json` 记录完整参数。

这是 PLA 试配版本，不是经过强度、卡扣寿命、连接器净空和实物装配验证的最终量产件。近端 57 mm 仍只是几何推算。
"""
    (OUT / "README_交付说明.md").write_text(readme, encoding="utf-8")
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = ROOT / "output" / "laser100m_HWT906P_T5_STEP_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file(): z.write(f, f"{OUT.name}/{f.name}")
        z.write(Path(__file__), "source/laser100m_T5.py"); z.write(PARAM, "source/laser100m_T5_parameters.json")
        z.write(CAD / "laser100m_T2.py", "source/laser100m_T2.py"); z.write(CAD / "laser100m_T2_parameters.json", "source/laser100m_T2_parameters.json")
    print(json.dumps({"output": str(OUT), "assembly": str(OUT / "laser100m_HWT906P_T5_assembly.step"), "zip": str(bundle), "housing_width_mm": 46.0, "wall_mm": 1.6, "needle_exposed_mm": 200.0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
