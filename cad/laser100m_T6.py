"""Build T6: PLA-printable three-piece needle with alignment pins and glue joints."""
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
OUT = ROOT / "output" / "cad_laser100m_T6_segmented_needle"
PARAM = CAD / "laser100m_T6_parameters.json"


def load_t5():
    spec = importlib.util.spec_from_file_location("laser100m_T5_base", CAD / "laser100m_T5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = load_t5()


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


def cone_x(x, length, r1, r2, z):
    return cq.Workplane(obj=cq.Solid.makeCone(r1, r2, length, cq.Vector(x, 0, z), cq.Vector(1, 0, 0)))


def needle_segments(p):
    n = p["needle"]
    z = n["axis_z"]
    shaft_r = p["needle_profile"]["body_diameter_mm"] / 2
    pin_r = p["needle_profile"]["alignment_pin_diameter_mm"] / 2
    hole_r = p["needle_profile"]["alignment_hole_diameter_mm"] / 2
    x_root_end, x_mid_end = p["needle_profile"]["joint_x_mm"]
    insertion_end = n["exit_x"] + n["insertion"]

    # Root segment: insertion tenon, reinforced shoulder, and the first shaft section.
    root = t5.base.cyl_x(x_root_end, (-24.0 - x_root_end) + .2, shaft_r, z)
    root = root.union(cone_x(-24.2, 14.4, shaft_r, 4.25, z))
    root = root.union(t5.base.cyl_x(-10.0, 4.2, 4.25, z))
    root = root.union(cone_x(-6.2, 3.2, 4.25, n["socket_diameter"] / 2 - .1, z))
    root = root.union(t5.base.cyl_x(-3.2, insertion_end - (-3.2) + .2, n["socket_diameter"] / 2 - .1, z))
    root = root.cut(t5.base.cyl_x(x_root_end - .2, p["needle_profile"]["joint_depth_mm"] + .2, hole_r, z))

    # Middle segment: Ø5 shaft with a blind alignment socket at each end.
    mid = t5.base.cyl_x(x_mid_end, x_root_end - x_mid_end, shaft_r, z)
    depth = p["needle_profile"]["joint_depth_mm"]
    mid = mid.cut(t5.base.cyl_x(x_root_end - depth, depth + .2, hole_r, z))
    mid = mid.cut(t5.base.cyl_x(x_mid_end - .2, depth + .2, hole_r, z))

    # Tip segment: ballpoint-like rounded nose and a long shallow taper.
    tip = n["exit_x"] - n["exposed_length"]
    ball_r = p["needle_profile"]["ball_diameter_mm"] / 2
    ball_center = tip + ball_r
    tip = cq.Workplane("XY").sphere(ball_r).translate((ball_center, 0, z))
    tip = tip.union(cone_x(ball_center - .2, x_mid_end - (ball_center - .2), ball_r * .995, shaft_r, z))
    tip = tip.cut(t5.base.cyl_x(x_mid_end - depth - .2, depth + .2, hole_r, z))

    # Separate dowels are intentionally smaller than the blind holes for PLA fit.
    pin1 = t5.base.cyl_x(x_root_end - depth, 2 * depth, pin_r, z)
    pin2 = t5.base.cyl_x(x_mid_end - depth, 2 * depth, pin_r, z)
    return {
        "REF_04A_needle_root_segment": solid(root),
        "REF_04B_needle_mid_segment": solid(mid),
        "REF_04C_needle_tip_segment": solid(tip),
        "REF_04D_alignment_pin_joint_1": solid(pin1),
        "REF_04E_alignment_pin_joint_2": solid(pin2),
    }


def exact_bounds(shape):
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    b = Bnd_Box(); BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    return b.Get()


def dims(shape):
    b = exact_bounds(shape)
    return [round(b[i + 3] - b[i], 6) for i in range(3)]


def overlap(a, b):
    return sum(s.Volume() for s in a.intersect(b).Solids())


def roundtrip(path, source, expected_solids):
    imported = cq.importers.importStep(str(path)).val()
    delta = abs(imported.Volume() - source.Volume())
    bd = max(abs(a - b) for a, b in zip(exact_bounds(imported), exact_bounds(source)))
    if not imported.isValid() or len(imported.Solids()) != expected_solids or delta > max(1e-4, source.Volume() * 1e-8) or bd > 1e-5:
        raise RuntimeError(f"STEP reimport failed: {path.name}; valid={imported.isValid()}; solids={len(imported.Solids())}; volume_delta={delta}; bbox_delta={bd}")
    return imported, {"valid": True, "solid_count": expected_solids, "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def check_mesh(path, shape):
    import trimesh
    cq.exporters.export(shape, str(path), tolerance=.003 if "needle" in path.name or "pin" in path.name else .03, angularTolerance=.03 if "needle" in path.name or "pin" in path.name else .08)
    mesh = trimesh.load_mesh(str(path))
    if "needle" in path.name or "pin" in path.name:
        mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices(); mesh.export(str(path)); mesh = trimesh.load_mesh(str(path))
    result = {"watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent), "components": len(mesh.split())}
    if not result["watertight"] or not result["winding_consistent"] or result["components"] != 1:
        raise RuntimeError(f"Invalid STL: {path.name}: {result}")
    return result


def render_preview(parts):
    import vtk
    from PIL import Image, ImageDraw, ImageFont
    colors = [(0.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.60, .67, .70), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42)]
    names = list(parts)
    def frame(path, selected, size, camera, target, scale, opacity=None):
        ren = vtk.vtkRenderer(); ren.SetBackground(.953, .965, .973); opacity = opacity or {}
        for idx, name in enumerate(names):
            if name not in selected: continue
            verts, tris = parts[name].tessellate(.06)
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for v in verts: pts.InsertNextPoint(*v.toTuple())
            for tri in tris:
                cell = vtk.vtkTriangle()
                for j, k in enumerate(tri): cell.GetPointIds().SetId(j, k)
                cells.InsertNextCell(cell)
            poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(cells)
            norm = vtk.vtkPolyDataNormals(); norm.SetInputData(poly); norm.SetFeatureAngle(40); norm.Update()
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(norm.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*colors[idx]); actor.GetProperty().SetOpacity(opacity.get(name, 1.0)); ren.AddActor(actor)
        win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(*size); win.AddRenderer(ren)
        cam = ren.GetActiveCamera(); cam.SetPosition(*camera); cam.SetFocalPoint(*target); cam.SetViewUp(0, 0, 1); cam.ParallelProjectionOn(); cam.SetParallelScale(scale); ren.ResetCameraClippingRange(); win.Render()
        cap = vtk.vtkWindowToImageFilter(); cap.SetInput(win); cap.SetInputBufferTypeToRGB(); cap.ReadFrontBufferOff(); cap.Update()
        writer = vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(cap.GetOutputPort()); writer.Write(); win.Finalize()
    frame(OUT / "T6_assembly_preview.png", names, (1800, 850), (-300, -360, 220), (-95, 0, 0), 75)
    exploded = {}
    for name, shape in parts.items():
        dz = 62 if name == "02_upper_cover" else (32 if name == "03_sensor_carrier" else (18 if name.startswith("REF_04") else 0))
        exploded[name] = shape.translate((0, 0, dz))
    original = dict(parts); parts.clear(); parts.update(exploded)
    frame(OUT / "T6_segmented_needle_exploded.png", [n for n in names if n.startswith("REF_04") or n.startswith("01_") or n.startswith("02_") or n.startswith("03_")], (1500, 900), (-110, -170, 130), (-30, 0, 8), 70, {"01_lower_chassis": .18, "02_upper_cover": .20, "03_sensor_carrier": .25})
    parts.clear(); parts.update(original)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    def font(n): return ImageFont.truetype(font_path, n)
    with Image.open(OUT / "T6_assembly_preview.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248)); c.paste(im, (0, 65)); d = ImageDraw.Draw(c)
        d.text((900, 24), "T6 | 三段式 PLA 针体 + 定位销 + 胶粘", font=font(30), fill="#163e50", anchor="mt")
        d.text((900, 812), "外露 200 mm | 主体 Ø5 mm | 三段水平打印后胶粘校直", font=font(23), fill="#29424f", anchor="mt")
        c.save(OUT / "T6_assembly_preview_labeled.png")
    with Image.open(OUT / "T6_segmented_needle_exploded.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248)); c.paste(im, (0, 65)); d = ImageDraw.Draw(c)
        d.text((750, 24), "T6 针体分段与定位销", font=font(29), fill="#163e50", anchor="mt")
        d.text((750, 865), "根部段 / 中段 / 针尖段；两处 Ø3 mm 定位销，胶粘后形成一根直针", font=font(21), fill="#29424f", anchor="mt")
        c.save(OUT / "T6_segmented_needle_exploded_labeled.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    p = merge(base_params, json.loads(PARAM.read_text(encoding="utf-8")))
    shells, env = t5.build_parts(p)
    needles = needle_segments(p)
    parts = {"01_lower_chassis": shells["01_lower_chassis"], "02_upper_cover": shells["02_upper_cover"], "03_sensor_carrier": shells["03_sensor_carrier"], **needles, "REF_05_HWT906P_approximate": shells["REF_05_HWT906P_approximate"], "REF_06_laser100m_envelope": shells["REF_06_laser100m_envelope"]}
    structural = {k: v for k, v in parts.items() if k.startswith(("01_", "02_", "03_"))}
    checks = {}
    for (an, a), (bn, b) in itertools.combinations(parts.items(), 2):
        ov = overlap(a, b)
        # Sensor references are proxies; the print parts themselves must remain clear.
        if ov > 1e-5 and not (an.startswith("REF_05") or bn.startswith("REF_05") or an.startswith("REF_06") or bn.startswith("REF_06")):
            raise RuntimeError(f"Unexpected overlap {an}/{bn}: {ov}")
        checks[f"{an} / {bn}"] = round(ov, 8)
    for name, shape in structural.items():
        for label, ref in (("laser_full", env["laser_full_envelope"]), ("optical", env["optical_corridor"])):
            ov = overlap(ref, shape)
            if ov > 1e-5: raise RuntimeError(f"{label}/{name}: {ov}")
            checks[f"{label} / {name}"] = round(ov, 8)
    rows, rts, meshes = {}, {}, {}
    for name, shape in parts.items():
        rows[name] = {"valid": shape.isValid(), "solid_count": len(shape.Solids()), "dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 6)}
        path = OUT / f"{name}.step"; cq.exporters.export(shape, str(path)); _, rts[path.name] = roundtrip(path, shape, 1)
        meshes[name] = check_mesh(OUT / f"{name}.stl", shape)
    printed_needle = cq.Compound.makeCompound([needles[k] for k in needles])
    cq.exporters.export(printed_needle, str(OUT / "T6_needle_print_set.step")); _, rts["T6_needle_print_set.step"] = roundtrip(OUT / "T6_needle_print_set.step", printed_needle, 5)
    printed_shell = cq.Compound.makeCompound(list(structural.values()))
    compound = cq.Compound.makeCompound(list(parts.values()))
    palette = [(.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42)]
    assembly = cq.Assembly(name="T6_PLA_SEGMENTED_NEEDLE")
    shell_assembly = cq.Assembly(name="T6_THREE_PRINTED_HOUSING_COMPONENTS")
    exploded = cq.Assembly(name="T6_EXPLODED_SEGMENTED_NEEDLE")
    exploded_shapes = []
    for (name, shape), color in zip(parts.items(), palette):
        assembly.add(shape, name=name, color=cq.Color(*color))
        if name in structural: shell_assembly.add(shape, name=name, color=cq.Color(*color))
        dz = 62 if name == "02_upper_cover" else (32 if name == "03_sensor_carrier" else (18 if name.startswith("REF_04") else 0))
        shifted = shape.translate((0, 0, dz)); exploded_shapes.append(shifted); exploded.add(shifted, name=name, color=cq.Color(*color))
    assembly.export(str(OUT / "laser100m_HWT906P_T6_assembly.step")); shell_assembly.export(str(OUT / "T6_shell_only.step")); exploded.export(str(OUT / "T6_exploded_assembly.step")); cq.exporters.export(compound, str(OUT / "laser100m_HWT906P_T6_multibody.step"))
    _, rts["laser100m_HWT906P_T6_assembly.step"] = roundtrip(OUT / "laser100m_HWT906P_T6_assembly.step", compound, len(parts))
    _, rts["T6_shell_only.step"] = roundtrip(OUT / "T6_shell_only.step", printed_shell, 3)
    _, rts["T6_exploded_assembly.step"] = roundtrip(OUT / "T6_exploded_assembly.step", cq.Compound.makeCompound(exploded_shapes), len(parts))
    _, rts["laser100m_HWT906P_T6_multibody.step"] = roundtrip(OUT / "laser100m_HWT906P_T6_multibody.step", compound, len(parts))
    distances = {"setback_from_rod_exit_mm": 7.0, "initial_geometric_distance_mm": 207.0, "distance_after_150mm_stroke_mm": 57.0, "near_range_verified": False}
    report = {"revision": p["revision"], "housing_envelope_mm": [58.0, 46.0, 47.0], "parts": rows, "step_roundtrip": rts, "stl_checks": meshes, "zero_overlap_checks_mm3": checks, "pla_clip_design": p["pla_clip_design"], "needle_profile": p["needle_profile"], "geometric_distances": distances, "limitations": ["Three needle segments are intended for horizontal PLA printing and epoxy/CA assembly; print orientation and adhesive cure need physical trial.", "Alignment pin holes use a nominal 0.6 mm diameter clearance; printer shrinkage may require the fit coupon.", "Sensor outlines, connector clearance, PLA strength and repeated bending have not been physically validated.", "The 57 mm near-range distance is geometric only and hardware near-range validity is unverified."]}
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# T6 PLA 三段式针体 STEP

本版采用三段式 PLA 针体：根部段、中段、针尖段，三段分别水平打印，再用两根 Ø3 mm 定位销校直并胶粘。外露长度 200 mm，主体 Ø5 mm，根部最大约 Ø8.5 mm，针尖为约 Ø4.4 mm 圆球。针体插入段仍为 Ø4 mm × 12 mm。

外壳沿用 T5：宽度 46 mm，内腔左右各放宽 1 mm，壁厚约 1.6 mm；外盖和中间载架采用刚性导向及胶粘定位，不使用 PLA 弹性卡扣。

打印件入口：`REF_04A_needle_root_segment.stl`、`REF_04B_needle_mid_segment.stl`、`REF_04C_needle_tip_segment.stl`、`REF_04D_alignment_pin_joint_1.stl`、`REF_04E_alignment_pin_joint_2.stl`。整套针体也可用 `T6_needle_print_set.step` 导入切片软件。

建议先打印定位销和两处接头试配，再打印三段针体；孔径按实际打印机调整。三段接缝用环氧胶或瞬干胶固定，固化时用直尺、玻璃板或简易 V 形槽校直。不要把装配后的长针竖直打印。

这是 PLA 试配版本，尚未验证强度、胶接寿命和实际打印参数。`validation.json` 记录了 STEP 重导入、干涉和 STL 网格检查。
"""
    (OUT / "README_交付说明.md").write_text(readme, encoding="utf-8")
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_preview(parts)
    bundle = ROOT / "output" / "laser100m_HWT906P_T6_STEP_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file(): z.write(f, f"{OUT.name}/{f.name}")
        for src in (Path(__file__), PARAM, CAD / "laser100m_T5.py", CAD / "laser100m_T2.py", CAD / "laser100m_T2_parameters.json"):
            z.write(src, f"source/{src.name}")
    print(json.dumps({"output": str(OUT), "assembly": str(OUT / "laser100m_HWT906P_T6_assembly.step"), "needle_print_set": str(OUT / "T6_needle_print_set.step"), "zip": str(bundle), "segment_count": 3, "alignment_pin_diameter_mm": 3.0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
