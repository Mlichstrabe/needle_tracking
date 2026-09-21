"""Build T8: T7 needle geometry plus a slicer-safe separated print layout."""
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
OUT = ROOT / "output" / "cad_laser100m_T8_separated_print_layout"
PARAM = CAD / "laser100m_T8_print_layout_parameters.json"


def load_t7():
    spec = importlib.util.spec_from_file_location("laser100m_T7_base", CAD / "laser100m_T7.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t7 = load_t7()


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def exact_bounds(shape):
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
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
    return {"valid": True, "solid_count": expected_solids, "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def place_on_bed(shape, x_start, y_center):
    """Translate a solid so its minimum Z is zero and its row is isolated."""
    b = exact_bounds(shape)
    shape_y_center = (b[1] + b[4]) / 2
    return shape.translate((x_start - b[0], y_center - shape_y_center, -b[2]))


def render_layout(layout):
    import vtk
    from PIL import Image, ImageDraw, ImageFont

    colors = [(0.56, .62, .66), (.43, .50, .55), (.31, .42, .48), (.19, .59, .43), (.19, .59, .43)]
    names = list(layout)

    def frame(path, camera, target, scale, size=(1500, 850)):
        ren = vtk.vtkRenderer()
        ren.SetBackground(.953, .965, .973)
        for idx, name in enumerate(names):
            verts, tris = layout[name].tessellate(.04)
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for v in verts:
                pts.InsertNextPoint(*v.toTuple())
            for tri in tris:
                cell = vtk.vtkTriangle()
                for j, k in enumerate(tri):
                    cell.GetPointIds().SetId(j, k)
                cells.InsertNextCell(cell)
            poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(cells)
            norm = vtk.vtkPolyDataNormals(); norm.SetInputData(poly); norm.SetFeatureAngle(40); norm.Update()
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(norm.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*colors[idx]); ren.AddActor(actor)
        win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(*size); win.AddRenderer(ren)
        cam = ren.GetActiveCamera(); cam.SetPosition(*camera); cam.SetFocalPoint(*target); cam.SetViewUp(0, 0, 1); cam.ParallelProjectionOn(); cam.SetParallelScale(scale); ren.ResetCameraClippingRange(); win.Render()
        cap = vtk.vtkWindowToImageFilter(); cap.SetInput(win); cap.SetInputBufferTypeToRGB(); cap.ReadFrontBufferOff(); cap.Update()
        writer = vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(cap.GetOutputPort()); writer.Write(); win.Finalize()

    frame(OUT / "T8_print_layout_preview.png", (65, -150, 180), (65, 0, 0), 110)
    frame(OUT / "T8_print_layout_top_preview.png", (65, -5, 260), (65, 0, 0), 110)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    def font(n): return ImageFont.truetype(font_path, n)
    with Image.open(OUT / "T8_print_layout_preview.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248)); c.paste(im, (0, 65)); d = ImageDraw.Draw(c)
        d.text((750, 24), "T8 | 三段针体独立排版，平面朝下", font=font(30), fill="#163e50", anchor="mt")
        d.text((750, 812), "根部 / 中段 / 针尖分列打印；定位销单独放置，避免整根 212 mm 长条跨床", font=font(22), fill="#29424f", anchor="mt")
        c.save(OUT / "T8_print_layout_preview_labeled.png")
    with Image.open(OUT / "T8_print_layout_top_preview.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248)); c.paste(im, (0, 65)); d = ImageDraw.Draw(c)
        d.text((750, 24), "T8 打印排版俯视图", font=font(30), fill="#163e50", anchor="mt")
        d.text((750, 812), "每个针体最大长度 88 mm，行间留有间距；不要加载总装 STEP 切片", font=font(22), fill="#29424f", anchor="mt")
        c.save(OUT / "T8_print_layout_top_preview_labeled.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    t7_params = json.loads((CAD / "laser100m_T7_parameters.json").read_text(encoding="utf-8"))
    layout_params = json.loads(PARAM.read_text(encoding="utf-8"))
    p = merge(merge(base_params, t7_params), layout_params)
    shells, env = t7.t6.t5.build_parts(p)
    needles, flat_z = t7.needle_segments(p)
    body_names = ["REF_04A_needle_root_segment", "REF_04B_needle_mid_segment", "REF_04C_needle_tip_segment"]
    pin_names = ["REF_04D_alignment_pin_joint_1", "REF_04E_alignment_pin_joint_2"]
    rows = p["print_layout"]["body_y_centers_mm"]
    layout = {
        "T8_PRINT_root_segment": place_on_bed(needles[body_names[0]], p["print_layout"]["body_x_start_mm"], rows["root"]),
        "T8_PRINT_mid_segment": place_on_bed(needles[body_names[1]], p["print_layout"]["body_x_start_mm"], rows["mid"]),
        "T8_PRINT_tip_segment": place_on_bed(needles[body_names[2]], p["print_layout"]["body_x_start_mm"], rows["tip"]),
        "T8_PRINT_alignment_pin_1": place_on_bed(needles[pin_names[0]], p["print_layout"]["pin_x_start_mm"], p["print_layout"]["pin_y_centers_mm"][0]),
        "T8_PRINT_alignment_pin_2": place_on_bed(needles[pin_names[1]], p["print_layout"]["pin_x_start_mm"], p["print_layout"]["pin_y_centers_mm"][1]),
    }
    layout_checks = {}
    for name, shape in layout.items():
        b = exact_bounds(shape)
        if b[2] < -1e-6:
            raise RuntimeError(f"Layout part below print bed: {name}, zmin={b[2]}")
        layout_checks[name] = {"dimensions_mm": dims(shape), "bbox_min_mm": [round(b[i], 6) for i in range(3)], "bbox_max_mm": [round(b[i + 3], 6) for i in range(3)], "solid_count": len(shape.Solids()), "valid": shape.isValid()}
    for (an, a), (bn, b) in itertools.combinations(layout.items(), 2):
        ov = overlap(a, b)
        if ov > 1e-5:
            raise RuntimeError(f"Print layout overlap {an}/{bn}: {ov}")
    layout_compound = cq.Compound.makeCompound(list(layout.values()))
    layout_step = OUT / "T8_needle_print_layout.step"
    cq.exporters.export(layout_compound, str(layout_step))
    rts = {layout_step.name: roundtrip(layout_step, layout_compound, 5)}
    meshes = {}
    stl_names = {
        "T8_PRINT_root_segment": "T8_needle_root_segment_flat_print.stl",
        "T8_PRINT_mid_segment": "T8_needle_mid_segment_flat_print.stl",
        "T8_PRINT_tip_segment": "T8_needle_tip_segment_flat_print.stl",
        "T8_PRINT_alignment_pin_1": "T8_alignment_pin_1_flat_print.stl",
        "T8_PRINT_alignment_pin_2": "T8_alignment_pin_2_flat_print.stl",
    }
    for name, filename in stl_names.items():
        meshes[filename] = t7.t6.check_mesh(OUT / filename, layout[name])

    # Export the unchanged T7 geometry as a self-contained reference assembly.
    parts = {"01_lower_chassis": shells["01_lower_chassis"], "02_upper_cover": shells["02_upper_cover"], "03_sensor_carrier": shells["03_sensor_carrier"], **needles, "REF_05_HWT906P_approximate": shells["REF_05_HWT906P_approximate"], "REF_06_laser100m_envelope": shells["REF_06_laser100m_envelope"]}
    structural = {k: v for k, v in parts.items() if k.startswith(("01_", "02_", "03_"))}
    checks = {}
    for (an, a), (bn, b) in itertools.combinations(parts.items(), 2):
        ov = overlap(a, b)
        if ov > 1e-5 and not (an.startswith("REF_05") or bn.startswith("REF_05") or an.startswith("REF_06") or bn.startswith("REF_06")):
            raise RuntimeError(f"Unexpected assembly overlap {an}/{bn}: {ov}")
        checks[f"{an} / {bn}"] = round(ov, 8)
    for name, shape in structural.items():
        for label, ref in (("laser_full", env["laser_full_envelope"]), ("optical", env["optical_corridor"])):
            ov = overlap(ref, shape)
            if ov > 1e-5: raise RuntimeError(f"{label}/{name}: {ov}")
            checks[f"{label} / {name}"] = round(ov, 8)
    part_rows = {}
    for name, shape in parts.items():
        part_rows[name] = {"valid": shape.isValid(), "solid_count": len(shape.Solids()), "dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 6)}
        path = OUT / f"{name}.step"; cq.exporters.export(shape, str(path)); rts[path.name] = roundtrip(path, shape, 1)
        meshes[path.with_suffix(".stl").name] = t7.t6.check_mesh(path.with_suffix(".stl"), shape)
    printed_shell = cq.Compound.makeCompound(list(structural.values()))
    compound = cq.Compound.makeCompound(list(parts.values()))
    palette = [(.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42)]
    assembly = cq.Assembly(name="T8_PLA_FLAT_BOTTOM_NEEDLE_ASSEMBLY")
    shell_assembly = cq.Assembly(name="T8_THREE_PRINTED_HOUSING_COMPONENTS")
    for (name, shape), color in zip(parts.items(), palette):
        assembly.add(shape, name=name, color=cq.Color(*color))
        if name in structural: shell_assembly.add(shape, name=name, color=cq.Color(*color))
    assembly_path = OUT / "laser100m_HWT906P_T8_assembly.step"; shell_path = OUT / "T8_shell_only.step"; multibody_path = OUT / "laser100m_HWT906P_T8_multibody.step"
    assembly.export(str(assembly_path)); shell_assembly.export(str(shell_path)); cq.exporters.export(compound, str(multibody_path))
    rts[assembly_path.name] = roundtrip(assembly_path, compound, len(parts)); rts[shell_path.name] = roundtrip(shell_path, printed_shell, 3); rts[multibody_path.name] = roundtrip(multibody_path, compound, len(parts))
    # Render the unchanged assembly in this output directory, then rename the inherited labels to T8.
    t7.OUT = OUT
    t7.render_preview(parts)
    for src, dst in ((
        ("T7_assembly_preview.png", "T8_assembly_preview.png"),
        ("T7_assembly_preview_labeled.png", "T8_assembly_preview_labeled.png"),
        ("T7_flat_bottom_needle_exploded.png", "T8_exploded_assembly_preview.png"),
        ("T7_flat_bottom_needle_exploded_labeled.png", "T8_exploded_assembly_preview_labeled.png"),
        ("T7_flat_bottom_needle_detail.png", "T8_needle_detail.png"),
        ("T7_flat_bottom_needle_detail_labeled.png", "T8_needle_detail_labeled.png"),
        ("T7_mid_segment_end_profile.png", "T8_mid_segment_end_profile.png"),
        ("T7_mid_segment_end_profile_labeled.png", "T8_mid_segment_end_profile_labeled.png"),
    )):
        source = t7.OUT / src
        if source.exists(): source.replace(OUT / dst)

    report = {
        "revision": p["revision"], "housing_envelope_mm": [58.0, 46.0, 47.0], "assembly_parts": part_rows, "print_layout_parts": layout_checks,
        "step_roundtrip": rts, "stl_checks": meshes, "zero_overlap_checks_mm3": checks, "layout_zero_overlap_checks_mm3": {f"{an} / {bn}": round(overlap(a, b), 8) for (an, a), (bn, b) in itertools.combinations(layout.items(), 2)},
        "needle_profile": p["needle_profile"], "flat_bottom_z_mm": flat_z, "print_layout": p["print_layout"],
        "limitations": ["Use T8_needle_print_layout.step or the five *_flat_print.stl files for slicing; the T8 assembly STEP intentionally retains assembly coordinates.", "The separate layout solves the long-compound placement problem shown in the slicer screenshot; brim, bed adhesion and PLA shrinkage still need a physical trial.", "The T7/T8 housing is a reference fit model; sensor outlines, connector clearance and repeated PLA loading are not physically validated."],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# T8 分离排版版 PLA 针体

你截图中的问题是切片软件把装配坐标中的三段针体当成一条约 212 mm 的长条来摆放。T8 保留 T7 的平底 3/4 截面和 200 mm 有效长度，但额外提供独立打印排版：根部段、中段、针尖段在 STEP 中分成三行，平面全部位于 `z=0`，定位销单独放在旁边。

## 切片入口

优先加载 `T8_needle_print_layout.step`，或分别加载：

- `T8_needle_root_segment_flat_print.stl`
- `T8_needle_mid_segment_flat_print.stl`
- `T8_needle_tip_segment_flat_print.stl`
- `T8_alignment_pin_1_flat_print.stl`
- `T8_alignment_pin_2_flat_print.stl`

不要把 `laser100m_HWT906P_T8_assembly.step` 直接送入切片。该文件用于检查装配关系，保留针体原始坐标；切片时会再次形成跨床的长条。

三段针体的平面朝下打印，每段最大长度不超过 88 mm，行间距约 25 mm。建议使用 3–5 mm brim、较慢外壁速度，并在打印完成后用玻璃板或 V 形槽校直，再用定位销和环氧胶或瞬干胶接合。

T8 外壳沿用 T5/T7：宽度 46 mm、左右内腔各放宽 1 mm、壁厚约 1.6 mm。验证结果写入 `validation.json`。
"""
    (OUT / "README_交付说明.md").write_text(readme, encoding="utf-8")
    render_layout(layout)
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = ROOT / "output" / "laser100m_HWT906P_T8_separated_print_layout_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file(): z.write(f, f"{OUT.name}/{f.name}")
        for src in (Path(__file__), PARAM, CAD / "laser100m_T7.py", CAD / "laser100m_T7_parameters.json", CAD / "laser100m_T6.py", CAD / "laser100m_T6_parameters.json", CAD / "laser100m_T5.py", CAD / "laser100m_T2.py", CAD / "laser100m_T2_parameters.json"):
            z.write(src, f"source/{src.name}")
    print(json.dumps({"output": str(OUT), "assembly": str(assembly_path), "needle_print_layout": str(layout_step), "zip": str(bundle), "layout_parts": len(layout), "max_body_length_mm": max(layout_checks[n]["dimensions_mm"][0] for n in layout_checks if "segment" in n)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
