"""Build T7: T6 housing with a continuous flat-bottom 3/4-section PLA needle."""
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
OUT = ROOT / "output" / "cad_laser100m_T7_flat_bottom_needle"
PARAM = CAD / "laser100m_T7_parameters.json"


def load_t6():
    spec = importlib.util.spec_from_file_location("laser100m_T6_base", CAD / "laser100m_T6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t6 = load_t6()


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


def flatten_body(shape, flat_z):
    """Keep the upper 3/4-like section above one common, print-bed plane."""
    half_space = t6.t5.base.box(-500, 500, -500, 500, flat_z, 500)
    return solid(shape.intersect(half_space.val()))


def needle_segments(p):
    round_needles = t6.needle_segments(p)
    flat_z = p["needle"]["axis_z"] + p["needle_profile"]["flat_bottom_z_offset_mm"]
    body_names = (
        "REF_04A_needle_root_segment",
        "REF_04B_needle_mid_segment",
        "REF_04C_needle_tip_segment",
    )
    out = dict(round_needles)
    for name in body_names:
        out[name] = flatten_body(round_needles[name], flat_z)
    return out, flat_z


def render_preview(parts):
    import vtk
    from PIL import Image, ImageDraw, ImageFont

    colors = [
        (0.17, .19, .21), (.22, .24, .26), (.25, .47, .54),
        (.62, .66, .68), (.62, .66, .68), (.62, .66, .68),
        (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42),
    ]
    names = list(parts)

    def frame(path, selected, size, camera, target, scale, opacity=None):
        ren = vtk.vtkRenderer()
        ren.SetBackground(.953, .965, .973)
        opacity = opacity or {}
        for idx, name in enumerate(names):
            if name not in selected:
                continue
            verts, tris = parts[name].tessellate(.06)
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for v in verts:
                pts.InsertNextPoint(*v.toTuple())
            for tri in tris:
                cell = vtk.vtkTriangle()
                for j, k in enumerate(tri):
                    cell.GetPointIds().SetId(j, k)
                cells.InsertNextCell(cell)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetPolys(cells)
            norm = vtk.vtkPolyDataNormals()
            norm.SetInputData(poly)
            norm.SetFeatureAngle(40)
            norm.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(norm.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*colors[idx])
            actor.GetProperty().SetOpacity(opacity.get(name, 1.0))
            ren.AddActor(actor)
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetSize(*size)
        win.AddRenderer(ren)
        cam = ren.GetActiveCamera()
        cam.SetPosition(*camera)
        cam.SetFocalPoint(*target)
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOn()
        cam.SetParallelScale(scale)
        ren.ResetCameraClippingRange()
        win.Render()
        cap = vtk.vtkWindowToImageFilter()
        cap.SetInput(win)
        cap.SetInputBufferTypeToRGB()
        cap.ReadFrontBufferOff()
        cap.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(cap.GetOutputPort())
        writer.Write()
        win.Finalize()

    frame(OUT / "T7_assembly_preview.png", names, (1800, 850), (-300, -360, 220), (-95, 0, 0), 75)
    exploded = {}
    for name, shape in parts.items():
        dz = 62 if name == "02_upper_cover" else (32 if name == "03_sensor_carrier" else (18 if name.startswith("REF_04") else 0))
        exploded[name] = shape.translate((0, 0, dz))
    original = dict(parts)
    parts.clear()
    parts.update(exploded)
    frame(OUT / "T7_flat_bottom_needle_exploded.png",
          [n for n in names if n.startswith("REF_04") or n.startswith("01_") or n.startswith("02_") or n.startswith("03_")],
          (1500, 900), (-110, -170, 130), (-30, 0, 8), 70,
          {"01_lower_chassis": .18, "02_upper_cover": .20, "03_sensor_carrier": .25})
    parts.clear()
    parts.update(original)
    frame(OUT / "T7_flat_bottom_needle_detail.png",
          [n for n in names if n.startswith("REF_04")], (1700, 520), (-130, -150, 105), (-88, 0, 4), 28)
    frame(OUT / "T7_mid_segment_end_profile.png", ["REF_04B_needle_mid_segment"],
          (720, 720), (-155, -0.01, 4.2), (-115, 0, 4.2), 8)

    font_path = "C:/Windows/Fonts/msyh.ttc"
    def font(n):
        return ImageFont.truetype(font_path, n)

    with Image.open(OUT / "T7_assembly_preview.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248))
        c.paste(im, (0, 65))
        d = ImageDraw.Draw(c)
        d.text((900, 24), "T7 | 平底 3/4 截面 PLA 针体 + 原外壳", font=font(30), fill="#163e50", anchor="mt")
        d.text((900, 812), "外露 200 mm | 主体 Ø5 mm | 底部为连续打印平面", font=font(23), fill="#29424f", anchor="mt")
        c.save(OUT / "T7_assembly_preview_labeled.png")
    with Image.open(OUT / "T7_flat_bottom_needle_exploded.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248))
        c.paste(im, (0, 65))
        d = ImageDraw.Draw(c)
        d.text((750, 24), "T7 针体分段：连续平底便于 PLA 打印", font=font(29), fill="#163e50", anchor="mt")
        d.text((750, 865), "根部段 / 中段 / 针尖段；平面朝下打印，接缝用定位销和胶粘固定", font=font(21), fill="#29424f", anchor="mt")
        c.save(OUT / "T7_flat_bottom_needle_exploded_labeled.png")
    with Image.open(OUT / "T7_flat_bottom_needle_detail.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248))
        c.paste(im, (0, 65))
        d = ImageDraw.Draw(c)
        d.text((850, 24), "针体细节：共同平底 + 圆珠笔头过渡", font=font(29), fill="#163e50", anchor="mt")
        d.text((850, 490), "平面贴热床可抑制滚动；圆润针尖保留，底面同步切平", font=font(21), fill="#29424f", anchor="mt")
        c.save(OUT / "T7_flat_bottom_needle_detail_labeled.png")
    with Image.open(OUT / "T7_mid_segment_end_profile.png").convert("RGB") as im:
        c = Image.new("RGB", im.size, (243, 246, 248))
        c.paste(im, (0, 65))
        d = ImageDraw.Draw(c)
        d.text((360, 24), "Ø5 mm 轴段端面：约 3/4 圆截面", font=font(29), fill="#163e50", anchor="mt")
        d.text((360, 690), "下方直边为打印底面，定位销孔仍保持圆孔", font=font(21), fill="#29424f", anchor="mt")
        c.save(OUT / "T7_mid_segment_end_profile_labeled.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    p = merge(base_params, json.loads(PARAM.read_text(encoding="utf-8")))
    shells, env = t6.t5.build_parts(p)
    needles, flat_z = needle_segments(p)
    parts = {
        "01_lower_chassis": shells["01_lower_chassis"],
        "02_upper_cover": shells["02_upper_cover"],
        "03_sensor_carrier": shells["03_sensor_carrier"],
        **needles,
        "REF_05_HWT906P_approximate": shells["REF_05_HWT906P_approximate"],
        "REF_06_laser100m_envelope": shells["REF_06_laser100m_envelope"],
    }
    structural = {k: v for k, v in parts.items() if k.startswith(("01_", "02_", "03_"))}
    checks = {}
    for (an, a), (bn, b) in itertools.combinations(parts.items(), 2):
        ov = overlap(a, b)
        if ov > 1e-5 and not (an.startswith("REF_05") or bn.startswith("REF_05") or an.startswith("REF_06") or bn.startswith("REF_06")):
            raise RuntimeError(f"Unexpected overlap {an}/{bn}: {ov}")
        checks[f"{an} / {bn}"] = round(ov, 8)
    for name, shape in structural.items():
        for label, ref in (("laser_full", env["laser_full_envelope"]), ("optical", env["optical_corridor"])):
            ov = overlap(ref, shape)
            if ov > 1e-5:
                raise RuntimeError(f"{label}/{name}: {ov}")
            checks[f"{label} / {name}"] = round(ov, 8)

    rows, rts, meshes = {}, {}, {}
    for name, shape in parts.items():
        rows[name] = {"valid": shape.isValid(), "solid_count": len(shape.Solids()), "dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 6)}
        path = OUT / f"{name}.step"
        cq.exporters.export(shape, str(path))
        rts[path.name] = roundtrip(path, shape, 1)
        meshes[name] = t6.check_mesh(OUT / f"{name}.stl", shape)

    printed_needle = cq.Compound.makeCompound([needles[k] for k in needles])
    cq.exporters.export(printed_needle, str(OUT / "T7_needle_print_set.step"))
    rts["T7_needle_print_set.step"] = roundtrip(OUT / "T7_needle_print_set.step", printed_needle, 5)
    printed_shell = cq.Compound.makeCompound(list(structural.values()))
    compound = cq.Compound.makeCompound(list(parts.values()))
    palette = [(.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42)]
    assembly = cq.Assembly(name="T7_PLA_FLAT_BOTTOM_NEEDLE")
    shell_assembly = cq.Assembly(name="T7_THREE_PRINTED_HOUSING_COMPONENTS")
    exploded = cq.Assembly(name="T7_EXPLODED_FLAT_BOTTOM_NEEDLE")
    exploded_shapes = []
    for (name, shape), color in zip(parts.items(), palette):
        assembly.add(shape, name=name, color=cq.Color(*color))
        if name in structural:
            shell_assembly.add(shape, name=name, color=cq.Color(*color))
        dz = 62 if name == "02_upper_cover" else (32 if name == "03_sensor_carrier" else (18 if name.startswith("REF_04") else 0))
        shifted = shape.translate((0, 0, dz))
        exploded_shapes.append(shifted)
        exploded.add(shifted, name=name, color=cq.Color(*color))
    assembly_path = OUT / "laser100m_HWT906P_T7_assembly.step"
    shell_path = OUT / "T7_shell_only.step"
    exploded_path = OUT / "T7_exploded_assembly.step"
    multibody_path = OUT / "laser100m_HWT906P_T7_multibody.step"
    assembly.export(str(assembly_path))
    shell_assembly.export(str(shell_path))
    exploded.export(str(exploded_path))
    cq.exporters.export(compound, str(multibody_path))
    rts[assembly_path.name] = roundtrip(assembly_path, compound, len(parts))
    rts[shell_path.name] = roundtrip(shell_path, printed_shell, 3)
    rts[exploded_path.name] = roundtrip(exploded_path, cq.Compound.makeCompound(exploded_shapes), len(parts))
    rts[multibody_path.name] = roundtrip(multibody_path, compound, len(parts))

    n = p["needle"]
    profile = p["needle_profile"]
    report = {
        "revision": p["revision"],
        "housing_envelope_mm": [58.0, 46.0, 47.0],
        "parts": rows,
        "step_roundtrip": rts,
        "stl_checks": meshes,
        "zero_overlap_checks_mm3": checks,
        "pla_clip_design": p["pla_clip_design"],
        "needle_profile": profile,
        "flat_bottom_geometry": {
            "axis_z_mm": n["axis_z"],
            "flat_bottom_z_mm": flat_z,
            "flat_bottom_offset_from_axis_mm": profile["flat_bottom_z_offset_mm"],
            "nominal_shaft_width_mm": profile["body_diameter_mm"],
            "nominal_shaft_remaining_height_mm": round(profile["body_diameter_mm"] / 2 - profile["flat_bottom_z_offset_mm"], 6),
            "definition": "one common longitudinal planar face on all three needle body segments; alignment pins remain round for the existing round clearance holes",
        },
        "geometric_distances": {"setback_from_rod_exit_mm": 7.0, "initial_geometric_distance_mm": 207.0, "distance_after_150mm_stroke_mm": 57.0, "near_range_verified": False},
        "limitations": [
            "The 3/4-section wording is implemented as one common planar cut at axis_z - 1.0 mm; the nominal Ø5 shaft retains a 3.5 mm vertical height.",
            "The same print-bed plane also cuts the enlarged root reinforcement; it is intentionally wider and flatter for PLA stability.",
            "Three needle segments are intended for horizontal PLA printing and epoxy/CA assembly; print orientation and adhesive cure need physical trial.",
            "Alignment pin holes use a nominal 0.6 mm diameter clearance; printer shrinkage may require a fit coupon.",
            "Sensor outlines, connector clearance, PLA strength and repeated bending have not been physically validated.",
        ],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# T7 平底 3/4 截面 PLA 针体 STEP

本版沿用 T6 外壳和三段式针体接口，但将根部段、中段、针尖段都与同一个纵向平面相交，形成连续平底。平面朝下放置在打印床上，可显著降低细长针体和圆珠笔头在打印过程中滚动、滑移的概率。

针体保留：外露 200 mm、主体标称 Ø5 mm、根部最大约 Ø8.5 mm、圆珠笔头约 Ø4.4 mm、Ø4 mm × 12 mm 插入段、两处定位销接头。T7 的统一平底面位于 `axis_z - 1.0 mm`；对于标称 Ø5 mm 轴段，剩余高度约 3.5 mm。定位销保持圆柱形，以继续匹配 Ø3.6 mm 圆孔。

打印时将 `REF_04A_needle_root_segment.stl`、`REF_04B_needle_mid_segment.stl`、`REF_04C_needle_tip_segment.stl` 的平面朝下，建议关闭针体下方支撑；定位销单独打印。三段接缝先干装，再用环氧胶或瞬干胶固定，固化时放在玻璃板、直尺或 V 形槽上校直。

外壳仍为宽度 46 mm、内腔左右各放宽 1 mm、壁厚约 1.6 mm 的 T5 方案；外盖和中间载架仍采用刚性导向与胶粘定位。

这是为 PLA 打印稳定性做的结构试配版本。`validation.json` 已记录 STEP 重导入、干涉、STL 网格和尺寸检查；真实打印仍需根据喷嘴、层高、收缩和胶接强度做一次试配。
"""
    (OUT / "README_交付说明.md").write_text(readme, encoding="utf-8")
    render_preview(parts)
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = ROOT / "output" / "laser100m_HWT906P_T7_flat_bottom_STEP_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file():
                z.write(f, f"{OUT.name}/{f.name}")
        for src in (Path(__file__), PARAM, CAD / "laser100m_T6.py", CAD / "laser100m_T6_parameters.json", CAD / "laser100m_T5.py", CAD / "laser100m_T2.py", CAD / "laser100m_T2_parameters.json"):
            z.write(src, f"source/{src.name}")
    print(json.dumps({"output": str(OUT), "assembly": str(assembly_path), "needle_print_set": str(OUT / "T7_needle_print_set.step"), "zip": str(bundle), "flat_bottom_z_mm": flat_z, "segment_count": 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
