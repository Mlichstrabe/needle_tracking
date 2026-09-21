"""Build T9: flat-bottom alignment pins and removable tip adhesion feet."""
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
OUT = ROOT / "output" / "cad_laser100m_T9_connector_tip_adhesion"
PARAM = CAD / "laser100m_T9_connector_tip_adhesion_parameters.json"


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, CAD / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t8 = load_module("laser100m_T8_print_layout.py", "laser100m_T8_base")
t7 = t8.t7


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


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
    return {"valid": True, "solid_count": expected_solids, "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def flatten_pin(pin, p):
    n = p["needle"]
    offset = p["connector_print"]["pin_flat_bottom_offset_from_axis_mm"]
    flat_z = n["axis_z"] + offset
    half_space = t7.t6.t5.base.box(-500, 500, -500, 500, flat_z, 500)
    return t7.solid(pin.intersect(half_space.val()))


def add_tip_feet(tip, p):
    n = p["needle"]
    profile = p["needle_profile"]
    flat_z = n["axis_z"] + profile["flat_bottom_z_offset_mm"]
    cfg = p["tip_print_adhesion"]
    feet = None
    for x0, x1 in cfg["tab_x_ranges_mm"]:
        foot = t7.t6.t5.base.box(x0, x1, -cfg["tab_width_y_mm"] / 2, cfg["tab_width_y_mm"] / 2, flat_z - cfg["tab_height_z_mm"], flat_z + cfg["tab_overlap_into_body_mm"])
        feet = foot if feet is None else feet.union(foot)
    return t7.solid(cq.Workplane(obj=tip).union(feet))


def relabel_preview(path, title, footer):
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(path).convert("RGB") as im:
        bg = (243, 246, 248)
        c = Image.new("RGB", im.size, bg); c.paste(im, (0, 0)); d = ImageDraw.Draw(c)
        font_path = "C:/Windows/Fonts/msyh.ttc"
        title_font = ImageFont.truetype(font_path, 29)
        footer_font = ImageFont.truetype(font_path, 21)
        d.rectangle((0, 0, im.width, 65), fill=bg)
        d.rectangle((0, im.height - 48, im.width, im.height), fill=bg)
        d.text((im.width / 2, 25), title, font=title_font, fill="#163e50", anchor="mt")
        d.text((im.width / 2, im.height - 38), footer, font=footer_font, fill="#29424f", anchor="mt")
        c.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    t7_params = json.loads((CAD / "laser100m_T7_parameters.json").read_text(encoding="utf-8"))
    t9_params = json.loads(PARAM.read_text(encoding="utf-8"))
    p = merge(merge(base_params, t7_params), t9_params)
    shells, env = t7.t6.t5.build_parts(p)
    clean_needles, flat_z = t7.needle_segments(p)
    body_names = ["REF_04A_needle_root_segment", "REF_04B_needle_mid_segment", "REF_04C_needle_tip_segment"]
    pin_names = ["REF_04D_alignment_pin_joint_1", "REF_04E_alignment_pin_joint_2"]
    flat_pins = {name: flatten_pin(clean_needles[name], p) for name in pin_names}
    assembly_needles = dict(clean_needles); assembly_needles.update(flat_pins)
    print_tip = add_tip_feet(clean_needles[body_names[2]], p)
    layout = {
        "T9_PRINT_root_segment": t8.place_on_bed(clean_needles[body_names[0]], 15.0, -35.0),
        "T9_PRINT_mid_segment": t8.place_on_bed(clean_needles[body_names[1]], 15.0, 0.0),
        "T9_PRINT_tip_segment_with_feet": t8.place_on_bed(print_tip, 15.0, 35.0),
        "T9_PRINT_alignment_pin_1_flat": t8.place_on_bed(flat_pins[pin_names[0]], 120.0, -35.0),
        "T9_PRINT_alignment_pin_2_flat": t8.place_on_bed(flat_pins[pin_names[1]], 120.0, 0.0),
    }
    layout_checks = {}
    for name, shape in layout.items():
        b = exact_bounds(shape)
        if b[2] < -1e-6: raise RuntimeError(f"Print layout below bed: {name}, zmin={b[2]}")
        layout_checks[name] = {"dimensions_mm": dims(shape), "bbox_min_mm": [round(b[i], 6) for i in range(3)], "bbox_max_mm": [round(b[i + 3], 6) for i in range(3)], "solid_count": len(shape.Solids()), "valid": shape.isValid()}
    layout_ov = {}
    for (an, a), (bn, b) in itertools.combinations(layout.items(), 2):
        ov = overlap(a, b)
        if ov > 1e-5: raise RuntimeError(f"Print layout overlap {an}/{bn}: {ov}")
        layout_ov[f"{an} / {bn}"] = round(ov, 8)
    layout_compound = cq.Compound.makeCompound(list(layout.values()))
    layout_step = OUT / "T9_needle_print_layout.step"; cq.exporters.export(layout_compound, str(layout_step))
    rts = {layout_step.name: roundtrip(layout_step, layout_compound, 5)}
    meshes = {}
    layout_stl_names = {
        "T9_PRINT_root_segment": "T9_needle_root_segment_flat_print.stl",
        "T9_PRINT_mid_segment": "T9_needle_mid_segment_flat_print.stl",
        "T9_PRINT_tip_segment_with_feet": "T9_needle_tip_segment_with_adhesion_feet.stl",
        "T9_PRINT_alignment_pin_1_flat": "T9_alignment_pin_1_flat_bottom.stl",
        "T9_PRINT_alignment_pin_2_flat": "T9_alignment_pin_2_flat_bottom.stl",
    }
    for name, filename in layout_stl_names.items():
        meshes[filename] = t7.t6.check_mesh(OUT / filename, layout[name])
    clean_tip_print = t8.place_on_bed(clean_needles[body_names[2]], 15.0, 75.0)
    clean_tip_path = OUT / "T9_needle_tip_segment_clean_flat_print.stl"
    meshes[clean_tip_path.name] = t7.t6.check_mesh(clean_tip_path, clean_tip_print)

    parts = {"01_lower_chassis": shells["01_lower_chassis"], "02_upper_cover": shells["02_upper_cover"], "03_sensor_carrier": shells["03_sensor_carrier"], **{**clean_needles, **flat_pins}, "REF_05_HWT906P_approximate": shells["REF_05_HWT906P_approximate"], "REF_06_laser100m_envelope": shells["REF_06_laser100m_envelope"]}
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
    compound = cq.Compound.makeCompound(list(parts.values())); printed_shell = cq.Compound.makeCompound(list(structural.values()))
    palette = [(.17, .19, .21), (.22, .24, .26), (.25, .47, .54), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.62, .66, .68), (.19, .59, .43), (.38, .40, .42)]
    assembly = cq.Assembly(name="T9_PLA_FLAT_CONNECTOR_NEEDLE"); shell_assembly = cq.Assembly(name="T9_THREE_PRINTED_HOUSING_COMPONENTS")
    for (name, shape), color in zip(parts.items(), palette):
        assembly.add(shape, name=name, color=cq.Color(*color))
        if name in structural: shell_assembly.add(shape, name=name, color=cq.Color(*color))
    assembly_path = OUT / "laser100m_HWT906P_T9_assembly.step"; shell_path = OUT / "T9_shell_only.step"; multibody_path = OUT / "laser100m_HWT906P_T9_multibody.step"
    assembly.export(str(assembly_path)); shell_assembly.export(str(shell_path)); cq.exporters.export(compound, str(multibody_path))
    rts[assembly_path.name] = roundtrip(assembly_path, compound, len(parts)); rts[shell_path.name] = roundtrip(shell_path, printed_shell, 3); rts[multibody_path.name] = roundtrip(multibody_path, compound, len(parts))
    t7.OUT = OUT; t7.render_preview(parts)
    for src, dst in ((
        ("T7_assembly_preview.png", "T9_assembly_preview.png"), ("T7_assembly_preview_labeled.png", "T9_assembly_preview_labeled.png"),
        ("T7_flat_bottom_needle_exploded.png", "T9_exploded_assembly_preview.png"), ("T7_flat_bottom_needle_exploded_labeled.png", "T9_exploded_assembly_preview_labeled.png"),
        ("T7_flat_bottom_needle_detail.png", "T9_needle_detail.png"), ("T7_flat_bottom_needle_detail_labeled.png", "T9_needle_detail_labeled.png"),
        ("T7_mid_segment_end_profile.png", "T9_mid_segment_end_profile.png"), ("T7_mid_segment_end_profile_labeled.png", "T9_mid_segment_end_profile_labeled.png"),
    )):
        source = OUT / src
        if source.exists(): source.replace(OUT / dst)
    t8.OUT = OUT; t8.render_layout(layout)
    for src, dst in (("T8_print_layout_preview.png", "T9_print_layout_preview.png"), ("T8_print_layout_preview_labeled.png", "T9_print_layout_preview_labeled.png"), ("T8_print_layout_top_preview.png", "T9_print_layout_top_preview.png"), ("T8_print_layout_top_preview_labeled.png", "T9_print_layout_top_preview_labeled.png")):
        source = OUT / src
        if source.exists(): source.replace(OUT / dst)
    relabel_preview(OUT / "T9_assembly_preview_labeled.png", "T9 | 平底连接件 + 针尖防滑脚 + 原外壳", "外露 200 mm | 主体 Ø5 mm | 连接件平底打印")
    relabel_preview(OUT / "T9_exploded_assembly_preview_labeled.png", "T9 装配分解视图", "平底定位销；针尖三处可掰除防滑脚")
    relabel_preview(OUT / "T9_needle_detail_labeled.png", "T9 针体细节：平底 + 防滑脚", "打印后折除防滑脚，清理底面再胶粘")
    relabel_preview(OUT / "T9_mid_segment_end_profile_labeled.png", "T9 Ø5 mm 轴段端面：约 3/4 圆截面", "下方直边为打印底面，定位销改为同向平底")
    relabel_preview(OUT / "T9_print_layout_preview_labeled.png", "T9 | 三段针体 + 平底连接件独立排版", "针尖防滑脚与定位销均有床面接触，避免圆柱滚动")
    relabel_preview(OUT / "T9_print_layout_top_preview_labeled.png", "T9 打印排版俯视图", "根部 / 中段 / 针尖分列；定位销单独放置")
    report = {
        "revision": p["revision"], "housing_envelope_mm": [58.0, 46.0, 47.0], "assembly_parts": part_rows, "print_layout_parts": layout_checks,
        "step_roundtrip": rts, "stl_checks": meshes, "zero_overlap_checks_mm3": checks, "layout_zero_overlap_checks_mm3": layout_ov,
        "flat_bottom_z_mm": flat_z, "connector_print": p["connector_print"], "tip_print_adhesion": p["tip_print_adhesion"],
        "limitations": ["The three tip feet are sacrificial print aids and must be snapped or cut off after printing.", "The connector pins use a flat-bottom 3/4-like section but retain round Ø3.6 mm holes; adhesive is part of the joint design.", "Use T9_needle_print_layout.step or the T9 flat-print STL files for slicing; the T9 assembly STEP retains assembly coordinates and is for CAD inspection.", "Brim, bed adhesion, PLA shrinkage and final glue strength still need physical trial."],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# T9 平底连接件与针尖防滑打印版

这版针对实际打印反馈做了两项改动：

1. 两根 Ø3 mm 定位销改为带宽平底的 3/4-like D 截面。平面朝下打印，装入原 Ø3.6 mm 圆孔后用胶固定。
2. 针尖段在平底下增加三个可掰除的防滑脚，分别位于前端、中部和连接端。它们扩大首层接触面积，打印完成后从薄连接处折下并清理底面。

## 正确的切片入口

优先使用 `T9_needle_print_layout.step`，或分别加载：

- `T9_needle_root_segment_flat_print.stl`
- `T9_needle_mid_segment_flat_print.stl`
- `T9_needle_tip_segment_with_adhesion_feet.stl`
- `T9_alignment_pin_1_flat_bottom.stl`
- `T9_alignment_pin_2_flat_bottom.stl`

`T9_needle_tip_segment_clean_flat_print.stl` 是不带防滑脚的备用版本。`laser100m_HWT906P_T9_assembly.step` 只用于 CAD 查看，不要直接送入切片。

建议针尖段和定位销单独打印，不与长针或外壳混在同一盘；平面朝下，使用 3–5 mm brim，外壁速度适当降低。针尖段打印后先确认三个防滑脚完整，再折除并打磨底面；两根定位销直接从平面侧打印，装配时用环氧胶或瞬干胶固定。

外壳沿用 T5/T7/T8：宽度 46 mm，内腔左右各放宽 1 mm，壁厚约 1.6 mm，采用刚性导向和胶粘定位。验证结果写入 `validation.json`。
"""
    (OUT / "README_交付说明.md").write_text(readme, encoding="utf-8")
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = ROOT / "output" / "laser100m_HWT906P_T9_connector_tip_adhesion_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file(): z.write(f, f"{OUT.name}/{f.name}")
        for src in (Path(__file__), PARAM, CAD / "laser100m_T8_print_layout.py", CAD / "laser100m_T8_print_layout_parameters.json", CAD / "laser100m_T7.py", CAD / "laser100m_T7_parameters.json", CAD / "laser100m_T6.py", CAD / "laser100m_T6_parameters.json", CAD / "laser100m_T5.py", CAD / "laser100m_T2.py", CAD / "laser100m_T2_parameters.json"):
            z.write(src, f"source/{src.name}")
    print(json.dumps({"output": str(OUT), "assembly": str(assembly_path), "needle_print_layout": str(layout_step), "zip": str(bundle), "tip_feet": 3, "flat_pin_count": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
