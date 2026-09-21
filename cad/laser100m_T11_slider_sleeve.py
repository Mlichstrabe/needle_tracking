"""T11: removable PLA slider sleeve for measuring simulated needle insertion depth.

The T10 one-piece needle is kept unchanged.  The sleeve is installed from the
needle tip and slides on the flat-bottom 6 mm body.  Its proximal flat target
face is offset to the existing laser optical axis, so the range module measures
the sleeve position rather than the slender needle surface.

Run:
    uv run --python 3.12 --offline --with cadquery==2.6.1 --with trimesh \
      --with matplotlib --with numpy python -X utf8 cad/laser100m_T11_slider_sleeve.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


CAD = Path(__file__).resolve().parent
ROOT = CAD.parent
OUT = ROOT / "output" / "cad_laser100m_T11_slider_sleeve"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = load_module("t11_t5_base", CAD / "laser100m_T5.py")
t10 = load_module("t11_t10_needle", CAD / "laser100m_T10_onepiece.py")


P = {
    "revision": "T11-slider-sleeve",
    "units": "mm",
    "needle_body_width": 6.0,
    "needle_body_height": 4.0,
    "needle_axis_z": 4.2,
    "needle_bottom_z": 3.2,
    "channel_x_length": 14.0,
    "channel_outer_width": 8.4,
    "channel_inner_width": 6.6,
    "channel_outer_bottom_z": 2.0,
    "channel_inner_floor_z": 3.0,
    "channel_top_z": 7.4,
    "channel_side_wall": 0.9,
    "channel_lip_width": 0.5,
    "target_panel_x_thickness": 2.2,
    "target_panel_y_width": 16.0,
    "target_panel_z_bottom": -12.5,
    "target_panel_z_top": 2.6,
    "target_center_y": -3.0,
    "laser_center_y": -3.0,
    "laser_center_z": -9.7,
    "target_face_datum_x": 0.0,
    "fit_clearance_each_side": 0.3,
    "lead_in_length": 1.0,
    "print_foot_thickness": 0.6,
    "print_foot_x0": -4.0,
    "print_foot_x1": 1.8,
    "print_foot_y_margin": 0.5,
    "sample_target_front_x": -100.0,
    "safe_near_root_target_front_x": -22.0,
    "safe_near_tip_target_front_x": -196.0,
    "needle_exposed_length": 200.0,
}


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    return t5.base.box(x0, x1, y0, y1, z0, z1)


def solid(shape):
    value = shape.val() if isinstance(shape, cq.Workplane) else shape
    ss = value.Solids()
    if len(ss) != 1 or not value.isValid():
        raise ValueError(f"Expected one valid solid; solids={len(ss)}, valid={value.isValid()}")
    return ss[0]


def overlap(a, b):
    return sum(s.Volume() for s in a.intersect(b).Solids())


def exact_bounds(shape):
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    return b.Get()


def dims(shape):
    b = exact_bounds(shape)
    return [round(b[i + 3] - b[i], 6) for i in range(3)]


def roundtrip(path: Path, source, expected_solids: int):
    imported = cq.importers.importStep(str(path)).val()
    delta = abs(imported.Volume() - source.Volume())
    bd = max(abs(a - b) for a, b in zip(exact_bounds(imported), exact_bounds(source)))
    if (
        not imported.isValid()
        or len(imported.Solids()) != expected_solids
        or delta > max(1e-4, source.Volume() * 1e-8)
        or bd > 1e-5
    ):
        raise RuntimeError(
            f"STEP reimport failed: {path.name}; valid={imported.isValid()}; "
            f"solids={len(imported.Solids())}; volume_delta={delta}; bbox_delta={bd}"
        )
    return {"valid": True, "solid_count": expected_solids,
            "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def mesh_of(shape, tolerance: float = 0.02):
    vertices, faces = shape.tessellate(tolerance, 0.08)
    mesh = trimesh.Trimesh(
        vertices=[v.toTuple() for v in vertices], faces=faces, process=True
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split()) != 1:
        raise RuntimeError("STL mesh is not one watertight component")
    return mesh


def build_sleeve():
    """Build clean assembly geometry with target face at x=0."""
    cx0, cx1 = -16.0, -2.0
    # Flat floor and two full side guides. The upper lips retain the rounded
    # 3/4-cylinder body while leaving a generous open lead-in at the tip.
    parts = [
        box(cx0, cx1, -4.2, 4.2, 2.0, 3.0),
        box(cx0, cx1, -4.2, -3.3, 3.0, 7.4),
        box(cx0, cx1, 3.3, 4.2, 3.0, 7.4),
        box(cx0, cx1, -3.3, -2.8, 6.8, 7.4),
        box(cx0, cx1, 2.8, 3.3, 6.8, 7.4),
    ]
    # The laser-facing surface is the proximal (larger-x) face. It extends
    # downward to the verified laser optical centre and is wide enough to be a
    # stable diffuse target. A 0.2 mm overlap guarantees a single solid.
    panel = box(-2.2, 0.0, -11.0, 5.0, -12.5, 2.6)
    parts.append(panel)
    shape = parts[0]
    for part in parts[1:]:
        shape = shape.union(part)
    clean = solid(shape)

    # A broad, sacrificial printing foot keeps the 2.2 mm target plate stable
    # on the PLA bed. It is below the target's clean assembly envelope and is
    # trimmed after printing.
    foot = box(
        P["print_foot_x0"], P["print_foot_x1"],
        -11.0 - P["print_foot_y_margin"], 5.0 + P["print_foot_y_margin"],
        -12.5, -12.5 + P["print_foot_thickness"],
    )
    printable = solid(cq.Workplane(obj=clean).union(foot))
    return clean, printable


def installed_needle():
    clean, _ = t10.build()
    return clean.translate((0, 0, t10.P["axis_z"] + t10.P["flat_offset"]))


def reference_parts():
    params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    params["housing"]["width"] = 46.0
    parts, env = t5.build_parts(params)
    needle = installed_needle()
    return parts, env, needle


def target_front(sleeve, x: float):
    return sleeve.translate((x - P["target_face_datum_x"], 0, 0))


def add_mesh(ax, mesh, color, alpha=1.0, edgecolor=None):
    faces = mesh.triangles
    poly = Poly3DCollection(faces, facecolor=color, edgecolor=edgecolor or color,
                           linewidths=0.12, alpha=alpha)
    ax.add_collection3d(poly)


def plot_mesh(ax, shape, color, alpha=1.0, tolerance=0.08):
    add_mesh(ax, mesh_of(shape, tolerance), color, alpha)


def preview(clean, printable, needle, parts, env):
    # Register a Windows CJK font when available so the preview does not show
    # missing-glyph boxes. The geometry and file generation do not depend on it.
    from matplotlib import font_manager
    font_path = Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Noto Sans SC"
    else:
        plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sleeve = target_front(clean, P["sample_target_front_x"])
    fig = plt.figure(figsize=(16, 10), facecolor="#f4f7fa", layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax = fig.add_subplot(gs[0, :], projection="3d")
    plot_mesh(ax, parts["01_lower_chassis"], "#7f8d96", 0.14)
    plot_mesh(ax, parts["03_sensor_carrier"], "#4b7789", 0.25)
    plot_mesh(ax, env["laser_full_envelope"], "#e49a40", 0.20)
    plot_mesh(ax, needle, "#46525a", 0.95)
    plot_mesh(ax, sleeve, "#d75b4b", 0.88)
    ax.set_title("T11 滑套装配参考｜滑套沿 T10 平底针体移动，红色为测距靶板", fontsize=15)
    ax.set_xlabel("轴向 X / mm"); ax.set_ylabel("横向 Y / mm"); ax.set_zlabel("高度 Z / mm")
    ax.set_xlim(-215, 12); ax.set_ylim(-25, 25); ax.set_zlim(-28, 22)
    ax.view_init(elev=22, azim=-68)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot([-203, 9], [4.2, 4.2], color="#46525a", lw=4, label="T10 针体轴线")
    x = P["sample_target_front_x"]
    ax2.add_patch(plt.Rectangle((x - 2.2, P["target_panel_z_bottom"]), 2.2,
                                P["target_panel_z_top"] - P["target_panel_z_bottom"],
                                color="#d75b4b", alpha=.8, label="测距靶板"))
    ax2.axhline(P["laser_center_z"], color="#e49a40", ls="--", lw=1.5, label="激光光轴 z=-9.7")
    ax2.axvline(4, color="#e49a40", ls=":", lw=1, label="激光模块前端 x=4")
    ax2.annotate("约 100 mm 位置", xy=(x, P["laser_center_z"]), xytext=(x+28, -14),
                 arrowprops={"arrowstyle": "->", "color": "#b84036"}, color="#b84036")
    ax2.set_xlim(-215, 12); ax2.set_ylim(-28, 12); ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X / mm"); ax2.set_ylabel("Z / mm"); ax2.grid(alpha=.2)
    ax2.legend(loc="upper left", fontsize=8); ax2.set_title("侧视：靶面始终垂直于激光测距方向")

    ax3 = fig.add_subplot(gs[1, 1])
    # Assembly cross section at the middle of the sleeve, shown in assembly Z.
    cross = mesh_of(clean, 0.04).section(plane_origin=[-9, 0, 0], plane_normal=[1, 0, 0])
    if cross is not None:
        for path in cross.discrete:
            ax3.plot(path[:, 1], path[:, 2], color="#d75b4b", lw=1.5)
    circle = plt.Circle((0, P["needle_axis_z"]), 3.0, fill=False, color="#46525a", lw=2)
    ax3.add_patch(circle)
    ax3.axhline(P["needle_bottom_z"], color="#46525a", lw=1, ls="--")
    ax3.text(-4.1, 2.75, "针体平底", fontsize=8, color="#46525a")
    ax3.set_xlim(-6, 6); ax3.set_ylim(1.2, 8.5); ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("Y / mm"); ax3.set_ylabel("Z / mm"); ax3.grid(alpha=.2)
    ax3.set_title("滑动截面：内宽 6.6 mm，单侧间隙 0.3 mm")
    fig.savefig(OUT / "T11_slider_sleeve_preview.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean, printable = build_sleeve()
    needle = installed_needle()
    parts, env, _ = reference_parts()

    # Fit checks at the near-root, mid-stroke and near-tip positions. The
    # sleeve surrounds the needle with clearance, so the solid intersection
    # must be zero at every position.
    fit_positions = [P["safe_near_root_target_front_x"], -100.0,
                     P["safe_near_tip_target_front_x"]]
    fit = {}
    for x in fit_positions:
        value = overlap(target_front(clean, x), needle)
        fit[str(x)] = round(value, 8)
        if value > 1e-5:
            raise RuntimeError(f"Sleeve/needle interference at target front x={x}: {value}")

    # The laser centreline enters the target panel at every tested position.
    beam_checks = {}
    for x in fit_positions:
        beam = solid(box(x - 0.05, 4.05, P["laser_center_y"] - 1.0,
                         P["laser_center_y"] + 1.0, P["laser_center_z"] - 1.0,
                         P["laser_center_z"] + 1.0))
        panel_hit = overlap(target_front(clean, x), beam)
        beam_checks[str(x)] = round(panel_hit, 6)
        if panel_hit <= 0:
            raise RuntimeError(f"Laser beam misses target panel at x={x}")

    files = {}
    checks = {"fit_sleeve_needle_overlap_mm3": fit,
              "laser_centerline_target_intersection_mm3": beam_checks}
    for name, shape in (("T11_SLIDER_SLEEVE_FINISHED", clean),
                        ("T11_SLIDER_SLEEVE_PRINT_WITH_FOOT", printable)):
        step = OUT / f"{name}.step"
        stl = OUT / f"{name}.stl"
        cq.exporters.export(shape, str(step))
        rt = roundtrip(step, shape, 1)
        mesh = mesh_of(shape)
        mesh.export(stl)
        reread = trimesh.load_mesh(stl)
        if not reread.is_watertight or not reread.is_winding_consistent or len(reread.split()) != 1:
            raise RuntimeError(f"STL reread failed: {stl.name}")
        files[name] = {"step": rt, "stl_watertight": True,
                       "stl_components": 1, "dimensions_mm": dims(shape),
                       "bounds_mm": reread.bounds.tolist(),
                       "volume_mm3": round(shape.Volume(), 5)}

    # Reference needle and a mid-stroke assembly are provided for CAD inspection.
    needle_step = OUT / "T10_REFERENCE_INSTALLED_NEEDLE.step"
    cq.exporters.export(needle, str(needle_step))
    needle_rt = roundtrip(needle_step, needle, 1)
    mid_sleeve = target_front(clean, P["sample_target_front_x"])
    assembly_shapes = [parts["01_lower_chassis"], parts["02_upper_cover"],
                       parts["03_sensor_carrier"], needle, mid_sleeve,
                       parts["REF_06_laser100m_envelope"]]
    assembly_compound = cq.Compound.makeCompound(assembly_shapes)
    assembly_step = OUT / "T11_REFERENCE_ASSEMBLY_100mm.step"
    cq.exporters.export(assembly_compound, str(assembly_step))
    assembly_rt = roundtrip(assembly_step, assembly_compound, len(assembly_shapes))
    named = cq.Assembly(name="T11_SLIDER_SLEEVE_REFERENCE")
    colors = [(0.38, .42, .45), (.56, .61, .64), (.25, .47, .54),
              (.20, .22, .24), (.82, .20, .15), (.86, .57, .16)]
    names = ["lower_chassis", "upper_cover", "sensor_carrier",
             "T10_needle", "slider_sleeve_at_100mm", "laser_envelope"]
    for n, shape, color in zip(names, assembly_shapes, colors):
        named.add(shape, name=n, color=cq.Color(*color))
    named_step = OUT / "T11_REFERENCE_ASSEMBLY_100mm_named.step"
    named.export(str(named_step))

    preview(clean, printable, needle, parts, env)

    range_info = {
        "laser_emitter_front_x_mm": 4.0,
        "target_front_x_near_root_mm": P["safe_near_root_target_front_x"],
        "target_front_x_near_tip_mm": P["safe_near_tip_target_front_x"],
        "geometric_range_near_root_mm": 4.0 - P["safe_near_root_target_front_x"],
        "geometric_range_near_tip_mm": 4.0 - P["safe_near_tip_target_front_x"],
        "depth_span_mm": P["safe_near_root_target_front_x"] - P["safe_near_tip_target_front_x"],
        "software_zero_offset_required": True,
    }
    report = {
        "parameters": P,
        "files": files,
        "reference_needle_step_roundtrip": needle_rt,
        "reference_assembly_step_roundtrip": assembly_rt,
        "range_geometry": range_info,
        "checks": checks,
        "physical_print_verified": False,
        "gcode_sliced": False,
        "limitations": [
            "激光模块的真实发射点、最小量程和发散角仍需实测。",
            "滑动间隙按 PLA FDM 名义尺寸设计，需用实际喷嘴和流量校正。",
            "打印脚为牺牲结构，打印后沿靶板底边切除。",
            "测距值到进针深度的零点和方向需要连接实际模块校准。",
        ],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(P, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = f"""# T11 滑套打印与装配说明

本版保留 T10 一体式平底针体，新增一个从针尖套入的 PLA 滑套。滑套沿针体轴向移动，后侧（靠外壳方向）的平面靶板供 100 m 测距模块测量。

## 推荐文件

- `T11_SLIDER_SLEEVE_PRINT_WITH_FOOT.stl`：首选打印文件，底部带 0.6 mm 厚牺牲打印脚，打印完成后沿靶板底边切除。
- `T11_SLIDER_SLEEVE_FINISHED.step` / `.stl`：去除打印脚后的成品几何。
- `T11_REFERENCE_ASSEMBLY_100mm_named.step`：带名称和颜色的装配参考，滑套示例位置为靶面 x=-100 mm。
- `T10_REFERENCE_INSTALLED_NEEDLE.step`：当前成功打印的 T10 针体装配坐标参考。

## 关键尺寸

滑道内宽 6.6 mm，针体主体宽 6.0 mm，单侧名义间隙 0.3 mm；内底面 z=3.0 mm，针体平底约 z=3.2 mm。通道长度 14 mm，顶部保留上翻导向唇以限制侧向摆动。测距靶板厚 2.2 mm、宽 16 mm，沿 z=-12.5 到 2.6 mm 延伸，靶面中心对准现有激光光轴 y=-3、z=-9.7 mm。

## 装配和使用

1. 从针尖将滑套沿平底面推入，平底朝向滑套底部；滑套根部到达 T10 根部加固段时停止。
2. 初次使用先手动把滑套放在针尖附近，连接测距模块，确认激光打在靶板中央，不照到针体或外壳边缘。
3. 以靶面靠近针尖的位置建立软件零点；当前几何参考范围约为 {range_info['geometric_range_near_root_mm']:.1f} 到 {range_info['geometric_range_near_tip_mm']:.1f} mm，实际模块需校准偏置和方向。
4. 滑动阻力过大时只处理内侧导向面，每次去除约 0.05 到 0.10 mm，再重新试配；不要扩大靶面。

## 打印建议

靶板底部牺牲脚朝下，滑道开口朝上，层高 0.20 mm，外墙至少 3 圈。先打印滑套单件确认滑动间隙，再装到整根针体。打印脚只用于首层稳定，装配前必须切除，不能把脚当作测量基准。

这是几何和 STEP/STL 数字验证结果，尚未包含实际 G-code、传感器最小量程、PLA 收缩、床面附着和连续往复寿命验证。
"""
    (OUT / "START_HERE_T11_滑套说明.md").write_text(readme, encoding="utf-8")

    manifest = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(OUT.iterdir())
        if f.is_file() and f.name not in {"sha256.json", "independent_validation.json"}
    }
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = ROOT / "output" / "T11_SLIDER_SLEEVE_STEP_STL_PACKAGE.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file():
                z.write(f, f"{OUT.name}/{f.name}")
        for src in [Path(__file__), CAD / "laser100m_T10_onepiece.py",
                    CAD / "T10_design.md", CAD / "laser100m_T5.py",
                    CAD / "laser100m_T2_parameters.json"]:
            z.write(src, f"source/{src.name}")

    print(json.dumps({
        "output": str(OUT), "bundle": str(bundle),
        "finished_dimensions_mm": dims(clean),
        "print_dimensions_mm": dims(printable),
        "fit_checks": fit, "beam_checks": beam_checks,
        "preview": str(OUT / "T11_slider_sleeve_preview.png"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
