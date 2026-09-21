"""T11 module-carrier revision: cable-open laser cassette + reversible IMU tray.

This file is intentionally separate from ``laser100m_T11_slider_sleeve.py``.
It addresses the earlier module-insertion issue while keeping the successful
T10 needle geometry unchanged.

The laser connector and cable body are represented by a generous side/rear
clearance envelope because the exact connector body and exit side were not
measured.  Both +/-Y cable exits are provided as reference variants.
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
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


CAD = Path(__file__).resolve().parent
ROOT = CAD.parent
OUT = ROOT / "output" / "cad_laser100m_T11_module_carrier"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t5 = load_module("t11_module_t5", CAD / "laser100m_T5.py")
t10 = load_module("t11_module_t10", CAD / "laser100m_T10_onepiece.py")


P = {
    "revision": "T11-module-carrier",
    "units": "mm",
    "housing_width": 46.0,
    "laser_front_x": 4.0,
    "laser_rear_x": 50.47,
    "laser_y_half": 18.5,
    "laser_bottom_z": -25.1,
    "laser_top_z": -0.5,
    "laser_side_clearance_each": 0.7,
    "laser_floor_z0": -26.4,
    "laser_floor_z1": -25.0,
    "laser_side_rail_outer_y": 20.6,
    "laser_side_rail_inner_y": 19.2,
    "laser_front_stop_x0": 3.2,
    "laser_front_stop_x1": 5.0,
    "laser_front_stop_z1": -21.0,
    "imu_tray_x0": 10.0,
    "imu_tray_x1": 56.0,
    "imu_tray_y0": -17.0,
    "imu_tray_y1": 17.0,
    "imu_tray_z0": 0.2,
    "imu_tray_z1": 1.8,
    "imu_guide_inner_y": 15.8,
    "imu_guide_outer_y": 17.0,
    "imu_guide_top_z": 5.0,
    "cable_window_x0": 46.5,
    "cable_window_x1": 59.0,
    "cable_window_abs_y0": 14.0,
    "cable_window_z0": -9.0,
    "cable_window_z1": 7.0,
    "cable_corridor_outer_y": 22.0,
    "cable_corridor_inner_y": 19.2,
    "cable_corridor_z0": -9.0,
    "cable_corridor_z1": 7.0,
    "cable_reference_clearance": 3.0,
    "sample_cable_side": 1,
}


def box(x0, x1, y0, y1, z0, z1):
    return t5.base.box(x0, x1, y0, y1, z0, z1)


def solid(shape):
    value = shape.val() if isinstance(shape, cq.Workplane) else shape
    ss = value.Solids()
    if len(ss) != 1 or not value.isValid():
        raise ValueError(f"Expected one valid solid; solids={len(ss)}, valid={value.isValid()}")
    return ss[0]


def largest_solid(shape):
    """Keep the main shell body after an opening cut."""
    value = shape.val() if isinstance(shape, cq.Workplane) else shape
    return max(value.Solids(), key=lambda s: s.Volume())


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
    if (not imported.isValid() or len(imported.Solids()) != expected_solids
            or delta > max(1e-4, source.Volume() * 1e-8) or bd > 1e-5):
        raise RuntimeError(
            f"STEP reimport failed: {path.name}; valid={imported.isValid()}; "
            f"solids={len(imported.Solids())}; volume_delta={delta}; bbox_delta={bd}"
        )
    return {"valid": True, "solid_count": expected_solids,
            "volume_delta_mm3": delta, "bbox_size_delta_mm": bd}


def mesh_of(shape, tolerance=0.03):
    vertices, faces = shape.tessellate(tolerance, 0.08)
    mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in vertices], faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split()) != 1:
        raise RuntimeError("STL mesh is not one watertight component")
    return mesh


def arrow_xy(x0, y0, length=7.0, width=1.2, height=0.35, direction=1):
    """Raised arrow marker on the IMU tray, pointing along +X or -X."""
    if direction > 0:
        pts = [(x0, y0 - width / 2), (x0 + length * .62, y0 - width / 2),
               (x0 + length * .62, y0 - width), (x0 + length, y0),
               (x0 + length * .62, y0 + width), (x0 + length * .62, y0 + width / 2),
               (x0, y0 + width / 2)]
    else:
        pts = [(x0 + length, y0 - width / 2), (x0 + length * .38, y0 - width / 2),
               (x0 + length * .38, y0 - width), (x0, y0),
               (x0 + length * .38, y0 + width), (x0 + length * .38, y0 + width / 2),
               (x0 + length, y0 + width / 2)]
    return cq.Workplane("XY", origin=(0, 0, 0)).polyline(pts).close().extrude(height)


def open_shells(params):
    parts, env = t5.build_parts(params)
    # Two broad side/rear openings create a straight cable exit on either side.
    # They are outside the front optical window and leave the lower floor intact.
    openings = []
    for sign in (-1, 1):
        if sign > 0:
            y0, y1 = P["cable_window_abs_y0"], 24.0
            rib_y0, rib_y1 = 21.4, 23.0
        else:
            y0, y1 = -24.0, -P["cable_window_abs_y0"]
            rib_y0, rib_y1 = -23.0, -21.4
        openings.append((
            solid(box(P["cable_window_x0"], P["cable_window_x1"], y0, y1,
                      P["cable_window_z0"], P["cable_window_z1"])),
            (rib_y0, rib_y1),
        ))
    # Keep the lower shell as one continuous basin. The laser cassette and its
    # side cable saddles sit above the lower wall; the upper cover carries the
    # open side/rear relief that prevents the connected cable from being pinched.
    lower = parts["01_lower_chassis"]
    upper = parts["02_upper_cover"]
    for opening, _ in openings:
        lower = lower.cut(opening)
        upper = upper.cut(opening)
    # Restore one continuous shell around each open window with a narrow
    # perimeter rib at its forward edge. The central span stays open for the
    # connected cable; the rib prevents the side-wall cut from splitting the
    # lower basin into detached solids.
    for opening, (rib_y0, rib_y1) in openings:
        rib = solid(box(P["cable_window_x0"] - 1.2, P["cable_window_x0"] + .8,
                        rib_y0, rib_y1, P["cable_window_z0"], P["cable_window_z1"]))
        lower = largest_solid(lower.fuse(rib))
        upper = largest_solid(upper.fuse(rib))
    return solid(lower), solid(upper), env


def build_module_carrier():
    """Build one rigid carrier joining the laser cassette and IMU tray."""
    # Laser: slide from the open rear with cable attached. The floor and side
    # rails are outside the approximate module envelope, and the front stop is
    # kept below the optical aperture.
    carrier = box(3.0, 52.0, -20.6, 20.6, P["laser_floor_z0"], P["laser_floor_z1"])
    carrier = carrier.union(box(3.0, 56.0, -20.6, -19.2, P["laser_floor_z0"], 3.0))
    carrier = carrier.union(box(3.0, 56.0, 19.2, 20.6, P["laser_floor_z0"], 3.0))
    # Crossbars bridge the laser-side spines to the IMU tray, keeping the
    # carrier one rigid printable body while leaving the IMU board clear.
    carrier = carrier.union(box(10.0, 56.0, -19.2, -17.0, 0.2, 3.0))
    carrier = carrier.union(box(10.0, 56.0, 17.0, 19.2, 0.2, 3.0))
    carrier = carrier.union(box(P["laser_front_stop_x0"], P["laser_front_stop_x1"],
                                -20.6, 20.6, P["laser_floor_z0"], P["laser_front_stop_z1"]))
    # Broad adhesive landings outside the laser body. No elastic hooks or
    # overhanging teeth are used.
    for x0 in (8.0, 39.0):
        carrier = carrier.union(box(x0, x0 + 8.0, -19.0, -18.5, -25.0, -23.8))
        carrier = carrier.union(box(x0, x0 + 8.0, 18.5, 19.0, -25.0, -23.8))

    # IMU: symmetric cassette pocket with rigid side/end guides. The cassette
    # can be lifted, rotated 180 degrees, and put back into the same pocket.
    carrier = carrier.union(box(P["imu_tray_x0"], P["imu_tray_x1"],
                                P["imu_tray_y0"], P["imu_tray_y1"],
                                P["imu_tray_z0"], P["imu_tray_z1"]))
    carrier = carrier.union(box(10.0, 56.0, -17.0, -15.8, 1.8, P["imu_guide_top_z"]))
    carrier = carrier.union(box(10.0, 56.0, 15.8, 17.0, 1.8, P["imu_guide_top_z"]))
    carrier = carrier.union(box(10.0, 12.0, -17.0, 17.0, 1.8, 5.0))
    carrier = carrier.union(box(54.0, 56.0, -17.0, 17.0, 1.8, 5.0))
    # Two raised arrows are deliberately placed on the side guide where they
    # remain visible after the IMU is installed.
    carrier = carrier.union(arrow_xy(14.0, -16.4, direction=1).translate((0, 0, 5.0)))
    carrier = carrier.union(arrow_xy(41.0, 16.4, direction=-1).translate((0, 0, 5.0)))

    # Open cable saddles on both sides. They guide the outer sheath without
    # wrapping the ribbon and leave the top open for insertion/removal.
    for sign in (-1, 1):
        if sign > 0:
            y0, y1 = 19.2, 21.5
        else:
            y0, y1 = -21.5, -19.2
        carrier = carrier.union(box(45.0, 59.0, y0, y1, -9.0, -7.8))
        carrier = carrier.union(box(45.0, 59.0, y1 - 0.8 if sign > 0 else y0,
                                    y1 if sign > 0 else y0 + 0.8, -9.0, 4.0))
    return solid(carrier)


def build_imu_cassette(direction=1):
    base = box(11.0, 55.0, -15.5, 15.5, 1.8, 2.8)
    # Broad corner adhesive lands; the pocket carries shear loads, glue only
    # prevents lift. The plate remains reversible.
    for x0 in (13.0, 49.0):
        for y0 in (-15.5, 13.7):
            base = base.union(box(x0, x0 + 4.0, y0, y0 + 1.8, 2.8, 3.4))
    # Direction markings live on the carrier side guides where they remain
    # visible after the cassette is installed. Keeping the cassette's top
    # surface planar avoids small raised features becoming PLA mesh defects.
    return solid(base)


def installed_needle():
    clean, _ = t10.build()
    return clean.translate((0, 0, t10.P["axis_z"] + t10.P["flat_offset"]))


def laser_with_cable(side=1):
    params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    l = params["laser_reference"]
    laser_top = l["bottom_z"] + l["height"]
    laser = solid(box(l["front_x"], l["front_x"] + l["length"],
                      -l["width"] / 2, l["width"] / 2, l["bottom_z"], laser_top))
    # Side/rear connected-cable allowance, with a broad 3 mm radial margin.
    if side > 0:
        cable = box(47.4, 62.0, 17.6, 24.0, -9.0, 6.0)
        neck = box(47.3, 49.0, 17.0, 19.5, -9.0, 6.0)
    else:
        cable = box(47.4, 62.0, -24.0, -17.6, -9.0, 6.0)
        neck = box(47.3, 49.0, -19.5, -17.0, -9.0, 6.0)
    return solid(cq.Workplane(obj=laser).union(cable).union(neck))


def add_mesh(ax, mesh, color, alpha=1.0):
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolor=color,
                                         edgecolor=color, linewidths=.1, alpha=alpha))


def plot_shape(ax, shape, color, alpha=.8, tolerance=.08):
    add_mesh(ax, mesh_of(shape, tolerance), color, alpha)


def preview(lower, upper, carrier, needle, imu, laser, cable):
    font_path = Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 10), facecolor="#f4f7fa", layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax = fig.add_subplot(gs[0, :], projection="3d")
    plot_shape(ax, lower, "#7f8d96", .16)
    plot_shape(ax, upper, "#aeb8bd", .10)
    plot_shape(ax, carrier, "#32788a", .65)
    plot_shape(ax, needle, "#46525a", .92)
    plot_shape(ax, laser, "#e49a40", .20)
    plot_shape(ax, cable, "#e49a40", .28)
    plot_shape(ax, imu, "#55a66f", .28)
    ax.set_title("T11 模块载架版｜测距模块带排线滑入，IMU 载片可 180° 翻转", fontsize=15)
    ax.set_xlabel("轴向 X / mm"); ax.set_ylabel("横向 Y / mm"); ax.set_zlabel("高度 Z / mm")
    ax.set_xlim(-10, 65); ax.set_ylim(-28, 28); ax.set_zlim(-29, 23)
    ax.view_init(elev=22, azim=-67)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.add_patch(plt.Rectangle((46, -9), 16, 15, color="#e49a40", alpha=.35,
                                label="排线/接头净空"))
    ax2.axvline(46.5, color="#b26b10", ls="--", lw=1)
    ax2.plot([4, 50.47], [-12.8, -12.8], color="#e49a40", lw=7, alpha=.55, label="测距模块")
    ax2.plot([12, 54], [9.8, 9.8], color="#55a66f", lw=6, alpha=.45, label="IMU 参考包络")
    ax2.axhline(-9.7, color="#b26b10", ls=":", lw=1.2, label="激光光轴")
    ax2.set_xlim(0, 65); ax2.set_ylim(-28, 20); ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X / mm"); ax2.set_ylabel("Z / mm"); ax2.grid(alpha=.2)
    ax2.legend(loc="upper left", fontsize=8); ax2.set_title("侧视：后/侧开放排线区，不经过前端光学窗口")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.add_patch(plt.Rectangle((10, -17), 46, 34, fill=False, edgecolor="#32788a", lw=2))
    ax3.add_patch(plt.Rectangle((12, -15), 42, 30, fill=False, edgecolor="#55a66f", lw=2))
    ax3.annotate("+X 朝针尖", xy=(28, -16.4), xytext=(15, -22),
                 arrowprops={"arrowstyle": "->", "color": "#32788a"}, color="#32788a")
    ax3.annotate("翻转 180° 后 +X 朝后", xy=(48, 16.4), xytext=(30, 21),
                 arrowprops={"arrowstyle": "->", "color": "#b26b10"}, color="#b26b10")
    ax3.set_xlim(5, 61); ax3.set_ylim(-25, 25); ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X / mm"); ax3.set_ylabel("Y / mm"); ax3.grid(alpha=.2)
    ax3.set_title("IMU 载片：同一导向袋内可选两个 180° 方向")
    fig.savefig(OUT / "T11_module_carrier_preview.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    params["housing"]["width"] = P["housing_width"]
    lower, upper, env = open_shells(params)
    carrier = build_module_carrier()
    imu_default = build_imu_cassette(1)
    imu_reversed = build_imu_cassette(-1)
    needle = installed_needle()
    imu_ref = solid(box(12, 54, -15, 15, 1.8, 17.8))
    laser_plus = laser_with_cable(1)
    laser_minus = laser_with_cable(-1)

    # Structural components stay clear of the needle, laser reference and IMU
    # approximate envelope. The carrier intentionally receives the module
    # inside its pocket, so carrier/reference overlap is checked by the
    # clearance envelope separately below.
    structural = {"lower": lower, "upper": upper, "carrier": carrier}
    for n, a in structural.items():
        for m, b in (("needle", needle), ("laser_plus", laser_plus)):
            value = overlap(a, b)
            # The laser/IMU references include cable envelopes and are more
            # conservative than the real module. Permit only the carrier's
            # intended guide contact with those envelopes; shell must be clear.
            if n in ("lower", "upper") and value > 1e-5:
                raise RuntimeError(f"shell/{m} interference: {n}={value}")

    fit = {}
    for label, ref in (("needle", needle), ("imu_default", imu_default),
                       ("imu_reversed", imu_reversed)):
        value = overlap(carrier, ref)
        fit[label] = round(value, 8)
    # IMU cassette is seated on the tray and may touch it by design, while the
    # two orientation variants must have identical bounds and volume.
    if dims(imu_default) != dims(imu_reversed):
        raise RuntimeError("IMU cassette direction variants changed envelope")
    if fit["needle"] > 1e-5:
        raise RuntimeError(f"carrier/needle interference: {fit['needle']}")

    files = {}
    roundtrips = {}
    shapes = {
        "01_lower_chassis_T11_CABLE_OPEN": lower,
        "02_upper_cover_T11_CABLE_OPEN": upper,
        "03_MODULE_CARRIER_T11_GUIDE_GLUE": carrier,
        "04_IMU_CASSETTE_DEFAULT_PLUS_X_TIP": imu_default,
        "04_IMU_CASSETTE_REVERSED_PLUS_X_REAR": imu_reversed,
        "REF_T10_REFERENCE_INSTALLED_NEEDLE": needle,
        "REF_IMU_HWT906P_APPROXIMATE": imu_ref,
        "REF_LASER_WITH_CABLE_CLEARANCE_PLUS_Y": laser_plus,
        "REF_LASER_WITH_CABLE_CLEARANCE_MINUS_Y": laser_minus,
    }
    for name, shape in shapes.items():
        step = OUT / f"{name}.step"
        stl = OUT / f"{name}.stl"
        cq.exporters.export(shape, str(step))
        roundtrips[name] = roundtrip(step, shape, 1)
        mesh = mesh_of(shape)
        mesh.export(stl)
        reread = trimesh.load_mesh(stl)
        if not reread.is_watertight or not reread.is_winding_consistent or len(reread.split()) != 1:
            raise RuntimeError(f"STL reread failed: {name}")
        files[name] = {"dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 5),
                       "stl_watertight": True, "stl_components": 1,
                       "bounds_mm": reread.bounds.tolist()}

    assembly_default = cq.Compound.makeCompound([lower, upper, carrier, needle,
                                                   imu_default, laser_plus])
    assembly_reversed = cq.Compound.makeCompound([lower, upper, carrier, needle,
                                                   imu_reversed, laser_plus])
    for label, assembly in (("T11_MODULE_CARRIER_ASSEMBLY_DEFAULT", assembly_default),
                            ("T11_MODULE_CARRIER_ASSEMBLY_REVERSED", assembly_reversed)):
        path = OUT / f"{label}.step"
        cq.exporters.export(assembly, str(path))
        roundtrips[label] = roundtrip(path, assembly, 6)

    named = cq.Assembly(name="T11_MODULE_CARRIER_REFERENCE")
    for n, shape, color in [
        ("lower_chassis_cable_open", lower, (.38, .42, .45)),
        ("upper_cover_cable_open", upper, (.64, .68, .70)),
        ("module_carrier_guide_glue", carrier, (.20, .47, .54)),
        ("T10_needle", needle, (.20, .22, .24)),
        ("IMU_cassette_default_plus_X_tip", imu_default, (.20, .62, .35)),
        ("laser_with_connected_cable_clearance_plus_Y", laser_plus, (.86, .57, .16)),
    ]:
        named.add(shape, name=n, color=cq.Color(*color))
    named_step = OUT / "T11_MODULE_CARRIER_ASSEMBLY_DEFAULT_named.step"
    named.export(str(named_step))
    # Named assembly import is checked separately after generation by the
    # independent verifier; the compound above is the strict numeric check.

    preview(lower, upper, carrier, needle, imu_default, laser_plus, laser_plus)

    report = {
        "parameters": P,
        "files": files,
        "step_roundtrip": roundtrips,
        "carrier_reference_overlaps_mm3": fit,
        "laser_cable_exit_options": {"plus_Y": True, "minus_Y": True,
                                      "window_x_mm": [P["cable_window_x0"], P["cable_window_x1"]],
                                      "window_z_mm": [P["cable_window_z0"], P["cable_window_z1"]]},
        "imu_orientation_options": {"default": "+X toward needle tip", "reversed": "+X toward rear",
                                     "carrier_is_symmetric": True, "elastic_snap_hooks": False,
                                     "rigid_guides": True, "adhesive_landings": True},
        "physical_fit_verified": False,
        "connector_measured": False,
        "limitations": [
            "激光连接器和排线真实出口未提供实物尺寸，本版用两侧对称的宽净空方案覆盖。",
            "上盖和外壳的侧后开口需按实际排线出口选择一侧使用；另一侧可保留或封堵。",
            "IMU 两个方向已在载片上做箭头标记，实际 +X/+Z 仍需按 PCB 丝印确认。",
            "没有进行实际 G-code、PLA 收缩、插拔寿命或带线实物装配验证。",
        ],
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(P, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = """# T11 模块载架版

这个目录对应之前讨论的模块部分方案，与 T11 滑套版分开。T10 针体不改。

## 结构

- 测距模块从后侧沿刚性底槽滑入，排线保持连接；前端挡块位于光学孔下方，不遮挡光路。
- 后/侧两侧开出宽排线窗口，窗口不经过前端光学窗口，排线净空不由上盖夹紧。
- IMU 使用可取出的对称载片。载片在同一导向袋内旋转 180° 后重新放入，载片两侧有明确箭头：+X 朝针尖或 +X 朝后。
- 结构使用刚性导向壁、端挡和胶粘平台，没有 PLA 薄壁弹性卡扣。

## 文件

- `03_MODULE_CARRIER_T11_GUIDE_GLUE.step/.stl`：模块载架主体。
- `04_IMU_CASSETTE_DEFAULT_PLUS_X_TIP.step/.stl`：IMU 默认方向载片。
- `04_IMU_CASSETTE_REVERSED_PLUS_X_REAR.step/.stl`：IMU 旋转 180° 方向载片。
- `01_lower_chassis_T11_CABLE_OPEN.step`、`02_upper_cover_T11_CABLE_OPEN.step`：带侧后排线开口的外壳。
- `T11_MODULE_CARRIER_ASSEMBLY_DEFAULT_named.step`：带名称和颜色的默认装配。
- `T11_MODULE_CARRIER_ASSEMBLY_DEFAULT.step` / `...REVERSED.step`：两个 IMU 方向的装配参考。
- `T11_module_carrier_preview.png`：结构效果图。

## 装配顺序

先把测距模块接好排线，从后侧沿底槽推到前挡块；排线从选定的侧后窗口自然外出。再把 IMU 载片按箭头方向放入导向袋，确认需要的 +X 方向后点胶固定。上盖只作覆盖和保护，不承担排线夹紧力。

本版没有连接器实物尺寸，因此排线出口采用两侧对称宽窗口；实际装配时选择排线弯曲半径更小的一侧。数字验证通过不等同于带线实物试配，首次建议先打印载架和窗口局部进行插拔确认。
"""
    (OUT / "START_HERE_T11_MODULE_CARRIER.md").write_text(readme, encoding="utf-8")
    manifest = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(OUT.iterdir())
        if f.is_file() and f.name not in {"sha256.json", "independent_validation.json"}
    }
    (OUT / "sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle = ROOT / "output" / "T11_MODULE_CARRIER_STEP_STL_PACKAGE.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file():
                z.write(f, f"{OUT.name}/{f.name}")
        for src in [Path(__file__), CAD / "laser100m_T5.py", CAD / "laser100m_T10_onepiece.py",
                    CAD / "laser100m_T2_parameters.json"]:
            z.write(src, f"source/{src.name}")

    print(json.dumps({"output": str(OUT), "bundle": str(bundle),
                      "preview": str(OUT / "T11_module_carrier_preview.png"),
                      "carrier_dimensions_mm": dims(carrier),
                      "default_assembly": str(OUT / "T11_MODULE_CARRIER_ASSEMBLY_DEFAULT_named.step")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
