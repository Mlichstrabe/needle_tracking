"""Render a module-only exploded view for the T11 carrier assembly."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


CAD = Path(__file__).resolve().parent
ROOT = CAD.parent
OUT = ROOT / "output" / "cad_laser100m_T11_module_carrier"


def draw_shape(ax, shape, color, alpha=0.85):
    mesh = shape
    ax.add_collection3d(
        Poly3DCollection(mesh.triangles, facecolor=color, edgecolor=color,
                         linewidths=0.15, alpha=alpha)
    )


def center(shape):
    b = shape.bounds
    return tuple(((b[0] + b[1]) / 2).tolist())


def load_stl(name: str):
    path = OUT / name
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Use the already validated STL outputs so the preview does not depend on
    # CadQuery/OCP being importable in the plotting environment.
    lower = load_stl("01_lower_chassis_T11_CABLE_OPEN.stl")
    upper = load_stl("02_upper_cover_T11_CABLE_OPEN.stl")
    carrier = load_stl("03_MODULE_CARRIER_T11_GUIDE_GLUE.stl")
    laser = load_stl("REF_LASER_WITH_CABLE_CLEARANCE_PLUS_Y.stl")
    imu = load_stl("04_IMU_CASSETTE_DEFAULT_PLUS_X_TIP.stl")

    font_path = Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 9), facecolor="#f4f7fa", layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    # Display-only offsets: keep each part in its assembly orientation while
    # separating it vertically so the insertion order is unambiguous.
    items = [
        ("下壳｜固定外壳", lower, (0, 0, 0), "#7f8d96", 0.30, (78, -29, 2)),
        ("刚性模块载架｜导向+胶粘", carrier, (0, 0, 22), "#32788a", 0.88, (78, -15, 24)),
        ("测距模块｜侧后方出线", laser, (2, 9, 48), "#e49a40", 0.78, (78, -1, 50)),
        ("IMU载片｜+X朝针尖", imu, (2, 0, 72), "#55a66f", 0.92, (78, 14, 74)),
        ("上盖｜最后合上", upper, (0, 0, 98), "#aeb8bd", 0.28, (78, 29, 101)),
    ]

    centers = []
    for label, shape, offset, color, alpha, label_pos in items:
        moved = shape.copy()
        moved.apply_translation(offset)
        draw_shape(ax, moved, color, alpha)
        c = center(moved)
        centers.append(c)
        # Place concise labels at staggered positions to keep the stack readable.
        label_x, label_y, label_z = label_pos
        ax.plot([c[0], label_x - 3], [c[1], label_y], [c[2], label_z],
                color=color, lw=0.9, alpha=0.9)
        ax.text(label_x, label_y, label_z, label, fontsize=9, ha="left", va="center",
                bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": color, "alpha": .90})

    # Dashed insertion direction and numbered assembly order.
    for i in range(len(centers) - 1):
        ax.plot([30, 30], [28, 28], [centers[i][2] + 2, centers[i + 1][2] - 2],
                color="#77858c", ls="--", lw=1.0)
    ax.text(30, 30, 52, "装配方向 ↑", color="#5d6a70", fontsize=10,
            ha="center", va="center")

    ax.set_title("T11 模块部分爆炸图｜测距模块与 IMU 的装配关系", fontsize=16)
    ax.set_xlabel("轴向 X / mm")
    ax.set_ylabel("横向 Y / mm")
    ax.set_zlabel("装配高度 Z / mm")
    ax.set_xlim(0, 102)
    ax.set_ylim(-42, 42)
    ax.set_zlim(-30, 132)
    ax.view_init(elev=23, azim=-64)

    # A simple side section showing what slides into what.
    path = OUT / "T11_module_only_exploded_preview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
