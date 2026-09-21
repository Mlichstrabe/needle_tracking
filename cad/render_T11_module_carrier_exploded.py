"""Render an exploded-view image for the T11 module-carrier assembly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


m = load_module("t11_module_carrier_for_exploded", CAD / "laser100m_T11_module_carrier.py")


def add_mesh(ax, mesh, color, alpha=0.8):
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolor=color,
                                         edgecolor=color, linewidths=.12, alpha=alpha))


def draw_shape(ax, shape, color, alpha=.8):
    add_mesh(ax, m.mesh_of(shape, .08), color, alpha)


def center(shape):
    b = m.exact_bounds(shape)
    return ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = __import__("json").loads((CAD / "laser100m_T2_parameters.json").read_text(encoding="utf-8"))
    params["housing"]["width"] = 46.0
    lower, upper, env = m.open_shells(params)
    carrier = m.build_module_carrier()
    imu = m.build_imu_cassette(1)
    needle = m.installed_needle()
    laser = m.laser_with_cable(1)

    # Vertical exploded stack. The offsets are display-only and are not CAD
    # assembly transforms; each component keeps its original coordinate frame.
    items = [
        ("下壳\nCable-open", lower, (0, 0, 0), "#7f8d96", .30),
        ("模块载架\nRigid guide + glue", carrier, (0, 0, 11), "#32788a", .82),
        ("测距模块 +\n排线净空", laser, (5, 15, 20), "#e49a40", .42),
        ("T10 针体\n保持不变", needle, (-8, -26, 27), "#46525a", .92),
        ("IMU 载片\n+X 朝针尖", imu, (0, -14, 35), "#55a66f", .92),
        ("上盖\nCable-open", upper, (0, 0, 48), "#aeb8bd", .28),
    ]

    font_path = Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 10), facecolor="#f4f7fa", layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, .85])
    ax = fig.add_subplot(gs[0, :], projection="3d")
    centers = []
    for label, shape, offset, color, alpha in items:
        moved = shape.translate(offset)
        draw_shape(ax, moved, color, alpha)
        c = center(moved)
        centers.append(c)
        label_y = c[1] + (10 if c[1] < -10 else 12)
        ax.plot([c[0], c[0]], [c[1], label_y], [c[2], c[2]], color=color, lw=.8, alpha=.8)
        ax.text(c[0], label_y, c[2], label, fontsize=9, ha="center", va="center",
                bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": color, "alpha": .86})
    # Exploded direction guide lines.
    base_x = 31
    for i in range(len(centers) - 1):
        ax.plot([base_x, base_x], [25, 25], [centers[i][2], centers[i + 1][2]],
                color="#8a969d", ls="--", lw=1)
    ax.text(31, 27, centers[0][2] + 4, "爆炸方向 / Z", color="#5d6a70", fontsize=9)
    ax.set_title("T11 模块载架版爆炸图｜排线开放、IMU 载片可 180° 翻转", fontsize=16)
    ax.set_xlabel("轴向 X / mm"); ax.set_ylabel("横向 Y / mm"); ax.set_zlabel("高度 Z / mm")
    ax.set_xlim(-225, 72); ax.set_ylim(-50, 60); ax.set_zlim(-30, 86)
    ax.view_init(elev=24, azim=-67)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.add_patch(plt.Rectangle((46.5, -9), 12.5, 16, color="#e49a40", alpha=.30,
                                label="侧后排线窗口"))
    ax2.axvline(46.5, color="#b26b10", ls="--", lw=1)
    ax2.axhline(-9.7, color="#b26b10", ls=":", lw=1.2, label="激光光轴")
    ax2.text(47, 8.5, "排线向 +Y / -Y 侧自然外出", color="#b26b10", fontsize=9)
    ax2.set_xlim(0, 65); ax2.set_ylim(-28, 20); ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X / mm"); ax2.set_ylabel("Z / mm"); ax2.grid(alpha=.2)
    ax2.legend(loc="upper left", fontsize=8); ax2.set_title("排线路径示意")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.add_patch(plt.Rectangle((10, -17), 46, 34, fill=False, edgecolor="#32788a", lw=2,
                                label="载架导向袋"))
    ax3.add_patch(plt.Rectangle((11, -15.5), 44, 31, fill=False, edgecolor="#55a66f", lw=2,
                                label="IMU 载片"))
    ax3.annotate("默认：+X 朝针尖", xy=(24, -16.2), xytext=(11, -22),
                 arrowprops={"arrowstyle": "->", "color": "#32788a"}, color="#32788a")
    ax3.annotate("翻转 180°：+X 朝后", xy=(47, 16.2), xytext=(31, 22),
                 arrowprops={"arrowstyle": "->", "color": "#b26b10"}, color="#b26b10")
    ax3.set_xlim(5, 61); ax3.set_ylim(-25, 25); ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X / mm"); ax3.set_ylabel("Y / mm"); ax3.grid(alpha=.2)
    ax3.legend(loc="lower right", fontsize=8); ax3.set_title("IMU 两个安装方向")

    path = OUT / "T11_module_carrier_exploded_preview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
