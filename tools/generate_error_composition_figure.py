"""生成「当前项目误差构成」示意图，保存为 PNG。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "docs" / "figures" / "undated" / "project_error_composition.png"


def _configure_chinese():
    from matplotlib import font_manager, rcParams

    for name in ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei"):
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["font.family"] = "sans-serif"
            break
    rcParams["axes.unicode_minus"] = False


def main():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    _configure_chinese()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # (名称, 典型量级/表现, 影响等级0-3, 处理状态, 颜色)
    items = [
        (
            "B 坐标映射\n(IMU→场景)",
            "顺/逆时反、固定≈90°偏、\n45°像90° → 需交换XY+偏置",
            3,
            "部分处理\n(UI可调，未持久化)",
            "#ef5350",
        ),
        (
            "C 针轴几何\n(安装角)",
            "代码针轴45° vs 实物≈31°\n→ 倾角/幅度系统性偏差",
            3,
            "未处理\n(标定向导已回滚)",
            "#e53935",
        ),
        (
            "磁/融合模式\n(九轴 vs 六轴)",
            "室内磁干扰：九轴≈六轴或更差\n120s实测：六轴 4/5 项更优",
            2,
            "已实验结论\n(训练用六轴)",
            "#ff9800",
        ),
        (
            "显示平滑\n(跟手 vs 稳)",
            "EMA 平滑略滞后；\n默认「中」可能觉钝",
            1,
            "可调\n(弱/中/强/关)",
            "#ffb74d",
        ),
        (
            "A 传感器本体\n(芯片融合输出)",
            "静止倾角噪声通常 <1°\nYaw 受磁/环境波动大",
            1,
            "已验证\n(imu_accuracy_test)",
            "#66bb6a",
        ),
        (
            "几何一致\n(针长等)",
            "针长已统一 200mm\n(曾 100/162 不一致)",
            0,
            "已修复",
            "#42a5f5",
        ),
    ]

    fig = plt.figure(figsize=(14, 9), facecolor="#1a1d23")
    fig.suptitle(
        "needle_tracking（无配准版）— 当前误差构成",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    fig.text(
        0.5,
        0.92,
        "主程序针向 = 芯片四元数(0x59) → imu_kinematics 映射 → 3D 针体 ｜ 无光学真值",
        ha="center",
        fontsize=11,
        color="#b0bec5",
    )

    ax = fig.add_axes([0.06, 0.08, 0.58, 0.78])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(items) + 1)
    ax.axis("off")
    ax.set_title("按对「对准/跟手」影响排序（上→下）", fontsize=13, color="white", pad=12)

    impact_labels = ["已消除", "低", "中", "高"]
    impact_colors = ["#546e7a", "#66bb6a", "#ff9800", "#ef5350"]

    y = len(items)
    for name, desc, impact, status, color in items:
        h = 0.85
        # 影响条
        bar_w = 0.35 + impact * 1.15
        rect = FancyBboxPatch(
            (0.2, y - h / 2),
            bar_w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(0.35, y, name, va="center", ha="left", fontsize=11, color="white", fontweight="bold")
        ax.text(bar_w + 0.55, y + 0.12, desc, va="center", ha="left", fontsize=9.5, color="#eceff1")
        ax.text(bar_w + 0.55, y - 0.22, f"状态：{status}", va="center", ha="left", fontsize=9, color="#90a4ae")

        badge_x = bar_w + 0.15
        ax.add_patch(
            FancyBboxPatch(
                (badge_x, y - 0.28),
                0.9,
                0.35,
                boxstyle="round,pad=0.01",
                facecolor=impact_colors[impact],
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax.text(badge_x + 0.45, y - 0.1, f"影响{impact_labels[impact]}", ha="center", va="center", fontsize=8, color="white")

        y -= 1.15

    # 右侧：数据流 + 结论
    ax2 = fig.add_axes([0.68, 0.08, 0.28, 0.78])
    ax2.axis("off")
    ax2.set_title("数据流 & 结论", fontsize=13, color="white", pad=12)

    flow = [
        "IMU 芯片",
        "↓ 内部融合(建议六轴)",
        "四元数 0x59",
        "↓",
        "imu_kinematics",
        "  · 针体方向 45°模型",
        "  · 镜像/交换XY/偏置",
        "↓",
        "可选方向平滑",
        "↓",
        "3D 针体 + 对准角",
    ]
    box = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.02",
        facecolor="#263238",
        edgecolor="#546e7a",
        linewidth=1.5,
    )
    ax2.add_patch(box)
    for i, line in enumerate(flow):
        ax2.text(0.08, 0.92 - i * 0.085, line, fontsize=10, color="#cfd8dc", va="top", family="sans-serif")

    conclusions = [
        "── 工程结论 ──",
        "① 优先修 B+C（映射+针轴）",
        "② 六轴优于九轴(本场景)",
        "③ 磁融合仅离线研究",
        "④ 无光学真值→难定量验",
    ]
    for j, line in enumerate(conclusions):
        c = "#81c784" if j > 0 else "#ffd54f"
        ax2.text(0.08, 0.22 - j * 0.055, line, fontsize=9.5, color=c, va="top")

    legend_patches = [
        mpatches.Patch(color=impact_colors[3], label="高：主导误差，需标定/固化"),
        mpatches.Patch(color=impact_colors[2], label="中：环境/模式相关"),
        mpatches.Patch(color=impact_colors[1], label="低：芯片/显示层"),
        mpatches.Patch(color=impact_colors[0], label="已消除"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        facecolor="#263238",
        edgecolor="#546e7a",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"已保存: {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
