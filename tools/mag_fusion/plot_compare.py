"""
对比 q_chip / q_6dof / q_pc：针轴夹角曲线、指标柱状图、文字结论。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.imu_kinematics import needle_axis_scene_normalized  # noqa: E402

from tools.mag_fusion.fusion import quat_angle_deg  # noqa: E402
from tools.mag_fusion.metrics import QuaternionTrackMetrics, verdict_text  # noqa: E402


def _configure_matplotlib_chinese() -> Optional[str]:
    try:
        from matplotlib import font_manager, rcParams
    except ImportError:
        return None
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen:
        rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
    return chosen


def _angle_series(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    n = min(len(q_a), len(q_b))
    out = np.full(n, np.nan)
    for i in range(n):
        if np.all(np.isfinite(q_a[i])) and np.all(np.isfinite(q_b[i])):
            out[i] = quat_angle_deg(q_a[i], q_b[i])
    return out


def plot_comparison(
    t: np.ndarray,
    q_chip: np.ndarray,
    q_pc: np.ndarray,
    q_6dof: np.ndarray,
    mag: np.ndarray,
    w_mag: np.ndarray,
    metrics: Dict[str, object],
    *,
    title: str = "离线融合对比",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("请安装 matplotlib: pip install matplotlib") from e

    _configure_matplotlib_chinese()

    chip: QuaternionTrackMetrics = metrics["chip"]
    pc: QuaternionTrackMetrics = metrics["pc"]
    six: QuaternionTrackMetrics = metrics["six_dof"]

    ang_pc = _angle_series(q_pc, q_chip)
    ang_6 = _angle_series(q_6dof, q_chip)
    mag_abs = np.linalg.norm(mag, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, ang_pc, color="#81c784", linewidth=1.2, label="优化九轴 vs 芯片九轴")
    ax.plot(t, ang_6, color="#ffb74d", linewidth=1.0, alpha=0.85, label="纯六轴 vs 芯片九轴")
    ax.set_title("相对芯片姿态的夹角")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("夹角 (°)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, mag_abs, color="#4fc3f7", linewidth=1.0, label="|B|")
    ax2 = ax.twinx()
    ax2.plot(t, w_mag, color="#ce93d8", linewidth=0.9, alpha=0.8, label="w_mag")
    ax.set_title("磁场模长与磁修正权重")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("|B|")
    ax2.set_ylabel("w_mag")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    labels = ["针轴抖动RMS°", "Yaw最大步°", "静止漂移°", "大跳变次数"]
    vals_chip = [chip.needle_jitter_deg_rms, chip.yaw_max_step_deg, chip.yaw_drift_still_deg, chip.yaw_big_jumps]
    vals_pc = [pc.needle_jitter_deg_rms, pc.yaw_max_step_deg, pc.yaw_drift_still_deg, pc.yaw_big_jumps]
    vals_6 = [six.needle_jitter_deg_rms, six.yaw_max_step_deg, six.yaw_drift_still_deg, six.yaw_big_jumps]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, vals_chip, w, label="芯片九轴", color="#4fc3f7")
    ax.bar(x, vals_pc, w, label="优化九轴", color="#81c784")
    ax.bar(x + w, vals_6, w, label="纯六轴", color="#ffb74d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("稳定性指标（越小越稳；静止漂移体现九轴长期优势）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.05, 0.95, verdict_text(metrics), va="top", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _needle_jitter_series(quats: np.ndarray) -> np.ndarray:
    """逐帧针轴方向相对全程均值方向的夹角（度）；越平越稳。"""
    axes = []
    for q in quats:
        d = needle_axis_scene_normalized(q) if np.all(np.isfinite(q)) else None
        axes.append(d if d is not None else [np.nan, np.nan, np.nan])
    a = np.array(axes, dtype=float)
    valid = np.all(np.isfinite(a), axis=1)
    out = np.full(len(a), np.nan)
    if int(np.sum(valid)) < 2:
        return out
    mean = np.nanmean(a[valid], axis=0)
    mn = np.linalg.norm(mean)
    if mn < 1e-6:
        return out
    mean = mean / mn
    dots = np.clip(np.sum(a[valid] * mean, axis=1), -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(dots))
    return out


def _yaw_offset_series(quats: np.ndarray) -> np.ndarray:
    """Yaw 相对起始基线的偏移量随时间（unwrap 后减去起始窗口均值）。"""
    from tools.mag_fusion.metrics import _unwrap_yaw_series, _yaw_from_quat

    valid = np.array([np.all(np.isfinite(q)) for q in quats])
    out = np.full(len(quats), np.nan)
    if int(np.sum(valid)) < 5:
        return out
    yaw = np.array([_yaw_from_quat(q) if v else np.nan for q, v in zip(quats, valid)])
    yu = _unwrap_yaw_series(yaw[valid])
    k = max(3, len(yu) // 20)
    base = float(np.mean(yu[:k]))
    out[valid] = yu - base
    return out


def _improvement_pct(a: float, b: float) -> float:
    """B 相对 A 的改善百分比（指标越小越好；正=B 更好）。"""
    if a is None or b is None or not np.isfinite(a) or a <= 1e-9:
        return 0.0
    return (a - b) / a * 100.0


def _needle_step_deg(quats: np.ndarray) -> np.ndarray:
    """相邻帧针轴方向夹角（度）。"""
    axes = []
    for q in quats:
        d = needle_axis_scene_normalized(q) if np.all(np.isfinite(q)) else None
        axes.append(d)
    n = len(axes)
    out = np.full(n, np.nan)
    for i in range(1, n):
        a, b = axes[i - 1], axes[i]
        if a is None or b is None:
            continue
        da, db = np.array(a, float), np.array(b, float)
        dot = float(np.clip(np.dot(da, db), -1.0, 1.0))
        out[i] = float(np.degrees(np.arccos(dot)))
    return out


def _yaw_step_series(quats: np.ndarray) -> np.ndarray:
    """相邻帧 Yaw 变化（unwrap 后，度）。"""
    from tools.mag_fusion.metrics import _unwrap_yaw_series, _yaw_from_quat

    valid = np.array([np.all(np.isfinite(q)) for q in quats])
    if int(np.sum(valid)) < 2:
        return np.full(len(quats), np.nan)
    yaw = np.array([_yaw_from_quat(q) if v else np.nan for q, v in zip(quats, valid)])
    yu = _unwrap_yaw_series(yaw[valid])
    steps = np.full(len(quats), np.nan)
    idx = np.where(valid)[0]
    for j in range(1, len(yu)):
        steps[idx[j]] = abs(float(yu[j] - yu[j - 1]))
    return steps


def _set_fig_title(fig, title: str) -> None:
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def _draw_events(ax, events: Optional[list], ylim=None):
    if not events:
        return
    for te, ev in events:
        ax.axvline(te, color="#ff9800", linestyle="--", linewidth=1.2, alpha=0.85)
        y = ax.get_ylim()[1] if ylim is None else ylim[1]
        ax.text(te, y, f" {ev}", fontsize=8, color="#ff9800", rotation=90, va="top")


def _mark_spikes(ax, t, y, thresh: float, color: str):
    m = np.isfinite(y) & (y >= thresh)
    if int(np.sum(m)) > 0:
        ax.scatter(t[m], y[m], s=36, c=color, zorder=5, edgecolors="white", linewidths=0.5)


def _pair_verdict_lines(
    label_a: str,
    label_b: str,
    metrics_a: QuaternionTrackMetrics,
    metrics_b: QuaternionTrackMetrics,
) -> list:
    va = [
        metrics_a.needle_jitter_deg_rms,
        metrics_a.yaw_max_step_deg,
        metrics_a.yaw_drift_still_deg,
        metrics_a.yaw_end_drift_deg,
        metrics_a.yaw_big_jumps,
    ]
    vb = [
        metrics_b.needle_jitter_deg_rms,
        metrics_b.yaw_max_step_deg,
        metrics_b.yaw_drift_still_deg,
        metrics_b.yaw_end_drift_deg,
        metrics_b.yaw_big_jumps,
    ]
    names = ["针轴抖动", "Yaw最大步", "静止漂移", "终端偏移", "大跳变"]
    lines = [f"对比: {label_a}  vs  {label_b}", ""]
    wins_b = 0
    valid_items = 0
    for nm, a_v, b_v in zip(names, va, vb):
        imp = _improvement_pct(a_v, b_v)
        if a_v <= 1e-9 and b_v <= 1e-9:
            verdict = "持平(均≈0)"
        else:
            valid_items += 1
            if b_v < a_v:
                wins_b += 1
                verdict = f"{label_b} 更好 (↓{imp:.0f}%)"
            elif b_v > a_v:
                verdict = f"{label_a} 更好 (↑{-imp:.0f}%)"
            else:
                verdict = "持平"
        lines.append(f"• {nm}: {a_v:.3f} vs {b_v:.3f} → {verdict}")
    lines.append("")
    if valid_items:
        lines.append(f"综合: {label_b} 在 {wins_b}/{valid_items} 项更优。")
    lines.append("")
    lines.append("看图提示: 窗口②帧间跳变=突变最明显；窗口③Yaw漂移=长测谁更漂。")
    return lines


def plot_three_comparison(
    t: np.ndarray,
    tracks: list,
    *,
    mag: Optional[np.ndarray] = None,
    title: str = "三轨迹对比",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> int:
    """Three-track comparison for six-axis, adaptive fusion, and always-on magnetic fusion."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("请安装 matplotlib: pip install matplotlib") from e

    _configure_matplotlib_chinese()
    colors = ["#ffb74d", "#4fc3f7", "#81c784"]
    figs = []

    def _save(fig, suffix: str):
        if save_path:
            p = Path(save_path)
            fig.savefig(p.with_name(f"{p.stem}_{suffix}{p.suffix}"), dpi=130)

    series = []
    for (label, quats, metrics), color in zip(tracks, colors):
        series.append(
            {
                "label": label,
                "q": quats,
                "m": metrics,
                "color": color,
                "jit": _needle_jitter_series(quats),
                "yaw": _yaw_offset_series(quats),
                "step_n": _needle_step_deg(quats),
                "step_y": _yaw_step_series(quats),
            }
        )

    fig1, ax1 = plt.subplots(figsize=(11, 4))
    _set_fig_title(fig1, f"① 针轴抖动 — {title}")
    fig1.suptitle(f"① 针轴方向抖动（越平越稳）\n{title}", fontsize=12)
    for s in series:
        ax1.plot(t, s["jit"], color=s["color"], lw=1.5, label=s["label"])
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("偏离均值方向 (°)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.35)
    fig1.tight_layout()
    figs.append(fig1)
    _save(fig1, "01_jitter")

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig2.suptitle(f"② 帧间跳变（三轨迹）\n{title}", fontsize=12)
    for s in series:
        ax2a.plot(t, s["step_n"], color=s["color"], lw=1.2, label=f"{s['label']} 针轴步进")
        ax2b.plot(t, s["step_y"], color=s["color"], lw=1.2, label=f"{s['label']} Yaw步进")
    ax2a.set_ylabel("针轴 帧间角 (°)")
    ax2a.legend(loc="upper right", fontsize=8)
    ax2a.grid(alpha=0.35)
    ax2b.set_xlabel("时间 (s)")
    ax2b.set_ylabel("Yaw 帧间变化 (°)")
    ax2b.legend(loc="upper right", fontsize=8)
    ax2b.grid(alpha=0.35)
    fig2.tight_layout()
    figs.append(fig2)
    _save(fig2, "02_frame_steps")

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    _set_fig_title(fig3, f"③ Yaw漂移 — {title}")
    fig3.suptitle(f"③ Yaw 相对起始偏移（三轨迹）\n{title}", fontsize=12)
    for s in series:
        ax3.plot(t, s["yaw"], color=s["color"], lw=1.8, label=s["label"])
    ax3.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax3.set_xlabel("时间 (s)")
    ax3.set_ylabel("Yaw 偏移 (°)")
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.35)
    fig3.tight_layout()
    figs.append(fig3)
    _save(fig3, "03_yaw_drift")

    if mag is not None:
        fig4, ax4 = plt.subplots(figsize=(11, 4))
        fig4.suptitle(f"④ 磁场 |B|\n{title}", fontsize=12)
        mag_abs = np.linalg.norm(mag, axis=1)
        b0 = float(np.nanmedian(mag_abs[: max(50, len(mag_abs) // 20)]))
        ax4.plot(t, mag_abs, color="#ce93d8", lw=1.5, label="|B|")
        if b0 > 0:
            ax4.axhline(b0, color="#4fc3f7", ls="--", alpha=0.7, label=f"基线≈{b0:.0f}")
            ax4.axhline(b0 * 1.3, color="#ef5350", ls=":", alpha=0.7, label="1.3×基线")
        ax4.set_xlabel("时间 (s)")
        ax4.set_ylabel("|B|")
        ax4.legend(loc="upper right")
        ax4.grid(alpha=0.35)
        fig4.tight_layout()
        figs.append(fig4)
        _save(fig4, "04_mag")

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    fig5.suptitle(f"⑤ 三轨迹指标对比（越小越稳）\n{title}", fontsize=12)
    bar_labels = ["针轴抖动RMS", "Yaw最大步", "静止漂移", "终端偏移", "大跳变"]
    x = np.arange(len(bar_labels))
    width = 0.25
    for i, s in enumerate(series):
        m = s["m"]
        vals = [
            m.needle_jitter_deg_rms,
            m.yaw_max_step_deg,
            m.yaw_drift_still_deg,
            m.yaw_end_drift_deg,
            m.yaw_big_jumps,
        ]
        bars = ax5.bar(x + (i - 1) * width, vals, width, color=s["color"], label=s["label"], alpha=0.9)
        for bar, val in zip(bars, vals):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax5.set_xticks(x)
    ax5.set_xticklabels(bar_labels)
    ax5.legend(loc="upper right")
    ax5.grid(axis="y", alpha=0.3)
    fig5.tight_layout()
    figs.append(fig5)
    _save(fig5, "05_bars")

    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)
    return len(figs)


def plot_pair_comparison(
    t: np.ndarray,
    q_a: np.ndarray,
    q_b: np.ndarray,
    label_a: str,
    label_b: str,
    metrics_a: QuaternionTrackMetrics,
    metrics_b: QuaternionTrackMetrics,
    *,
    mag: Optional[np.ndarray] = None,
    events: Optional[list] = None,
    title: str = "两轨迹对比",
    save_path: Optional[Path] = None,
    show: bool = True,
    multi_window: bool = True,
    jump_highlight_deg: float = 2.0,
) -> int:
    """
    两轨迹对比。multi_window=True 时拆成多个独立窗口，突变用「帧间跳变」图放大显示。
    返回打开的窗口数量。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("请安装 matplotlib: pip install matplotlib") from e

    _configure_matplotlib_chinese()

    col_a, col_b = "#4fc3f7", "#81c784"
    jit_a = _needle_jitter_series(q_a)
    jit_b = _needle_jitter_series(q_b)
    step_n_a = _needle_step_deg(q_a)
    step_n_b = _needle_step_deg(q_b)
    step_y_a = _yaw_step_series(q_a)
    step_y_b = _yaw_step_series(q_b)
    off_a = _yaw_offset_series(q_a)
    off_b = _yaw_offset_series(q_b)
    diff_off = np.abs(off_a - off_b)

    figs = []

    def _save(fig, suffix: str):
        if save_path:
            p = Path(save_path)
            fig.savefig(p.with_name(f"{p.stem}_{suffix}{p.suffix}"), dpi=130)

    if not multi_window:
        # 兼容：单窗口 2×2（旧版）
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14)
        figs.append(fig)
        axes[0, 0].plot(t, jit_a, color=col_a, lw=1.2, label=label_a)
        axes[0, 0].plot(t, jit_b, color=col_b, lw=1.2, label=label_b)
        axes[0, 1].plot(t, off_a, color=col_a, lw=1.3, label=label_a)
        axes[0, 1].plot(t, off_b, color=col_b, lw=1.3, label=label_b)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return 1

    # —— 窗口 1：针轴偏离均值 ——
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    _set_fig_title(fig1, f"① 针轴抖动 — {title}")
    fig1.suptitle(f"① 针轴方向抖动（越平越稳）\n{title}", fontsize=12)
    ax1.plot(t, jit_a, color=col_a, lw=1.5, label=label_a)
    ax1.plot(t, jit_b, color=col_b, lw=1.5, label=label_b)
    _draw_events(ax1, events)
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("偏离均值方向 (°)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.35)
    fig1.tight_layout()
    figs.append(fig1)
    _save(fig1, "01_jitter")

    # —— 窗口 2：帧间跳变（突变放大）——
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig2.suptitle(f"② 帧间跳变（突变最明显，>={jump_highlight_deg}° 打点）\n{title}", fontsize=12)
    ax2a.plot(t, step_n_a, color=col_a, lw=1.2, label=f"{label_a} 针轴步进")
    ax2a.plot(t, step_n_b, color=col_b, lw=1.2, label=f"{label_b} 针轴步进")
    _mark_spikes(ax2a, t, step_n_a, jump_highlight_deg, col_a)
    _mark_spikes(ax2a, t, step_n_b, jump_highlight_deg, col_b)
    ax2a.axhline(jump_highlight_deg, color="gray", ls=":", alpha=0.6)
    ax2a.set_ylabel("针轴 帧间角 (°)")
    ax2a.legend(loc="upper right", fontsize=9)
    ax2a.grid(alpha=0.35)
    ax2b.plot(t, step_y_a, color=col_a, lw=1.2, label=f"{label_a} Yaw步进")
    ax2b.plot(t, step_y_b, color=col_b, lw=1.2, label=f"{label_b} Yaw步进")
    _mark_spikes(ax2b, t, step_y_a, jump_highlight_deg, col_a)
    _mark_spikes(ax2b, t, step_y_b, jump_highlight_deg, col_b)
    ax2b.axhline(jump_highlight_deg, color="gray", ls=":", alpha=0.6)
    _draw_events(ax2b, events)
    ax2b.set_xlabel("时间 (s)")
    ax2b.set_ylabel("Yaw 帧间变化 (°)")
    ax2b.legend(loc="upper right", fontsize=9)
    ax2b.grid(alpha=0.35)
    fig2.tight_layout()
    figs.append(fig2)
    _save(fig2, "02_frame_steps")

    # —— 窗口 3：Yaw 漂移 + 两轨差值 ——
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig3.suptitle(f"③ Yaw 漂移与两轨差距（长测看谁更漂）\n{title}", fontsize=12)
    ax3a.plot(t, off_a, color=col_a, lw=2.0, label=label_a)
    ax3a.plot(t, off_b, color=col_b, lw=2.0, label=label_b)
    ax3a.axhline(0, color="gray", lw=0.8, alpha=0.5)
    _draw_events(ax3a, events)
    ax3a.set_ylabel("Yaw 相对起始偏移 (°)")
    ax3a.legend(loc="upper right")
    ax3a.grid(alpha=0.35)
    ax3b.fill_between(t, 0, diff_off, color="#ff7043", alpha=0.35, label="|A−B| Yaw偏移差")
    ax3b.plot(t, diff_off, color="#d84315", lw=1.5)
    ax3b.set_xlabel("时间 (s)")
    ax3b.set_ylabel("|两轨 Yaw 偏移差| (°)")
    ax3b.legend(loc="upper right")
    ax3b.grid(alpha=0.35)
    fig3.tight_layout()
    figs.append(fig3)
    _save(fig3, "03_yaw_drift")

    # —— 窗口 4：长测静止段放大（≥60s 时 30–90s）——
    t_end = float(t[-1]) if len(t) else 0.0
    if t_end >= 55.0:
        mask = (t >= 28.0) & (t <= min(92.0, t_end))
        if int(np.sum(mask)) > 30:
            fig4, ax4 = plt.subplots(figsize=(11, 4))
            fig4.suptitle(f"④ 静止段放大 (~30–90s，看长期漂移)\n{title}", fontsize=12)
            ax4.plot(t[mask], off_a[mask], color=col_a, lw=2.2, label=label_a)
            ax4.plot(t[mask], off_b[mask], color=col_b, lw=2.2, label=label_b)
            ax4.axhline(0, color="gray", ls="--", alpha=0.5)
            ax4.set_xlabel("时间 (s)")
            ax4.set_ylabel("Yaw 偏移 (°)")
            ax4.legend()
            ax4.grid(alpha=0.35)
            fig4.tight_layout()
            figs.append(fig4)
            _save(fig4, "04_yaw_zoom")

    # —— 窗口 5：磁场（若有）——
    if mag is not None:
        mag_abs = np.linalg.norm(mag, axis=1)
        b0 = float(np.nanmedian(mag_abs[: max(50, len(mag_abs) // 20)]))
        fig5, ax5 = plt.subplots(figsize=(11, 4))
        fig5.suptitle(f"⑤ 磁场 |B|（飙高=磁干扰，九轴易受影响）\n{title}", fontsize=12)
        ax5.plot(t, mag_abs, color="#ce93d8", lw=1.5, label="|B|")
        if b0 > 0:
            ax5.axhline(b0, color="#4fc3f7", ls="--", alpha=0.7, label=f"基线≈{b0:.0f}")
            ax5.axhline(b0 * 1.3, color="#ef5350", ls=":", alpha=0.7, label="1.3×基线(脏磁)")
        _draw_events(ax5, events)
        ax5.set_xlabel("时间 (s)")
        ax5.set_ylabel("|B|")
        ax5.legend()
        ax5.grid(alpha=0.35)
        fig5.tight_layout()
        figs.append(fig5)
        _save(fig5, "05_mag")

    # —— 窗口 6：柱状图（带数值）——
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    fig6.suptitle(f"⑥ 指标对比（柱上数字，越小越稳）\n{title}", fontsize=12)
    bar_labels = ["针轴抖动RMS", "Yaw最大步", "静止漂移", "终端偏移", "大跳变"]
    va = [
        metrics_a.needle_jitter_deg_rms,
        metrics_a.yaw_max_step_deg,
        metrics_a.yaw_drift_still_deg,
        metrics_a.yaw_end_drift_deg,
        metrics_a.yaw_big_jumps,
    ]
    vb = [
        metrics_b.needle_jitter_deg_rms,
        metrics_b.yaw_max_step_deg,
        metrics_b.yaw_drift_still_deg,
        metrics_b.yaw_end_drift_deg,
        metrics_b.yaw_big_jumps,
    ]
    x = np.arange(len(bar_labels))
    w = 0.38
    b1 = ax6.bar(x - w / 2, va, w, label=label_a, color=col_a)
    b2 = ax6.bar(x + w / 2, vb, w, label=label_b, color=col_b)
    ax6.set_xticks(x)
    ax6.set_xticklabels(bar_labels, fontsize=9)
    ax6.set_title("稳定性指标")
    ax6.legend()
    ax6.grid(axis="y", alpha=0.35)

    def _bar_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _bar_labels(b1)
    _bar_labels(b2)
    fig6.tight_layout()
    figs.append(fig6)
    _save(fig6, "06_bars")

    # —— 窗口 7：文字结论 ——
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    fig7.suptitle(f"⑦ 结论 — {title}", fontsize=13)
    ax7.axis("off")
    lines = _pair_verdict_lines(label_a, label_b, metrics_a, metrics_b)
    ax7.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="sans-serif")
    fig7.tight_layout()
    figs.append(fig7)
    _save(fig7, "07_summary")

    if show:
        plt.show()
    else:
        for f in figs:
            plt.close(f)

    return len(figs)
