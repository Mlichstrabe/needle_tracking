from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "rangefinder_test_logs"
OUT_DIR = ROOT / "output" / "rangefinder_ppt_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 160,
        "savefig.dpi": 220,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def values(rows: list[dict[str, str]]) -> np.ndarray:
    key = "raw_mm" if rows and rows[0].get("raw_mm") not in (None, "") else "distance_mm"
    return np.array([float(row[key]) for row in rows], dtype=float)


def corrected_values(rows: list[dict[str, str]]) -> np.ndarray:
    if not rows or rows[0].get("corrected_distance_mm") in (None, ""):
        raise ValueError("CSV does not contain corrected_distance_mm")
    return np.array([float(row["corrected_distance_mm"]) for row in rows], dtype=float)


def mean_std(xs: np.ndarray) -> tuple[float, float]:
    return float(np.mean(xs)), float(np.std(xs))


def is_original_uncompensated_run(path: Path) -> bool:
    summary_path = path.with_suffix(".json")
    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8") as f:
        test = json.load(f).get("test", {})
    configured_offset = test.get("software_offset_mm", test.get("near_offset_max_mm"))
    return configured_offset == 0


def find_original_uncompensated(ref: int) -> Path:
    files = [
        path for path in sorted(LOG_DIR.glob(f"laser100m_ref{ref}mm_20260913_*.csv"))
        if is_original_uncompensated_run(path)
    ]
    if not files:
        raise FileNotFoundError(f"missing uncompensated {ref} mm CSV")
    return files[-1]


# Preserve the three original 57 mm runs used by the group-meeting figure.
# Later compensated trials are a different experiment condition and must not be merged here.
run57_files = [
    path for path in sorted(LOG_DIR.glob("laser100m_ref57mm_20260913_*.csv"))
    if is_original_uncompensated_run(path)
]
run57 = [read_csv(path) for path in run57_files]
xs57 = [values(rows) for rows in run57]
all57 = np.concatenate(xs57)

# Same-condition batch collected with the approved piecewise correction:
# +2 mm through 60 mm raw distance, tapered to 0 between 60 and 70 mm.
ppt_batch_files = {
    57: "laser100m_ref57mm_20260913_172956_362630.csv",
    60: "laser100m_ref60mm_20260913_173047_997530.csv",
    65: "laser100m_ref65mm_20260913_173207_923920.csv",
    70: "laser100m_ref70mm_20260913_173326_436542.csv",
    100: "laser100m_ref100mm_20260913_173416_662834.csv",
    150: "laser100m_ref150mm_20260913_182433_315729.csv",
    200: "laser100m_ref200mm_20260913_182554_848987.csv",
    300: "laser100m_ref300mm_20260913_182651_341954.csv",
}
static: dict[int, np.ndarray] = {
    ref: corrected_values(read_csv(LOG_DIR / filename))
    for ref, filename in ppt_batch_files.items()
}

summary = []
for ref in ppt_batch_files:
    xs = static[ref]
    m, sd = mean_std(xs)
    errors = xs - ref
    summary.append(
        {
            "ref": ref,
            "n": len(xs),
            "runs": 1,
            "mean": m,
            "std": sd,
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Static charts, separated from the summary table for direct PPT placement.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), facecolor="white", constrained_layout=True)
refs = np.array([s["ref"] for s in summary])
means = np.array([s["mean"] for s in summary])
sds = np.array([s["std"] for s in summary])
bias = np.array([s["bias"] for s in summary])

ax1.plot([40, 320], [40, 320], "--", color="#7f8c8d", lw=2.0, label="理想线 y=x")
ax1.errorbar(refs, means, yerr=sds, fmt="o", ms=9, capsize=5, lw=2.0, color="#1f77b4", ecolor="#1f77b4", label="均值 ± 1σ")
ax1.set(xlim=(40, 320), ylim=(40, 320), xlabel="参考距离 / mm", ylabel="补偿后测量距离 / mm", title="补偿后距离与参考距离")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="lower right", frameon=False)

ax2.axhline(0, color="#7f8c8d", ls="--", lw=2)
ax2.plot(refs, bias, "o-", color="#c0392b", lw=2.5, ms=9)
label_offsets = {
    57: (0, -20),
    60: (-12, 14),
    65: (14, -20),
    70: (0, -20),
    100: (0, -20),
    150: (0, -20),
    200: (0, 14),
    300: (0, -20),
}
for r, e in zip(refs, bias):
    dx, dy = label_offsets[int(r)]
    ax2.annotate(f"{e:+.2f}", (r, e), xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=13)
ax2.set(xlim=(40, 320), ylim=(-1.0, 1.5), xlabel="参考距离 / mm", ylabel="补偿后偏差 / mm", title="补偿后的距离相关偏差")
ax2.grid(True, alpha=0.25)
fig.suptitle("100 m激光测距模块：静态距离验证（+2 mm分段补偿）", fontsize=27, fontweight="bold", x=0.05, ha="left")
save(fig, "rangefinder_static_charts")


# Standalone summary table with MAE-based relative error.
fig_table, ax_table = plt.subplots(figsize=(16, 6.4), facecolor="white", constrained_layout=True)
ax_table.axis("off")
columns = ["参考距离", "有效帧", "均值 / mm", "标准差 / mm", "偏差 / mm", "MAE / mm", "RMSE / mm", "相对误差 / %"]
cell_text = []
for s in summary:
    label = f"{s['ref']} mm"
    relative_error = s["mae"] / s["ref"] * 100
    cell_text.append([label, str(s["n"]), f"{s['mean']:.2f}", f"{s['std']:.2f}", f"{s['bias']:+.2f}", f"{s['mae']:.2f}", f"{s['rmse']:.2f}", f"{relative_error:.2f}%"])
table = ax_table.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center", colWidths=[0.18, 0.095, 0.115, 0.125, 0.115, 0.105, 0.105, 0.13])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1, 2.0)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#ccd5dc")
    if row == 0:
        cell.set_facecolor("#e9eff4")
        cell.set_text_props(weight="bold")
    elif row == 1:
        cell.set_facecolor("#fff0f0")
fig_table.suptitle("静态距离实验数据汇总（+2 mm分段补偿后）", fontsize=27, fontweight="bold", x=0.05, ha="left")
fig_table.text(0.05, 0.06, "相对误差 = MAE ÷ 参考距离 × 100%；各距离测试1次，每次约30 s。", fontsize=15, color="#4f5b66")
save(fig_table, "rangefinder_static_table")


# 16:9 57 mm repeatability figure: distribution + run-level bias.
fig = plt.figure(figsize=(16, 9), facecolor="white", constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
ax_bias = fig.add_subplot(gs[0, 1])
line_colors = ["#1f77b4", "#e67e22", "#7b61a8"]
positions = np.arange(1, 4)

bp = ax.boxplot(
    xs57,
    positions=positions,
    widths=0.52,
    patch_artist=True,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#17202a", "markersize": 8},
    medianprops={"color": "#17202a", "linewidth": 2.2},
    whiskerprops={"color": "#65727d", "linewidth": 1.6},
    capprops={"color": "#65727d", "linewidth": 1.6},
    flierprops={"marker": "o", "markersize": 5, "alpha": 0.45},
)
for patch, color in zip(bp["boxes"], line_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.24)
    patch.set_edgecolor(color)
    patch.set_linewidth(2.2)

rng = np.random.default_rng(20260913)
for idx, (x, color) in enumerate(zip(xs57, line_colors), start=1):
    sampled = x[::5]
    jitter = rng.normal(0, 0.055, len(sampled))
    ax.scatter(np.full(len(sampled), idx) + jitter, sampled, s=22, color=color, alpha=0.35, edgecolors="none")
    m = float(np.mean(x))
    ax.annotate(f"均值 {m:.2f} mm", (idx, m), xytext=(0, -34), textcoords="offset points", ha="center", fontsize=14, color=color, fontweight="bold")

ax.axhline(57, color="#c0392b", ls="--", lw=2.5, label="参考值 57 mm")
ax.set_xticks(positions, ["第1次", "第2次", "第3次"])
ax.set(ylim=(50, 58.2), xlabel="独立采集批次", ylabel="原始距离 / mm", title="每次采集的数据分布")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="upper right", frameon=False)

means57 = np.array([np.mean(x) for x in xs57])
stds57 = np.array([np.std(x) for x in xs57])
bias57 = means57 - 57
bars = ax_bias.bar(positions, bias57, color=line_colors, alpha=0.8, width=0.62)
ax_bias.errorbar(positions, bias57, yerr=stds57, fmt="none", ecolor="#17202a", elinewidth=2, capsize=6)
ax_bias.axhline(0, color="#7f8c8d", ls="--", lw=2.2)
for i, (bar, b, sd) in enumerate(zip(bars, bias57, stds57)):
    ax_bias.text(bar.get_x() + bar.get_width() / 2, b + 0.18, f"{b:+.2f} mm", ha="center", va="bottom", fontsize=16, color="white", fontweight="bold")
    ax_bias.text(i + 1, -4.55, f"σ={sd:.2f} mm\n有效 {len(xs57[i])}/{len(xs57[i])}", ha="center", va="top", fontsize=13, color=line_colors[i])
ax_bias.set_xticks(positions, ["第1次", "第2次", "第3次"])
ax_bias.set(ylim=(-5.25, 0.7), xlabel="独立采集批次", ylabel="均值相对57 mm的偏差 / mm", title="每次均值偏差（误差棒：±1σ）")
ax_bias.grid(True, axis="y", alpha=0.25)

fig.suptitle("57 mm近距离重复性：短时稳定，但批次均值存在漂移（未补偿）", fontsize=27, fontweight="bold", x=0.05, ha="left")
save(fig, "rangefinder_57mm_repeatability")


with (OUT_DIR / "rangefinder_static_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["reference_mm", "n", "runs", "mean_mm", "std_mm", "bias_mm", "mae_mm", "rmse_mm", "relative_error_percent"])
    for s in summary:
        writer.writerow([s["ref"], s["n"], s["runs"], f"{s['mean']:.6f}", f"{s['std']:.6f}", f"{s['bias']:.6f}", f"{s['mae']:.6f}", f"{s['rmse']:.6f}", f"{s['mae'] / s['ref'] * 100:.6f}"])

print(f"Generated figures in {OUT_DIR}")
