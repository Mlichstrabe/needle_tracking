#!/usr/bin/env python
"""对比 IMU 芯片九轴四元数 vs PC 原融合算法（连续自适应磁）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.fusion import AdaptiveMagConfig, quat_angle_deg, run_fusion_series
from tools.mag_fusion.metrics import compute_track
from tools.mag_fusion.plot_compare import plot_pair_comparison
from tools.mag_fusion.replay import load_csv

LOG_DIR = _ROOT / "imu_calibration_logs"
FIG_DIR = _ROOT / "docs" / "figures"


def _latest_csv() -> Path:
    cands = list(LOG_DIR.glob("mag_*.csv"))
    if not cands:
        raise SystemExit(f"未找到 CSV，请先录制: {LOG_DIR}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _figure_dir_for_csv(csv_path: Path) -> Path:
    parts = csv_path.stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return FIG_DIR / f"{parts[-2]}_{parts[-1]}"
    return FIG_DIR / "undated"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="芯片九轴 vs 原融合算法")
    p.add_argument("--csv", type=str, default="", help="输入 CSV，默认最新 mag_*.csv")
    args = p.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    table = load_csv(csv_path)
    cfg = AdaptiveMagConfig()
    q_adapt, _, w_mag = run_fusion_series(
        table.t, table.gyro, table.acc, table.mag, table.q_chip, cfg=cfg
    )
    q_chip = table.q_chip

    m_chip = compute_track("IMU自带九轴", q_chip, gyro=table.gyro)
    m_adapt = compute_track("原融合算法", q_adapt, gyro=table.gyro)

    angs = []
    for i in range(len(table.t)):
        if np.all(np.isfinite(q_chip[i])) and np.all(np.isfinite(q_adapt[i])):
            angs.append(quat_angle_deg(q_chip[i], q_adapt[i]))

    print("========== 芯片九轴 vs 原融合算法 ==========")
    print(f"CSV: {csv_path.name}  时长≈{table.t[-1] - table.t[0]:.1f}s  帧数={len(table.t)}")
    print(f"原融合 磁权重 mean/max = {float(np.mean(w_mag)):.3f} / {float(np.max(w_mag)):.3f}")
    print()
    print(f"{'指标':<16} {'IMU九轴':>12} {'原融合':>12}")
    print("-" * 42)
    rows = [
        ("抖动RMS(°)", m_chip.needle_jitter_deg_rms, m_adapt.needle_jitter_deg_rms),
        ("Yaw最大步(°)", m_chip.yaw_max_step_deg, m_adapt.yaw_max_step_deg),
        ("静止漂移(°)", m_chip.yaw_drift_still_deg, m_adapt.yaw_drift_still_deg),
        ("终端偏移(°)", m_chip.yaw_end_drift_deg, m_adapt.yaw_end_drift_deg),
        ("大跳变(次)", float(m_chip.yaw_big_jumps), float(m_adapt.yaw_big_jumps)),
    ]
    for name, a, b in rows:
        print(f"{name:<16} {a:12.3f} {b:12.3f}")
    if angs:
        print(f"\n全程最大夹角: {max(angs):.4f}°  末帧夹角: {angs[-1]:.4f}°")

    fig_dir = _figure_dir_for_csv(csv_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"chip_vs_adaptive_{csv_path.stem}.png"
    plot_pair_comparison(
        table.t,
        q_chip,
        q_adapt,
        "IMU自带九轴",
        "原融合算法",
        m_chip,
        m_adapt,
        mag=table.mag,
        title=f"芯片九轴 vs 原融合 · {csv_path.stem}",
        save_path=out,
        show=False,
        multi_window=True,
    )
    print(f"\n图表目录: {fig_dir}")
    print(f"  结论图: chip_vs_adaptive_{csv_path.stem}_07_summary.png")
    print(f"  柱状图: chip_vs_adaptive_{csv_path.stem}_06_bars.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
