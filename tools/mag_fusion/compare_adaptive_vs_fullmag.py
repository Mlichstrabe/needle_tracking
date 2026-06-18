#!/usr/bin/env python
"""
对比两种 PC 端磁融合策略（同一 Mahony + 同一 kp_mag_max）:

  • 自适应加权融合 —— |B|/变化率/磁倾角 软权重门控（阻带脏磁）
  • 常开磁融合     —— 有磁场即 w_mag=1，始终按 Mahony 磁航向修正（无阻带）

用法:
    python tools/mag_fusion/compare_adaptive_vs_fullmag.py
    python tools/mag_fusion/compare_adaptive_vs_fullmag.py --csv imu_calibration_logs/mag_ab_A_xxx.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.fusion import AdaptiveMagConfig, quat_angle_deg, run_adaptive_vs_fullmag
from tools.mag_fusion.metrics import compute_track
from tools.mag_fusion.plot_compare import plot_pair_comparison
from tools.mag_fusion.replay import load_csv

LOG_DIR = _ROOT / "imu_calibration_logs"
FIG_DIR = _ROOT / "docs" / "figures"

LABEL_ADAPT = "自适应v2(芯片引导)"
LABEL_FULL = "常开磁融合(无阻带)"
LABEL_CHIP = "IMU自带九轴"


def _latest_csv() -> Path:
    cands = list(LOG_DIR.glob("mag_*.csv"))
    if not cands:
        raise SystemExit(f"未找到 CSV: {LOG_DIR}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _figure_dir_for_csv(csv_path: Path) -> Path:
    parts = csv_path.stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return FIG_DIR / f"{parts[-2]}_{parts[-1]}"
    return FIG_DIR / "undated"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="自适应加权 vs 常开磁融合")
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--kp-mag", type=float, default=-1.0, help="默认用 v2 内置 0.18")
    p.add_argument("--eps", type=float, default=0.12)
    p.add_argument("--legacy", action="store_true", help="同时打印旧版纯 Mahony 自适应")
    args = p.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    table = load_csv(csv_path)
    kw = {"eps": args.eps}
    if args.kp_mag >= 0:
        kw["kp_mag_max"] = args.kp_mag
    cfg = AdaptiveMagConfig(**kw)
    out = run_adaptive_vs_fullmag(
        table.t,
        table.gyro,
        table.acc,
        table.mag,
        table.q_chip,
        cfg=cfg,
        include_legacy=args.legacy,
    )

    m_ad = compute_track(LABEL_ADAPT, out["q_adaptive"], gyro=table.gyro)
    m_full = compute_track(LABEL_FULL, out["q_full_mag"], gyro=table.gyro)
    m_chip = compute_track(LABEL_CHIP, out["q_chip"], gyro=table.gyro)
    m_6 = compute_track("纯六轴", out["q_6dof"], gyro=table.gyro)

    def _max_ang(qa, qb):
        xs = [
            quat_angle_deg(qa[i], qb[i])
            for i in range(len(table.t))
            if np.all(np.isfinite(qa[i])) and np.all(np.isfinite(qb[i]))
        ]
        return (max(xs), xs[-1] if xs else 0.0) if xs else (0.0, 0.0)

    af_max, af_end = _max_ang(out["q_adaptive"], out["q_full_mag"])
    ac_max, ac_end = _max_ang(out["q_adaptive"], out["q_chip"])

    print("========== 自适应 v2 vs 常开磁 vs 芯片九轴 ==========")
    print(f"CSV: {csv_path.name}  时长≈{table.t[-1] - table.t[0]:.1f}s  帧={len(table.t)}")
    print(
        f"v2: chip_guided={cfg.use_chip_guided}  still_only_mag={cfg.mag_only_when_still}  "
        f"kp_mag={cfg.kp_mag_max}  eps={cfg.eps}"
    )
    print(
        f"磁权重 w: v2 mean/max={float(np.mean(out['w_adaptive'])):.3f}/"
        f"{float(np.max(out['w_adaptive'])):.3f}  "
        f"常开 mean/max={float(np.mean(out['w_full_mag'])):.3f}/1.000"
    )
    print()
    print(f"{'指标':<16} {'v2自适应':>12} {'常开磁':>12} {LABEL_CHIP:>12}")
    print("-" * 56)
    for name, a, b, c in [
        ("抖动RMS(°)", m_ad.needle_jitter_deg_rms, m_full.needle_jitter_deg_rms, m_chip.needle_jitter_deg_rms),
        ("Yaw最大步(°)", m_ad.yaw_max_step_deg, m_full.yaw_max_step_deg, m_chip.yaw_max_step_deg),
        ("静止漂移(°)", m_ad.yaw_drift_still_deg, m_full.yaw_drift_still_deg, m_chip.yaw_drift_still_deg),
        ("终端偏移(°)", m_ad.yaw_end_drift_deg, m_full.yaw_end_drift_deg, m_chip.yaw_end_drift_deg),
        ("大跳变(次)", float(m_ad.yaw_big_jumps), float(m_full.yaw_big_jumps), float(m_chip.yaw_big_jumps)),
    ]:
        print(f"{name:<16} {a:12.3f} {b:12.3f} {c:12.3f}")
    print(
        f"\n夹角: v2-常开 最大={af_max:.2f} 末={af_end:.2f} | "
        f"v2-芯片 最大={ac_max:.2f} 末={ac_end:.2f}"
    )
    if args.legacy and "q_adaptive_legacy" in out:
        m_leg = compute_track("旧版自适应", out["q_adaptive_legacy"], gyro=table.gyro)
        lg_max, _ = _max_ang(out["q_adaptive_legacy"], out["q_full_mag"])
        print(
            f"旧版自适应: 抖动={m_leg.needle_jitter_deg_rms:.3f}  "
            f"对常开最大夹角={lg_max:.2f}  w_mean={float(np.mean(out['w_adaptive_legacy'])):.3f}"
        )

    fig_dir = _figure_dir_for_csv(csv_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"adaptive_vs_fullmag_{csv_path.stem}.png"
    plot_pair_comparison(
        table.t,
        out["q_adaptive"],
        out["q_full_mag"],
        LABEL_ADAPT,
        LABEL_FULL,
        m_ad,
        m_full,
        mag=table.mag,
        title=f"自适应v2 vs 常开磁 · {csv_path.stem}",
        save_path=png,
        show=False,
        multi_window=True,
    )
    print(f"\n图表: {fig_dir}/adaptive_vs_fullmag_{csv_path.stem}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
