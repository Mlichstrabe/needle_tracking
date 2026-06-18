#!/usr/bin/env python
"""
离线自适应磁力计融合 — 入口

用法（项目根目录）:
    python tools/mag_fusion/run_offline.py --csv imu_calibration_logs/mag_test_A_xxx.csv
    python tools/mag_fusion/run_offline.py --demo
    python tools/mag_fusion/run_offline.py --csv path.csv --plot --out path_fused.csv

所需 CSV 列（fusion 格式）:
    t, ax, ay, az, gx, gy, gz, mx, my, mz
    可选: qw, qx, qy, qz（芯片四元数，缺省则用 roll/pitch/yaw 或 identity）
    可选: event（分段统计）

参数:
    --eps          |B| 相对容差，默认 0.15
    --b0-window    估计 B0 / 磁倾角基线的静止窗口秒数，默认 3.0
    --kp-mag       磁慢校正最大增益，默认 0.30（越小越偏六轴的平滑，越大航向约束越强）
    --no-plot      不弹 matplotlib 窗口

优化九轴思路:
    磁只做「低增益慢校正」——短期跟陀螺（≈六轴的平滑），长期靠磁防漂移；
    磁脏时（|B| 偏离、跳变快、磁倾角异常）软性降权而非硬关，避免边界抖动。
    考核新增「静止航向漂移」：六轴静止会漂，优化九轴应被磁拉住 → 体现九轴长期优势。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.fusion import (
    AdaptiveMagConfig,
    PeriodicMagConfig,
    run_fusion_compare,
    run_fusion_series,
    synthesize_demo,
)
from tools.mag_fusion.metrics import compute_all_metrics, compute_track, verdict_text
from tools.mag_fusion.plot_compare import plot_comparison, plot_pair_comparison
from tools.mag_fusion.replay import load_csv, write_fusion_output


def _default_out(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_fused.csv")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="离线 6DOF + 自适应磁融合")
    p.add_argument("--csv", type=str, help="输入 CSV 路径")
    p.add_argument("--out", type=str, default="", help="输出 sidecar CSV")
    p.add_argument("--demo", action="store_true", help="使用内置合成数据自检")
    p.add_argument("--plot", action="store_true", help="显示对比图")
    p.add_argument("--no-plot", action="store_true", help="禁用绘图")
    p.add_argument("--eps", type=float, default=0.15, help="|B| 相对容差")
    p.add_argument("--b0-window", type=float, default=3.0, help="B0 估计窗口 (s)")
    p.add_argument("--kp-mag", type=float, default=0.30, help="磁修正最大增益（慢校正，建议 0.2~0.4）")
    p.add_argument("--mag-interval", type=float, default=30.0, help="新融合：脉冲磁校正间隔 (s)")
    p.add_argument(
        "--compare-algos",
        action="store_true",
        help="同时输出「原连续融合」与「新30s脉冲融合」对比",
    )
    args = p.parse_args(argv)

    cfg = AdaptiveMagConfig(eps=args.eps, b0_window_s=args.b0_window, kp_mag_max=args.kp_mag)

    table = None
    if args.demo:
        d = synthesize_demo()
        t, gyro, acc, mag, q_chip = d["t"], d["gyro"], d["acc"], d["mag"], d["q_chip"]
        src = Path("(synthetic)")
        events = []
        schema = "demo"
    elif args.csv:
        table = load_csv(args.csv)
        t, gyro, acc, mag, q_chip = table.t, table.gyro, table.acc, table.mag, table.q_chip
        events = table.events
        src = table.source_path
        schema = table.schema
    else:
        p.error("请指定 --csv 或 --demo")
        return 2

    q_pc, q_6, w_mag = run_fusion_series(t, gyro, acc, mag, q_chip, cfg=cfg)
    metrics = compute_all_metrics(
        t, q_chip, q_pc, q_6, mag, w_mag, events=events or None, gyro=gyro
    )

    print(f"来源: {src}")
    print(f"样本数: {len(t)}  schema={schema}")
    print(verdict_text(metrics))

    if args.compare_algos:
        periodic_cfg = PeriodicMagConfig(
            eps=cfg.eps,
            b0_window_s=cfg.b0_window_s,
            kp_mag_max=cfg.kp_mag_max,
            interval_s=args.mag_interval,
        )
        cmp = run_fusion_compare(
            t, gyro, acc, mag, q_chip, adaptive_cfg=cfg, periodic_cfg=periodic_cfg
        )
        ma = compute_track("原九轴(连续)", cmp["q_adaptive"], gyro=gyro)
        mp = compute_track("新融合(30s)", cmp["q_periodic"], gyro=gyro)
        m6 = compute_track("六轴", cmp["q_6dof"], gyro=gyro)
        print("\n--- 新旧融合对比 ---")
        print(
            f"新融合 脉冲触发={cmp['pulse_fired']} 跳过(磁脏)={cmp['pulse_skipped']} "
            f"间隔={periodic_cfg.interval_s:.0f}s"
        )
        print(
            f"抖动RMS°: 新={mp.needle_jitter_deg_rms:.3f} 原={ma.needle_jitter_deg_rms:.3f} "
            f"六轴={m6.needle_jitter_deg_rms:.3f}"
        )
        print(
            f"静止Yaw漂移°: 新={mp.yaw_drift_still_deg:.2f} 原={ma.yaw_drift_still_deg:.2f} "
            f"六轴={m6.yaw_drift_still_deg:.2f}"
        )
        print(
            f"终端Yaw偏移°: 新={mp.yaw_end_drift_deg:.2f} 原={ma.yaw_end_drift_deg:.2f} "
            f"六轴={m6.yaw_end_drift_deg:.2f}"
        )
        show_cmp = (bool(args.plot) and not args.no_plot) or args.demo
        if show_cmp:
            plot_pair_comparison(
                t,
                cmp["q_periodic"],
                cmp["q_adaptive"],
                "新融合(30s)",
                "原九轴(连续)",
                mp,
                ma,
                mag=mag,
                events=events or None,
                title="新旧磁融合对比",
                show=True,
                multi_window=True,
            )
        return 0

    if not args.demo:
        out = Path(args.out) if args.out else _default_out(Path(args.csv))
        write_fusion_output(table, q_pc, q_6, w_mag, out)
        print(f"已写出: {out}")
    else:
        out = None

    do_plot = bool(args.plot) and not args.no_plot
    if args.demo:
        do_plot = not args.no_plot
    if do_plot:
        plot_comparison(
            t,
            q_chip,
            q_pc,
            q_6,
            mag,
            w_mag,
            metrics,
            title=f"离线融合 — {src.name if hasattr(src, 'name') else src}",
            save_path=out.with_suffix(".png") if out else None,
            show=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
