#!/usr/bin/env python
"""
录制一段 IMU 数据，并对比：
  • 自适应加权磁融合（|B|/变化率/倾角 软门控）
  • 常开磁融合（有磁就用，无阻带）

用法（项目根目录，先关闭 main.py 释放串口）:

  # 推荐：120s 分段引导 + 自动对比 + 保存图表
  python tools/mag_fusion/record_compare_weighted_mag.py --port COM3 --guided

  # 简单：录 60s 后对比
  python tools/mag_fusion/record_compare_weighted_mag.py --port COM3 --duration 60

  # 已有 CSV，只对比
  python tools/mag_fusion/record_compare_weighted_mag.py --csv imu_calibration_logs/xxx.csv

Wit 上位机请保持九轴模式，并勾选输出：四元数(0x59)、磁场(0x54)、加速度(0x51)、陀螺(0x52)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.compare_adaptive_vs_fullmag import main as compare_main
from tools.mag_fusion.live_capture_compare import capture, resolve_serial_port

# 与 mag_ab_compare 一致的 120s 引导
GUIDED_120: List[Tuple[float, float, str, str]] = [
    (0, 10, "① 0–10s 静止基线", "IMU 放稳，完全不动（估磁场基线）"),
    (10, 30, "② 10–30s 缓慢转动", "缓慢小幅转动针体/IMU"),
    (30, 90, "③ 30–90s 再静止", "放稳不动，观察长期漂移"),
    (90, 100, "④ 90–100s 磁干扰", "缓慢把金属靠近 IMU"),
    (100, 120, "⑤ 100–120s 移开恢复", "金属慢慢移开"),
]


def _capture_with_guidance(
    port: str,
    duration_s: float,
    baud: int,
    guided: bool,
) -> Path:
    if not guided:
        return capture(port, duration_s, baud)

    # 分段提示版：先打印全程说明，再在 capture 外包一层计时提示
    print("\n===== 120s 分段引导（请按阶段操作）=====")
    for t0, t1, title, hint in GUIDED_120:
        print(f"  {title}: {hint}")
    print("========================================\n")

    import threading
    from tools.mag_fusion.live_capture_compare import HeadlessImuReader, CSV_FIELDS, LOG_DIR
    import csv
    from datetime import datetime

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    reader = HeadlessImuReader()
    print(f"连接 {port} @ {baud} …")
    try:
        reader.connect(port, baud)
    except Exception as e:
        from tools.mag_fusion.live_capture_compare import list_serial_ports_hint

        raise SystemExit(
            f"串口连接失败: {e}\n请先关闭 main.py / mag_ab_compare 等占用串口的程序。\n"
            f"{list_serial_ports_hint()}"
        ) from e

    t_start = time.time()
    t_end = t_start + duration_s
    announced = set()
    last_n = 0

    def _guide_loop():
        while time.time() < t_end:
            elapsed = time.time() - t_start
            for i, (t0, t1, title, hint) in enumerate(GUIDED_120):
                if t0 <= elapsed < t1 and i not in announced:
                    announced.add(i)
                    print(f"\n>>> [{elapsed:.0f}s] {title}\n    {hint}\n", flush=True)
                    break
            time.sleep(0.3)

    guide_th = threading.Thread(target=_guide_loop, daemon=True)
    guide_th.start()

    print(f"开始录制 {duration_s:.0f}s …")
    while time.time() < t_end:
        time.sleep(1.0)
        with reader._lock:
            n = len(reader.rows)
        if n != last_n:
            print(f"  … {n} 帧", flush=True)
            last_n = n

    reader.disconnect()
    if len(reader.rows) < 50:
        raise SystemExit(f"帧数过少 ({len(reader.rows)})，请检查 Wit 是否输出 0x59/0x54。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"mag_weighted_test_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(reader.rows)
    print(f"\n已保存: {out}  ({len(reader.rows)} 帧)")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="录制并对比：自适应加权 vs 常开磁融合")
    p.add_argument(
        "--port",
        default="auto",
        help="串口，默认 auto（自动选 CH340 等）；也可 COM6",
    )
    p.add_argument("--duration", type=float, default=60.0, help="录制秒数（--guided 时默认 120）")
    p.add_argument("--guided", action="store_true", help="120s 分段引导录制")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--csv", type=str, default="", help="跳过录制，只对比已有 CSV")
    p.add_argument("--kp-mag", type=float, default=0.30)
    p.add_argument("--eps", type=float, default=0.15)
    p.add_argument("--show", action="store_true", help="弹出 matplotlib 窗口（默认只存图）")
    args = p.parse_args(argv)

    if args.csv:
        csv_path = args.csv
    else:
        port = resolve_serial_port(args.port)
        dur = 120.0 if args.guided else args.duration
        csv_path = str(_capture_with_guidance(port, dur, args.baud, args.guided))

    if not args.show:
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")

    compare_argv = ["--csv", csv_path, "--kp-mag", str(args.kp_mag), "--eps", str(args.eps)]
    return compare_main(compare_argv)


if __name__ == "__main__":
    raise SystemExit(main())
