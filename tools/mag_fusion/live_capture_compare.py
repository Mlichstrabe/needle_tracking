#!/usr/bin/env python
"""
连接 IMU 串口录制一段数据，并离线对比「新融合(30s)」与「原九轴(连续)」。

用法（项目根目录，请先关闭占用串口的主程序）:
    python tools/mag_fusion/live_capture_compare.py --port COM3 --duration 60
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.fusion import PeriodicMagConfig, run_fusion_compare, AdaptiveMagConfig
from tools.mag_fusion.metrics import compute_track
from tools.mag_fusion.plot_compare import plot_pair_comparison
from tools.mag_fusion.replay import load_csv

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None  # type: ignore

LOG_DIR = _ROOT / "imu_calibration_logs"

_IMU_PORT_KEYWORDS = ("ch340", "cp210", "usb-serial", "usb serial", "wit", "jy901", "wt901")


def list_serial_ports_hint() -> str:
    """列出当前可用串口，供错误提示。"""
    if list_ports is None:
        return "（未安装 pyserial，无法枚举串口）"
    ports = list(list_ports.comports())
    if not ports:
        return "（当前未检测到任何 COM 口 — 请插 USB 线、装 CH340 驱动）"
    lines = ["当前可用串口:"]
    for p in ports:
        lines.append(f"  {p.device}  —  {p.description or '未知设备'}")
    return "\n".join(lines)


def resolve_serial_port(requested: str = "auto") -> str:
    """
    解析串口名。requested='auto' 时优先选 CH340/CP210 等 IMU 常见适配器。
    """
    if list_ports is None:
        if requested.lower() == "auto":
            raise SystemExit("未安装 pyserial，请: pip install pyserial")
        return requested

    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit(
            "未检测到串口。\n" + list_serial_ports_hint()
        )

    if requested.lower() != "auto":
        want = requested.upper()
        by_name = {p.device.upper(): p.device for p in ports}
        if want in by_name:
            return by_name[want]
        raise SystemExit(
            f"找不到串口 {requested}。\n{list_serial_ports_hint()}"
        )

    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in _IMU_PORT_KEYWORDS):
            print(f"自动选择串口: {p.device} ({p.description})")
            return p.device

    print(f"自动选择串口: {ports[0].device} ({ports[0].description})")
    return ports[0].device
FIG_DIR = _ROOT / "docs" / "figures"

CSV_FIELDS = [
    "t", "roll", "pitch", "yaw", "yaw_step",
    "mx", "my", "mz", "mag_abs",
    "qw", "qx", "qy", "qz",
    "ax", "ay", "az", "gx", "gy", "gz",
]


class HeadlessImuReader:
    """JY901 协议，无 Qt。"""

    def __init__(self):
        self._ser = None
        self._run = False
        self._th: Optional[threading.Thread] = None
        self.cache = {
            "acc": None,
            "gyro": None,
            "euler": None,
            "quat": None,
            "mag": None,
        }
        self.rows: List[dict] = []
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self.quat_count = 0
        self.mag_count = 0

    def connect(self, port: str, baud: int = 115200) -> None:
        if serial is None:
            raise RuntimeError("未安装 pyserial")
        self._ser = serial.Serial(port, baud, timeout=0.5)
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def disconnect(self):
        self._run = False
        if self._th:
            self._th.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def _loop(self):
        buf = bytearray()
        while self._run and self._ser:
            try:
                if self._ser.in_waiting:
                    buf.extend(self._ser.read(self._ser.in_waiting))
                while len(buf) >= 11:
                    if buf[0] != 0x55:
                        buf.pop(0)
                        continue
                    frame = bytes(buf[:11])
                    if (sum(frame[:10]) & 0xFF) != frame[10]:
                        buf.pop(0)
                        continue
                    buf = buf[11:]
                    self._parse(frame[1], frame)
            except Exception:
                break

    def _parse(self, typ: int, frame: bytes):
        try:
            if typ == 0x51:
                ax = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 16 * 9.8
                ay = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 16 * 9.8
                az = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 16 * 9.8
                self.cache["acc"] = np.array([ax, ay, az])
            elif typ == 0x52:
                gx = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 2000 * np.pi / 180
                gy = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 2000 * np.pi / 180
                gz = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 2000 * np.pi / 180
                self.cache["gyro"] = np.array([gx, gy, gz])
            elif typ == 0x53:
                roll = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 180
                pitch = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 180
                yaw = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 180
                self.cache["euler"] = np.array([roll, pitch, yaw])
            elif typ == 0x54:
                mx = struct.unpack("<h", frame[2:4])[0]
                my = struct.unpack("<h", frame[4:6])[0]
                mz = struct.unpack("<h", frame[6:8])[0]
                self.cache["mag"] = np.array([mx, my, mz], dtype=float)
                self.mag_count += 1
            elif typ == 0x59:
                q = np.array([
                    struct.unpack("<h", frame[2:4])[0] / 32768.0,
                    struct.unpack("<h", frame[4:6])[0] / 32768.0,
                    struct.unpack("<h", frame[6:8])[0] / 32768.0,
                    struct.unpack("<h", frame[8:10])[0] / 32768.0,
                ])
                self.cache["quat"] = q
                self.quat_count += 1
                self._record()
        except Exception:
            pass

    def _record(self):
        if self.cache["quat"] is None:
            return
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        t_rel = now - self._t0
        q = self.cache["quat"]
        acc = self.cache["acc"]
        gyro = self.cache["gyro"]
        mag = self.cache["mag"]
        euler = self.cache["euler"]
        row = {
            "t": t_rel,
            "roll": float(euler[0]) if euler is not None else "",
            "pitch": float(euler[1]) if euler is not None else "",
            "yaw": float(euler[2]) if euler is not None else "",
            "yaw_step": "",
            "mx": float(mag[0]) if mag is not None else "",
            "my": float(mag[1]) if mag is not None else "",
            "mz": float(mag[2]) if mag is not None else "",
            "mag_abs": float(np.linalg.norm(mag)) if mag is not None else "",
            "qw": float(q[0]), "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3]),
            "ax": float(acc[0]) if acc is not None else "",
            "ay": float(acc[1]) if acc is not None else "",
            "az": float(acc[2]) if acc is not None else "",
            "gx": float(gyro[0]) if gyro is not None else "",
            "gy": float(gyro[1]) if gyro is not None else "",
            "gz": float(gyro[2]) if gyro is not None else "",
        }
        with self._lock:
            self.rows.append(row)


def capture(port: str, duration_s: float, baud: int = 115200) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    reader = HeadlessImuReader()
    print(f"连接 {port} @ {baud} …")
    try:
        reader.connect(port, baud)
    except Exception as e:
        raise SystemExit(
            f"串口连接失败: {e}\n"
            "若主程序已连接 IMU，请先关闭 main.py 再重试。\n"
            f"{list_serial_ports_hint()}"
        ) from e

    print(f"录制 {duration_s:.0f}s — 请保持 IMU 尽量静止（前 10s 估磁基线）…")
    t_end = time.time() + duration_s
    last_n = 0
    while time.time() < t_end:
        time.sleep(1.0)
        with reader._lock:
            n = len(reader.rows)
        if n != last_n:
            print(f"  … {n} 帧  四元数={reader.quat_count}  磁={reader.mag_count}", flush=True)
            last_n = n

    reader.disconnect()
    if len(reader.rows) < 50:
        raise SystemExit(f"帧数过少 ({len(reader.rows)})，请检查 Wit 是否输出四元数(0x59)与磁场(0x54)。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"mag_live_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(reader.rows)
    print(f"已保存: {out}  ({len(reader.rows)} 帧)")
    return out


def analyze(csv_path: Path, interval_s: float) -> dict:
    table = load_csv(csv_path)
    adaptive = AdaptiveMagConfig()
    periodic = PeriodicMagConfig(interval_s=interval_s)
    cmp = run_fusion_compare(
        table.t, table.gyro, table.acc, table.mag, table.q_chip,
        adaptive_cfg=adaptive, periodic_cfg=periodic,
    )
    ma = compute_track("原九轴(连续)", cmp["q_adaptive"], gyro=table.gyro)
    mp = compute_track("新融合(30s)", cmp["q_periodic"], gyro=table.gyro)
    m6 = compute_track("六轴", cmp["q_6dof"], gyro=table.gyro)
    return {"table": table, "cmp": cmp, "ma": ma, "mp": mp, "m6": m6}


def _figure_dir_for_csv(csv_path: Path) -> Path:
    parts = csv_path.stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return FIG_DIR / f"{parts[-2]}_{parts[-1]}"
    return FIG_DIR / "undated"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="录制 IMU 并对比新旧磁融合")
    p.add_argument("--port", default="COM3", help="串口，默认 COM3")
    p.add_argument("--duration", type=float, default=60.0, help="录制时长(秒)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--csv", type=str, default="", help="跳过录制，直接分析已有 CSV")
    p.add_argument("--mag-interval", type=float, default=30.0)
    args = p.parse_args(argv)

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = capture(args.port, args.duration, args.baud)

    res = analyze(csv_path, args.mag_interval)
    cmp, ma, mp, m6 = res["cmp"], res["ma"], res["mp"], res["m6"]
    table = res["table"]

    print("\n========== 真机数据 · 新旧融合对比 ==========")
    print(f"CSV: {csv_path.name}  时长≈{table.t[-1]-table.t[0]:.1f}s  帧数={len(table.t)}")
    print(f"新融合 脉冲触发={cmp['pulse_fired']}  跳过(磁脏)={cmp['pulse_skipped']}")
    print(f"磁权重均值: 新={float(np.mean(cmp['w_periodic'])):.3f}  原={float(np.mean(cmp['w_adaptive'])):.3f}")
    print()
    print(f"{'指标':<16} {'新融合(30s)':>12} {'原九轴':>12} {'六轴':>12}")
    print("-" * 56)
    print(f"{'抖动RMS°':<16} {mp.needle_jitter_deg_rms:12.3f} {ma.needle_jitter_deg_rms:12.3f} {m6.needle_jitter_deg_rms:12.3f}")
    print(f"{'静止Yaw漂移°':<16} {mp.yaw_drift_still_deg:12.2f} {ma.yaw_drift_still_deg:12.2f} {m6.yaw_drift_still_deg:12.2f}")
    print(f"{'终端Yaw偏移°':<16} {mp.yaw_end_drift_deg:12.2f} {ma.yaw_end_drift_deg:12.2f} {m6.yaw_end_drift_deg:12.2f}")
    print(f"{'Yaw大跳变次数':<16} {mp.yaw_big_jumps:12d} {ma.yaw_big_jumps:12d} {m6.yaw_big_jumps:12d}")

    fig_dir = _figure_dir_for_csv(csv_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_base = fig_dir / f"fusion_live_{csv_path.stem}"
    plot_pair_comparison(
        table.t,
        cmp["q_periodic"],
        cmp["q_adaptive"],
        "新融合(30s)",
        "原九轴(连续)",
        mp,
        ma,
        mag=table.mag,
        title=f"真机 · {csv_path.stem}",
        save_path=png_base.with_suffix(".png"),
        show=False,
        multi_window=True,
    )
    summary = png_base.with_name(f"{png_base.name}_07_summary.png")
    bars = png_base.with_name(f"{png_base.name}_06_bars.png")
    print(f"\n图表已保存: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
