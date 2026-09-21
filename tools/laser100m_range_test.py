#!/usr/bin/env python
"""100米激光测距模块的独立采集与静态精度测试。

默认依次引导测试100、200、300 mm；60 mm及以下补偿+2 mm，
60–70 mm线性衰减至0，70 mm及以上不补偿，并保留原始值：
    python tools/laser100m_range_test.py --port COM3

单点快速测试：
    python tools/laser100m_range_test.py --port COM3 --reference-mm 100 --duration 10

脚本将模块设置为20 Hz连续测量，保存逐帧CSV和汇总JSON。它不接入主窗口，
也不会把测得的一维距离解释为CT空间针尖位置。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - runtime dependency error
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "rangefinder_test_logs"
ERRORS = {
    10: "电量低",
    14: "计算错误",
    15: "超出量程",
    16: "信号弱或测量时间过长",
    18: "环境光太强",
    26: "超出显示范围",
}
FREQUENCY_CODES = {3: 0x00, 5: 0x05, 10: 0x0A, 20: 0x14}


def applied_offset_mm(
    distance: Optional[float],
    max_offset_mm: float,
    full_until_mm: float = 60.0,
    zero_from_mm: float = 70.0,
) -> Optional[float]:
    """Return the continuous near-range offset for one numeric measurement."""
    if distance is None:
        return None
    if distance <= full_until_mm:
        return max_offset_mm
    if distance >= zero_from_mm:
        return 0.0
    return max_offset_mm * (zero_from_mm - distance) / (zero_from_mm - full_until_mm)


def corrected_distance(
    distance: Optional[float],
    max_offset_mm: float,
    full_until_mm: float = 60.0,
    zero_from_mm: float = 70.0,
) -> Optional[float]:
    """Correct numeric measurements; error frames remain missing."""
    offset = applied_offset_mm(distance, max_offset_mm, full_until_mm, zero_from_mm)
    return None if distance is None or offset is None else distance + offset


def command(*values: int) -> bytes:
    data = bytes(values)
    return data + bytes(((-sum(data)) & 0xFF,))


@dataclass(frozen=True)
class RangeFrame:
    status: str
    distance_mm: Optional[float]
    error_code: Optional[int]
    error_text: str
    raw_hex: str


class Laser100mParser:
    """增量解析连续测量回复，并在噪声或坏校验后重新同步。"""

    def __init__(self, address: int = 0x80) -> None:
        self.address = address
        self._buffer = bytearray()
        self.bad_checksum = 0
        self.bad_payload = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[RangeFrame]:
        self._buffer.extend(data)
        frames: list[RangeFrame] = []
        header = bytes((self.address, 0x06, 0x83))
        while True:
            start = self._buffer.find(header)
            if start < 0:
                keep = min(len(self._buffer), len(header) - 1)
                self.discarded_bytes += len(self._buffer) - keep
                if keep:
                    del self._buffer[:-keep]
                else:
                    self._buffer.clear()
                break
            if start:
                self.discarded_bytes += start
                del self._buffer[:start]

            matched = False
            checksum_candidate = False
            for frame_size in (11, 12):
                if len(self._buffer) < frame_size:
                    continue
                raw = bytes(self._buffer[:frame_size])
                payload = raw[3:-1]
                numeric = re.fullmatch(rb"\d{3}\.\d{3,4}", payload)
                device_error = re.fullmatch(rb"ERR-{1,3}\d{2}", payload)
                if not (numeric or device_error):
                    continue
                if sum(raw) & 0xFF:
                    checksum_candidate = True
                    continue

                del self._buffer[:frame_size]
                text = payload.decode("ascii")
                if numeric:
                    frames.append(RangeFrame("ok", float(text) * 1000.0, None, "", raw.hex(" ")))
                else:
                    code = int(text[-2:])
                    frames.append(RangeFrame("device_error", None, code, ERRORS.get(code, "未知错误码"), raw.hex(" ")))
                matched = True
                break
            if matched:
                continue
            if len(self._buffer) < 12:
                break
            if checksum_candidate:
                self.bad_checksum += 1
            else:
                self.bad_payload += 1
            del self._buffer[0]
        return frames


def describe(values: Iterable[float], reference_mm: Optional[float]) -> dict:
    xs = list(values)
    result = {
        "count": len(xs), "mean_mm": None, "median_mm": None, "std_mm": None,
        "min_mm": None, "max_mm": None, "peak_to_peak_mm": None,
        "bias_mm": None, "mae_mm": None, "rmse_mm": None,
    }
    if not xs:
        return result
    result.update({
        "mean_mm": statistics.fmean(xs),
        "median_mm": statistics.median(xs),
        "std_mm": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min_mm": min(xs),
        "max_mm": max(xs),
        "peak_to_peak_mm": max(xs) - min(xs),
    })
    if reference_mm is not None:
        errors = [value - reference_mm for value in xs]
        result.update({
            "bias_mm": statistics.fmean(errors),
            "mae_mm": statistics.fmean(abs(error) for error in errors),
            "rmse_mm": math.sqrt(statistics.fmean(error * error for error in errors)),
        })
    return result


def interval_stats(timestamps: list[float]) -> dict:
    gaps = [(b - a) * 1000.0 for a, b in zip(timestamps, timestamps[1:])]
    if not gaps:
        return {"count": 0, "mean_ms": None, "median_ms": None, "std_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(gaps),
        "mean_ms": statistics.fmean(gaps),
        "median_ms": statistics.median(gaps),
        "std_ms": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
        "min_ms": min(gaps),
        "max_ms": max(gaps),
    }


def available_ports() -> str:
    if list_ports is None:
        return "pyserial未安装。"
    ports = list(list_ports.comports())
    return "\n".join(f"  {p.device}: {p.description or '未知设备'}" for p in ports) or "未检测到串口。"


def wait_for_bytes(ser, expected: bytes, timeout_s: float) -> tuple[bool, bytes]:
    received = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        received.extend(ser.read(ser.in_waiting or 1))
        if expected in received:
            return True, bytes(received)
    return False, bytes(received)


def configure_and_start(ser, address: int, frequency_hz: int, setup_timeout_s: float) -> bool:
    frequency_command = command(0xFA, 0x04, 0x0A, FREQUENCY_CODES[frequency_hz])
    ser.reset_input_buffer()
    ser.write(frequency_command)
    ser.flush()
    confirmed, received = wait_for_bytes(ser, bytes.fromhex("FA 04 8A 78"), setup_timeout_s)
    if confirmed:
        print(f"  模块已确认 {frequency_hz} Hz 设置。")
    else:
        suffix = f"，期间收到：{received.hex(' ')}" if received else ""
        print(f"  警告：未收到频率设置应答{suffix}；继续检查连续测量输出。")
    ser.reset_input_buffer()
    ser.write(command(address, 0x06, 0x03))
    ser.flush()
    return confirmed


def write_outputs(rows: list[dict], summary: dict, label: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = LOG_DIR / f"laser100m_{label}_{stamp}"
    csv_path, json_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json")
    fieldnames = ["host_elapsed_s", "wall_time", "status", "distance_mm", "corrected_distance_mm", "offset_mm", "error_code", "error_text", "raw_hex"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return csv_path, json_path


def capture_point(args: argparse.Namespace) -> int:
    parser = Laser100mParser(args.address)
    rows: list[dict] = []
    timestamps: list[float] = []
    first_frame_at: Optional[float] = None
    capture_started: Optional[float] = None
    setup_confirmed = False
    print(f"连接 {args.port} @ {args.baud}；目标采集 {args.duration:g} s，设置 {args.frequency} Hz。")
    print(
        f"近距离软件补偿：≤{args.offset_full_until_mm:g} mm 时 {args.offset_mm:+g} mm，"
        f"到 {args.offset_zero_from_mm:g} mm 线性衰减为0；不写入硬件距离修正。"
    )
    try:
        with serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=1) as ser:
            ser.dtr = False
            ser.rts = False
            setup_confirmed = configure_and_start(ser, args.address, args.frequency, args.setup_timeout)
            print(f"  等待首个连续测量帧（最长 {args.first_frame_timeout:g} s）…")
            wait_started = time.monotonic()
            wait_deadline = wait_started + args.first_frame_timeout
            last_report = wait_started
            while True:
                now = time.monotonic()
                if capture_started is None and now >= wait_deadline:
                    break
                if capture_started is not None and now - capture_started >= args.duration:
                    break
                for frame in parser.feed(ser.read(ser.in_waiting or 1)):
                    received_at = time.monotonic()
                    if first_frame_at is None:
                        first_frame_at = received_at
                        capture_started = received_at
                        print("  已收到首帧，正式计时采集。")
                    elapsed = received_at - capture_started
                    offset = applied_offset_mm(
                        frame.distance_mm,
                        args.offset_mm,
                        args.offset_full_until_mm,
                        args.offset_zero_from_mm,
                    )
                    rows.append({
                        "host_elapsed_s": round(elapsed, 6),
                        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                        **asdict(frame),
                        "corrected_distance_mm": corrected_distance(
                            frame.distance_mm,
                            args.offset_mm,
                            args.offset_full_until_mm,
                            args.offset_zero_from_mm,
                        ),
                        "offset_mm": offset,
                    })
                    timestamps.append(elapsed)
                now = time.monotonic()
                if capture_started is not None and now - last_report >= 1.0:
                    valid = [row["distance_mm"] for row in rows if row["status"] == "ok"]
                    errors = len(rows) - len(valid)
                    latest = rows[-1] if rows else None
                    current = (f"原始={latest['distance_mm']:.1f} mm 补偿后={latest['corrected_distance_mm']:.1f} mm"
                               if latest and latest['status'] == 'ok' else "无有效距离（当前为错误帧）")
                    print(f"  帧数={len(rows)} 有效={len(valid)} 错误={errors} {current}", flush=True)
                    last_report = now
    except serial.SerialException as exc:
        print(f"无法打开/读取 {args.port}: {exc}\n可用串口：\n{available_ports()}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已由用户停止，保存已采集数据。")

    valid = [float(row["distance_mm"]) for row in rows if row["status"] == "ok"]
    errors: dict[str, int] = {}
    for row in rows:
        if row["status"] == "device_error":
            key = f"ERR-{row['error_code']} {row['error_text']}"
            errors[key] = errors.get(key, 0) + 1
    measured_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    observed_rate = (len(timestamps) - 1) / measured_span if measured_span > 0 else None
    valid_rate = (len(valid) - 1) / measured_span if measured_span > 0 and len(valid) > 1 else None
    reference = args.reference_mm
    summary = {
        "test": {
            "port": args.port, "baud": args.baud, "address": args.address,
            "requested_frequency_hz": args.frequency, "duration_s": args.duration,
            "reference_mm": reference, "frequency_setting_confirmed": setup_confirmed,
            "near_offset_max_mm": args.offset_mm,
            "offset_full_until_mm": args.offset_full_until_mm,
            "offset_zero_from_mm": args.offset_zero_from_mm,
            "correction_formula": (
                "raw<=full_until: +max_offset; full_until<raw<zero_from: "
                "+max_offset*(zero_from-raw)/(zero_from-full_until); raw>=zero_from: +0"
            ),
            "reference_source": "operator_entered_ruler_distance",
        },
        "protocol": {
            "total_frames": len(rows), "valid_distance_frames": len(valid),
            "device_error_frames": len(rows) - len(valid), "device_errors": errors,
            "observed_total_rate_hz": observed_rate, "observed_valid_rate_hz": valid_rate,
            "bad_checksum": parser.bad_checksum, "bad_payload": parser.bad_payload,
            "discarded_bytes": parser.discarded_bytes,
            "valid_reply_ratio": len(valid) / len(rows) if rows else None,
        },
        "intervals": interval_stats(timestamps),
        "statistics": describe(valid, reference),
        "corrected_statistics": describe(
            (
                corrected_distance(
                    value,
                    args.offset_mm,
                    args.offset_full_until_mm,
                    args.offset_zero_from_mm,
                )
                for value in valid
            ),
            reference,
        ),
        "correction_comparison": [
            {"name": "raw", **describe(valid, reference)},
            {
                "name": "configured_piecewise",
                **describe(
                    (
                        corrected_distance(
                            value,
                            args.offset_mm,
                            args.offset_full_until_mm,
                            args.offset_zero_from_mm,
                        )
                        for value in valid
                    ),
                    reference,
                ),
            },
        ],
    }
    label = f"ref{reference:g}mm" if reference is not None else "unreferenced"
    csv_path, json_path = write_outputs(rows, summary, label)
    print(f"完成：总帧={len(rows)}，有效距离={len(valid)}，设备错误={len(rows)-len(valid)}。")
    if rows:
        print(f"已收到回复中有效距离占比={len(valid)/len(rows):.1%}（不是丢包率）。")
    if observed_rate is not None:
        print(f"实际总帧率={observed_rate:.2f} Hz；有效距离帧率={valid_rate:.2f} Hz" if valid_rate is not None else f"实际总帧率={observed_rate:.2f} Hz；无有效距离帧率")
    stats = summary["statistics"]
    if valid:
        print(f"距离：均值={stats['mean_mm']:.3f} mm，标准差={stats['std_mm']:.3f} mm，峰峰值={stats['peak_to_peak_mm']:.3f} mm")
        if reference is not None:
            print(f"误差：偏差={stats['bias_mm']:.3f} mm，MAE={stats['mae_mm']:.3f} mm，RMSE={stats['rmse_mm']:.3f} mm")
        corrected = summary['corrected_statistics']
        print(f"补偿后均值={corrected['mean_mm']:.3f} mm；标准差={corrected['std_mm']:.3f} mm")
        if reference is not None:
            print("同一批数据的修正前后比较（mm）：")
            for candidate in summary['correction_comparison']:
                print(f"  {candidate['name']}：偏差={candidate['bias_mm']:+.3f}，MAE={candidate['mae_mm']:.3f}，RMSE={candidate['rmse_mm']:.3f}")
    elif errors:
        print(f"无有效距离；错误统计：{errors}")
    else:
        print("未收到任何连续测量帧。")
    print(f"原始数据：{csv_path}\n汇总结果：{json_path}")
    return 0 if rows else 1


def parse_distances(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--distances 必须是逗号分隔的数字，例如 50,100,200,300") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("测试距离必须是大于0的数字。")
    return values


def run_guided(args: argparse.Namespace) -> int:
    try:
        distances = parse_distances(args.distances)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("\n100米模块引导式静态精度测试")
    print(f"测试点：{', '.join(f'{value:g}' for value in distances)} mm；每点记录 {args.duration:g} s。")
    print("请用尺/卡尺确定传感器测距基准面到平整靶面的距离，并保持正对、静止。")
    print("按尺具摆放参考距离，不要调整靶面去凑补偿后的读数；50 mm可通过--distances单独探测。")
    completed = 0
    for index, reference in enumerate(distances, 1):
        while True:
            answer = input(f"\n[{index}/{len(distances)}] 摆到 {reference:g} mm 后按Enter开始，输入q结束：").strip().lower()
            if answer in {"", "q", "quit", "exit"}:
                break
            print("请按Enter开始，或输入q结束。")
        if answer in {"q", "quit", "exit"}:
            break
        args.reference_mm = reference
        result = capture_point(args)
        if result == 2:
            return result
        completed += 1
    print(f"测试结束，已完成 {completed}/{len(distances)} 个距离点。")
    return 0 if completed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="100米激光模块独立采集与静态精度测试")
    parser.add_argument("--port", required=True, help="串口，例如COM3")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x80)
    parser.add_argument("--frequency", type=int, choices=sorted(FREQUENCY_CODES), default=20)
    parser.add_argument("--duration", type=float, default=30.0, help="收到首帧后的采集时长")
    parser.add_argument("--reference-mm", type=float, help="单点参考距离；指定后跳过引导流程")
    parser.add_argument("--distances", default="100,200,300", help="引导测试距离，默认100,200,300；可选50,60,70,80探测下限")
    parser.add_argument("--offset-mm", type=float, default=2.0, help="近距离最大软件补偿，默认+2 mm；0关闭")
    parser.add_argument("--offset-full-until-mm", type=float, default=60.0, help="完整应用近距离补偿的原始距离上限，默认60 mm")
    parser.add_argument("--offset-zero-from-mm", type=float, default=70.0, help="补偿线性衰减至0的原始距离，默认70 mm")
    parser.add_argument("--setup-timeout", type=float, default=8.0, help="等待频率设置应答的最长时间")
    parser.add_argument("--first-frame-timeout", type=float, default=10.0, help="等待首个连续帧的最长时间")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if serial is None:
        print("缺少pyserial：请先执行 pip install pyserial", file=sys.stderr)
        return 2
    if not 0 <= args.address <= 255 or any(not math.isfinite(v) or v <= 0 for v in (args.duration, args.setup_timeout, args.first_frame_timeout)):
        print("地址需在0..255；时长参数必须大于0。", file=sys.stderr)
        return 2
    if not all(math.isfinite(v) for v in (args.offset_mm, args.offset_full_until_mm, args.offset_zero_from_mm)):
        print("补偿参数必须是有限数字。", file=sys.stderr)
        return 2
    if args.offset_full_until_mm < 0 or args.offset_zero_from_mm <= args.offset_full_until_mm:
        print("补偿区间必须满足0 <= 完整补偿上限 < 补偿归零距离。", file=sys.stderr)
        return 2
    if args.reference_mm is not None and (not math.isfinite(args.reference_mm) or args.reference_mm <= 0):
        print("--reference-mm必须大于0。", file=sys.stderr)
        return 2
    return capture_point(args) if args.reference_mm is not None else run_guided(args)


if __name__ == "__main__":
    raise SystemExit(main())
