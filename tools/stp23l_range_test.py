#!/usr/bin/env python
"""STP-23L 串口测距模块的独立采集与静态精度测试。

引导式示例（先关闭可能占用该 COM 口的上位机）：
    python tools/stp23l_range_test.py --port COM5

程序会依次提示摆放 50、100、200、300 mm 的平整靶面。每次摆好后按
Enter 才开始采集。若只想复测一个距离，可使用 ``--reference-mm 200``。

脚本不依赖主程序，不会修改任何项目配置。采集结果写入
``rangefinder_test_logs/``（该目录已由 .gitignore 中的 CSV 规则忽略）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - runtime environment error
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "rangefinder_test_logs"
FRAME_HEAD = b"\xAA\xAA\xAA\xAA"
PACK_GET_DISTANCE = 0x02
POINT_COUNT = 12
POINT_STRUCT = struct.Struct("<hHIBIh")
PAYLOAD_STRUCT = struct.Struct("<" + "hHIBIh" * POINT_COUNT + "I")
PAYLOAD_BYTES = PAYLOAD_STRUCT.size  # 12 * 15 + 4 = 184


@dataclass(frozen=True)
class Measurement:
    distance_mm: int
    noise: int
    peak: int
    confidence: int
    integration: int
    ref_tof: int


@dataclass(frozen=True)
class RangeFrame:
    module_timestamp: int
    points: tuple[Measurement, ...]


class Stp23lParser:
    """增量解析官方 TRNet 串口帧，并在损坏数据后重新同步。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.bad_checksum = 0
        self.bad_length = 0
        self.non_distance_packets = 0

    @staticmethod
    def _checksum(frame_without_checksum: bytes) -> int:
        # 官方 SDK：从 device_address 起对 6 + payload_len 字节求和。
        return sum(frame_without_checksum[4:]) & 0xFF

    def feed(self, data: bytes) -> list[RangeFrame]:
        self._buffer.extend(data)
        frames: list[RangeFrame] = []
        while True:
            start = self._buffer.find(FRAME_HEAD)
            if start < 0:
                del self._buffer[:-3]
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 11:
                break

            payload_len = struct.unpack_from("<H", self._buffer, 8)[0]
            frame_len = 11 + payload_len
            if payload_len > 320:
                self.bad_length += 1
                del self._buffer[0]
                continue
            if len(self._buffer) < frame_len:
                break

            raw = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]
            if self._checksum(raw[:-1]) != raw[-1]:
                self.bad_checksum += 1
                continue
            if raw[5] != PACK_GET_DISTANCE:
                self.non_distance_packets += 1
                continue
            if payload_len != PAYLOAD_BYTES:
                self.bad_length += 1
                continue

            values = PAYLOAD_STRUCT.unpack(raw[10:-1])
            points = tuple(
                Measurement(*values[offset:offset + 6])
                for offset in range(0, POINT_COUNT * 6, 6)
            )
            frames.append(RangeFrame(values[-1], points))
        return frames


def distance_candidates(frame: RangeFrame, min_confidence: int, max_noise: Optional[int]) -> dict[str, Optional[float]]:
    distances = [p.distance_mm for p in frame.points if p.distance_mm >= 0]
    valid = [
        p.distance_mm
        for p in frame.points
        if p.distance_mm >= 0
        and p.confidence >= min_confidence
        and (max_noise is None or p.noise <= max_noise)
    ]
    return {
        "first_mm": float(distances[0]) if distances else None,
        "median_mm": float(statistics.median(distances)) if distances else None,
        "quality_median_mm": float(statistics.median(valid)) if valid else None,
        "valid_points": float(len(valid)),
    }


def describe(values: Iterable[float], reference_mm: Optional[float]) -> dict[str, Optional[float]]:
    xs = list(values)
    if not xs:
        return {"count": 0, "mean_mm": None, "std_mm": None, "min_mm": None, "max_mm": None, "bias_mm": None, "rmse_mm": None}
    mean = statistics.fmean(xs)
    result: dict[str, Optional[float]] = {
        "count": len(xs),
        "mean_mm": mean,
        "std_mm": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min_mm": min(xs),
        "max_mm": max(xs),
        "bias_mm": None,
        "rmse_mm": None,
    }
    if reference_mm is not None:
        errors = [x - reference_mm for x in xs]
        result["bias_mm"] = statistics.fmean(errors)
        result["rmse_mm"] = math.sqrt(statistics.fmean(e * e for e in errors))
    return result


def available_ports() -> str:
    if list_ports is None:
        return "pyserial 未安装，无法枚举串口。"
    ports = list(list_ports.comports())
    if not ports:
        return "未检测到串口。"
    return "\n".join(f"  {p.device}: {p.description or '未知设备'}" for p in ports)


def write_outputs(rows: list[dict], summary: dict, label: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = LOG_DIR / f"stp23l_{label}_{stamp}"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return csv_path, json_path


def run(args: argparse.Namespace) -> int:
    if serial is None:
        print("缺少 pyserial：请先执行 pip install pyserial", file=sys.stderr)
        return 2
    if args.reference_mm is not None and not 0 <= args.reference_mm <= 300:
        print("本轮测试范围限定为 0–300 mm。", file=sys.stderr)
        return 2
    if args.duration <= 0:
        print("--duration 必须大于 0。", file=sys.stderr)
        return 2

    parser = Stp23lParser()
    rows: list[dict] = []
    t_start = time.monotonic()
    last_report = t_start
    print(f"连接 {args.port} @ {args.baud}；采集 {args.duration:g} s。")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
            ser.reset_input_buffer()
            while time.monotonic() - t_start < args.duration:
                chunk = ser.read(ser.in_waiting or 1)
                for frame in parser.feed(chunk):
                    candidates = distance_candidates(frame, args.min_confidence, args.max_noise)
                    row = {
                        "host_elapsed_s": round(time.monotonic() - t_start, 6),
                        "module_timestamp": frame.module_timestamp,
                        **candidates,
                    }
                    for index, point in enumerate(frame.points):
                        for key, value in asdict(point).items():
                            row[f"p{index + 1}_{key}"] = value
                    rows.append(row)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    current = rows[-1]["median_mm"] if rows else None
                    print(f"  帧数={len(rows)} 当前中位数={current} mm 校验失败={parser.bad_checksum}")
                    last_report = now
    except serial.SerialException as exc:
        print(f"无法打开 {args.port}: {exc}\n可用串口：\n{available_ports()}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已由用户停止采集，保存已收集的数据。")

    if not rows:
        print("未收到有效 STP-23L 测距帧。请确认上位机已关闭、COM 口与 230400 波特率正确。", file=sys.stderr)
        return 1

    selected_key = "quality_median_mm" if args.min_confidence > 0 or args.max_noise is not None else "median_mm"
    selected = [row[selected_key] for row in rows if row[selected_key] is not None]
    reference = args.reference_mm
    summary = {
        "test": {
            "port": args.port, "baud": args.baud, "duration_s": args.duration,
            "reference_mm": reference, "min_confidence": args.min_confidence,
            "max_noise": args.max_noise, "selected_distance": selected_key,
        },
        "protocol": {
            "valid_distance_frames": len(rows), "estimated_frame_rate_hz": len(rows) / max(time.monotonic() - t_start, 1e-9),
            "bad_checksum": parser.bad_checksum, "bad_length": parser.bad_length,
            "non_distance_packets": parser.non_distance_packets,
        },
        "statistics": {
            "first_mm": describe((r["first_mm"] for r in rows if r["first_mm"] is not None), reference),
            "median_mm": describe((r["median_mm"] for r in rows if r["median_mm"] is not None), reference),
            "quality_median_mm": describe((r["quality_median_mm"] for r in rows if r["quality_median_mm"] is not None), reference),
        },
    }
    label = f"ref{reference:g}mm" if reference is not None else "unreferenced"
    csv_path, json_path = write_outputs(rows, summary, label)
    stats = summary["statistics"][selected_key]
    print(f"完成：{len(rows)} 帧，{summary['protocol']['estimated_frame_rate_hz']:.2f} Hz")
    print(f"{selected_key}: 均值={stats['mean_mm']:.2f} mm，标准差={stats['std_mm']:.2f} mm，RMSE={stats['rmse_mm']}")
    print(f"原始数据：{csv_path}\n汇总结果：{json_path}")
    return 0


def parse_distances(text: str) -> list[float]:
    """将 ``50,100,200,300`` 形式的用户输入转为经验证的距离列表。"""
    try:
        distances = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--distances 必须是逗号分隔的数字，例如 50,100,200,300") from exc
    if not distances:
        raise ValueError("--distances 至少需要一个距离。")
    if any(not 0 <= value <= 300 for value in distances):
        raise ValueError("本轮测试范围限定为 0–300 mm。")
    return distances


def run_guided(args: argparse.Namespace) -> int:
    """按预设距离逐点等待用户确认，再复用单点采集和统计逻辑。"""
    try:
        distances = parse_distances(args.distances)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("\nSTP-23L 引导式静态精度测试")
    print(f"将依次测试：{', '.join(f'{value:g}' for value in distances)} mm")
    print(f"每点记录 {args.duration:g} 秒。请让传感器和靶面保持静止、正对。")
    print("任一步输入 q 后回车可结束测试；已经完成的点会保留结果。\n")

    completed = 0
    for index, reference_mm in enumerate(distances, start=1):
        while True:
            answer = input(
                f"[{index}/{len(distances)}] 请将平整靶面放在 {reference_mm:g} mm，"
                "稳定后按 Enter 开始（输入 q 结束）："
            ).strip().lower()
            if answer in {"q", "quit", "exit"}:
                print(f"测试结束，已完成 {completed}/{len(distances)} 个距离点。")
                return 0
            if not answer:
                break
            print("请输入 Enter 开始，或输入 q 结束。")

        print(f"开始记录 {reference_mm:g} mm …")
        args.reference_mm = reference_mm
        result = run(args)
        if result != 0:
            print(f"{reference_mm:g} mm 测试失败，停止后续步骤。", file=sys.stderr)
            return result
        completed += 1
        print(f"✓ {reference_mm:g} mm 完成。\n")

    print("所有距离点测试完成。请把每个 JSON 汇总文件发给我，我们再据此确定滤波策略。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STP-23L 独立串口采集与静态精度测试")
    parser.add_argument("--port", required=True, help="STP-23L 的 COM 口，例如 COM5")
    parser.add_argument("--baud", type=int, default=230400, help="串口波特率，默认 230400")
    parser.add_argument("--duration", type=float, default=30.0, help="采集时长（秒），默认 30")
    parser.add_argument(
        "--reference-mm", type=float,
        help="仅测试一个尺/卡尺实测靶距（指定后跳过引导流程）",
    )
    parser.add_argument(
        "--distances", default="50,100,200,300",
        help="引导流程的距离列表（mm），默认 50,100,200,300",
    )
    parser.add_argument("--min-confidence", type=int, default=0, help="质量筛选的最小置信度，默认不筛选")
    parser.add_argument("--max-noise", type=int, help="质量筛选的最大噪声，默认不筛选")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(run(parsed_args) if parsed_args.reference_mm is not None else run_guided(parsed_args))
