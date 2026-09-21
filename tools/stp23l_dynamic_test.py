#!/usr/bin/env python
"""STP-23L 手动阶段控制的动态往返测试。

使用方法（先关闭占用串口的上位机）：
    python tools/stp23l_dynamic_test.py --port COM4

操作者通过 Enter 明确控制接近和返回的开始/结束。脚本不会假定人工运动
严格匀速，也不会把规划直线当成真实位置。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tools.stp23l_range_test import (
        Stp23lParser,
        available_ports,
        distance_candidates,
        serial,
    )
except ModuleNotFoundError:
    from stp23l_range_test import (
        Stp23lParser,
        available_ports,
        distance_candidates,
        serial,
    )


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "rangefinder_test_logs"


def describe(values: list[float], target_mm: Optional[float] = None) -> dict:
    if not values:
        return {"count": 0}
    mean = statistics.fmean(values)
    result = {
        "count": len(values),
        "mean_mm": mean,
        "median_mm": statistics.median(values),
        "std_mm": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_mm": min(values),
        "max_mm": max(values),
    }
    if target_mm is not None:
        result["bias_mm"] = mean - target_mm
    return result


def monotonic_metrics(values: list[float], direction: str, tolerance_mm: float) -> dict:
    if len(values) < 2:
        return {"steps": 0, "opposite_steps": 0, "opposite_ratio": None}
    diffs = [b - a for a, b in zip(values, values[1:])]
    opposite = (
        sum(delta > tolerance_mm for delta in diffs)
        if direction == "decreasing"
        else sum(delta < -tolerance_mm for delta in diffs)
    )
    return {
        "steps": len(diffs),
        "opposite_steps": opposite,
        "opposite_ratio": opposite / len(diffs),
        "tolerance_mm": tolerance_mm,
    }


class QualityGate:
    """拒绝盲区、超范围和单帧大跳变，并允许稳定后重新捕获目标。"""

    def __init__(
        self,
        min_mm: float,
        max_mm: float,
        jump_mm: float,
        reacquire_frames: int,
        reacquire_spread_mm: float,
    ) -> None:
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.jump_mm = jump_mm
        self.reacquire_frames = reacquire_frames
        self.reacquire_spread_mm = reacquire_spread_mm
        self.last_good: Optional[float] = None
        self._candidates: list[float] = []
        self._target_lost = False

    def reset(self, initial_mm: Optional[float] = None) -> None:
        self.last_good = initial_mm
        self._candidates.clear()
        self._target_lost = False

    def update(self, raw_mm: float, allow_reacquire: bool = True) -> tuple[Optional[float], str]:
        if self._target_lost:
            return None, "target_lost"
        if raw_mm < self.min_mm:
            self._candidates.clear()
            return None, "below_min_range"
        if raw_mm > self.max_mm:
            self._candidates.clear()
            return None, "above_max_range"
        if self.last_good is None or abs(raw_mm - self.last_good) <= self.jump_mm:
            self.last_good = raw_mm
            self._candidates.clear()
            return raw_mm, "accepted"

        if not allow_reacquire:
            self._candidates.clear()
            self._target_lost = True
            return None, "target_lost"

        if not self._candidates or abs(raw_mm - self._candidates[-1]) <= self.reacquire_spread_mm:
            self._candidates.append(raw_mm)
        else:
            self._candidates = [raw_mm]
        if len(self._candidates) >= self.reacquire_frames:
            self.last_good = statistics.median(self._candidates[-self.reacquire_frames:])
            self._candidates.clear()
            return self.last_good, "reacquired"
        return None, "jump_rejected"


class OnlineFilter:
    def __init__(self, window: int, alpha: float) -> None:
        self.window = window
        self.alpha = alpha
        self._recent: list[float] = []
        self._ema: Optional[float] = None

    def reset(self) -> None:
        self._recent.clear()
        self._ema = None

    def update(self, accepted_mm: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        if accepted_mm is None:
            return None, None
        self._recent.append(float(accepted_mm))
        self._recent = self._recent[-self.window:]
        med = float(statistics.median(self._recent))
        self._ema = med if self._ema is None else self.alpha * med + (1.0 - self.alpha) * self._ema
        return med, self._ema


class TargetStability:
    """要求测量值连续处于目标容差内达到指定时长。"""

    def __init__(self, target_mm: float, tolerance_mm: float, stable_seconds: float) -> None:
        self.target_mm = target_mm
        self.tolerance_mm = tolerance_mm
        self.stable_seconds = stable_seconds
        self._stable_since: Optional[float] = None

    def update(self, value_mm: Optional[float], now: float) -> tuple[bool, float]:
        if value_mm is None or abs(value_mm - self.target_mm) > self.tolerance_mm:
            self._stable_since = None
            return False, 0.0
        if self._stable_since is None:
            self._stable_since = now
        stable_for = now - self._stable_since
        return stable_for >= self.stable_seconds, stable_for


def wait_for_stable_target(
    ser,
    parser: Stp23lParser,
    target_mm: float,
    label: str,
    args: argparse.Namespace,
) -> float:
    """实时预览距离，直到目标附近连续稳定指定时间。"""
    tracker = TargetStability(target_mm, args.target_tolerance_mm, args.stable_seconds)
    latest_raw: Optional[float] = None
    latest_stable_for = 0.0
    last_measurement_at: Optional[float] = None
    last_report = 0.0
    print(
        f"\n[{label}锁定] 请调整靶面，使读数保持在 "
        f"{target_mm:g} ± {args.target_tolerance_mm:g} mm。"
    )
    print(f"连续稳定 {args.stable_seconds:g} 秒后会自动开始记录；Ctrl+C 可取消。")
    ser.reset_input_buffer()

    while True:
        chunk = ser.read(ser.in_waiting or 1)
        for frame in parser.feed(chunk):
            candidates = distance_candidates(frame, args.min_confidence, args.max_noise)
            raw = candidates["median_mm"]
            now = time.monotonic()
            latest_raw = None if raw is None else float(raw)
            last_measurement_at = now
            in_valid_range = (
                latest_raw is not None
                and args.valid_min_mm <= latest_raw <= args.valid_max_mm
            )
            locked, latest_stable_for = tracker.update(
                latest_raw if in_valid_range else None,
                now,
            )
            if locked and latest_raw is not None:
                print(f"\a  已锁定：{latest_raw:.1f} mm，开始记录。")
                ser.reset_input_buffer()
                return latest_raw

        now = time.monotonic()
        if last_measurement_at is not None and now - last_measurement_at > 0.5:
            tracker.update(None, now)
            latest_stable_for = 0.0
        if now - last_report >= 0.3:
            if latest_raw is None or last_measurement_at is None or now - last_measurement_at > 0.5:
                status = "无有效测距数据"
            elif not args.valid_min_mm <= latest_raw <= args.valid_max_mm:
                status = f"信号无效（当前 {latest_raw:.1f} mm）"
            elif abs(latest_raw - target_mm) > args.target_tolerance_mm:
                status = f"未对准（当前 {latest_raw:.1f} mm）"
            else:
                status = (
                    f"锁定中（当前 {latest_raw:.1f} mm，"
                    f"{min(latest_stable_for, args.stable_seconds):.1f}/{args.stable_seconds:g} s）"
                )
            print(f"  {status}", flush=True)
            last_report = now


def _enter_waiter(event: threading.Event) -> None:
    try:
        input()
    except EOFError:
        pass
    event.set()


def capture_phase(
    ser,
    parser: Stp23lParser,
    rows: list[dict],
    phase: str,
    elapsed_offset: float,
    gate: QualityGate,
    smoother: OnlineFilter,
    args: argparse.Namespace,
    duration_s: Optional[float] = None,
) -> float:
    """采集一个固定时长阶段，或采集到用户按 Enter 为止。"""
    stop_event = threading.Event()
    if duration_s is None:
        print("移动过程中会实时显示距离；到达目标后按 Enter 停止本阶段。", flush=True)
        threading.Thread(target=_enter_waiter, args=(stop_event,), daemon=True).start()

    ser.reset_input_buffer()
    phase_started = time.monotonic()
    last_report = phase_started
    while True:
        phase_elapsed = time.monotonic() - phase_started
        if duration_s is not None and phase_elapsed >= duration_s:
            break
        if duration_s is None and stop_event.is_set():
            break

        chunk = ser.read(ser.in_waiting or 1)
        for frame in parser.feed(chunk):
            candidates = distance_candidates(frame, args.min_confidence, args.max_noise)
            raw = candidates["median_mm"]
            if raw is None:
                continue
            accepted, status = gate.update(float(raw), allow_reacquire=False)
            median_n, smooth = smoother.update(accepted)
            row = {
                "elapsed_s": round(elapsed_offset + phase_elapsed, 6),
                "phase": phase,
                "phase_elapsed_s": round(phase_elapsed, 6),
                "module_timestamp": frame.module_timestamp,
                **candidates,
                "gate_status": status,
                "accepted_mm": accepted,
                "median_filtered_mm": median_n,
                "ema_mm": smooth,
            }
            for index, point in enumerate(frame.points):
                for key, value in asdict(point).items():
                    row[f"p{index + 1}_{key}"] = value
            rows.append(row)

        now = time.monotonic()
        if now - last_report >= 0.5:
            if rows and rows[-1]["phase"] == phase:
                latest = rows[-1]
                raw_text = f"{float(latest['median_mm']):6.1f}"
                smooth_value = latest["ema_mm"]
                smooth_text = "--" if smooth_value is None else f"{float(smooth_value):6.1f}"
                if latest["gate_status"] == "target_lost":
                    status_text = "目标丢失，本阶段剩余数据无效"
                elif latest["gate_status"] in {"accepted", "reacquired"}:
                    status_text = "有效"
                else:
                    status_text = "信号无效"
                print(
                    f"  当前={raw_text} mm  滤波={smooth_text} mm  "
                    f"状态={status_text}（{latest['gate_status']}）",
                    flush=True,
                )
            last_report = now
    return time.monotonic() - phase_started


def phase_values(rows: list[dict], phase: str, accepted_only: bool = True) -> list[float]:
    key = "accepted_mm" if accepted_only else "median_mm"
    return [
        float(row[key])
        for row in rows
        if row["phase"] == phase and row[key] not in (None, "")
    ]


def analyze(rows: list[dict], args: argparse.Namespace, parser: Stp23lParser, recorded_s: float) -> dict:
    raw = [float(row["median_mm"]) for row in rows]
    raw_jumps = [abs(b - a) for a, b in zip(raw, raw[1:])]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["gate_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    invalid_run = 0
    max_invalid_run = 0
    for row in rows:
        if row["accepted_mm"] in (None, ""):
            invalid_run += 1
            max_invalid_run = max(max_invalid_run, invalid_run)
        else:
            invalid_run = 0

    approach = phase_values(rows, "approach")
    retreat = phase_values(rows, "retreat")
    approach_target_lost = any(
        row["phase"] == "approach" and row["gate_status"] == "target_lost"
        for row in rows
    )
    retreat_target_lost = any(
        row["phase"] == "retreat" and row["gate_status"] == "target_lost"
        for row in rows
    )
    return {
        "test": {
            "port": args.port,
            "baud": args.baud,
            "far_mm": args.far_mm,
            "near_mm": args.near_mm,
            "hold_seconds": args.hold_seconds,
            "control": "user presses Enter to start and stop each motion phase",
            "valid_range_mm": [args.valid_min_mm, args.valid_max_mm],
            "jump_reject_mm": args.jump_reject_mm,
            "median_window": args.median_window,
            "ema_alpha": args.ema_alpha,
            "target_tolerance_mm": args.target_tolerance_mm,
            "stable_seconds": args.stable_seconds,
            "recording_auto_reacquire": False,
        },
        "protocol": {
            "valid_frames": len(rows),
            "recorded_seconds": recorded_s,
            "estimated_frame_rate_hz": len(rows) / max(recorded_s, 1e-9),
            "bad_checksum": parser.bad_checksum,
            "bad_length": parser.bad_length,
            "non_distance_packets": parser.non_distance_packets,
        },
        "quality_gate": {
            "status_counts": status_counts,
            "accepted_ratio": (
                sum(status in {"accepted", "reacquired"} for status in (row["gate_status"] for row in rows))
                / len(rows)
            ),
            "max_consecutive_invalid_frames": max_invalid_run,
            "estimated_max_invalid_seconds": max_invalid_run * recorded_s / max(len(rows), 1),
        },
        "static_segments": {
            "start_hold": describe(phase_values(rows, "start_hold"), args.far_mm),
            "near_hold": describe(phase_values(rows, "near_hold"), args.near_mm),
            "end_hold": describe(phase_values(rows, "end_hold"), args.far_mm),
        },
        "motion": {
            "approach": {
                **monotonic_metrics(approach, "decreasing", args.monotonic_tolerance_mm),
                "target_lost": approach_target_lost,
                "valid": not approach_target_lost,
            },
            "retreat": {
                **monotonic_metrics(retreat, "increasing", args.monotonic_tolerance_mm),
                "target_lost": retreat_target_lost,
                "valid": not retreat_target_lost,
            },
            "round_trip_valid": not approach_target_lost and not retreat_target_lost,
            "raw_jump_count_gt_10mm": sum(value > 10.0 for value in raw_jumps),
            "raw_jump_count_gt_20mm": sum(value > 20.0 for value in raw_jumps),
            "raw_jump_count_gt_50mm": sum(value > 50.0 for value in raw_jumps),
            "raw_max_frame_jump_mm": max(raw_jumps) if raw_jumps else 0.0,
        },
    }


def write_plot(rows: list[dict], output: Path, boundaries: list[float], args: argparse.Namespace) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    x = [float(row["elapsed_s"]) for row in rows]
    raw = [float(row["median_mm"]) for row in rows]
    accepted = [
        float(row["accepted_mm"])
        if row["accepted_mm"] not in (None, "")
        else float("nan")
        for row in rows
    ]
    rejected_x = [float(row["elapsed_s"]) for row in rows if row["accepted_mm"] in (None, "")]
    rejected = [float(row["median_mm"]) for row in rows if row["accepted_mm"] in (None, "")]
    smoothed = [
        float(row["ema_mm"])
        if row["accepted_mm"] not in (None, "") and row["ema_mm"] not in (None, "")
        else float("nan")
        for row in rows
    ]

    plt.figure(figsize=(11, 6))
    plt.plot(x, raw, color="#9ca3af", linewidth=0.8, label="raw frame median")
    plt.plot(x, accepted, color="#2563eb", linewidth=1.1, label="accepted")
    plt.plot(x, smoothed, color="#dc2626", linewidth=1.6, label="median + EMA")
    if rejected_x:
        plt.scatter(rejected_x, rejected, color="#f59e0b", marker="x", s=20, label="rejected")
    plt.axhline(args.valid_min_mm, color="#f59e0b", linestyle="--", linewidth=0.8)
    plt.axhline(args.valid_max_mm, color="#f59e0b", linestyle="--", linewidth=0.8)
    for boundary in boundaries:
        plt.axvline(boundary, color="#d1d5db", linewidth=0.8)
    plt.xlabel("recorded time (s)")
    plt.ylabel("distance (mm)")
    plt.title("STP-23L manually controlled dynamic test")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def _prompt_enter(message: str) -> bool:
    answer = input(message).strip().lower()
    if answer in {"q", "quit", "exit"}:
        return False
    if answer:
        print("请输入 Enter 继续，或输入 q 退出。", file=sys.stderr)
        return _prompt_enter(message)
    return True


def run(args: argparse.Namespace) -> int:
    if serial is None:
        print("缺少 pyserial：请先执行 pip install pyserial", file=sys.stderr)
        return 2
    if not 70 <= args.near_mm < args.far_mm <= 300:
        print("要求 70 <= near-mm < far-mm <= 300。建议使用80和300 mm。", file=sys.stderr)
        return 2
    if args.hold_seconds <= 0:
        print("--hold-seconds 必须大于0。", file=sys.stderr)
        return 2
    if args.median_window < 1 or args.median_window % 2 == 0:
        print("--median-window 必须是正奇数。", file=sys.stderr)
        return 2
    if not 0 < args.ema_alpha <= 1:
        print("--ema-alpha 必须在 (0, 1]。", file=sys.stderr)
        return 2
    if args.target_tolerance_mm <= 0:
        print("--target-tolerance-mm 必须大于0。", file=sys.stderr)
        return 2
    if args.stable_seconds <= 0:
        print("--stable-seconds 必须大于0。", file=sys.stderr)
        return 2

    print("\nSTP-23L 手动动态往返测试")
    print("每次运动都由你按 Enter 开始，到达目标后再按 Enter 结束。")
    print(
        f"有效范围门限：{args.valid_min_mm:g}–{args.valid_max_mm:g} mm；"
        f"单帧跳变门限：{args.jump_reject_mm:g} mm。\n"
        f"本轮滤波：{args.median_window}帧中值 + EMA α={args.ema_alpha:g}。\n"
        f"目标锁定：±{args.target_tolerance_mm:g} mm 内连续稳定 {args.stable_seconds:g} 秒。\n"
        "记录阶段自动重捕获：关闭；超过跳变门限后本阶段持续标记为目标丢失。"
    )
    if not _prompt_enter(
        f"请把靶面稳定放在 {args.far_mm:g} mm，准备好后按 Enter（q退出）："
    ):
        print("已取消。")
        return 0

    parser = Stp23lParser()
    gate = QualityGate(
        args.valid_min_mm,
        args.valid_max_mm,
        args.jump_reject_mm,
        args.reacquire_frames,
        args.reacquire_spread_mm,
    )
    smoother = OnlineFilter(args.median_window, args.ema_alpha)
    rows: list[dict] = []
    boundaries: list[float] = []
    recorded_s = 0.0

    try:
        with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
            locked_mm = wait_for_stable_target(ser, parser, args.far_mm, "起点", args)
            gate.reset(locked_mm)
            smoother.reset()
            print(f"\n[1/5] 记录 {args.far_mm:g} mm 起点静止段 {args.hold_seconds:g} 秒")
            recorded_s += capture_phase(
                ser, parser, rows, "start_hold", recorded_s, gate, smoother, args,
                duration_s=args.hold_seconds,
            )
            boundaries.append(recorded_s)

            if not _prompt_enter(
                f"\n[2/5] 按 Enter 开始向 {args.near_mm:g} mm 移动（q结束并保存）："
            ):
                raise KeyboardInterrupt
            print(f"\a开始接近；实时读数到约 {args.near_mm:g} mm 时按 Enter。")
            recorded_s += capture_phase(
                ser, parser, rows, "approach", recorded_s, gate, smoother, args,
            )
            boundaries.append(recorded_s)

            locked_mm = wait_for_stable_target(ser, parser, args.near_mm, "近端", args)
            gate.reset(locked_mm)
            smoother.reset()
            print(f"\n[3/5] 记录 {args.near_mm:g} mm 近端静止段 {args.hold_seconds:g} 秒")
            recorded_s += capture_phase(
                ser, parser, rows, "near_hold", recorded_s, gate, smoother, args,
                duration_s=args.hold_seconds,
            )
            boundaries.append(recorded_s)

            if not _prompt_enter(
                f"\n[4/5] 按 Enter 开始返回 {args.far_mm:g} mm（q结束并保存）："
            ):
                raise KeyboardInterrupt
            print(f"\a开始返回；实时读数到约 {args.far_mm:g} mm 时按 Enter。")
            recorded_s += capture_phase(
                ser, parser, rows, "retreat", recorded_s, gate, smoother, args,
            )
            boundaries.append(recorded_s)

            locked_mm = wait_for_stable_target(ser, parser, args.far_mm, "终点", args)
            gate.reset(locked_mm)
            smoother.reset()
            print(f"\n[5/5] 记录 {args.far_mm:g} mm 终点静止段 {args.hold_seconds:g} 秒")
            recorded_s += capture_phase(
                ser, parser, rows, "end_hold", recorded_s, gate, smoother, args,
                duration_s=args.hold_seconds,
            )
    except serial.SerialException as exc:
        print(f"无法打开 {args.port}: {exc}\n可用串口：\n{available_ports()}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n测试提前结束，保存已记录的数据。")

    if not rows:
        print("没有收到有效测距帧。", file=sys.stderr)
        return 1

    summary = analyze(rows, args, parser, recorded_s)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = LOG_DIR / f"stp23l_dynamic_manual_{stamp}"
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    plot_path = prefix.with_suffix(".png")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plotted = write_plot(rows, plot_path, boundaries, args)

    protocol = summary["protocol"]
    quality = summary["quality_gate"]
    motion = summary["motion"]
    print(f"\n完成：{protocol['valid_frames']}帧，{protocol['estimated_frame_rate_hz']:.2f} Hz")
    print(f"质量门接受率：{quality['accepted_ratio']:.1%}")
    if motion["round_trip_valid"]:
        print("动态往返有效性：通过（接近和返回阶段均未锁存目标丢失）")
    else:
        failed_phases = []
        if not motion["approach"]["valid"]:
            failed_phases.append("接近")
        if not motion["retreat"]["valid"]:
            failed_phases.append("返回")
        print(f"动态往返有效性：不通过（{'、'.join(failed_phases)}阶段发生目标丢失）")
    print(f"数据：{csv_path}\n汇总：{json_path}")
    if plotted is not None:
        print(f"曲线：{plotted}")
    else:
        print("未安装matplotlib，已跳过曲线图片。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STP-23L 手动阶段控制动态往返测试")
    parser.add_argument("--port", required=True, help="STP-23L串口，例如COM4")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--far-mm", type=float, default=300.0)
    parser.add_argument("--near-mm", type=float, default=80.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--valid-min-mm", type=float, default=70.0)
    parser.add_argument("--valid-max-mm", type=float, default=320.0)
    parser.add_argument("--jump-reject-mm", type=float, default=30.0)
    parser.add_argument("--reacquire-frames", type=int, default=3)
    parser.add_argument("--reacquire-spread-mm", type=float, default=8.0)
    parser.add_argument("--median-window", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--target-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--stable-seconds", type=float, default=2.0)
    parser.add_argument("--monotonic-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--max-noise", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
