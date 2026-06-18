#!/usr/bin/env python
"""Batch benchmark: adaptive fusion vs always-on magnetometer fusion."""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.mag_fusion.fusion import AdaptiveMagConfig, run_adaptive_vs_fullmag

# The offline benchmark only needs core.imu_kinematics through metrics.py.
# Avoid forcing serial/PyQt hardware dependencies in minimal Python runtimes.
if "core.device_manager" not in sys.modules:
    _dm = types.ModuleType("core.device_manager")
    _dm.DeviceManager = object
    sys.modules["core.device_manager"] = _dm
if "core.dicom_loader" not in sys.modules:
    _dl = types.ModuleType("core.dicom_loader")
    _dl.DicomModelLoader = object
    sys.modules["core.dicom_loader"] = _dl

from tools.mag_fusion.metrics import compute_track
from tools.mag_fusion.replay import load_csv


DEFAULT_PATTERN = "imu_calibration_logs/mag_ab_A_*.csv"


def _safe_segment_name(name: str) -> str:
    return name.replace(",", "_").replace(" ", "_")


def _segments(t: np.ndarray, events: list[tuple[float, str]]) -> list[tuple[str, float, float]]:
    out = [("all", float(t[0]), float(t[-1]))]
    if events:
        bounds = [(float(events[0][0]), events[0][1])]
        bounds.extend((float(ts), name) for ts, name in events[1:])
        for i, (start, name) in enumerate(bounds):
            end = float(bounds[i + 1][0]) if i + 1 < len(bounds) else float(t[-1])
            if end - start >= 0.5:
                out.append((_safe_segment_name(name), start, end))
        return out

    if float(t[-1]) >= 110.0:
        out.extend(
            [
                ("dist_50_80", 50.0, 80.0),
                ("dist_80_110", 80.0, 110.0),
            ]
        )
    return out


def _ratio(a: float, b: float) -> float:
    if abs(b) <= 1e-9:
        return float("nan")
    return float(a / b)


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _print_row(
    csv_name: str,
    duration: float,
    segment: str,
    metric: str,
    adaptive: float,
    fullmag: float,
) -> None:
    ratio = _ratio(adaptive, fullmag)
    print(
        f"{csv_name},{duration:.1f},{segment},{metric},"
        f"{_fmt(adaptive)},{_fmt(fullmag)},{_fmt(ratio)}"
    )


def benchmark_file(
    csv_path: Path,
    cfg: AdaptiveMagConfig,
) -> tuple[list[float], list[float], list[float]]:
    table = load_csv(csv_path)
    out = run_adaptive_vs_fullmag(
        table.t,
        table.gyro,
        table.acc,
        table.mag,
        table.q_chip,
        cfg=cfg,
    )

    duration = float(table.t[-1] - table.t[0])
    primary_ratios: list[float] = []
    disturbed_primary_ratios: list[float] = []
    reference_ratios: list[float] = []
    for name, t0, t1 in _segments(table.t, table.events):
        mask = (table.t >= t0) & (table.t <= t1)
        if int(np.sum(mask)) < 50:
            continue

        adaptive = compute_track(
            "adaptive",
            out["q_adaptive"][mask],
            gyro=table.gyro[mask],
            ref=table.q_chip[mask],
        )
        fullmag = compute_track(
            "fullmag",
            out["q_full_mag"][mask],
            gyro=table.gyro[mask],
            ref=table.q_chip[mask],
        )

        rows = [
            ("jitter", adaptive.needle_jitter_deg_rms, fullmag.needle_jitter_deg_rms),
            ("yaw_step", adaptive.yaw_max_step_deg, fullmag.yaw_max_step_deg),
            ("end_drift", adaptive.yaw_end_drift_deg, fullmag.yaw_end_drift_deg),
            ("vs_chip_mean", adaptive.vs_chip_mean_deg, fullmag.vs_chip_mean_deg),
        ]
        for metric, a, b in rows:
            r = _ratio(a, b)
            if np.isfinite(r):
                if metric == "vs_chip_mean":
                    reference_ratios.append(r)
                else:
                    primary_ratios.append(r)
                    if "disturbed" in name:
                        disturbed_primary_ratios.append(r)
            _print_row(csv_path.name, duration, name, metric, a, b)
    return primary_ratios, disturbed_primary_ratios, reference_ratios


def _print_summary(name: str, ratios: list[float]) -> None:
    finite = np.array([x for x in ratios if np.isfinite(x)], dtype=float)
    if len(finite):
        print(
            f"{name},,,,,"
            f"median_ratio={np.median(finite):.3f},"
            f"mean_ratio={np.mean(finite):.3f},"
            f"p75_ratio={np.percentile(finite, 75):.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive fusion against always-on magnetometer fusion."
    )
    parser.add_argument("--csv", default="", help="Single CSV file. If omitted, use --pattern.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for batch mode.")
    parser.add_argument("--fallback", type=float, default=0.50, help="Dirty-magnet six-axis fallback share.")
    parser.add_argument("--eps", type=float, default=0.12)
    args = parser.parse_args(argv)

    cfg = AdaptiveMagConfig(eps=args.eps, dirty_six_axis_fallback=args.fallback)
    if args.csv:
        paths = [Path(args.csv)]
    else:
        paths = sorted(Path(".").glob(args.pattern))

    if not paths:
        raise SystemExit(f"No CSV files matched: {args.csv or args.pattern}")

    print("csv,duration_s,segment,metric,adaptive,fullmag,ratio")
    primary_ratios: list[float] = []
    disturbed_primary_ratios: list[float] = []
    reference_ratios: list[float] = []
    for path in paths:
        primary, disturbed_primary, reference = benchmark_file(path, cfg)
        primary_ratios.extend(primary)
        disturbed_primary_ratios.extend(disturbed_primary)
        reference_ratios.extend(reference)

    _print_summary("summary_primary_metrics", primary_ratios)
    _print_summary("summary_disturbed_primary_metrics", disturbed_primary_ratios)
    _print_summary("summary_vs_chip_reference", reference_ratios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
