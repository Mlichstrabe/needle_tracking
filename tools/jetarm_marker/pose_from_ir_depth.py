#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 IR 图像中的 4 个反光 marker 融合深度图，估计 marker 三维坐标、针轴和临时针尖。"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.camera_math import backproject_mm, depth_median_window  # noqa: E402
from tools.jetarm_marker.rosbag_io import (  # noqa: E402
    TOPIC_DEPTH,
    TOPIC_DEPTH_INFO,
    TOPIC_IR_INFO,
    FrameRecord,
    camera_info_to_dict,
    load_image_frames,
    read_message_at_index,
)


def _load_marker_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _nearest_frame(frames: Sequence[FrameRecord], timestamps: Sequence[int], target_ns: int) -> Tuple[int, FrameRecord]:
    pos = bisect.bisect_left(timestamps, target_ns)
    candidates = []
    if pos < len(frames):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        raise RuntimeError("没有可用的 depth 帧")
    best = min(candidates, key=lambda idx: abs(timestamps[idx] - target_ns))
    return best, frames[best]


def _parse_marker_uv(row: Dict[str, str], marker_index: int) -> Tuple[Optional[float], Optional[float], bool]:
    valid = row.get(f"m{marker_index}_valid", "0") == "1"
    u_text = row.get(f"m{marker_index}_u", "")
    v_text = row.get(f"m{marker_index}_v", "")
    if not valid or not u_text or not v_text:
        return None, None, False
    return float(u_text), float(v_text), True


def _marker_3d_from_depth(
    row: Dict[str, str],
    depth_frame: FrameRecord,
    depth_info: Dict[str, Any],
    *,
    half_win: int,
    min_depth_pixels: int,
    z_min_mm: float,
    z_max_mm: float,
) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    for i in range(4):
        u, v, marker_valid = _parse_marker_uv(row, i)
        item: Dict[str, Any] = {
            "u": u,
            "v": v,
            "depth_mm": None,
            "depth_pixels": 0,
            "x": None,
            "y": None,
            "z": None,
            "valid": False,
        }
        if marker_valid and u is not None and v is not None:
            z_mm, n_valid = depth_median_window(
                depth_frame.array,
                u,
                v,
                half_win=half_win,
                z_min_mm=z_min_mm,
                z_max_mm=z_max_mm,
                encoding=depth_frame.encoding,
            )
            item["depth_mm"] = z_mm
            item["depth_pixels"] = n_valid
            if z_mm is not None and n_valid >= min_depth_pixels:
                x, y, z = backproject_mm(
                    u,
                    v,
                    z_mm,
                    depth_info["fx"],
                    depth_info["fy"],
                    depth_info["cx"],
                    depth_info["cy"],
                )
                item.update({"x": x, "y": y, "z": z, "valid": True})
        markers.append(item)
    return markers


def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return vec / norm


def _estimate_axis_and_tip(
    markers: Sequence[Dict[str, Any]],
    *,
    axis_start_marker: int,
    axis_end_marker: int,
    tip_offset_mm: float,
    needle_length_mm: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": False,
        "axis": None,
        "tip": None,
        "tail": None,
        "axis_start": None,
        "axis_end": None,
    }
    if axis_start_marker >= len(markers) or axis_end_marker >= len(markers):
        return result
    start = markers[axis_start_marker]
    end = markers[axis_end_marker]
    if not start["valid"] or not end["valid"]:
        return result
    p_start = np.asarray([start["x"], start["y"], start["z"]], dtype=np.float64)
    p_end = np.asarray([end["x"], end["y"], end["z"]], dtype=np.float64)
    axis = _unit(p_end - p_start)
    if axis is None:
        return result
    tip = p_end + axis * tip_offset_mm
    tail = tip - axis * needle_length_mm
    result.update(
        {
            "valid": True,
            "axis": axis,
            "tip": tip,
            "tail": tail,
            "axis_start": p_start,
            "axis_end": p_end,
        }
    )
    return result


def _axis_length_ratio_2d(markers: Sequence[Dict[str, Any]]) -> Optional[float]:
    """2D 长度比：针轴 m3–m1 / 横档 m0–m2。"""
    if len(markers) < 4:
        return None
    required = [markers[i] for i in (0, 1, 2, 3)]
    if any(marker["u"] is None or marker["v"] is None for marker in required):
        return None
    p0 = np.asarray([markers[0]["u"], markers[0]["v"]], dtype=np.float64)
    p1 = np.asarray([markers[1]["u"], markers[1]["v"]], dtype=np.float64)
    p2 = np.asarray([markers[2]["u"], markers[2]["v"]], dtype=np.float64)
    p3 = np.asarray([markers[3]["u"], markers[3]["v"]], dtype=np.float64)
    cross_len = float(np.linalg.norm(p0 - p2))
    axis_len = float(np.linalg.norm(p3 - p1))
    if cross_len <= 1e-9:
        return None
    return axis_len / cross_len


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_pose_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "frame_id",
        "ir_index",
        "timestamp_ns",
        "depth_index",
        "depth_dt_ms",
        "axis_length_ratio_2d",
        "valid",
        "confidence",
    ]
    for i in range(4):
        header += [
            f"m{i}_u",
            f"m{i}_v",
            f"m{i}_depth_mm",
            f"m{i}_depth_pixels",
            f"m{i}_x_cam_mm",
            f"m{i}_y_cam_mm",
            f"m{i}_z_cam_mm",
            f"m{i}_valid",
        ]
    header += [
        "tip_x_cam_mm",
        "tip_y_cam_mm",
        "tip_z_cam_mm",
        "tail_x_cam_mm",
        "tail_y_cam_mm",
        "tail_z_cam_mm",
        "axis_x_cam",
        "axis_y_cam",
        "axis_z_cam",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in results:
            row = [
                item["frame_id"],
                item["ir_index"],
                item["timestamp_ns"],
                item["depth_index"],
                _fmt_float(item["depth_dt_ms"]),
                _fmt_float(item["axis_length_ratio_2d"], 3),
                int(item["valid"]),
                _fmt_float(item["confidence"], 3),
            ]
            for marker in item["markers"]:
                row += [
                    _fmt_float(marker["u"]),
                    _fmt_float(marker["v"]),
                    _fmt_float(marker["depth_mm"], 2),
                    marker["depth_pixels"],
                    _fmt_float(marker["x"], 2),
                    _fmt_float(marker["y"], 2),
                    _fmt_float(marker["z"], 2),
                    int(marker["valid"]),
                ]
            if item["valid"]:
                tip = item["tip"]
                tail = item["tail"]
                axis = item["axis"]
                row += [
                    _fmt_float(tip[0]),
                    _fmt_float(tip[1]),
                    _fmt_float(tip[2]),
                    _fmt_float(tail[0]),
                    _fmt_float(tail[1]),
                    _fmt_float(tail[2]),
                    _fmt_float(axis[0], 6),
                    _fmt_float(axis[1], 6),
                    _fmt_float(axis[2], 6),
                ]
            else:
                row += ["", "", "", "", "", "", "", "", ""]
            writer.writerow(row)


def estimate_ir_depth_pose(
    *,
    bag_dir: Path,
    marker_csv: Path,
    output_csv: Path,
    axis_start_marker: int,
    axis_end_marker: int,
    tip_offset_mm: float,
    needle_length_mm: float,
    depth_half_window: int,
    min_depth_pixels: int,
    min_axis_length_ratio: float,
    z_min_mm: float,
    z_max_mm: float,
) -> Dict[str, Any]:
    marker_rows = _load_marker_rows(marker_csv)
    depth_frames = load_image_frames(bag_dir, TOPIC_DEPTH)
    if not depth_frames:
        raise RuntimeError(f"bag 中没有 depth 图像: {bag_dir}")
    depth_timestamps = [frame.timestamp_ns for frame in depth_frames]
    depth_info = camera_info_to_dict(read_message_at_index(bag_dir, TOPIC_DEPTH_INFO, 0))
    ir_info = camera_info_to_dict(read_message_at_index(bag_dir, TOPIC_IR_INFO, 0))
    if (depth_info["width"], depth_info["height"]) != (ir_info["width"], ir_info["height"]):
        raise RuntimeError(
            "当前脚本只支持 IR 和 depth 分辨率一致的 bag；"
            f"IR={ir_info['width']}x{ir_info['height']}, depth={depth_info['width']}x{depth_info['height']}"
        )

    results: List[Dict[str, Any]] = []
    for row in marker_rows:
        timestamp_ns = int(row["timestamp_ns"])
        depth_index, depth_frame = _nearest_frame(depth_frames, depth_timestamps, timestamp_ns)
        markers = _marker_3d_from_depth(
            row,
            depth_frame,
            depth_info,
            half_win=depth_half_window,
            min_depth_pixels=min_depth_pixels,
            z_min_mm=z_min_mm,
            z_max_mm=z_max_mm,
        )
        axis_ratio_2d = _axis_length_ratio_2d(markers)
        pose = _estimate_axis_and_tip(
            markers,
            axis_start_marker=axis_start_marker,
            axis_end_marker=axis_end_marker,
            tip_offset_mm=tip_offset_mm,
            needle_length_mm=needle_length_mm,
        )
        ratio_valid = axis_ratio_2d is not None and axis_ratio_2d >= min_axis_length_ratio
        valid_marker_count = sum(1 for marker in markers if marker["valid"])
        item: Dict[str, Any] = {
            "frame_id": int(row["frame_id"]),
            "ir_index": int(row["ir_index"]),
            "timestamp_ns": timestamp_ns,
            "depth_index": depth_index,
            "depth_dt_ms": (depth_frame.timestamp_ns - timestamp_ns) / 1e6,
            "axis_length_ratio_2d": axis_ratio_2d,
            "markers": markers,
            "valid": bool(pose["valid"]) and ratio_valid,
            "confidence": valid_marker_count / 4.0,
            "axis": pose["axis"],
            "tip": pose["tip"],
            "tail": pose["tail"],
        }
        results.append(item)

    write_pose_csv(output_csv, results)
    valid_results = [item for item in results if item["valid"]]
    marker_valid_counts = [
        sum(1 for item in results if item["markers"][i]["valid"])
        for i in range(4)
    ]
    axis_ratio_rejects = sum(
        1
        for item in results
        if item["axis_length_ratio_2d"] is not None and item["axis_length_ratio_2d"] < min_axis_length_ratio
    )
    tip_jitter_mm_mean = None
    tip_motion_extent_mm = None
    if valid_results:
        tips = np.asarray([item["tip"] for item in valid_results], dtype=np.float64)
        tip_jitter_mm_mean = float(np.mean(np.linalg.norm(tips - tips.mean(axis=0), axis=1)))
        tip_motion_extent_mm = float(np.linalg.norm(tips.max(axis=0) - tips.min(axis=0)))

    summary = {
        "bag": str(Path(bag_dir).resolve()),
        "marker_csv": str(Path(marker_csv).resolve()),
        "output_csv": str(Path(output_csv).resolve()),
        "frames": len(results),
        "valid_pose_frames": len(valid_results),
        "valid_ratio": len(valid_results) / max(len(results), 1),
        "marker_valid_counts": marker_valid_counts,
        "axis_start_marker": axis_start_marker,
        "axis_end_marker": axis_end_marker,
        "tip_offset_mm": tip_offset_mm,
        "needle_length_mm": needle_length_mm,
        "depth_half_window": depth_half_window,
        "min_depth_pixels": min_depth_pixels,
        "min_axis_length_ratio": min_axis_length_ratio,
        "axis_ratio_rejects": axis_ratio_rejects,
        "tip_jitter_mm_mean": tip_jitter_mm_mean,
        "tip_motion_extent_mm": tip_motion_extent_mm,
        "note": "tip_offset_mm 已按当前手工测量值设置；仍需更精确测量球心到针尖距离和 marker 局部几何后才能用于精度结论。",
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _print_chinese_summary(summary: Dict[str, Any]) -> None:
    print("IR + depth 三维针姿态估计完成")
    print(f"输入 bag：{summary['bag']}")
    print(f"marker CSV：{summary['marker_csv']}")
    print(f"输出 CSV：{summary['output_csv']}")
    print(f"总帧数：{summary['frames']}")
    print(f"有效 3D 姿态帧：{summary['valid_pose_frames']}")
    print(f"有效率：{summary['valid_ratio'] * 100:.1f}%")
    print(f"每个 marker 的有效深度帧数：{summary['marker_valid_counts']}")
    print(f"针轴近似：m{summary['axis_start_marker']} -> m{summary['axis_end_marker']}")
    print(f"临时针尖外推距离：{summary['tip_offset_mm']:.1f} mm")
    print(f"m1 几何门控阈值：m2-m1 / m0-m3 >= {summary['min_axis_length_ratio']:.2f}")
    print(f"被几何门控剔除的帧数：{summary['axis_ratio_rejects']}")
    if summary["tip_jitter_mm_mean"] is not None:
        print(f"针尖平均抖动/离散量：{summary['tip_jitter_mm_mean']:.2f} mm")
    if summary["tip_motion_extent_mm"] is not None:
        print(f"针尖运动包围盒对角线：{summary['tip_motion_extent_mm']:.2f} mm")
    print("注意：当前针尖位置已使用手工测量距离，但仍不是最终精度结论；下一步需要实测 marker 球心局部几何。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 IR marker CSV 和 depth 图像估计 4 个 marker 的三维坐标、针轴和临时针尖。"
    )
    parser.add_argument("--bag", type=Path, required=True, help="包含 IR 和 depth 话题的 rosbag 目录")
    parser.add_argument("--markers", type=Path, required=True, help="detect_ir_markers.py 输出的 marker CSV")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 pose CSV 路径")
    parser.add_argument("--axis-start-marker", type=int, default=3, help="针轴起点 marker，默认 m3")
    parser.add_argument("--axis-end-marker", type=int, default=1, help="针轴朝针尖一侧的 marker，默认 m1")
    parser.add_argument("--tip-offset-mm", type=float, default=141.0, help="从 m1 球心沿针轴外推到针尖的距离 (mm)")
    parser.add_argument("--needle-length-mm", type=float, default=162.0, help="用于显示的针长")
    parser.add_argument("--depth-half-window", type=int, default=13, help="取深度中位数的半窗口像素")
    parser.add_argument("--min-depth-pixels", type=int, default=3, help="窗口内至少需要的有效深度像素")
    parser.add_argument(
        "--min-axis-length-ratio",
        type=float,
        default=0.55,
        help="m2-m1 与 m0-m3 的最小 2D 长度比例，用于过滤 m1 被下方关节点误识别的帧",
    )
    parser.add_argument("--z-min-mm", type=float, default=100.0, help="有效深度下限")
    parser.add_argument("--z-max-mm", type=float, default=2000.0, help="有效深度上限")
    args = parser.parse_args()

    out = args.output or (
        _REPO_ROOT
        / "data"
        / "jetarm_marker"
        / "ir_depth_pose"
        / f"{args.bag.name}_ir_depth_pose.csv"
    )

    try:
        summary = estimate_ir_depth_pose(
            bag_dir=args.bag,
            marker_csv=args.markers,
            output_csv=out,
            axis_start_marker=args.axis_start_marker,
            axis_end_marker=args.axis_end_marker,
            tip_offset_mm=args.tip_offset_mm,
            needle_length_mm=args.needle_length_mm,
            depth_half_window=args.depth_half_window,
            min_depth_pixels=args.min_depth_pixels,
            min_axis_length_ratio=args.min_axis_length_ratio,
            z_min_mm=args.z_min_mm,
            z_max_mm=args.z_max_mm,
        )
    except Exception as exc:
        print(f"IR + depth 三维估计失败：{exc}", file=sys.stderr)
        return 1

    _print_chinese_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
