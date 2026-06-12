#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 4–7：深度融合 + Kabsch 姿态 + 针尖/针轴 + scene 映射 → pose CSV。

用法:
  python tools/jetarm_marker/pose_from_markers.py \\
      --bag data/jetarm_marker/bags/marker_static_clean_01 \\
      --track data/jetarm_marker/tracking/marker_static_clean_01_track2d.csv \\
      --geometry data/jetarm_marker/geometry/needle_geometry.json \\
      --scene data/jetarm_marker/geometry/scene_transform.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.camera_math import (  # noqa: E402
    backproject_mm,
    depth_median_window,
    kabsch_rotation,
    rgb_uv_to_depth_uv,
)
from tools.jetarm_marker.geometry_io import (  # noqa: E402
    camera_to_scene,
    camera_vec_to_scene,
    load_needle_geometry,
    load_scene_transform,
)
from tools.jetarm_marker.rosbag_io import (  # noqa: E402
    TOPIC_DEPTH,
    TOPIC_DEPTH_INFO,
    TOPIC_RGB_INFO,
    camera_info_to_dict,
    load_image_frames,
    read_message_at_index,
)


def _load_track_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def estimate_pose_row(
    row: Dict[str, Any],
    depth_arr: np.ndarray,
    depth_enc: str,
    depth_info: Dict[str, float],
    rgb_info: Dict[str, float],
    geom: Dict[str, Any],
    scene_r: np.ndarray,
    scene_t: np.ndarray,
    scene_scale: float,
) -> Dict[str, Any]:
    obs = []
    local = []
    marker_meta = []
    for i in range(4):
        valid = int(row.get(f"m{i}_valid", 0)) == 1
        u = float(row[f"m{i}_u"])
        v = float(row[f"m{i}_v"])
        du, dv = rgb_uv_to_depth_uv(
            u, v,
            (rgb_info["width"], rgb_info["height"]),
            (depth_info["width"], depth_info["height"]),
        )
        z_mm, n_valid = depth_median_window(depth_arr, du, dv, encoding=depth_enc)
        x = y = None
        m_valid = False
        if z_mm is not None and n_valid >= 3:
            x, y, z = backproject_mm(du, dv, z_mm, depth_info["fx"], depth_info["fy"], depth_info["cx"], depth_info["cy"])
            if valid:
                obs.append([x, y, z])
                local.append(geom["markers_local_mm"][i])
                m_valid = True
        marker_meta.append({
            "u": u, "v": v, "depth_mm": z_mm, "x": x, "y": y, "z": z_mm if z_mm else None, "valid": m_valid and valid,
        })

    out: Dict[str, Any] = {
        "frame_valid": False,
        "confidence": 0.0,
        "markers": marker_meta,
    }
    if len(obs) < 3:
        return out

    obs_np = np.asarray(obs, dtype=np.float64)
    local_np = np.asarray(local, dtype=np.float64)
    r, t = kabsch_rotation(local_np, obs_np)
    tip_cam = r @ geom["tip_offset_marker_mm"] + t
    axis_cam = r @ geom["axis_marker_unit"]
    axis_cam = axis_cam / max(np.linalg.norm(axis_cam), 1e-12)

    tip_scene = camera_to_scene(tip_cam, scene_r, scene_t, scene_scale)
    axis_scene = camera_vec_to_scene(axis_cam, scene_r, scene_scale)

    conf = len(obs) / 4.0
    out.update({
        "frame_valid": int(row.get("frame_valid", 0)) == 1 and len(obs) >= 3,
        "confidence": conf,
        "tip_x_cam": float(tip_cam[0]),
        "tip_y_cam": float(tip_cam[1]),
        "tip_z_cam": float(tip_cam[2]),
        "axis_x_cam": float(axis_cam[0]),
        "axis_y_cam": float(axis_cam[1]),
        "axis_z_cam": float(axis_cam[2]),
        "tip_x_scene": float(tip_scene[0]),
        "tip_y_scene": float(tip_scene[1]),
        "tip_z_scene": float(tip_scene[2]),
        "axis_x_scene": float(axis_scene[0]),
        "axis_y_scene": float(axis_scene[1]),
        "axis_z_scene": float(axis_scene[2]),
    })
    return out


def write_pose_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "frame_id", "rgb_index", "timestamp_ns", "valid", "confidence",
        "m0_u", "m0_v", "m0_depth", "m0_x", "m0_y", "m0_z", "m0_valid",
        "m1_u", "m1_v", "m1_depth", "m1_x", "m1_y", "m1_z", "m1_valid",
        "m2_u", "m2_v", "m2_depth", "m2_x", "m2_y", "m2_z", "m2_valid",
        "m3_u", "m3_v", "m3_depth", "m3_x", "m3_y", "m3_z", "m3_valid",
        "tip_x_scene", "tip_y_scene", "tip_z_scene",
        "axis_x_scene", "axis_y_scene", "axis_z_scene",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for item in rows:
            base = item["track"]
            pose = item["pose"]
            row = [
                base["frame_id"],
                base["rgb_index"],
                base["timestamp_ns"],
                int(pose.get("frame_valid", False)),
                f"{pose.get('confidence', 0):.3f}",
            ]
            for i in range(4):
                m = pose["markers"][i]
                row += [
                    f"{m['u']:.3f}",
                    f"{m['v']:.3f}",
                    "" if m["depth_mm"] is None else f"{m['depth_mm']:.2f}",
                    "" if m["x"] is None else f"{m['x']:.2f}",
                    "" if m["y"] is None else f"{m['y']:.2f}",
                    "" if m["z"] is None else f"{m['z']:.2f}",
                    int(m["valid"]),
                ]
            if pose.get("frame_valid"):
                row += [
                    f"{pose['tip_x_scene']:.3f}",
                    f"{pose['tip_y_scene']:.3f}",
                    f"{pose['tip_z_scene']:.3f}",
                    f"{pose['axis_x_scene']:.4f}",
                    f"{pose['axis_y_scene']:.4f}",
                    f"{pose['axis_z_scene']:.4f}",
                ]
            else:
                row += ["", "", "", "", "", ""]
            w.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="marker 深度融合与姿态估计")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    geom = load_needle_geometry(args.geometry)
    scene_r, scene_t, scene_scale = load_scene_transform(args.scene)
    rgb_info = camera_info_to_dict(read_message_at_index(args.bag, TOPIC_RGB_INFO, 0))
    depth_info = camera_info_to_dict(read_message_at_index(args.bag, TOPIC_DEPTH_INFO, 0))

    from tools.jetarm_marker.rosbag_io import collect_timestamps, TOPIC_RGB  # noqa: WPS433

    track_rows = _load_track_rows(args.track)
    rgb_indices = [int(r["rgb_index"]) for r in track_rows]

    depth_recs = load_image_frames(args.bag, TOPIC_DEPTH)
    rgb_ts = collect_timestamps(args.bag, TOPIC_RGB)

    depth_cache: Dict[int, Tuple[np.ndarray, str]] = {}
    for idx in set(rgb_indices):
        if idx >= len(rgb_ts) or not depth_recs:
            continue
        target_ts = rgb_ts[idx]
        best = min(depth_recs, key=lambda rec: abs(rec.timestamp_ns - target_ts))
        depth_cache[idx] = (best.array, best.encoding)

    results = []
    for row in track_rows:
        idx = int(row["rgb_index"])
        depth_arr, depth_enc = depth_cache.get(idx, (np.zeros((1, 1)), "16UC1"))
        pose = estimate_pose_row(row, depth_arr, depth_enc, depth_info, rgb_info, geom, scene_r, scene_t, scene_scale)
        results.append({"track": row, "pose": pose})

    bag_name = args.bag.name
    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "tracking" / f"{bag_name}_pose.csv"
    )
    write_pose_csv(out, results)

    valid_count = sum(1 for r in results if r["pose"].get("frame_valid"))
    tips = [r["pose"] for r in results if r["pose"].get("frame_valid")]
    jitter = None
    if tips:
        arr = np.asarray([[p["tip_x_scene"], p["tip_y_scene"], p["tip_z_scene"]] for p in tips])
        jitter = float(np.mean(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))

    summary = {
        "output": str(out.resolve()),
        "frames": len(results),
        "valid_pose_frames": valid_count,
        "tip_jitter_mm_mean": jitter,
        "geometry_measured": geom["measured"],
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
