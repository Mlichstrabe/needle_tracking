#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3：LK 光流跟踪 4 marker，输出 2D 轨迹 CSV + 可选标注视频。

用法:
  python tools/jetarm_marker/track_markers.py \\
      --bag data/jetarm_marker/bags/marker_static_clean_01 \\
      --init data/jetarm_marker/inits/marker_static_clean_01_init.json \\
      --start 120 --end 380
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.geometry_io import load_json  # noqa: E402
from tools.jetarm_marker.rosbag_io import TOPIC_RGB, load_image_frames  # noqa: E402

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _initial_distances(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(float(np.linalg.norm(pts[i] - pts[j])))
    return np.asarray(d, dtype=np.float64)


def _rigid_ok(curr: np.ndarray, init: np.ndarray, dist0: np.ndarray, tol_ratio: float) -> bool:
    d1 = _initial_distances(curr)
    if len(d1) != len(dist0):
        return False
    rel = np.abs(d1 - dist0) / np.maximum(dist0, 1.0)
    return bool(np.max(rel) <= tol_ratio)


def track_sequence(
    frames_gray: List[np.ndarray],
    init_points: np.ndarray,
    *,
    jump_px: float = 40.0,
    rigid_tol: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      points: (T, 4, 2)
      valid: (T, 4) bool
      frame_valid: (T,) bool
    """
    t_count = len(frames_gray)
    n_markers = init_points.shape[0]
    out = np.zeros((t_count, n_markers, 2), dtype=np.float64)
    valid = np.zeros((t_count, n_markers), dtype=bool)
    frame_valid = np.zeros(t_count, dtype=bool)

    prev_gray = frames_gray[0]
    curr_pts = init_points.astype(np.float32).reshape(-1, 1, 2)
    out[0] = init_points
    valid[0] = True
    frame_valid[0] = True
    dist0 = _initial_distances(init_points)

    for t in range(1, t_count):
        next_gray = frames_gray[t]
        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, curr_pts, None, **LK_PARAMS)
        if nxt is None:
            out[t] = out[t - 1]
            valid[t] = False
            frame_valid[t] = False
            prev_gray = next_gray
            continue

        pts = nxt.reshape(-1, 2)
        status = st.reshape(-1).astype(bool)
        prev_pts = out[t - 1]

        for i in range(n_markers):
            if not status[i]:
                pts[i] = prev_pts[i]
                status[i] = False
            elif float(np.linalg.norm(pts[i] - prev_pts[i])) > jump_px:
                pts[i] = prev_pts[i]
                status[i] = False

        ok_markers = int(status.sum())
        rigid = _rigid_ok(pts[status], init_points[status], dist0, rigid_tol) if ok_markers >= 3 else False
        fv = ok_markers >= 3 and rigid

        out[t] = pts
        valid[t] = status
        frame_valid[t] = fv
        curr_pts = pts.astype(np.float32).reshape(-1, 1, 2)
        prev_gray = next_gray

    return out, valid, frame_valid


def write_tracking_csv(
    path: Path,
    *,
    frame_indices: List[int],
    timestamps_ns: List[int],
    points: np.ndarray,
    valid: np.ndarray,
    frame_valid: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame_id", "rgb_index", "timestamp_ns", "frame_valid"]
    for i in range(4):
        header += [f"m{i}_u", f"m{i}_v", f"m{i}_valid"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fid, (idx, ts, fv) in enumerate(zip(frame_indices, timestamps_ns, frame_valid)):
            row = [fid, idx, ts, int(fv)]
            for i in range(4):
                row += [
                    f"{points[fid, i, 0]:.3f}",
                    f"{points[fid, i, 1]:.3f}",
                    int(valid[fid, i]),
                ]
            w.writerow(row)


def write_preview_video(path: Path, frames_bgr: List[np.ndarray], points: np.ndarray, valid: np.ndarray) -> None:
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (w, h),
    )
    for t, img in enumerate(frames_bgr):
        vis = img.copy()
        for i in range(4):
            u, v = points[t, i]
            color = (0, 255, 0) if valid[t, i] else (0, 0, 255)
            cv2.circle(vis, (int(u), int(v)), 6, color, 2)
            cv2.putText(vis, f"m{i}", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        writer.write(vis)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="LK 光流跟踪 4 marker")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="含首含尾；-1 表示到最后一帧")
    parser.add_argument("--jump-px", type=float, default=40.0, help="单帧 marker 最大允许跳变像素")
    parser.add_argument("--rigid-tol", type=float, default=0.15, help="2D marker 间距相对变化容忍度")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()

    init_data = load_json(args.init)
    init_pts = np.asarray(
        [[m["u"], m["v"]] for m in sorted(init_data["markers"], key=lambda x: x["id"])],
        dtype=np.float64,
    )
    if init_pts.shape != (4, 2):
        print("init JSON 需要 4 个 marker", file=sys.stderr)
        return 1

    start = max(0, args.start)
    end = args.end
    records = load_image_frames(args.bag, TOPIC_RGB, start_index=start, end_index=end if end >= 0 else None)
    if len(records) < 2:
        print("帧数不足", file=sys.stderr)
        return 1

    grays = []
    rgbs = []
    indices = []
    timestamps = []
    for i, rec in enumerate(records):
        indices.append(start + i)
        timestamps.append(rec.timestamp_ns)
        img = rec.array
        if img.ndim == 3:
            rgbs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        else:
            g = img.astype(np.uint8)
            grays.append(g)
            rgbs.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))

    points, valid, frame_valid = track_sequence(
        grays,
        init_pts,
        jump_px=args.jump_px,
        rigid_tol=args.rigid_tol,
    )

    bag_name = init_data.get("bag", args.bag.name)
    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "tracking" / f"{bag_name}_track2d.csv"
    )
    write_tracking_csv(
        out,
        frame_indices=indices,
        timestamps_ns=timestamps,
        points=points,
        valid=valid,
        frame_valid=frame_valid,
    )

    meta = {
        "bag": str(args.bag.resolve()),
        "init": str(args.init.resolve()),
        "start_index": start,
        "end_index": indices[-1],
        "frames": len(indices),
        "valid_frames": int(frame_valid.sum()),
        "output_csv": str(out.resolve()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.video:
        write_preview_video(args.video, rgbs, points, valid)
    else:
        vid = out.parent / f"{bag_name}_track_preview.mp4"
        write_preview_video(vid, rgbs, points, valid)
        print(f"预览视频: {vid}")

    print(f"跟踪 CSV: {out}")
    print(f"有效帧: {int(frame_valid.sum())}/{len(frame_valid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
