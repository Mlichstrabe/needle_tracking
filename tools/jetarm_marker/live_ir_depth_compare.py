#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IR + depth 并排对照（单 TCP 连接）：左 IR 反光点检测，右 depth 伪彩 + 各 marker 深度读数。

  ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
  python tools/jetarm_marker/live_ir_depth_compare.py --host 192.168.55.1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.ir_depth_stream_protocol import (  # noqa: E402
    IrDepthFrame,
    connect_ir_depth_stream,
    iter_ir_depth_frames,
)
from tools.jetarm_marker.ir_marker_detect import (  # noqa: E402
    MARKER_COLORS_BGR,
    DetectParams,
    MarkerTracker,
    draw_live_overlay,
    ir_array_to_u8,
)
from tools.jetarm_marker.anchor_geometry import load_anchor_geometry  # noqa: E402
from tools.jetarm_marker.live_pose_estimate import (  # noqa: E402
    _markers_3d_from_uv,
    estimate_live_needle_pose,
    load_depth_camera_info,
)
from tools.jetarm_marker.rosbag_io import TOPIC_DEPTH, TOPIC_IR, load_image_frames  # noqa: E402

MARKER_COLORS = MARKER_COLORS_BGR


def depth_to_bgr(depth: np.ndarray, *, encoding: str = "16UC1") -> np.ndarray:
    d = depth.astype(np.float32)
    if encoding == "32FC1":
        d = d * 1000.0
    valid = d[(d > 50) & (d < 2000)]
    if valid.size == 0:
        out = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
        out[:] = (40, 40, 40)
        return out
    lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
    norm = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    invalid = (d <= 0) | (d < 50) | (d > 2000)
    bgr[invalid] = (255, 0, 255)
    return bgr


def draw_depth_panel(
    depth_bgr: np.ndarray,
    detect,
    marker_meta: List[dict],
) -> np.ndarray:
    vis = depth_bgr.copy()
    if detect.selected is not None:
        for i, (u, v) in enumerate(detect.selected):
            color = MARKER_COLORS[i] if detect.geometry_valid else (0, 0, 255)
            ui, vi = int(round(u)), int(round(v))
            cv2.circle(vis, (ui, vi), 10, color, 2)
            meta = marker_meta[i] if i < len(marker_meta) else {}
            z = meta.get("depth_mm")
            npx = meta.get("depth_pixels", 0)
            ok = meta.get("valid", False)
            if z is None:
                label = f"m{i}: no depth ({npx}px)"
            else:
                label = f"m{i}: {z:.0f}mm ({npx}px)" + (" OK" if ok else " FAIL")
            cv2.putText(vis, label, (ui + 12, vi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        p0, p1, p2, p3 = detect.selected
        cv2.line(vis, tuple(p0.astype(int)), tuple(p2.astype(int)), (200, 200, 200), 1)
        cv2.line(vis, tuple(p3.astype(int)), tuple(p1.astype(int)), (0, 255, 180), 2)
    cv2.putText(vis, "DEPTH (magenta=invalid)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def _iter_tcp_frames(host: str, port: int) -> Iterator[IrDepthFrame]:
    sock = connect_ir_depth_stream(host, port)
    try:
        yield from iter_ir_depth_frames(sock)
    finally:
        sock.close()


def _iter_bag_frames(bag_dir: Path) -> Iterator[IrDepthFrame]:
    ir_frames = load_image_frames(bag_dir, TOPIC_IR)
    depth_frames = load_image_frames(bag_dir, TOPIC_DEPTH)
    if not ir_frames or not depth_frames:
        raise RuntimeError("bag 需要 IR + depth")
    depth_ts = [f.timestamp_ns for f in depth_frames]
    for ir in ir_frames:
        best = min(range(len(depth_frames)), key=lambda i: abs(depth_ts[i] - ir.timestamp_ns))
        yield IrDepthFrame(
            gray=ir_array_to_u8(ir.array),
            depth=depth_frames[best].array.copy(),
        )


def run_compare(
    frame_iter: Iterator[IrDepthFrame],
    *,
    params: DetectParams,
    depth_info: dict,
    depth_half_window: int,
    min_depth_pixels: int,
    tip_offset_mm: float,
    needle_length_mm: float,
    axis_start_marker: int,
    axis_end_marker: int,
    window_name: str,
) -> int:
    tracker = MarkerTracker(params=params)
    encoding = depth_info.get("encoding", "16UC1")
    win = depth_half_window
    fps = 0.0
    last_t = time.perf_counter()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in frame_iter:
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        detect = tracker.process(frame.gray, enforce_match_gate=False)
        marker_meta: List[dict] = []
        pose = None
        if frame.depth is not None and detect.selected is not None:
            marker_meta = _markers_3d_from_uv(
                detect.selected,
                frame.depth,
                depth_info,
                half_win=win,
                min_depth_pixels=min_depth_pixels,
                z_min_mm=50.0,
                z_max_mm=2000.0,
            )
            pose = estimate_live_needle_pose(
                detect,
                frame.depth,
                depth_info,
                axis_start_marker=axis_start_marker,
                axis_end_marker=axis_end_marker,
                tip_offset_mm=tip_offset_mm,
                needle_length_mm=needle_length_mm,
                depth_half_window=win,
                min_depth_pixels=min_depth_pixels,
            )

        ir_panel = draw_live_overlay(detect, fps=fps)
        cv2.putText(ir_panel, "IR + markers", (8, ir_panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if frame.depth is None:
            depth_panel = np.zeros_like(ir_panel)
            cv2.putText(depth_panel, "NO DEPTH", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            depth_panel = draw_depth_panel(depth_to_bgr(frame.depth, encoding=encoding), detect, marker_meta)

        if ir_panel.shape[:2] != depth_panel.shape[:2]:
            depth_panel = cv2.resize(depth_panel, (ir_panel.shape[1], ir_panel.shape[0]))

        combined = np.hstack([ir_panel, depth_panel])

        n_valid = sum(1 for m in marker_meta if m.get("valid"))
        tip_line = "tip=—"
        if pose is not None and pose.valid and pose.tip is not None:
            t = pose.tip
            tip_line = f"tip=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
        elif pose is not None and pose.tip is not None:
            t = pose.tip
            tip_line = f"tip~=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm INVALID"

        status = (
            f"depth={'yes' if frame.depth is not None else 'NO'}  "
            f"3d_markers={n_valid}/4  pose={'OK' if pose and pose.valid else 'FAIL'}  "
            f"win={win}  {tip_line}"
        )
        cv2.putText(combined, status, (8, combined.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2)

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            tracker.reset()
        if key in (ord("+"), ord("=")):
            params.threshold_percentile = min(99.9, params.threshold_percentile + 0.5)
        if key == ord("-"):
            params.threshold_percentile = max(90.0, params.threshold_percentile - 0.5)
        if key == ord("]"):
            win = min(25, win + 1)
        if key == ord("["):
            win = max(1, win - 1)

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="IR + depth 并排对照")
    parser.add_argument("--source", choices=("tcp", "bag"), default="tcp")
    parser.add_argument("--host", default="192.168.55.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("bag_dir", nargs="?", type=Path)
    parser.add_argument("--depth-half-window", type=int, default=13, help="depth 中值窗口半径（离线默认同）")
    parser.add_argument("--min-depth-pixels", type=int, default=3)
    parser.add_argument("--tip-offset-mm", type=float, default=None)
    parser.add_argument("--threshold-percentile", type=float, default=98.5)
    args = parser.parse_args()

    anchor = load_anchor_geometry()
    tip_mm = float(args.tip_offset_mm if args.tip_offset_mm is not None else anchor["tip_offset_mm"])
    needle_mm = float(anchor["needle_length_mm"])
    axis_start = int(anchor.get("axis_start_marker", 3))
    axis_end = int(anchor.get("axis_end_marker", 1))

    params = DetectParams(threshold_percentile=args.threshold_percentile)
    depth_info = load_depth_camera_info()

    if args.source == "bag":
        if args.bag_dir is None:
            print("bag 模式需要 bag 目录", file=sys.stderr)
            return 2
        frames = _iter_bag_frames(args.bag_dir)
        title = f"IR+depth compare [bag] {args.bag_dir.name}"
    else:
        print(f"连接 {args.host}:{args.port} ...")
        try:
            frames = _iter_tcp_frames(args.host, args.port)
        except OSError as exc:
            print(f"连接失败: {exc}", file=sys.stderr)
            return 1
        title = f"IR+depth compare [tcp] {args.host}:{args.port}"

    print("左=IR检测  右=depth伪彩(紫=无效)  [+/-]阈值  [/[] depth窗口  [r]重置  [q]退出")
    return run_compare(
        frames,
        params=params,
        depth_info=depth_info,
        depth_half_window=args.depth_half_window,
        min_depth_pixels=args.min_depth_pixels,
        tip_offset_mm=tip_mm,
        needle_length_mm=needle_mm,
        axis_start_marker=axis_start,
        axis_end_marker=axis_end,
        window_name=title,
    )


if __name__ == "__main__":
    raise SystemExit(main())
