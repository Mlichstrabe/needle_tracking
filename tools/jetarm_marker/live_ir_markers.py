#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时 IR marker 2D 预览（Windows 开发机）。

  # 实时（JetArm 需先跑 ir_stream_server.py + depth_cam）
  python tools/jetarm_marker/live_ir_markers.py --host 192.168.55.1

  # 离线 bag 调参
  python tools/jetarm_marker/live_ir_markers.py --source bag data/jetarm_marker/bags/marker_move_rgb_ir_depth_01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.ir_marker_detect import (  # noqa: E402
  DetectParams,
  MarkerTracker,
  draw_live_overlay,
  ir_array_to_u8,
)
from tools.jetarm_marker.ir_depth_stream_protocol import connect_ir_depth_stream, iter_ir_depth_frames  # noqa: E402
from tools.jetarm_marker.rosbag_io import TOPIC_IR, load_image_frames  # noqa: E402


def _iter_bag_gray(bag_dir: Path) -> Iterator[np.ndarray]:
  for frame in load_image_frames(bag_dir, TOPIC_IR):
    yield ir_array_to_u8(frame.array)


def _iter_tcp_gray(host: str, port: int) -> Iterator[np.ndarray]:
    sock = connect_ir_depth_stream(host, port)
    try:
        for frame in iter_ir_depth_frames(sock):
            yield frame.gray
    finally:
        sock.close()


def run_preview(
  frame_iter: Iterator[np.ndarray],
  *,
  params: DetectParams,
  enforce_match_gate: bool,
  window_name: str,
) -> int:
  tracker = MarkerTracker(params=params)
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  fps = 0.0
  last_t = time.perf_counter()

  for gray in frame_iter:
    now = time.perf_counter()
    dt = now - last_t
    last_t = now
    if dt > 1e-6:
      fps = 0.9 * fps + 0.1 * (1.0 / dt)

    result = tracker.process(gray, enforce_match_gate=enforce_match_gate)
    vis = draw_live_overlay(result, fps=fps)
    cv2.imshow(window_name, vis)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
      break
    if key == ord("r"):
      tracker.reset()
    if key == ord("+") or key == ord("="):
      params.threshold_percentile = min(99.9, params.threshold_percentile + 0.5)
    if key == ord("-"):
      params.threshold_percentile = max(90.0, params.threshold_percentile - 0.5)

  cv2.destroyAllWindows()
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="实时 IR marker 2D 预览")
  parser.add_argument("--source", choices=("tcp", "bag"), default="tcp")
  parser.add_argument("--host", default="192.168.55.1", help="JetArm USB 网段地址")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("bag_dir", nargs="?", type=Path, help="--source bag 时必填")
  parser.add_argument("--threshold-percentile", type=float, default=98.5)
  parser.add_argument("--min-area", type=float, default=12.0)
  parser.add_argument("--max-area", type=float, default=1800.0)
  parser.add_argument("--min-circularity", type=float, default=0.15)
  parser.add_argument("--edge-margin", type=int, default=14)
  parser.add_argument("--max-match-px", type=float, default=70.0)
    parser.add_argument("--min-axis-ratio", type=float, default=0.55, help="m1 几何门控阈值")
    parser.add_argument("--max-rom-rms-mm", type=float, default=22.0, help="ROM 边长拟合 RMS 上限（mm）")
    parser.add_argument("--no-rom", action="store_true", help="禁用 ROM 匹配，回退 spread 启发式")
  parser.add_argument("--enforce-match-gate", action="store_true")
  args = parser.parse_args()

    params = DetectParams(
        threshold_percentile=args.threshold_percentile,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        edge_margin=args.edge_margin,
        max_match_px=args.max_match_px,
        min_axis_length_ratio_2d=args.min_axis_ratio,
        max_rom_rms_mm=args.max_rom_rms_mm,
        use_rom=not args.no_rom,
    )

  if args.source == "bag":
    if args.bag_dir is None:
      print("bag 模式需要 bag 目录参数", file=sys.stderr)
      return 2
    frame_iter = _iter_bag_gray(args.bag_dir)
    title = f"IR live preview [bag] {args.bag_dir.name}"
  else:
    print(f"连接 IR 流 {args.host}:{args.port} ...")
    try:
      frame_iter = _iter_tcp_gray(args.host, args.port)
    except OSError as exc:
      print(f"连接失败：{exc}", file=sys.stderr)
      print("请确认 JetArm 上 depth_cam 与 ir_stream_server 已启动。", file=sys.stderr)
      return 1
    title = f"IR live preview [tcp] {args.host}:{args.port}"

  try:
    return run_preview(
      frame_iter,
      params=params,
      enforce_match_gate=args.enforce_match_gate,
      window_name=title,
    )
  except KeyboardInterrupt:
    return 0
  except ConnectionError as exc:
    print(f"流中断：{exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
