#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时 IR + depth → 针尖/针轴 → needle_tracking 3D 视图（相机系 mm，先不看 CT 配准）。

  ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
  python tools/jetarm_marker/live_needle_gl.py --host 192.168.55.1

  离线 bag（需含 IR+depth）:
  python tools/jetarm_marker/live_needle_gl.py --source bag data/jetarm_marker/bags/marker_move_rgb_ir_depth_01
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph.opengl as gl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtGui import QVector3D  # noqa: E402
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget  # noqa: E402

from tools.jetarm_marker.ir_depth_stream_protocol import (  # noqa: E402
    IrDepthFrame,
    connect_ir_depth_stream,
    iter_ir_depth_frames,
)
from tools.jetarm_marker.ir_marker_detect import DetectParams, MarkerTracker  # noqa: E402
from tools.jetarm_marker.live_pose_estimate import estimate_live_needle_pose, load_depth_camera_info  # noqa: E402
from tools.jetarm_marker.rosbag_io import TOPIC_DEPTH, TOPIC_IR, load_image_frames  # noqa: E402
from tools.jetarm_marker.ir_marker_detect import ir_array_to_u8  # noqa: E402
from ui.widgets.gl_widget import GLVisualizationWidget  # noqa: E402


class LiveNeedleWindow(QMainWindow):
    def __init__(
        self,
        *,
        frame_queue: "queue.Queue[IrDepthFrame]",
        depth_info: dict,
        params: DetectParams,
        tip_offset_mm: float,
        needle_length_mm: float,
        bag_mode: bool,
    ):
        super().__init__()
        self.setWindowTitle("JetArm 实时针体（相机系）")
        self._queue = frame_queue
        self._depth_info = depth_info
        self._tracker = MarkerTracker(params=params)
        self._tip_offset_mm = tip_offset_mm
        self._needle_length_mm = needle_length_mm
        self._bag_mode = bag_mode
        self._bag_frames: List[Tuple[np.ndarray, np.ndarray]] = []
        self._bag_index = 0
        self._tips_history: List[np.ndarray] = []
        self._camera_fitted = False
        self._last_pose_valid = False
        self._fps = 0.0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._gl = GLVisualizationWidget()
        self._gl.set_marker_replay_mode(True)
        self._gl.needle_length = self._needle_length_mm
        self._init_marker_overlay()
        self._gl.view.setCameraPosition(distance=400, elevation=25, azimuth=45)
        layout.addWidget(self._gl, stretch=1)

        self._status = QLabel("等待帧…")
        layout.addWidget(self._status)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _init_marker_overlay(self) -> None:
        self._marker_colors = {
            0: (0.15, 0.45, 1.00, 1.0),
            1: (1.00, 0.85, 0.10, 1.0),
            2: (0.00, 0.95, 1.00, 1.0),
            3: (1.00, 0.25, 0.95, 1.0),
        }
        self._marker_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            size=9.0,
            pxMode=False,
        )
        self._gl.view.addItem(self._marker_scatter)
        self._marker_rods = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.75, 0.95, 1.0, 0.8),
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self._gl.view.addItem(self._marker_rods)

    def load_bag_frames(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self._bag_frames = pairs
        self._bag_index = 0

    def _update_marker_overlay(self, markers: List[Tuple[int, np.ndarray]]) -> None:
        if not markers:
            self._marker_scatter.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))
            return
        positions = np.asarray([p for _i, p in markers], dtype=np.float32)
        colors = np.asarray(
            [self._marker_colors.get(i, (1.0, 1.0, 1.0, 1.0)) for i, _p in markers],
            dtype=np.float32,
        )
        self._marker_scatter.setData(pos=positions, color=colors, size=9.0, pxMode=False)
        by_id = {i: p for i, p in markers}
        segments = []
        for a, b in ((0, 3), (2, 1)):
            if a in by_id and b in by_id:
                segments.extend([by_id[a], by_id[b]])
        if segments:
            self._marker_rods.setData(pos=np.asarray(segments, dtype=np.float32))
        else:
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))

    def _fit_camera_once(self, tip: np.ndarray, markers: List[Tuple[int, np.ndarray]]) -> None:
        if self._camera_fitted:
            return
        points = [tip]
        points.extend(p for _i, p in markers)
        all_points = np.asarray(points, dtype=np.float64)
        center = all_points.mean(axis=0)
        extent = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
        distance = float(np.clip(max(extent * 3.5, self._needle_length_mm * 2.5, 220.0), 220.0, 800.0))
        self._gl.view.opts["center"] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self._gl.view.setCameraPosition(distance=distance, elevation=22, azimuth=45)
        self._camera_fitted = True

    def _poll_frame(self) -> Optional[IrDepthFrame]:
        if self._bag_mode:
            if not self._bag_frames:
                return None
            gray, depth = self._bag_frames[self._bag_index % len(self._bag_frames)]
            self._bag_index += 1
            return IrDepthFrame(gray=gray, depth=depth)
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _tick(self) -> None:
        t0 = time.perf_counter()
        frame = self._poll_frame()
        if frame is None:
            return

        detect = self._tracker.process(frame.gray, enforce_match_gate=False)
        if frame.depth is None:
            from tools.jetarm_marker.live_pose_estimate import LiveNeedlePose

            pose = LiveNeedlePose(
                valid=False,
                tip=None,
                axis=None,
                tail=None,
                confidence=0.0,
                markers=[],
                axis_length_ratio_2d=detect.axis_length_ratio_2d,
                rom_rms_mm=detect.rom_rms_mm,
            )
        else:
            pose = estimate_live_needle_pose(
                detect,
                frame.depth,
                self._depth_info,
                tip_offset_mm=self._tip_offset_mm,
                needle_length_mm=self._needle_length_mm,
            )

        if pose.valid and pose.tip is not None and pose.axis is not None:
            self._gl.set_marker_needle_pose(pose.tip, pose.axis, confidence=pose.confidence)
            self._update_marker_overlay(pose.markers)
            self._fit_camera_once(pose.tip, pose.markers)
            self._tips_history.append(pose.tip.copy())
            if len(self._tips_history) > 200:
                self._tips_history.pop(0)
            self._last_pose_valid = True
            tip_text = f"tip=({pose.tip[0]:.0f},{pose.tip[1]:.0f},{pose.tip[2]:.0f}) mm"
        else:
            self._last_pose_valid = False
            tip_text = "tip=— (pose invalid)"

        dt = time.perf_counter() - t0
        if dt > 1e-6:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        rom = "n/a" if detect.rom_rms_mm is None else f"{detect.rom_rms_mm:.1f}"
        ratio = "n/a" if detect.axis_length_ratio_2d is None else f"{detect.axis_length_ratio_2d:.2f}"
        depth_ok = "yes" if frame.depth is not None else "no"
        self._status.setText(
            f"fps={self._fps:.1f}  candidates={detect.candidate_count}  depth={depth_ok}  "
            f"rom_rms={rom}  ratio={ratio}  pose={'OK' if pose.valid else 'FAIL'}  {tip_text}"
        )


def _stream_worker(host: str, port: int, out_q: "queue.Queue[IrDepthFrame]") -> None:
    while True:
        try:
            sock = connect_ir_depth_stream(host, port)
            for frame in iter_ir_depth_frames(sock):
                try:
                    while not out_q.empty():
                        out_q.get_nowait()
                except queue.Empty:
                    pass
                out_q.put(frame)
        except Exception as exc:
            print(f"[stream] 断开/错误: {exc}，3s 后重连…", file=sys.stderr)
            time.sleep(3.0)


def _load_bag_ir_depth_pairs(bag_dir: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    ir_frames = load_image_frames(bag_dir, TOPIC_IR)
    depth_frames = load_image_frames(bag_dir, TOPIC_DEPTH)
    if not ir_frames or not depth_frames:
        raise RuntimeError("bag 需要同时含 IR 与 depth")
    depth_ts = [f.timestamp_ns for f in depth_frames]
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for ir in ir_frames:
        best = min(range(len(depth_frames)), key=lambda i: abs(depth_ts[i] - ir.timestamp_ns))
        pairs.append((ir_array_to_u8(ir.array), depth_frames[best].array.copy()))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="实时 IR+depth 针体 3D 显示")
    parser.add_argument("--source", choices=("tcp", "bag"), default="tcp")
    parser.add_argument("--host", default="192.168.55.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("bag_dir", nargs="?", type=Path)
    parser.add_argument("--tip-offset-mm", type=float, default=140.0)
    parser.add_argument("--needle-length-mm", type=float, default=162.0)
    parser.add_argument("--max-rom-rms-mm", type=float, default=22.0)
    parser.add_argument("--threshold-percentile", type=float, default=98.5)
    args = parser.parse_args()

    params = DetectParams(
        threshold_percentile=args.threshold_percentile,
        max_rom_rms_mm=args.max_rom_rms_mm,
    )
    depth_info = load_depth_camera_info()

    frame_q: queue.Queue[IrDepthFrame] = queue.Queue(maxsize=2)
    bag_mode = args.source == "bag"

    app = QApplication(sys.argv)
    win = LiveNeedleWindow(
        frame_queue=frame_q,
        depth_info=depth_info,
        params=params,
        tip_offset_mm=args.tip_offset_mm,
        needle_length_mm=args.needle_length_mm,
        bag_mode=bag_mode,
    )

    if bag_mode:
        if args.bag_dir is None:
            print("bag 模式需要 bag 目录", file=sys.stderr)
            return 2
        win.load_bag_frames(_load_bag_ir_depth_pairs(args.bag_dir))
        win.setWindowTitle(f"JetArm 针体回放（bag） {args.bag_dir.name}")
    else:
        threading.Thread(target=_stream_worker, args=(args.host, args.port, frame_q), daemon=True).start()

    win.resize(1100, 820)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
