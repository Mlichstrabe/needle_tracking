#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IR + depth + 3D 三窗对照（单 TCP）：左 IR 检测 | 中 depth 读数 | 右 3D 针体。

  ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
  python tools/jetarm_marker/live_triple_view.py --host 192.168.55.1
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.needle_gl_view import (  # noqa: E402
    MarkerOverlay,
    configure_console_encoding,
    fit_gl_camera_to_points,
)
from tools.jetarm_marker.live_ir_depth_compare import (  # noqa: E402
    _iter_bag_frames,
    depth_to_bgr,
    draw_depth_panel,
)
from tools.jetarm_marker.ir_marker_detect import (  # noqa: E402
    DetectParams,
    MarkerTracker,
    draw_live_overlay,
    ir_array_to_u8,
)
from tools.jetarm_marker.ir_depth_stream_protocol import (  # noqa: E402
    IrDepthFrame,
    connect_ir_depth_stream,
    iter_ir_depth_frames,
)
from tools.jetarm_marker.live_pose_estimate import (  # noqa: E402
    _markers_3d_from_uv,
    estimate_live_needle_pose,
    load_depth_camera_info,
)
from tools.jetarm_marker.rosbag_io import TOPIC_DEPTH, TOPIC_IR, load_image_frames  # noqa: E402

configure_console_encoding()

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QImage, QKeyEvent, QPixmap  # noqa: E402
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget  # noqa: E402

from ui.widgets.gl_widget import GLVisualizationWidget  # noqa: E402


def _bgr_to_pixmap(bgr: np.ndarray, max_w: int = 640) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if w > max_w:
        pix = pix.scaledToWidth(max_w, Qt.SmoothTransformation)
    return pix


def _stream_worker(host: str, port: int, out_q: queue.Queue) -> None:
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
            print(f"[stream] {exc}，3s 后重连…", file=sys.stderr)
            time.sleep(3.0)


def _load_bag_pairs(bag_dir: Path) -> List[IrDepthFrame]:
    return list(_iter_bag_frames(bag_dir))


class TripleViewWindow(QMainWindow):
    def __init__(
        self,
        *,
        frame_queue: Optional[queue.Queue],
        bag_frames: Optional[List[IrDepthFrame]],
        depth_info: dict,
        params: DetectParams,
        depth_half_window: int,
        min_depth_pixels: int,
        tip_offset_mm: float,
        needle_length_mm: float,
    ):
        super().__init__()
        self.setWindowTitle("JetArm IR | Depth | 3D 对照")
        self._queue = frame_queue
        self._bag_frames = bag_frames or []
        self._bag_index = 0
        self._depth_info = depth_info
        self._params = params
        self._tracker = MarkerTracker(params=params)
        self._win = depth_half_window
        self._min_depth_pixels = min_depth_pixels
        self._tip_offset_mm = tip_offset_mm
        self._needle_length_mm = needle_length_mm
        self._camera_fitted = False
        self._fps = 0.0
        self._encoding = depth_info.get("encoding", "16UC1")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        row = QHBoxLayout()
        self._ir_label = QLabel("IR")
        self._ir_label.setAlignment(Qt.AlignCenter)
        self._ir_label.setMinimumWidth(320)
        self._depth_label = QLabel("Depth")
        self._depth_label.setAlignment(Qt.AlignCenter)
        self._depth_label.setMinimumWidth(320)

        self._gl = GLVisualizationWidget()
        self._gl.set_marker_replay_mode(True)
        self._gl.needle_length = self._needle_length_mm
        self._overlay = MarkerOverlay(self._gl)
        self._gl.view.setCameraPosition(distance=400, elevation=25, azimuth=45)

        row.addWidget(self._ir_label, stretch=1)
        row.addWidget(self._depth_label, stretch=1)
        row.addWidget(self._gl, stretch=2)
        root.addLayout(row, stretch=1)

        self._status = QLabel("等待帧…")
        root.addWidget(self._status)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
            return
        if key == Qt.Key_R:
            self._tracker.reset()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._params.threshold_percentile = min(99.9, self._params.threshold_percentile + 0.5)
        if key == Qt.Key_Minus:
            self._params.threshold_percentile = max(90.0, self._params.threshold_percentile - 0.5)
        if key == Qt.Key_BracketRight:
            self._win = min(25, self._win + 1)
        if key == Qt.Key_BracketLeft:
            self._win = max(1, self._win - 1)
        super().keyPressEvent(event)

    def _poll_frame(self) -> Optional[IrDepthFrame]:
        if self._bag_frames:
            if not self._bag_frames:
                return None
            f = self._bag_frames[self._bag_index % len(self._bag_frames)]
            self._bag_index += 1
            return f
        if self._queue is None:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _fit_camera_once(self, tip: np.ndarray, markers: List[Tuple[int, np.ndarray]]) -> None:
        if self._camera_fitted:
            return
        points = [tip] + [p for _i, p in markers]
        fit_gl_camera_to_points(
            self._gl,
            points,
            needle_length_mm=self._needle_length_mm,
            extent_scale=3.5,
            min_distance=220.0,
            max_distance=800.0,
        )
        self._camera_fitted = True

    def _tick(self) -> None:
        t0 = time.perf_counter()
        frame = self._poll_frame()
        if frame is None:
            return

        detect = self._tracker.process(frame.gray, enforce_match_gate=False)
        marker_meta: list = []
        pose = None
        if frame.depth is not None and detect.selected is not None:
            marker_meta = _markers_3d_from_uv(
                detect.selected,
                frame.depth,
                self._depth_info,
                half_win=self._win,
                min_depth_pixels=self._min_depth_pixels,
                z_min_mm=50.0,
                z_max_mm=2000.0,
            )
            pose = estimate_live_needle_pose(
                detect,
                frame.depth,
                self._depth_info,
                depth_half_window=self._win,
                min_depth_pixels=self._min_depth_pixels,
                tip_offset_mm=self._tip_offset_mm,
                needle_length_mm=self._needle_length_mm,
            )

        dt = time.perf_counter() - t0
        if dt > 1e-6:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        ir_bgr = draw_live_overlay(detect, fps=self._fps)
        cv2.putText(ir_bgr, "IR", (8, ir_bgr.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self._ir_label.setPixmap(_bgr_to_pixmap(ir_bgr))

        if frame.depth is None:
            depth_bgr = np.zeros((frame.gray.shape[0], frame.gray.shape[1], 3), dtype=np.uint8)
            cv2.putText(depth_bgr, "NO DEPTH", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            depth_bgr = draw_depth_panel(
                depth_to_bgr(frame.depth, encoding=self._encoding),
                detect,
                marker_meta,
            )
        self._depth_label.setPixmap(_bgr_to_pixmap(depth_bgr))

        if pose is not None and pose.valid and pose.tip is not None and pose.axis is not None:
            self._gl.set_marker_needle_pose(pose.tip, pose.axis, confidence=pose.confidence)
            self._overlay.update(pose.markers)
            self._fit_camera_once(pose.tip, pose.markers)

        n_valid = sum(1 for m in marker_meta if m.get("valid"))
        rom = "n/a" if detect.rom_rms_mm is None else f"{detect.rom_rms_mm:.1f}"
        tip_text = "tip=—"
        if pose is not None and pose.tip is not None:
            t = pose.tip
            tip_text = f"tip=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
        self._status.setText(
            f"fps={self._fps:.1f}  candidates={detect.candidate_count}  "
            f"gate={'PASS' if detect.geometry_valid else 'FAIL'}  rom={rom}  "
            f"3d={n_valid}/4  pose={'OK' if pose and pose.valid else 'FAIL'}  "
            f"win={self._win}  {tip_text}  |  [+/-]阈值  [/[] depth  [r]重置  [q]退出"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="IR + depth + 3D 三窗对照")
    parser.add_argument("--source", choices=("tcp", "bag"), default="tcp")
    parser.add_argument("--host", default="192.168.55.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("bag_dir", nargs="?", type=Path)
    parser.add_argument("--depth-half-window", type=int, default=13)
    parser.add_argument("--min-depth-pixels", type=int, default=3)
    parser.add_argument("--tip-offset-mm", type=float, default=140.0)
    parser.add_argument("--needle-length-mm", type=float, default=162.0)
    parser.add_argument("--threshold-percentile", type=float, default=98.5)
    args = parser.parse_args()

    params = DetectParams(threshold_percentile=args.threshold_percentile)
    depth_info = load_depth_camera_info()

    frame_q: Optional[queue.Queue] = None
    bag_frames: Optional[List[IrDepthFrame]] = None

    if args.source == "bag":
        if args.bag_dir is None:
            print("bag 模式需要 bag 目录", file=sys.stderr)
            return 2
        bag_frames = _load_bag_pairs(args.bag_dir)
        if not bag_frames:
            print("bag 无帧", file=sys.stderr)
            return 1
    else:
        frame_q = queue.Queue(maxsize=2)
        threading.Thread(target=_stream_worker, args=(args.host, args.port, frame_q), daemon=True).start()

    app = QApplication(sys.argv)
    win = TripleViewWindow(
        frame_queue=frame_q,
        bag_frames=bag_frames,
        depth_info=depth_info,
        params=params,
        depth_half_window=args.depth_half_window,
        min_depth_pixels=args.min_depth_pixels,
        tip_offset_mm=args.tip_offset_mm,
        needle_length_mm=args.needle_length_mm,
    )
    win.resize(1600, 720)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
