#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IR + depth + 3D 三窗对照（单 TCP）：左 IR 检测 | 中 depth 读数 | 右 3D 针体。

推荐（位移=IR+depth，姿态=IMU）:
  python tools/jetarm_marker/live_triple_view.py --host 192.168.55.1 --pose-mode hybrid --imu-port COM3

纯 marker 针轴（旧）:
  python tools/jetarm_marker/live_triple_view.py --pose-mode marker ...

  ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
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
from tools.jetarm_marker.anchor_geometry import load_anchor_geometry  # noqa: E402
from tools.jetarm_marker.hybrid_pose import (  # noqa: E402
    HybridPoseFusion,
    apply_imu_kinematics_from_repo,
    load_camera_scene_extrinsic,
    m1_position_cam,
)
from tools.jetarm_marker.imu_serial_reader import ImuSerialReader, probe_imu_port  # noqa: E402
from tools.jetarm_marker.live_pose_estimate import (  # noqa: E402
    _markers_3d_from_uv,
    anchor_marker_3d_from_uv,
    estimate_live_needle_pose,
    load_depth_camera_info,
)
from tools.jetarm_marker.rosbag_io import TOPIC_DEPTH, TOPIC_IR, load_image_frames  # noqa: E402

configure_console_encoding()

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap, QVector3D  # noqa: E402
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
        axis_start_marker: int = 3,
        axis_end_marker: int = 1,
        tip_anchor_marker: int = 1,
        pose_mode: str = "hybrid",
        imu_reader: Optional[ImuSerialReader] = None,
        fusion: Optional[HybridPoseFusion] = None,
        imu_needle_length_mm: float = 200.0,
    ):
        super().__init__()
        self._pose_mode = pose_mode
        self._imu = imu_reader
        self._fusion = fusion
        self._imu_needle_len = imu_needle_length_mm
        title = "JetArm 三窗 | 位移 IR+depth | 姿态 IMU" if pose_mode == "hybrid" else "JetArm IR | Depth | 3D (marker轴)"
        self.setWindowTitle(title)
        self._queue = frame_queue
        self._bag_frames = bag_frames or []
        self._bag_index = 0
        self._depth_info = depth_info
        self._params = params
        self._tracker = MarkerTracker(
            params=params,
            anchor_marker=tip_anchor_marker,
            anchor_fallback=(pose_mode == "hybrid"),
        )
        self._win = depth_half_window
        self._min_depth_pixels = min_depth_pixels
        self._tip_offset_mm = tip_offset_mm
        self._needle_length_mm = needle_length_mm
        self._axis_start_marker = axis_start_marker
        self._axis_end_marker = axis_end_marker
        self._tip_anchor_marker = tip_anchor_marker
        self._camera_fitted = False
        self._last_hybrid_tip: Optional[np.ndarray] = None
        self._fps = 0.0
        self._encoding = depth_info.get("encoding", "16UC1")
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[Path] = None
        self._recording = False

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
        gl_len = self._imu_needle_len if pose_mode == "hybrid" else self._needle_length_mm
        self._gl.needle_length = gl_len
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
            if self._fusion is not None:
                self._fusion.reset()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._params.threshold_percentile = min(99.9, self._params.threshold_percentile + 0.5)
        if key == Qt.Key_Minus:
            self._params.threshold_percentile = max(90.0, self._params.threshold_percentile - 0.5)
        if key == Qt.Key_BracketRight:
            self._win = min(25, self._win + 1)
        if key == Qt.Key_BracketLeft:
            self._win = max(1, self._win - 1)
        if key == Qt.Key_V:
            self._toggle_record()
        super().keyPressEvent(event)

    def _toggle_record(self) -> None:
        if self._recording:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self._recording = False
            print(f"[录制] 已保存: {self._record_path}")
            self._record_path = None
            return
        out_dir = _REPO_ROOT / "data" / "jetarm_marker" / "recordings"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("demo_%Y%m%d_%H%M%S.mp4")
        self._record_path = out_dir / name
        self._recording = True
        self._video_writer = None
        print(f"[录制] 开始 → {self._record_path}（再按 v 停止）")

    def _append_record_frame(self, ir_bgr: np.ndarray, depth_bgr: np.ndarray) -> None:
        if not self._recording:
            return
        h = max(ir_bgr.shape[0], depth_bgr.shape[0])
        ir = ir_bgr
        dep = depth_bgr
        if ir.shape[0] != h:
            ir = cv2.resize(ir, (int(ir.shape[1] * h / ir.shape[0]), h))
        if dep.shape[0] != h:
            dep = cv2.resize(dep, (int(dep.shape[1] * h / dep.shape[0]), h))
        panel_w = 480
        ir_s = cv2.resize(ir, (panel_w, h))
        dep_s = cv2.resize(dep, (panel_w, h))
        gl_img = self._gl.grab().toImage()
        gl_img = gl_img.convertToFormat(QImage.Format_RGB888)
        gw, gh = gl_img.width(), gl_img.height()
        ptr = gl_img.bits()
        ptr.setsize(gl_img.byteCount())
        gl_bgr = cv2.cvtColor(
            np.frombuffer(ptr, np.uint8).reshape(gh, gw, 3),
            cv2.COLOR_RGB2BGR,
        )
        gl_s = cv2.resize(gl_bgr, (panel_w, h))
        row = np.hstack([ir_s, dep_s, gl_s])
        bar_h = 44
        bar = np.zeros((bar_h, row.shape[1], 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        txt = self._status.text()[:120]
        if self._recording:
            cv2.putText(bar, "REC", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(bar, txt, (70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        frame = np.vstack([row, bar])
        if self._video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(self._record_path),
                fourcc,
                30.0,
                (frame.shape[1], frame.shape[0]),
            )
        self._video_writer.write(frame)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        if self._imu is not None:
            self._imu.stop()
        super().closeEvent(event)

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
        nl = self._imu_needle_len if self._pose_mode == "hybrid" else self._needle_length_mm
        fit_gl_camera_to_points(
            self._gl,
            points,
            needle_length_mm=nl,
            extent_scale=3.5,
            min_distance=220.0,
            max_distance=800.0,
        )
        if self._pose_mode == "hybrid":
            c = tip.reshape(3)
            self._gl.view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
            self._gl.view.setCameraPosition(distance=420, elevation=18, azimuth=-135)
            self._gl.view.opts["distance"] = 420
        self._camera_fitted = True

    def _tick(self) -> None:
        t0 = time.perf_counter()
        frame = self._poll_frame()
        if frame is None:
            return

        detect = self._tracker.process(frame.gray, enforce_match_gate=False)
        marker_meta: list = []
        pose = None
        hybrid_tip: Optional[np.ndarray] = None
        hybrid_axis: Optional[np.ndarray] = None
        hybrid_conf = 0.0
        hybrid_valid = False

        anchor_depth_ok = False
        m1_cam, m1_ok = None, False
        if frame.depth is not None and detect.selected is not None:
            if self._pose_mode == "hybrid":
                marker_meta = _markers_3d_from_uv(
                    detect.selected,
                    frame.depth,
                    self._depth_info,
                    half_win=self._win,
                    min_depth_pixels=self._min_depth_pixels,
                    z_min_mm=50.0,
                    z_max_mm=2000.0,
                )
                aidx = (
                    self._fusion.tip_anchor_marker
                    if self._fusion is not None
                    else self._tip_anchor_marker
                )
                m1_cam, m1_ok = m1_position_cam(marker_meta, aidx)
                anchor_depth_ok = m1_ok
                if not m1_ok:
                    m1_cam, anchor_meta = anchor_marker_3d_from_uv(
                        detect.selected,
                        frame.depth,
                        self._depth_info,
                        aidx,
                        half_win=self._win,
                        min_depth_pixels=self._min_depth_pixels,
                    )
                    m1_ok = anchor_depth_ok = bool(anchor_meta.get("valid"))
                    if aidx < 4 and anchor_meta.get("valid"):
                        marker_meta[aidx] = {**anchor_meta}
            else:
                marker_meta = _markers_3d_from_uv(
                    detect.selected,
                    frame.depth,
                    self._depth_info,
                    half_win=self._win,
                    min_depth_pixels=self._min_depth_pixels,
                    z_min_mm=50.0,
                    z_max_mm=2000.0,
                )

        if self._pose_mode == "hybrid" and self._fusion is not None:
            q = self._imu.get_quaternion() if self._imu is not None else None
            hybrid_tip, hybrid_axis, hybrid_conf, hybrid_valid = self._fusion.fuse(
                m1_cam=m1_cam,
                m1_visible=m1_ok,
                axis_scene=None,
                quaternion=q,
            )
            imu_ok = self._imu is not None and q is not None
            hybrid_valid = bool(hybrid_valid and imu_ok and m1_ok)
        elif frame.depth is not None and detect.selected is not None:
            pose = estimate_live_needle_pose(
                detect,
                frame.depth,
                self._depth_info,
                axis_start_marker=self._axis_start_marker,
                axis_end_marker=self._axis_end_marker,
                tip_offset_mm=self._tip_offset_mm,
                needle_length_mm=self._needle_length_mm,
                depth_half_window=self._win,
                min_depth_pixels=self._min_depth_pixels,
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

        if self._pose_mode == "hybrid" and hybrid_axis is not None:
            if hybrid_tip is not None:
                self._last_hybrid_tip = hybrid_tip.copy()
                self._gl.set_marker_needle_pose(hybrid_tip, hybrid_axis, confidence=hybrid_conf)
                c = hybrid_tip.reshape(3)
                self._gl.view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
                if not self._camera_fitted:
                    self._fit_camera_once(hybrid_tip, [])
                    self._camera_fitted = True
            else:
                tip_hold = (
                    self._last_hybrid_tip.copy()
                    if self._last_hybrid_tip is not None
                    else np.zeros(3, dtype=float)
                )
                self._gl.set_marker_needle_pose(tip_hold, hybrid_axis, confidence=0.2)
            # 3D 窗针位在 scene；marker 仍在相机系，避免错位不画 3D marker 点
            self._overlay.update([])
        elif pose is not None and pose.tip is not None and pose.axis is not None:
            if pose.valid:
                self._gl.set_marker_needle_pose(pose.tip, pose.axis, confidence=pose.confidence)
                self._overlay.update(pose.markers)
                self._fit_camera_once(pose.tip, pose.markers)

        n_valid = sum(1 for m in marker_meta if m.get("valid"))
        rom = "n/a" if detect.rom_rms_mm is None else f"{detect.rom_rms_mm:.1f}"
        tip_text = "tip=—"
        if self._pose_mode == "hybrid" and hybrid_tip is not None:
            t = hybrid_tip
            tip_text = f"tip_scene=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
        elif pose is not None and pose.tip is not None:
            t = pose.tip
            tip_text = f"tip=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
        imu_ok = self._imu is not None and self._imu.get_quaternion() is not None
        if self._pose_mode == "hybrid":
            aidx = self._tip_anchor_marker
            self._status.setText(
                f"fps={self._fps:.1f}  HYBRID  向=IMU  移=m{aidx}+depth  "
                f"track={detect.track_mode}  depth_m{aidx}={'OK' if anchor_depth_ok else 'FAIL'}  "
                f"imu={'OK' if imu_ok else 'NO'}  fuse={'OK' if hybrid_valid else 'PART'}  "
                f"{tip_text}  |  [v]录  [r]原点  [q]退"
            )
        else:
            m3_ok = (
                len(marker_meta) > 3
                and marker_meta[3].get("valid")
                and marker_meta[1].get("valid")
            )
            self._status.setText(
                f"fps={self._fps:.1f}  axis=m{self._axis_start_marker}→m{self._axis_end_marker}  "
                f"tip+{self._tip_offset_mm:.0f}mm@m1  m3m1_3d={'OK' if m3_ok else 'FAIL'}  "
                f"gate={'PASS' if detect.geometry_valid else 'FAIL'}  rom={rom}  "
                f"3d={n_valid}/4  pose={'OK' if pose and pose.valid else 'FAIL'}  "
                f"win={self._win}  {tip_text}  |  [+/-]阈值  [/[] depth  [r]重置  [q]退出"
            )

        self._append_record_frame(ir_bgr, depth_bgr)


def main() -> int:
    parser = argparse.ArgumentParser(description="IR + depth + 3D 三窗对照")
    parser.add_argument("--source", choices=("tcp", "bag"), default="tcp")
    parser.add_argument("--host", default="192.168.55.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("bag_dir", nargs="?", type=Path)
    parser.add_argument("--depth-half-window", type=int, default=13)
    parser.add_argument("--min-depth-pixels", type=int, default=3)
    parser.add_argument("--tip-offset-mm", type=float, default=None)
    parser.add_argument("--needle-length-mm", type=float, default=None)
    parser.add_argument("--threshold-percentile", type=float, default=98.5)
    parser.add_argument(
        "--pose-mode",
        choices=("hybrid", "marker"),
        default="hybrid",
        help="hybrid=位移 IR+depth + 姿态 IMU；marker=全由 marker 定轴",
    )
    parser.add_argument(
        "--imu-port",
        default=None,
        help="IMU 串口，如 COM3；hybrid 模式建议指定",
    )
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument(
        "--no-imu-auto",
        action="store_true",
        help="未指定 --imu-port 时不自动扫描 COM 口",
    )
    args = parser.parse_args()

    anchor = load_anchor_geometry()
    tip_mm = float(args.tip_offset_mm if args.tip_offset_mm is not None else anchor["tip_offset_mm"])
    needle_mm = float(
        args.needle_length_mm if args.needle_length_mm is not None else anchor["needle_length_mm"]
    )
    axis_start = int(anchor.get("axis_start_marker", 3))
    axis_end = int(anchor.get("axis_end_marker", 1))

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

    imu_reader: Optional[ImuSerialReader] = None
    fusion: Optional[HybridPoseFusion] = None
    imu_needle_len = float(needle_mm)
    if args.pose_mode == "hybrid":
        imu_needle_len = apply_imu_kinematics_from_repo()
        R_cs, t_cs = load_camera_scene_extrinsic()
        fusion = HybridPoseFusion(
            needle_length_mm=imu_needle_len,
            tip_offset_mm=tip_mm,
            tip_anchor_marker=int(anchor.get("tip_anchor_marker", 1)),
            R_cam_to_scene=R_cs,
            t_cam_to_scene=t_cs,
        )
        if args.imu_port:
            imu_reader = ImuSerialReader(args.imu_port, args.imu_baud)
            if not imu_reader.start():
                imu_reader = None
            elif imu_reader.get_quaternion() is None:
                time.sleep(0.8)
                if imu_reader.get_quaternion() is None:
                    print("[WARN] 该口无四元数，尝试自动探测…", file=sys.stderr)
                    imu_reader.stop()
                    imu_reader = probe_imu_port()
        elif not args.no_imu_auto:
            imu_reader = probe_imu_port()
        if imu_reader is None:
            print("[WARN] IMU 未连接：3D 针轴无法随 IMU 转动", file=sys.stderr)

    app = QApplication(sys.argv)
    win = TripleViewWindow(
        frame_queue=frame_q,
        bag_frames=bag_frames,
        depth_info=depth_info,
        params=params,
        depth_half_window=args.depth_half_window,
        min_depth_pixels=args.min_depth_pixels,
        tip_offset_mm=tip_mm,
        needle_length_mm=needle_mm,
        axis_start_marker=axis_start,
        axis_end_marker=axis_end,
        tip_anchor_marker=int(anchor.get("tip_anchor_marker", 1)),
        pose_mode=args.pose_mode,
        imu_reader=imu_reader,
        fusion=fusion,
        imu_needle_length_mm=imu_needle_len,
    )
    win.resize(1600, 720)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
