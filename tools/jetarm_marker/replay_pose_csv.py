#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回放 JetArm 针姿态 CSV，在 needle_tracking 的 3D 视图里显示一根运动的针。"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph.opengl as gl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_console_encoding() -> None:
    """避免 Windows PowerShell GBK 控制台因中文或符号打印导致 GUI 启动失败。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtGui import QVector3D  # noqa: E402
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget  # noqa: E402

from ui.widgets.gl_widget import GLVisualizationWidget  # noqa: E402


@dataclass
class PoseFrame:
    """一帧可回放的针姿态。坐标单位为 mm。"""

    source_index: int
    label: str
    tip: np.ndarray
    axis: np.ndarray
    confidence: float
    markers: List[Tuple[int, np.ndarray]]


def _has_values(row: dict, keys: Sequence[str]) -> bool:
    return all(row.get(key, "") not in ("", None) for key in keys)


def _as_vec(row: dict, keys: Sequence[str]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in keys], dtype=np.float64)


def _normalize_axis(axis: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return None
    return axis / norm


def _marker_points_from_row(row: dict) -> List[Tuple[int, np.ndarray]]:
    """读取一帧里的 m0-m3 三维 marker 点。当前优先支持 IR-depth camera 坐标。"""
    markers: List[Tuple[int, np.ndarray]] = []
    for i in range(4):
        if row.get(f"m{i}_valid") != "1":
            continue
        cam_keys = [f"m{i}_x_cam_mm", f"m{i}_y_cam_mm", f"m{i}_z_cam_mm"]
        old_keys = [f"m{i}_x", f"m{i}_y", f"m{i}_z"]
        if _has_values(row, cam_keys):
            markers.append((i, _as_vec(row, cam_keys)))
        elif _has_values(row, old_keys):
            markers.append((i, _as_vec(row, old_keys)))
    return markers


def _row_to_pose(row: dict, source_index: int) -> Optional[PoseFrame]:
    if row.get("valid") != "1":
        return None

    # 新路线：IR marker + depth，直接输出 camera 坐标系下的针尖和轴线。
    if _has_values(row, ["tip_x_cam_mm", "tip_y_cam_mm", "tip_z_cam_mm", "axis_x_cam", "axis_y_cam", "axis_z_cam"]):
        tip = _as_vec(row, ["tip_x_cam_mm", "tip_y_cam_mm", "tip_z_cam_mm"])
        axis = _normalize_axis(_as_vec(row, ["axis_x_cam", "axis_y_cam", "axis_z_cam"]))
        label = f"ir_index={row.get('ir_index', '')} depth_index={row.get('depth_index', '')}"
    # 旧路线：RGB 光流 + scene transform，输出 scene 坐标系下的针尖和轴线。
    elif _has_values(row, ["tip_x_scene", "tip_y_scene", "tip_z_scene", "axis_x_scene", "axis_y_scene", "axis_z_scene"]):
        tip = _as_vec(row, ["tip_x_scene", "tip_y_scene", "tip_z_scene"])
        axis = _normalize_axis(_as_vec(row, ["axis_x_scene", "axis_y_scene", "axis_z_scene"]))
        label = f"rgb_index={row.get('rgb_index', '')}"
    else:
        return None

    if axis is None:
        return None

    try:
        confidence = float(row.get("confidence") or 1.0)
    except ValueError:
        confidence = 1.0

    return PoseFrame(
        source_index=source_index,
        label=label,
        tip=tip,
        axis=axis,
        confidence=confidence,
        markers=_marker_points_from_row(row),
    )


def load_pose_frames(path: Path) -> List[PoseFrame]:
    """读取 CSV，只保留 valid=1 且含针尖/针轴字段的帧。"""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    frames: List[PoseFrame] = []
    for i, row in enumerate(rows):
        pose = _row_to_pose(row, i)
        if pose is not None:
            frames.append(pose)
    return frames


class PoseReplayWindow(QMainWindow):
    """离线 CSV 回放窗口，复用项目现有 GLVisualizationWidget。"""

    def __init__(self, frames: List[PoseFrame], *, fps: float = 15.0, needle_length_mm: float = 162.0):
        super().__init__()
        self.setWindowTitle("JetArm IR-depth 针姿态回放")
        self._frames = frames
        self._idx = 0
        self._needle_length_mm = float(needle_length_mm)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._gl = GLVisualizationWidget()
        self._gl.set_marker_replay_mode(True)
        self._gl.needle_length = self._needle_length_mm
        self._init_marker_overlay()
        self._fit_camera_to_frames(frames)
        layout.addWidget(self._gl, stretch=1)

        bar = QHBoxLayout()
        self._status = QLabel("就绪")
        btn_prev = QPushButton("上一帧")
        btn_next = QPushButton("下一帧")
        btn_play = QPushButton("播放")
        btn_stop = QPushButton("暂停")
        bar.addWidget(self._status, stretch=1)
        bar.addWidget(btn_prev)
        bar.addWidget(btn_next)
        bar.addWidget(btn_play)
        bar.addWidget(btn_stop)
        layout.addLayout(bar)

        btn_prev.clicked.connect(self._prev)
        btn_next.clicked.connect(self._next)
        btn_play.clicked.connect(self._play)
        btn_stop.clicked.connect(self._stop)

        self._timer = QTimer(self)
        self._timer.setInterval(max(int(1000 / fps), 20))
        self._timer.timeout.connect(self._tick)

        if frames:
            self._show_frame(0)

    def _init_marker_overlay(self) -> None:
        """在 3D 视图中添加 4 个 marker 点和两条支架线。"""
        self._marker_colors = {
            0: (0.15, 0.45, 1.00, 1.0),  # m0 蓝色：右侧球
            1: (1.00, 0.85, 0.10, 1.0),  # m1 黄色：针尖方向球
            2: (0.00, 0.95, 1.00, 1.0),  # m2 青色：针尾方向球
            3: (1.00, 0.25, 0.95, 1.0),  # m3 紫色：左侧球
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

    def _fit_camera_to_frames(self, frames: Sequence[PoseFrame]) -> None:
        if not frames:
            return
        points = [frame.tip for frame in frames]
        for frame in frames:
            points.extend(marker for _idx, marker in frame.markers)
        all_points = np.asarray(points, dtype=np.float64)
        center = all_points.mean(axis=0)
        extent = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
        distance = float(np.clip(max(extent * 3.0, self._needle_length_mm * 2.2, 180.0), 180.0, 700.0))
        self._gl.view.opts["center"] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self._gl.view.setCameraPosition(distance=distance, elevation=22, azimuth=45)
        self._gl.view.opts["distance"] = distance

    def _update_marker_overlay(self, frame: PoseFrame) -> None:
        if not frame.markers:
            self._marker_scatter.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))
            return

        positions = np.asarray([marker for _idx, marker in frame.markers], dtype=np.float32)
        colors = np.asarray([self._marker_colors.get(idx, (1.0, 1.0, 1.0, 1.0)) for idx, _marker in frame.markers], dtype=np.float32)
        self._marker_scatter.setData(pos=positions, color=colors, size=9.0, pxMode=False)

        by_id = {idx: marker for idx, marker in frame.markers}
        segments = []
        for a, b in ((0, 3), (2, 1)):
            if a in by_id and b in by_id:
                segments.extend([by_id[a], by_id[b]])
        if segments:
            self._marker_rods.setData(pos=np.asarray(segments, dtype=np.float32))
        else:
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))

    def _show_frame(self, i: int) -> None:
        if not self._frames:
            return
        i = max(0, min(i, len(self._frames) - 1))
        self._idx = i
        frame = self._frames[i]
        self._gl.set_marker_needle_pose(frame.tip, frame.axis, confidence=frame.confidence)
        self._update_marker_overlay(frame)
        self._status.setText(
            f"有效帧 {i + 1}/{len(self._frames)}  原始行={frame.source_index}  {frame.label}  "
            f"conf={frame.confidence:.2f}  markers={len(frame.markers)}  "
            "m0蓝 m1黄 m2青 m3紫"
        )

    def _prev(self) -> None:
        self._show_frame(self._idx - 1)

    def _next(self) -> None:
        self._show_frame(self._idx + 1)

    def _play(self) -> None:
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        if self._idx + 1 >= len(self._frames):
            self._timer.stop()
            return
        self._show_frame(self._idx + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="回放 JetArm marker pose CSV，在 3D 窗口中显示针。")
    parser.add_argument("pose_csv", type=Path, help="pose_from_ir_depth.py 或 pose_from_markers.py 输出的 CSV")
    parser.add_argument("--fps", type=float, default=15.0, help="播放帧率")
    parser.add_argument("--needle-length-mm", type=float, default=162.0, help="显示用针长")
    args = parser.parse_args()

    if not args.pose_csv.is_file():
        print(f"找不到 CSV：{args.pose_csv}", file=sys.stderr)
        return 1

    frames = load_pose_frames(args.pose_csv)
    if not frames:
        print("CSV 中没有可回放的 valid=1 针姿态帧。", file=sys.stderr)
        return 1

    print(f"已读取可回放帧：{len(frames)}")
    print(f"输入 CSV：{args.pose_csv.resolve()}")
    print("提示：窗口内只播放 valid=1 的帧，已剔除 m1 疑似误识别帧。")

    app = QApplication(sys.argv)
    win = PoseReplayWindow(frames, fps=args.fps, needle_length_mm=args.needle_length_mm)
    win.resize(1200, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
