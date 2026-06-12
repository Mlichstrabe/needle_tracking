#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JetArm marker 针体 3D 显示共用：CSV 解析、marker 叠加、回放窗口。"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph.opengl as gl

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QVector3D
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from ui.widgets.gl_widget import GLVisualizationWidget

MARKER_COLORS = {
    0: (0.15, 0.45, 1.00, 1.0),
    1: (1.00, 0.85, 0.10, 1.0),
    2: (0.00, 0.95, 1.00, 1.0),
    3: (1.00, 0.25, 0.95, 1.0),
}


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class PoseFrame:
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


def row_to_pose(row: dict, source_index: int) -> Optional[PoseFrame]:
    if row.get("valid") != "1":
        return None

    if _has_values(row, ["tip_x_cam_mm", "tip_y_cam_mm", "tip_z_cam_mm", "axis_x_cam", "axis_y_cam", "axis_z_cam"]):
        tip = _as_vec(row, ["tip_x_cam_mm", "tip_y_cam_mm", "tip_z_cam_mm"])
        axis = _normalize_axis(_as_vec(row, ["axis_x_cam", "axis_y_cam", "axis_z_cam"]))
        label = f"ir_index={row.get('ir_index', '')} depth_index={row.get('depth_index', '')}"
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
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    frames: List[PoseFrame] = []
    for i, row in enumerate(rows):
        pose = row_to_pose(row, i)
        if pose is not None:
            frames.append(pose)
    return frames


def fit_gl_camera_to_points(
    gl_widget: GLVisualizationWidget,
    points: Sequence[np.ndarray],
    *,
    needle_length_mm: float = 162.0,
    extent_scale: float = 3.0,
    min_distance: float = 180.0,
    max_distance: float = 700.0,
) -> None:
    if not points:
        return
    all_points = np.asarray(points, dtype=np.float64)
    center = all_points.mean(axis=0)
    extent = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    distance = float(np.clip(max(extent * extent_scale, needle_length_mm * 2.2, min_distance), min_distance, max_distance))
    gl_widget.view.opts["center"] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
    gl_widget.view.setCameraPosition(distance=distance, elevation=22, azimuth=45)
    gl_widget.view.opts["distance"] = distance


class MarkerOverlay:
    """在 GL 视图上绘制 m0–m3 散点与支架连线。"""

    def __init__(self, gl_widget: GLVisualizationWidget) -> None:
        self._gl = gl_widget
        self._marker_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            size=9.0,
            pxMode=False,
        )
        self._marker_rods = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.75, 0.95, 1.0, 0.8),
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self._gl.view.addItem(self._marker_scatter)
        self._gl.view.addItem(self._marker_rods)

    def update(self, markers: List[Tuple[int, np.ndarray]]) -> None:
        if not markers:
            self._marker_scatter.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))
            return

        positions = np.asarray([marker for _idx, marker in markers], dtype=np.float32)
        colors = np.asarray(
            [MARKER_COLORS.get(idx, (1.0, 1.0, 1.0, 1.0)) for idx, _marker in markers],
            dtype=np.float32,
        )
        self._marker_scatter.setData(pos=positions, color=colors, size=9.0, pxMode=False)

        by_id = {idx: marker for idx, marker in markers}
        segments = []
        for a, b in ((0, 3), (2, 1)):
            if a in by_id and b in by_id:
                segments.extend([by_id[a], by_id[b]])
        if segments:
            self._marker_rods.setData(pos=np.asarray(segments, dtype=np.float32))
        else:
            self._marker_rods.setData(pos=np.zeros((1, 3), dtype=np.float32))


class PoseReplayWindow(QMainWindow):
    """离线 CSV 回放窗口。"""

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
        self._overlay = MarkerOverlay(self._gl)
        if frames:
            points = [frame.tip for frame in frames]
            for frame in frames:
                points.extend(marker for _idx, marker in frame.markers)
            fit_gl_camera_to_points(self._gl, points, needle_length_mm=self._needle_length_mm)
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

    def _show_frame(self, i: int) -> None:
        if not self._frames:
            return
        i = max(0, min(i, len(self._frames) - 1))
        self._idx = i
        frame = self._frames[i]
        self._gl.set_marker_needle_pose(frame.tip, frame.axis, confidence=frame.confidence)
        self._overlay.update(frame.markers)
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
