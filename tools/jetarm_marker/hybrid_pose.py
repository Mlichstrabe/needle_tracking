#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合针姿：平移来自 IR+depth（m1 深度 + IMU 针轴），姿态来自 IMU。

  tip_cam = p_m1 + axis_scene @ tip_offset   （避免 m3–m1 深度噪声放大）
  tip_scene = R_cam_to_scene @ tip_cam - origin
  axis_scene = slerp(needle_axis_scene_normalized(q))
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_EXTRINSIC = _REPO / "data" / "jetarm_marker" / "geometry" / "camera_scene_extrinsic.json"


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    a = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return a / n


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    a = quat_normalize(q0)
    b = quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = a + float(t) * (b - a)
        return quat_normalize(out)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return quat_normalize(w0 * a + w1 * b)


def default_R_cam_to_scene() -> np.ndarray:
    """OpenCV 相机 (x右 y下 z前) → 显示 (x右 y前 z上)，并消除左右镜像感。"""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )


def load_camera_scene_extrinsic(path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    p = path or _DEFAULT_EXTRINSIC
    R = default_R_cam_to_scene()
    t = np.zeros(3, dtype=float)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("rotation"), list):
            R = np.asarray(data["rotation"], dtype=float).reshape(3, 3)
        if isinstance(data.get("translation_mm"), list):
            t = np.asarray(data["translation_mm"], dtype=float).reshape(3)
    return R, t


def transform_cam_to_scene(
    p_cam: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    return R @ np.asarray(p_cam, dtype=float).reshape(3) + t


def m1_position_cam(
    markers: Sequence[Dict[str, Any]], marker_index: int = 1
) -> Tuple[Optional[np.ndarray], bool]:
    if marker_index >= len(markers):
        return None, False
    m = markers[marker_index]
    if not m.get("valid"):
        return None, False
    return np.array([m["x"], m["y"], m["z"]], dtype=float), True


@dataclass
class AxisSmoother:
    alpha: float = 0.12

    _axis: Optional[np.ndarray] = field(default=None, repr=False)
    _quat: Optional[np.ndarray] = field(default=None, repr=False)

    def reset(self) -> None:
        self._axis = None
        self._quat = None

    def update_axis(self, axis: np.ndarray) -> np.ndarray:
        a = _unit(np.asarray(axis, dtype=float).reshape(3))
        if a is None:
            return self._axis if self._axis is not None else np.array([0.0, 0.0, -1.0])
        if self._axis is None:
            self._axis = a.copy()
        else:
            blend = (1.0 - self.alpha) * self._axis + self.alpha * a
            u = _unit(blend)
            if u is not None:
                self._axis = u
        return self._axis.copy()

    def update_quaternion(self, q: Sequence[float]) -> Optional[np.ndarray]:
        qn = quat_normalize(q)
        if self._quat is None:
            self._quat = qn
        else:
            self._quat = quat_slerp(self._quat, qn, self.alpha)
        from core.imu_kinematics import needle_axis_scene_normalized

        ax = needle_axis_scene_normalized(self._quat.tolist())
        if ax is None:
            return self._axis.copy() if self._axis is not None else None
        return self.update_axis(np.array(ax, dtype=float))


@dataclass
class TranslationSmoother:
    alpha: float = 0.42
    hold_frames: int = 8
    max_jump_mm: float = 45.0
    _tip: Optional[np.ndarray] = field(default=None, repr=False)
    _hold: int = 0

    def reset(self) -> None:
        self._tip = None
        self._hold = 0

    def update(self, tip_cam: Optional[np.ndarray], visible: bool) -> Optional[np.ndarray]:
        if tip_cam is not None and visible:
            t = np.asarray(tip_cam, dtype=float).reshape(3)
            if self._tip is not None:
                if float(np.linalg.norm(t - self._tip)) > self.max_jump_mm:
                    if self._hold > 0:
                        self._hold -= 1
                        return self._tip.copy()
            if self._tip is None:
                self._tip = t.copy()
            else:
                self._tip = (1.0 - self.alpha) * self._tip + self.alpha * t
            self._hold = self.hold_frames
            return self._tip.copy()
        if self._tip is not None and self._hold > 0:
            self._hold -= 1
            return self._tip.copy()
        return self._tip.copy() if self._tip is not None else None


@dataclass
class ScenePoseSmoother:
    alpha: float = 0.48
    _tip: Optional[np.ndarray] = field(default=None, repr=False)

    def reset(self) -> None:
        self._tip = None

    def update(self, tip_scene: np.ndarray) -> np.ndarray:
        t = np.asarray(tip_scene, dtype=float).reshape(3)
        if self._tip is None:
            self._tip = t.copy()
        else:
            self._tip = (1.0 - self.alpha) * self._tip + self.alpha * t
        return self._tip.copy()


@dataclass
class HybridPoseFusion:
    needle_length_mm: float = 200.0
    tip_offset_mm: float = 141.0
    tip_anchor_marker: int = 1
    R_cam_to_scene: np.ndarray = field(default_factory=default_R_cam_to_scene)
    t_cam_to_scene: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_subtract: bool = True
    _origin_scene: Optional[np.ndarray] = field(default=None, repr=False)
    translator: TranslationSmoother = field(default_factory=TranslationSmoother)
    axis_smoother: AxisSmoother = field(default_factory=AxisSmoother)
    scene_smoother: ScenePoseSmoother = field(default_factory=ScenePoseSmoother)

    def reset(self) -> None:
        self._origin_scene = None
        self.translator.reset()
        self.axis_smoother.reset()
        self.scene_smoother.reset()

    def fuse(
        self,
        *,
        m1_cam: Optional[np.ndarray],
        m1_visible: bool,
        axis_scene: Optional[np.ndarray],
        quaternion: Optional[Sequence[float]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, bool]:
        axis_out: Optional[np.ndarray] = None
        if quaternion is not None:
            axis_out = self.axis_smoother.update_quaternion(quaternion)
        elif axis_scene is not None:
            axis_out = self.axis_smoother.update_axis(axis_scene)
        elif self.axis_smoother._axis is not None:
            axis_out = self.axis_smoother._axis.copy()

        if axis_out is None:
            return None, None, 0.0, False

        tip_cam: Optional[np.ndarray] = None
        if m1_cam is not None and m1_visible:
            # axis_out 为场景系；偏移在相机系：R^T @ axis_scene
            axis_cam = self.R_cam_to_scene.T @ axis_out
            u = _unit(axis_cam)
            if u is None:
                u = axis_cam
            tip_cam = np.asarray(m1_cam, dtype=float) + u * float(self.tip_offset_mm)

        tip_c_smooth = self.translator.update(tip_cam, m1_visible and tip_cam is not None)
        if tip_c_smooth is None:
            return None, axis_out, 0.35, False

        tip_s = transform_cam_to_scene(tip_c_smooth, self.R_cam_to_scene, self.t_cam_to_scene)
        if self.origin_subtract:
            if self._origin_scene is None:
                self._origin_scene = tip_s.copy()
            tip_s = tip_s - self._origin_scene

        tip_s = self.scene_smoother.update(tip_s)
        conf = 0.92 if (m1_visible and quaternion is not None) else 0.55
        return tip_s, axis_out, conf, bool(m1_visible)


def apply_imu_kinematics_from_repo() -> float:
    from core.imu_geometry_config import apply_kinematics, load_config

    cfg = load_config()
    apply_kinematics(cfg)
    return float(cfg.get("needle_length_mm", 200.0))


def imu_axis_from_quaternion(quaternion: Sequence[float]) -> Optional[List[float]]:
    from core.imu_kinematics import needle_axis_scene_normalized

    return needle_axis_scene_normalized(quaternion)