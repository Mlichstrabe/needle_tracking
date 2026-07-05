"""由锚点深度 + 针轴方向估计 scene 系针尖（V1：共线近似，无 m2→IMU 旋转标定）。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def depth_median_window(
    depth_map_mm: np.ndarray,
    cx: int,
    cy: int,
    half_window: int,
    min_pixels: int,
    z_min: float,
    z_max: float,
) -> Optional[float]:
    """在 (cx,cy) 邻域取有效深度中值（mm）。"""
    h, w = depth_map_mm.shape[:2]
    x0 = max(0, cx - half_window)
    x1 = min(w, cx + half_window + 1)
    y0 = max(0, cy - half_window)
    y1 = min(h, cy + half_window + 1)
    patch = depth_map_mm[y0:y1, x0:x1].astype(float).ravel()
    valid = patch[(patch >= z_min) & (patch <= z_max) & np.isfinite(patch)]
    if valid.size < min_pixels:
        return None
    return float(np.median(valid))


def tip_from_anchor_camera(
    anchor_cam_mm: Sequence[float],
    axis_cam_unit: Sequence[float],
    m2_to_tip_mm: float,
) -> np.ndarray:
    """锚点沿针轴偏移 m2_to_tip_mm 得到针尖（相机系 mm）。"""
    a = np.asarray(anchor_cam_mm, dtype=float).reshape(3)
    d = np.asarray(axis_cam_unit, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return a.copy()
    d = d / n
    return a + d * float(m2_to_tip_mm)


def imu_only_observe_pose(
    axis_scene_unit: Sequence[float],
    tip_scene_mm: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """观察模式默认：针尖 scene 原点（或指定点），姿态来自 IMU。"""
    if tip_scene_mm is not None:
        tip = np.asarray(tip_scene_mm, dtype=float).reshape(3)
    else:
        tip = np.zeros(3, dtype=float)
    axis = np.asarray(axis_scene_unit, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        axis = np.array([0.0, 0.0, -1.0])
    else:
        axis = axis / n
    return tip, axis