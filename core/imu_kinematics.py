"""IMU 四元数到针体轴向（场景坐标系）的共用几何计算。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

_SQRT2_INV = 0.7071067811865476
# 针体在 IMU 坐标系中的参考方向
_NEEDLE_BODY_IMU: Tuple[float, float, float] = (_SQRT2_INV, _SQRT2_INV, 0.0)


def _rotate_vector_by_quaternion(
    q: Sequence[float], v: Sequence[float]
) -> Tuple[float, float, float]:
    q0, q1, q2, q3 = q
    bx, by, bz = v
    rx = (1 - 2 * (q2 * q2 + q3 * q3)) * bx + 2 * (q1 * q2 - q0 * q3) * by + 2 * (q1 * q3 + q0 * q2) * bz
    ry = 2 * (q1 * q2 + q0 * q3) * bx + (1 - 2 * (q1 * q1 + q3 * q3)) * by + 2 * (q2 * q3 - q0 * q1) * bz
    rz = 2 * (q1 * q3 - q0 * q2) * bx + 2 * (q2 * q3 + q0 * q1) * by + (1 - 2 * (q1 * q1 + q2 * q2)) * bz
    return rx, ry, rz


def needle_axis_scene_raw(quaternion: Sequence[float]) -> Tuple[float, float, float]:
    """四元数旋转针体参考方向，并映射到场景坐标系（与主窗口原逻辑一致）。"""
    raw_x, raw_y, raw_z = _rotate_vector_by_quaternion(quaternion, _NEEDLE_BODY_IMU)
    tip_x = raw_y
    tip_y = -raw_x
    tip_z = -raw_z
    return tip_x, tip_y, tip_z


def needle_axis_scene_normalized(quaternion: Sequence[float]) -> Optional[List[float]]:
    """归一化针轴方向；若退化则返回 None（保持上一帧方向时不更新）。"""
    tip_x, tip_y, tip_z = needle_axis_scene_raw(quaternion)
    length = (tip_x ** 2 + tip_y ** 2 + tip_z ** 2) ** 0.5
    if length <= 0.001:
        return None
    return [tip_x / length, tip_y / length, tip_z / length]


def needle_axis_for_position(quaternion: Sequence[float]) -> List[float]:
    """用于由针尖反推 IMU 位置时的方向；退化时返回默认向上。"""
    d = needle_axis_scene_normalized(quaternion)
    if d is None:
        return [0.0, 0.0, 1.0]
    return d


def imu_position_from_tip(
    tip_pos: Sequence[float], direction: Sequence[float], needle_length: float
) -> List[float]:
    """由针尖、针轴与长度反推 IMU 位置。"""
    return [
        float(tip_pos[0]) - direction[0] * needle_length,
        float(tip_pos[1]) - direction[1] * needle_length,
        float(tip_pos[2]) - direction[2] * needle_length,
    ]


def tip_position_from_fixed(
    fixed_tip: Union[None, np.ndarray, Sequence[float]],
) -> Tuple[Optional[List[float]], bool]:
    """
    从 GL 固定针尖解析 tip_pos。
    返回 (tip_pos 或 None, 是否应跳过本帧)。
    """
    if fixed_tip is None:
        return [0.0, 0.0, 0.0], False
    arr = np.asarray(fixed_tip, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return None, True
    flat = arr.flatten()
    return [float(flat[0]), float(flat[1]), float(flat[2])], False
