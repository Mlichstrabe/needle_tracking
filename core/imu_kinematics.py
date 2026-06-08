"""IMU 四元数到针体轴向（场景坐标系）的共用几何计算。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

_SQRT2_INV = 0.7071067811865476

# 针体在 IMU 体坐标中的参考方向（相对模块 X/Y 平面；可按实物安装角再调）
_NEEDLE_BODY_IMU: Tuple[float, float, float] = (_SQRT2_INV, _SQRT2_INV, 0.0)

# R(q)*v_body 后映射到场景坐标：
#   默认：scene = (sx*ry, sy*rx, sz*rz)
# 说明：
# - WT901 常见安装下，经常出现“左右/前后其中一个反向”，因此把符号与 X/Y 交换做成可调
_SCENE_AXIS_SCALE: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
_SCENE_SWAP_XY: bool = False
_SCENE_YAW_OFFSET_DEG: float = 0.0


def set_scene_axis_scale(sx: float, sy: float, sz: float) -> None:
    """运行时调整场景轴符号（用于修正左右/上下镜像）。"""
    global _SCENE_AXIS_SCALE
    _SCENE_AXIS_SCALE = (float(sx), float(sy), float(sz))


def scene_axis_scale() -> Tuple[float, float, float]:
    return _SCENE_AXIS_SCALE


def set_scene_mapping(*, swap_xy: bool, sx: float, sy: float, sz: float) -> None:
    """运行时调整场景映射（交换 XY + 各轴符号）。"""
    global _SCENE_SWAP_XY
    _SCENE_SWAP_XY = bool(swap_xy)
    set_scene_axis_scale(sx, sy, sz)


def scene_swap_xy() -> bool:
    return _SCENE_SWAP_XY


def set_scene_yaw_offset_deg(deg: float) -> None:
    """设置水平面内的常量偏置（用于修正固定约 90° 的朝向偏差）。"""
    global _SCENE_YAW_OFFSET_DEG
    _SCENE_YAW_OFFSET_DEG = float(deg)


def scene_yaw_offset_deg() -> float:
    return _SCENE_YAW_OFFSET_DEG


def _map_imu_rotation_to_scene(
    rot: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """将 R(q)*v_body 的结果映射到场景 XYZ。"""
    rx, ry, rz = rot
    sx, sy, sz = _SCENE_AXIS_SCALE
    if _SCENE_SWAP_XY:
        # scene X <- rx, scene Y <- ry
        x, y, z = sx * rx, sy * ry, sz * rz
    else:
        # 默认：scene X <- ry, scene Y <- rx
        x, y, z = sx * ry, sy * rx, sz * rz

    # 水平面内常量偏置：绕 Z 轴旋转
    a = np.radians(_SCENE_YAW_OFFSET_DEG)
    ca = float(np.cos(a))
    sa = float(np.sin(a))
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr, z


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
    """四元数旋转针体参考方向，并映射到场景坐标系。"""
    rot = _rotate_vector_by_quaternion(quaternion, _NEEDLE_BODY_IMU)
    return _map_imu_rotation_to_scene(rot)


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
