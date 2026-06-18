"""IMU 四元数 → 针体轴向（场景坐标系）。

几何链路::

    d_raw     = R(q) @ v_needle_body
    d_display = R_offset @ d_raw          # 竖直标定一次后持久化

- ``v_needle_body``：针轴在 IMU 体坐标 XY 平面，从 +X **顺时针** 121°
- 场景约定：Z 向上，针尖向下 = ``(0, 0, -1)``，针尖固定原点
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# 针轴：IMU 体坐标 XY 平面，从 +X 顺时针 121° + config 偏置
_NEEDLE_BODY_ANGLE_DEG: float = 121.0
_NEEDLE_BODY_BIAS_DEG: float = 0.0
_NEEDLE_ANGLE_CLOCKWISE_FROM_X: bool = True
_NEEDLE_BODY_IMU: Tuple[float, float, float] = (1.0, 0.0, 0.0)

_DISPLAY_OFFSET: np.ndarray = np.eye(3, dtype=float)
_DISPLAY_OFFSET_ENABLED: bool = False

_SCENE_Z_CCW_DEG: float = 0.0

SCENE_DOWN: Tuple[float, float, float] = (0.0, 0.0, -1.0)


def _refresh_needle_body_vector() -> None:
    global _NEEDLE_BODY_IMU
    rad = math.radians(_NEEDLE_BODY_ANGLE_DEG + _NEEDLE_BODY_BIAS_DEG)
    if _NEEDLE_ANGLE_CLOCKWISE_FROM_X:
        _NEEDLE_BODY_IMU = (math.cos(rad), -math.sin(rad), 0.0)
    else:
        _NEEDLE_BODY_IMU = (math.cos(rad), math.sin(rad), 0.0)


def set_needle_angle_clockwise_from_x(clockwise: bool) -> None:
    global _NEEDLE_ANGLE_CLOCKWISE_FROM_X
    _NEEDLE_ANGLE_CLOCKWISE_FROM_X = bool(clockwise)
    _refresh_needle_body_vector()


def needle_angle_clockwise_from_x() -> bool:
    return _NEEDLE_ANGLE_CLOCKWISE_FROM_X


def set_needle_body_angle_deg(deg: float) -> None:
    global _NEEDLE_BODY_ANGLE_DEG
    _NEEDLE_BODY_ANGLE_DEG = float(deg)
    _refresh_needle_body_vector()


def needle_body_angle_deg() -> float:
    return _NEEDLE_BODY_ANGLE_DEG


def set_needle_body_bias_deg(deg: float) -> None:
    global _NEEDLE_BODY_BIAS_DEG
    _NEEDLE_BODY_BIAS_DEG = float(deg)
    _refresh_needle_body_vector()


def needle_body_bias_deg() -> float:
    return _NEEDLE_BODY_BIAS_DEG


def needle_body_effective_angle_deg() -> float:
    return _NEEDLE_BODY_ANGLE_DEG + _NEEDLE_BODY_BIAS_DEG


def set_scene_z_ccw_deg(deg: float) -> None:
    global _SCENE_Z_CCW_DEG
    _SCENE_Z_CCW_DEG = float(deg)


def scene_z_ccw_deg() -> float:
    return _SCENE_Z_CCW_DEG


def scene_z_rotation_matrix() -> np.ndarray:
    a = math.radians(_SCENE_Z_CCW_DEG)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def apply_scene_z_rotation(v: Sequence[float]) -> np.ndarray:
    return scene_z_rotation_matrix() @ np.asarray(v, dtype=float).reshape(3)


def needle_body_vector_imu() -> Tuple[float, float, float]:
    return _NEEDLE_BODY_IMU


_refresh_needle_body_vector()


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("zero vector")
    return a / n


def _rotate_vector_by_quaternion(
    q: Sequence[float], v: Sequence[float]
) -> Tuple[float, float, float]:
    q0, q1, q2, q3 = q
    bx, by, bz = v
    rx = (1 - 2 * (q2 * q2 + q3 * q3)) * bx + 2 * (q1 * q2 - q0 * q3) * by + 2 * (q1 * q3 + q0 * q2) * bz
    ry = 2 * (q1 * q2 + q0 * q3) * bx + (1 - 2 * (q1 * q1 + q3 * q3)) * by + 2 * (q2 * q3 - q0 * q1) * bz
    rz = 2 * (q1 * q3 - q0 * q2) * bx + 2 * (q2 * q3 + q0 * q1) * by + (1 - 2 * (q1 * q1 + q2 * q2)) * bz
    return rx, ry, rz


def _rotation_matrix_from_to(src: Sequence[float], dst: Sequence[float]) -> np.ndarray:
    """将单位向量 src 旋转到 dst 的 3×3 旋转矩阵。"""
    a = _unit(src)
    b = _unit(dst)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1e-8:
        ortho = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(a, ortho))) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _unit(np.cross(a, ortho))
    else:
        axis = _unit(np.cross(a, b))
    angle = math.acos(dot)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3, dtype=float) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _matrix_to_rows(m: np.ndarray) -> List[List[float]]:
    return [[float(m[r, c]) for c in range(3)] for r in range(3)]


def _rows_to_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(3, 3)


def display_offset_enabled() -> bool:
    return _DISPLAY_OFFSET_ENABLED


def display_offset_matrix() -> np.ndarray:
    return _DISPLAY_OFFSET.copy()


def display_offset_dict() -> dict:
    return {
        "enabled": bool(_DISPLAY_OFFSET_ENABLED),
        "rotation": _matrix_to_rows(_DISPLAY_OFFSET),
    }


def apply_display_offset(data: Optional[dict]) -> None:
    global _DISPLAY_OFFSET, _DISPLAY_OFFSET_ENABLED
    if not data:
        _DISPLAY_OFFSET = np.eye(3, dtype=float)
        _DISPLAY_OFFSET_ENABLED = False
        return
    rot = data.get("rotation")
    if rot and isinstance(rot, (list, tuple)) and len(rot) == 3:
        _DISPLAY_OFFSET = _rows_to_matrix(rot)
        _DISPLAY_OFFSET_ENABLED = bool(data.get("enabled", True))
    else:
        _DISPLAY_OFFSET = np.eye(3, dtype=float)
        _DISPLAY_OFFSET_ENABLED = False


def clear_display_offset() -> None:
    apply_display_offset(None)


def needle_direction_raw(quaternion: Sequence[float]) -> np.ndarray:
    """四元数 → 针轴方向（未做竖直偏置）。"""
    return np.array(_rotate_vector_by_quaternion(quaternion, _NEEDLE_BODY_IMU), dtype=float)


def capture_vertical_display_offset(quaternion: Sequence[float]) -> np.ndarray:
    """
    针体竖直持握时标定：将当前 d_raw 映射到场景 (0,0,-1)。

    返回写入的 R_offset 矩阵。
    """
    global _DISPLAY_OFFSET, _DISPLAY_OFFSET_ENABLED
    d_raw = _unit(needle_direction_raw(quaternion))
    _DISPLAY_OFFSET = _rotation_matrix_from_to(d_raw, SCENE_DOWN)
    _DISPLAY_OFFSET_ENABLED = True
    return _DISPLAY_OFFSET.copy()


def needle_axis_scene_raw(quaternion: Sequence[float]) -> Tuple[float, float, float]:
    d = needle_direction_raw(quaternion)
    if _DISPLAY_OFFSET_ENABLED:
        d = _DISPLAY_OFFSET @ d
    d = apply_scene_z_rotation(d)
    return float(d[0]), float(d[1]), float(d[2])


def needle_axis_scene_normalized(quaternion: Sequence[float]) -> Optional[List[float]]:
    tip_x, tip_y, tip_z = needle_axis_scene_raw(quaternion)
    length = (tip_x ** 2 + tip_y ** 2 + tip_z ** 2) ** 0.5
    if length <= 0.001:
        return None
    return [tip_x / length, tip_y / length, tip_z / length]


def needle_tilt_from_scene_down_deg(direction: Sequence[float]) -> float:
    d = np.asarray(direction, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return float("nan")
    d = d / n
    down = np.array(SCENE_DOWN, dtype=float)
    dot = float(np.clip(np.dot(d, down), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def needle_axis_for_position(quaternion: Sequence[float]) -> List[float]:
    d = needle_axis_scene_normalized(quaternion)
    if d is None:
        return [0.0, 0.0, 1.0]
    return d


def imu_position_from_tip(
    tip_pos: Sequence[float], direction: Sequence[float], needle_length: float
) -> List[float]:
    return [
        float(tip_pos[0]) - direction[0] * needle_length,
        float(tip_pos[1]) - direction[1] * needle_length,
        float(tip_pos[2]) - direction[2] * needle_length,
    ]


def tip_position_from_fixed(
    fixed_tip: Union[None, np.ndarray, Sequence[float]],
) -> Tuple[Optional[List[float]], bool]:
    if fixed_tip is None:
        return [0.0, 0.0, 0.0], False
    arr = np.asarray(fixed_tip, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return None, True
    flat = arr.flatten()
    return [float(flat[0]), float(flat[1]), float(flat[2])], False


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
