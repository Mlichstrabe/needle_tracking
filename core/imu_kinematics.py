"""IMU 四元数 → 针体轴向（场景坐标系）。

几何链路::

    d_raw     = R(q) @ v_needle_body
    d_display = R_offset @ d_raw          # 竖直标定一次后持久化

- ``v_needle_body``：针轴在 IMU 体坐标 XY 平面，从 +X **顺时针** 121°
- 场景约定：Z 向上，针尖向下 = ``(0, 0, -1)``，针尖固定原点

状态管理：所有运行时配置集中在 ``_State`` 实例中。
模块级函数为向后兼容的快捷方式，委托至模块级单例 ``_state``。
测试时可直接实例化 ``_State()`` 并传入相关函数。
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SCENE_DOWN: Tuple[float, float, float] = (0.0, 0.0, -1.0)


# ═══════════════════════════════════════════════════════════
#  状态容器
# ═══════════════════════════════════════════════════════════

class _State:
    """不可变数学函数的可变配置参数。"""

    def __init__(self) -> None:
        self.needle_body_angle_deg: float = 121.0
        self.needle_body_bias_deg: float = 0.0
        self.needle_angle_clockwise_from_x: bool = True
        self.needle_body_imu: Tuple[float, float, float] = (1.0, 0.0, 0.0)

        self.display_offset: np.ndarray = np.eye(3, dtype=float)
        self.display_offset_enabled: bool = False

        self.scene_z_ccw_deg: float = 0.0

        self._refresh_needle_body_vector()

    def _refresh_needle_body_vector(self) -> None:
        rad = math.radians(self.needle_body_angle_deg + self.needle_body_bias_deg)
        if self.needle_angle_clockwise_from_x:
            self.needle_body_imu = (math.cos(rad), -math.sin(rad), 0.0)
        else:
            self.needle_body_imu = (math.cos(rad), math.sin(rad), 0.0)

    def scene_z_rotation_matrix(self) -> np.ndarray:
        a = math.radians(self.scene_z_ccw_deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def display_offset_dict(self) -> dict:
        return {
            "enabled": self.display_offset_enabled,
            "rotation": [[float(self.display_offset[r, c]) for c in range(3)] for r in range(3)],
        }

    def to_dict(self) -> Dict[str, Any]:
        """导出为配置文件兼容的字典。"""
        return {
            "needle_body_angle_deg": self.needle_body_angle_deg,
            "needle_body_bias_deg": self.needle_body_bias_deg,
            "scene_z_ccw_deg": self.scene_z_ccw_deg,
            "needle_angle_clockwise_from_x": self.needle_angle_clockwise_from_x,
            "display_offset": self.display_offset_dict(),
        }


# 模块级单例（向后兼容）
_state = _State()


def get_state() -> _State:
    """获取当前全局状态（用于持久化/测试）。"""
    return _state


def reset_state() -> None:
    """重置为默认值（用于测试清理）。"""
    global _state
    _state = _State()


# ═══════════════════════════════════════════════════════════
#  配置 getter/setter（委托至 _state）
# ═══════════════════════════════════════════════════════════

def set_needle_angle_clockwise_from_x(clockwise: bool) -> None:
    _state.needle_angle_clockwise_from_x = bool(clockwise)
    _state._refresh_needle_body_vector()


def needle_angle_clockwise_from_x() -> bool:
    return _state.needle_angle_clockwise_from_x


def set_needle_body_angle_deg(deg: float) -> None:
    _state.needle_body_angle_deg = float(deg)
    _state._refresh_needle_body_vector()


def needle_body_angle_deg() -> float:
    return _state.needle_body_angle_deg


def set_needle_body_bias_deg(deg: float) -> None:
    _state.needle_body_bias_deg = float(deg)
    _state._refresh_needle_body_vector()


def needle_body_bias_deg() -> float:
    return _state.needle_body_bias_deg


def needle_body_effective_angle_deg() -> float:
    return _state.needle_body_angle_deg + _state.needle_body_bias_deg


def set_scene_z_ccw_deg(deg: float) -> None:
    _state.scene_z_ccw_deg = float(deg)


def scene_z_ccw_deg() -> float:
    return _state.scene_z_ccw_deg


def scene_z_rotation_matrix() -> np.ndarray:
    return _state.scene_z_rotation_matrix()


def apply_scene_z_rotation(v: Sequence[float]) -> np.ndarray:
    return _state.scene_z_rotation_matrix() @ np.asarray(v, dtype=float).reshape(3)


def needle_body_vector_imu() -> Tuple[float, float, float]:
    return _state.needle_body_imu


def display_offset_enabled() -> bool:
    return _state.display_offset_enabled


def display_offset_matrix() -> np.ndarray:
    return _state.display_offset.copy()


def display_offset_dict() -> dict:
    return _state.display_offset_dict()


def apply_display_offset(data: Optional[dict]) -> None:
    if not data:
        _state.display_offset = np.eye(3, dtype=float)
        _state.display_offset_enabled = False
        return
    rot = data.get("rotation")
    if rot and isinstance(rot, (list, tuple)) and len(rot) == 3:
        _state.display_offset = np.asarray(rot, dtype=float).reshape(3, 3)
        _state.display_offset_enabled = bool(data.get("enabled", True))
    else:
        _state.display_offset = np.eye(3, dtype=float)
        _state.display_offset_enabled = False


def clear_display_offset() -> None:
    apply_display_offset(None)


# ═══════════════════════════════════════════════════════════
#  纯函数（无状态依赖）
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  业务函数（读取 _state）
# ═══════════════════════════════════════════════════════════

def needle_direction_raw(quaternion: Sequence[float]) -> np.ndarray:
    """四元数 → 针轴方向（未做竖直偏置）。"""
    return np.array(_rotate_vector_by_quaternion(quaternion, _state.needle_body_imu), dtype=float)


def capture_vertical_display_offset(quaternion: Sequence[float]) -> np.ndarray:
    """针体竖直持握时标定：将当前 d_raw 映射到场景 (0,0,-1)。"""
    d_raw = _unit(needle_direction_raw(quaternion))
    _state.display_offset = _rotation_matrix_from_to(d_raw, SCENE_DOWN)
    _state.display_offset_enabled = True
    return _state.display_offset.copy()


def needle_axis_scene_raw(quaternion: Sequence[float]) -> Tuple[float, float, float]:
    d = needle_direction_raw(quaternion)
    if _state.display_offset_enabled:
        d = _state.display_offset @ d
    d = _state.scene_z_rotation_matrix() @ d
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
