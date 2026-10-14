"""姿态观察模式的锁轴平移控制器。

该模块只处理状态和几何，不依赖 Qt。测距只产生沿锁定针轴的相对位移；
姿态超出角度阈值时冻结已应用位置，但不会解除锁定。
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


class ObservationTranslationController:
    IDLE = "idle"
    READY = "ready"
    LOCKED = "locked"
    ANGLE_PAUSED = "angle_paused"
    RANGE_BLOCKED = "range_blocked"
    WAITING_ZERO = "waiting_zero"

    def __init__(self, clock=None, require_stationary=True, enforce_angle_pause=True):
        self._clock = clock or time.monotonic
        # 穿刺观察只用于展示沿锁定针轴的测距位移时，可以放宽这两项约束；
        # 默认值保持原有控制器行为，便于独立测试和其他调用方复用。
        self.require_stationary = bool(require_stationary)
        self.enforce_angle_pause = bool(enforce_angle_pause)
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.reason = "等待 IMU、测距和零点"
        self.imu_connected = False
        self.range_connected = False
        self.range_valid = False
        self.baseline_valid = False
        self.gyro_calibrated = False
        self.gyro_norm_dps = math.inf
        self.stationary_since: Optional[float] = None
        self._previous_axis = None
        self._previous_axis_at: Optional[float] = None
        self.locked = False
        self.locked_axis = None
        self.current_axis = None
        self.angle_deg = None
        self.measured_displacement_mm = 0.0
        self.applied_displacement_mm = 0.0
        self._last_valid_displacement_mm = 0.0
        self._over_angle_since: Optional[float] = None
        self._within_angle_since: Optional[float] = None

    def _now(self, now):
        return float(self._clock() if now is None else now)

    @staticmethod
    def _unit(axis):
        if axis is None:
            return None
        arr = np.asarray(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-9 and np.all(np.isfinite(arr)) else None

    @property
    def is_stationary(self):
        return self.stationary_since is not None

    @property
    def tip_position(self):
        if self.locked_axis is None:
            return np.zeros(3, dtype=float)
        return self.locked_axis * float(self.applied_displacement_mm)

    def set_imu_connected(self, connected: bool):
        self.imu_connected = bool(connected)
        if not connected:
            self.stationary_since = None
            self._previous_axis = None
            self._previous_axis_at = None
            self.unlock("IMU 已断开")
        self._refresh_state()

    def set_range_status(self, connected: bool, valid: bool = False, reason: str = ""):
        self.range_connected = bool(connected)
        self.range_valid = bool(valid)
        if not valid and self.locked:
            self.state = self.RANGE_BLOCKED
            self.reason = reason or "等待有效测距"
        self._refresh_state(reason)

    def set_baseline(self, valid: bool, baseline_mm=None):
        had_baseline = self.baseline_valid
        self.baseline_valid = bool(valid and baseline_mm is not None)
        if not self.baseline_valid:
            self.measured_displacement_mm = 0.0
            self._last_valid_displacement_mm = 0.0
            if had_baseline and self.locked:
                self.unlock("测距零点丢失，已退出平移")
            self.state = self.WAITING_ZERO
            self.reason = "针轴已锁定，请记录测距零点" if self.locked else "请记录测距零点"
        else:
            self._refresh_state()

    def update_gyro(self, gyro_rad_s, calibrated: bool, now=None):
        now = self._now(now)
        self.gyro_calibrated = bool(calibrated)
        try:
            gyro = np.asarray(gyro_rad_s, dtype=float).reshape(3)
            self.gyro_norm_dps = float(np.linalg.norm(gyro) * 180.0 / math.pi)
        except (TypeError, ValueError):
            self.gyro_norm_dps = math.inf
        if self.gyro_calibrated and self.gyro_norm_dps <= 3.0:
            if self.stationary_since is None:
                self.stationary_since = now
        elif self.gyro_calibrated:
            self.stationary_since = None
        self._refresh_state()

    def update_axis(self, axis, now=None):
        now = self._now(now)
        self.current_axis = self._unit(axis)
        if not self.gyro_calibrated:
            if self.current_axis is None:
                self.stationary_since = None
            elif self._previous_axis is not None and self._previous_axis_at is not None:
                dt = now - self._previous_axis_at
                dot = float(np.clip(np.dot(self.current_axis, self._previous_axis), -1.0, 1.0))
                rate = math.degrees(math.acos(dot)) / dt if dt > 1e-3 else math.inf
                if rate <= 3.0:
                    if self.stationary_since is None:
                        self.stationary_since = now
                else:
                    self.stationary_since = None
            self._previous_axis = None if self.current_axis is None else self.current_axis.copy()
            self._previous_axis_at = now
        if self.locked_axis is not None and self.current_axis is not None:
            dot = float(np.clip(np.dot(self.current_axis, self.locked_axis), -1.0, 1.0))
            self.angle_deg = float(np.degrees(np.arccos(dot)))
        self._refresh_state()

    def update_range(self, measured_displacement_mm, valid: bool, now=None):
        del now
        self.range_valid = bool(valid and measured_displacement_mm is not None)
        if self.range_valid:
            value = float(measured_displacement_mm)
            if math.isfinite(value):
                # 保留零点两侧的有符号位移：正值和负值分别表示沿锁定针轴
                # 的两个相反方向。RangeDepthPanel 已经用“零点距离 - 当前距离”
                # 计算相对深度，这里不能再次把负值截成 0，否则只能单向平移。
                self.measured_displacement_mm = value
                self._last_valid_displacement_mm = self.measured_displacement_mm
            else:
                self.range_valid = False
        if not self.range_valid:
            if self.locked:
                self.state = self.RANGE_BLOCKED
                self.reason = "测距无效，位置已冻结"
            self._refresh_state(self.reason)
            return
        if self.locked and self.state == self.LOCKED:
            self.applied_displacement_mm = self.measured_displacement_mm
        self._refresh_state()

    def toggle_lock(self, now=None):
        now = self._now(now)
        if self.locked:
            self.unlock("已退出平移模式")
            self._refresh_state()
            return False
        if not self.imu_connected:
            self.state, self.reason = self.IDLE, "请先连接 IMU"
            return False
        if (
            self.require_stationary
            and (self.stationary_since is None or now - self.stationary_since < 0.3)
        ):
            self.state, self.reason = self.READY, "姿态需静止 0.3 秒后才能进入平移"
            return False
        axis = self._unit(self.current_axis)
        if axis is None:
            self.state, self.reason = self.READY, "等待有效针轴"
            return False
        self.locked_axis = axis.copy()
        self.locked = True
        self._over_angle_since = None
        self._within_angle_since = now
        if not self.range_connected or not self.range_valid:
            self.state, self.reason = self.RANGE_BLOCKED, "针轴已锁定，等待有效测距"
        elif not self.baseline_valid:
            self.state, self.reason = self.WAITING_ZERO, "针轴已锁定，请记录测距零点"
        else:
            self.applied_displacement_mm = self.measured_displacement_mm
            self.state, self.reason = self.LOCKED, "平移中 · 沿锁定针轴移动"
        return True

    def unlock(self, reason="已退出平移模式"):
        self.locked = False
        self.locked_axis = None
        self._over_angle_since = None
        self._within_angle_since = None
        self.applied_displacement_mm = 0.0
        self.state = self.IDLE
        self.reason = reason

    def tick(self, now=None):
        now = self._now(now)
        if not self.locked:
            self._refresh_state()
            return self.state
        if not self.range_valid or not self.baseline_valid:
            self.state = self.RANGE_BLOCKED if self.range_valid is False else self.WAITING_ZERO
            self.reason = "测距无效，位置已冻结" if not self.range_valid else "针轴已锁定，请记录测距零点"
            return self.state
        angle = self.angle_deg if self.angle_deg is not None else math.inf
        if self.enforce_angle_pause and angle > 10.0:
            self._within_angle_since = None
            if self._over_angle_since is None:
                self._over_angle_since = now
            if now - self._over_angle_since >= 0.2:
                self.state = self.ANGLE_PAUSED
                self.reason = "姿态偏离锁定针轴，位置已冻结"
            return self.state
        if not self.enforce_angle_pause:
            self._over_angle_since = None
            self._within_angle_since = now
            self.state = self.LOCKED
            self.reason = "平移中 · 沿锁定针轴移动"
            self.applied_displacement_mm = self.measured_displacement_mm
            return self.state

        self._over_angle_since = None
        if angle <= 8.0:
            if self._within_angle_since is None:
                self._within_angle_since = now
            if self.state in (self.RANGE_BLOCKED, self.WAITING_ZERO):
                self.state = self.LOCKED
                self.reason = "平移中 · 沿锁定针轴移动"
                self.applied_displacement_mm = self.measured_displacement_mm
            elif self.state == self.ANGLE_PAUSED and now - self._within_angle_since >= 0.3:
                self.state = self.LOCKED
                self.reason = "已恢复平移 · 正在补到当前测距"
                self.applied_displacement_mm = self.measured_displacement_mm
            elif self.state == self.LOCKED:
                self.applied_displacement_mm = self.measured_displacement_mm
        else:
            self._within_angle_since = None
        return self.state

    def _refresh_state(self, reason=""):
        if self.locked:
            return
        if not self.imu_connected:
            self.state, self.reason = self.IDLE, "等待连接 IMU"
        elif not self.range_connected or not self.range_valid:
            self.state, self.reason = self.RANGE_BLOCKED, reason or "等待有效测距"
        elif not self.baseline_valid:
            self.state, self.reason = self.WAITING_ZERO, "请记录测距零点"
        elif self.require_stationary and self.stationary_since is None:
            self.state, self.reason = self.READY, "按 Space 前请保持姿态静止"
        else:
            self.state, self.reason = self.READY, "可按 Space 进入平移"
