"""测距驱动的穿刺状态机。

该模块只处理状态和数值，不依赖 Qt 控件，也不产生任何位移。主窗口
负责把事件转换为界面更新和 Entry→Target 方向上的训练可视化。
"""

import math
import time


class PunctureDepthController:
    PREPARING = "PREPARING"
    WAITING_DEVICES = "WAITING_DEVICES"
    ALIGNING = "ALIGNING"
    PUNCTURING = "PUNCTURING"
    ANGLE_PAUSED = "ANGLE_PAUSED"
    RANGE_INVALID = "RANGE_INVALID"
    COMPLETE = "COMPLETE"

    ALIGN_THRESHOLD_DEG = 3.0
    PAUSE_THRESHOLD_DEG = 4.0
    ALIGN_HOLD_S = 0.30
    COMPLETE_VALID_SAMPLES = 3

    def __init__(self, clock=None):
        self._clock = clock or time.monotonic
        self.reset(clear_baseline=True)

    def reset(self, clear_baseline=True):
        """重置本次穿刺；默认同时要求重新记录测距零点。"""
        old_imu_connected = getattr(self, "imu_connected", False)
        old_range_connected = getattr(self, "range_connected", False)
        old_range_valid = getattr(self, "range_valid", False)
        self.state = self.PREPARING
        self.reason = "等待 CT、Entry 和规划深度"
        self.plan_depth_mm = 0.0
        self.measured_depth_mm = None
        self.applied_depth_mm = 0.0
        self.final_depth_mm = None
        self.complete_count = 0
        self.imu_connected = old_imu_connected
        self.range_connected = old_range_connected
        self.range_valid = old_range_valid
        self.baseline_valid = False if clear_baseline else getattr(self, "baseline_valid", False)
        self.baseline_mm = None if clear_baseline else getattr(self, "baseline_mm", None)
        self.angle_error_deg = None
        self._aligned_since = None
        self._puncture_started = False
        return []

    def _now(self, now):
        return self._clock() if now is None else float(now)

    def _set_state(self, state, reason, events):
        if self.state == state and self.reason == reason:
            return
        self.state = state
        self.reason = reason
        if state != self.PUNCTURING:
            self.complete_count = 0
        events.append({"type": "state_changed", "state": state, "reason": reason})

    def _ready_for_alignment(self):
        return (
            self.plan_depth_mm > 0.0
            and self.imu_connected
            and self.range_connected
            and self.range_valid
            and self.baseline_valid
            and self.measured_depth_mm is not None
        )

    def _ready_without_sample(self):
        return (
            self.plan_depth_mm > 0.0
            and self.imu_connected
            and self.range_connected
            and self.baseline_valid
        )

    def _state_after_losing_readiness(self, reason, events, error=False):
        if self.state == self.COMPLETE:
            return
        if self.plan_depth_mm <= 0.0:
            self._set_state(self.PREPARING, reason, events)
            return
        if error or self._puncture_started:
            self._set_state(self.RANGE_INVALID, reason, events)
        else:
            self._set_state(self.WAITING_DEVICES, reason, events)

    def set_plan_depth(self, plan_depth_mm):
        events = []
        try:
            plan = float(plan_depth_mm)
        except (TypeError, ValueError):
            plan = 0.0
        self.plan_depth_mm = max(plan, 0.0) if math.isfinite(plan) else 0.0
        self.measured_depth_mm = None
        self.applied_depth_mm = 0.0
        self.final_depth_mm = None
        self.complete_count = 0
        self._puncture_started = False
        self._aligned_since = None
        if self.plan_depth_mm <= 0.0:
            self._set_state(self.PREPARING, "等待有效 Entry→Target 规划深度", events)
        elif self._ready_for_alignment():
            self._set_state(self.ALIGNING, "等待姿态对准", events)
        elif self._ready_without_sample():
            self._set_state(self.WAITING_DEVICES, "等待有效测距读数", events)
        else:
            self._set_state(self.WAITING_DEVICES, "等待 IMU、测距和测距零点", events)
        return events

    def set_imu_connected(self, connected):
        events = []
        self.imu_connected = bool(connected)
        self._aligned_since = None
        if not self.imu_connected:
            if self.state != self.COMPLETE:
                self._set_state(self.WAITING_DEVICES, "IMU 未连接，停止深度应用", events)
        elif self.state != self.COMPLETE:
            if self._ready_for_alignment():
                self._set_state(self.ALIGNING, "等待姿态对准", events)
            elif self.plan_depth_mm > 0.0:
                self._set_state(self.WAITING_DEVICES, "等待有效设备和测距零点", events)
        return events

    def set_range_status(self, connected, valid=False, reason="", error=False):
        events = []
        self.range_connected = bool(connected)
        self.range_valid = bool(valid) and self.range_connected
        self._aligned_since = None
        if not self.range_connected:
            self._state_after_losing_readiness(reason or "测距模块未连接", events, error=error)
        elif not self.range_valid:
            self._state_after_losing_readiness(reason or "等待有效测距读数", events, error=error)
        elif self.state != self.COMPLETE and self._ready_for_alignment():
            self._set_state(self.ALIGNING, "等待姿态对准", events)
        return events

    def set_baseline(self, valid, baseline_mm=None):
        events = []
        self.baseline_valid = bool(valid)
        if self.baseline_valid:
            try:
                baseline = float(baseline_mm)
            except (TypeError, ValueError):
                self.baseline_valid = False
                baseline = None
            if baseline is None or not math.isfinite(baseline):
                self.baseline_valid = False
            self.baseline_mm = baseline if self.baseline_valid else None
        else:
            self.baseline_mm = None
        self._aligned_since = None
        if not self.baseline_valid:
            self._state_after_losing_readiness("测距零点无效，请重新记录", events, error=self._puncture_started)
        elif self.state != self.COMPLETE and self._ready_for_alignment():
            self._set_state(self.ALIGNING, "等待姿态对准", events)
        elif self.state != self.COMPLETE and self.plan_depth_mm > 0.0:
            self._set_state(self.WAITING_DEVICES, "等待有效测距读数", events)
        return events

    def update_range(self, raw_mm, filtered_mm, measured_depth_mm, age_s=0.0, valid=True, reason="", now=None):
        """接收一条已完成有效性判断的测距样本。"""
        events = []
        if not valid:
            self.range_valid = False
            self._aligned_since = None
            self._state_after_losing_readiness(reason or "测距数据无效", events, error=True)
            return events

        try:
            filtered = float(filtered_mm)
            measured = float(measured_depth_mm)
            age = float(age_s)
        except (TypeError, ValueError):
            return self.update_range(raw_mm, filtered_mm, measured_depth_mm, valid=False, reason="测距数据格式错误", now=now)
        if not all(math.isfinite(v) for v in (filtered, measured, age)) or age > 0.5:
            self.range_valid = False
            self._aligned_since = None
            self._state_after_losing_readiness("测距超时或数据无效", events, error=True)
            return events

        self.range_valid = True
        self.measured_depth_mm = measured
        self.applied_depth_mm = min(max(measured, 0.0), self.plan_depth_mm) if self.plan_depth_mm > 0 else 0.0

        if self.state == self.RANGE_INVALID and self._ready_for_alignment():
            self._set_state(self.ALIGNING, "测距恢复，等待姿态重新对准", events)
        elif self.state == self.WAITING_DEVICES and self._ready_for_alignment():
            self._set_state(self.ALIGNING, "等待姿态对准", events)

        if self.state == self.PUNCTURING:
            self._emit_depth(events)
        return events

    def update_angle(self, angle_error_deg, now=None):
        events = []
        try:
            angle = float(angle_error_deg)
        except (TypeError, ValueError):
            return events
        if not math.isfinite(angle):
            return events
        self.angle_error_deg = max(angle, 0.0)
        t = self._now(now)

        if self.state == self.COMPLETE:
            return events
        if not self._ready_for_alignment():
            self._aligned_since = None
            if self.plan_depth_mm > 0.0 and self.state not in (self.PREPARING, self.RANGE_INVALID):
                self._set_state(self.WAITING_DEVICES, "等待有效测距零点和数据", events)
            return events

        if self.state == self.PUNCTURING:
            if self.angle_error_deg > self.PAUSE_THRESHOLD_DEG:
                self._aligned_since = None
                self._set_state(self.ANGLE_PAUSED, "姿态偏离，深度应用暂停", events)
            return events

        if self.state == self.ANGLE_PAUSED:
            if self.angle_error_deg <= self.ALIGN_THRESHOLD_DEG:
                if self._aligned_since is None:
                    self._aligned_since = t
                elif t - self._aligned_since >= self.ALIGN_HOLD_S:
                    self._puncture_started = True
                    self._set_state(self.PUNCTURING, "姿态恢复，测距驱动进针", events)
                    self._emit_depth(events)
            else:
                self._aligned_since = None
            return events

        if self.state in (self.ALIGNING, self.WAITING_DEVICES, self.RANGE_INVALID):
            if self.angle_error_deg <= self.ALIGN_THRESHOLD_DEG:
                if self._aligned_since is None:
                    self._aligned_since = t
                elif t - self._aligned_since >= self.ALIGN_HOLD_S:
                    self._puncture_started = True
                    self._set_state(self.PUNCTURING, "姿态对准，测距驱动进针", events)
                    self._emit_depth(events)
            else:
                self._aligned_since = None
                if self.state != self.ALIGNING:
                    self._set_state(self.ALIGNING, "等待姿态进入 3° 阈值", events)
        return events

    def _emit_depth(self, events):
        events.append({
            "type": "depth_applied",
            "measured_depth_mm": self.measured_depth_mm,
            "applied_depth_mm": self.applied_depth_mm,
            "plan_depth_mm": self.plan_depth_mm,
        })
        if self.applied_depth_mm >= self.plan_depth_mm > 0.0:
            self.complete_count += 1
        else:
            self.complete_count = 0
        if self.complete_count >= self.COMPLETE_VALID_SAMPLES:
            self.final_depth_mm = self.applied_depth_mm
            self._set_state(self.COMPLETE, "连续有效测距达到规划深度", events)
            events.append({"type": "completed", "final_depth_mm": self.final_depth_mm})
