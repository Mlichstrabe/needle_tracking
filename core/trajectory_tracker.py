"""轨迹跟踪器 - 极限优化版（实验性）"""
import numpy as np
from collections import deque


class TrajectoryTracker:
    """极限滤波的轨迹跟踪器"""

    def __init__(self):
        # 位置和速度
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        # =====  优化后的参数 =====
        self.sensitivity = 1.8  # 从1.5提升到1.8
        self.dead_zone = 0.08  # ✅ 从0.25降低
        self.velocity_decay_moving = 0.99  # ✅ 新增：运动中衰减
        self.velocity_decay_static = 0.85  # ✅ 新增：静止中衰减
        self.static_threshold = 0.05  # ✅ 从0.18降低
        self.max_velocity = 250.0  # 从200提升到250
        self.boundary = 300.0

        # ===== 零速度更新（ZUPT） =====
        self.zupt_window_base = 8  # ✅ 从15降低到8（基础值）
        self.static_count = 0
        self.is_static = True

        # ===== 加速度偏置校准 =====
        self.acc_bias = np.array([0.0, 0.0, 0.0])
        self.calibrating = False
        self.calibration_samples = []
        self.calibration_count = 100

        # ===== 高通滤波 =====
        self.hp_alpha = 0.85  # 从0.8提升到0.85
        self.last_acc = None
        self.filtered_acc = np.array([0.0, 0.0, 0.0])

        # ===== 运动检测 =====
        self.acc_history = deque(maxlen=5)
        self.motion_threshold = 0.12  # ✅ 从0.3降低

        self.enabled = False

    def reset(self):
        """重置"""
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.is_static = True
        self.static_count = 0
        self.last_acc = None
        self.filtered_acc = np.array([0.0, 0.0, 0.0])
        self.acc_history.clear()
        print("[Tracker] 位置已重置")

    def enable(self):
        """启用"""
        self.enabled = True
        self.reset()
        print("[Tracker] 已启用 - 请保持静止进行自动校准...")
        self.start_calibration()

    def disable(self):
        """禁用"""
        self.enabled = False
        self.calibrating = False
        print("[Tracker] 已禁用")

    def start_calibration(self):
        """开始校准（采集静止时的加速度偏置）"""
        self.calibrating = True
        self.calibration_samples = []
        print("[Tracker] 开始校准，请保持静止3秒...")

    def update(self, acceleration, quaternion, dt):
        """更新位置"""
        if not self.enabled:
            return self.position.copy()

        if acceleration is None or quaternion is None:
            return self.position.copy()

        acc = np.array(acceleration, dtype=float)
        quat = np.array(quaternion, dtype=float)

        # ====== 1. 去除重力 ======
        linear_acc = self._remove_gravity(acc, quat)

        # ====== 2. 校准阶段 ======
        if self.calibrating:
            self.calibration_samples.append(linear_acc.copy())
            if len(self.calibration_samples) >= self.calibration_count:
                self._finish_calibration()
            return self.position.copy()

        # ====== 3. 去除偏置 ======
        linear_acc = linear_acc - self.acc_bias

        # ====== 4. 高通滤波（关键！去除缓慢漂移） ======
        linear_acc = self._highpass_filter(linear_acc)

        # ====== 5. 死区滤波 ======
        acc_mag = np.linalg.norm(linear_acc)
        if acc_mag < self.dead_zone:
            linear_acc = np.zeros(3)
            acc_mag = 0

        # ====== 6. 运动检测（基于加速度变化） ======
        is_moving = self._detect_motion(linear_acc)

        # ====== 7. 动态 ZUPT（零速度更新） ======
        vel_mag = np.linalg.norm(self.velocity)

        # 根据速度动态调整ZUPT窗口
        if vel_mag > 50:
            zupt_threshold = 15
        elif vel_mag > 20:
            zupt_threshold = 10
        else:
            zupt_threshold = self.zupt_window_base

        if acc_mag < self.static_threshold and not is_moving:
            self.static_count += 1
            if self.static_count >= zupt_threshold:
                # 确认静止，强制速度归零
                self.is_static = True
                self.velocity = np.zeros(3)
                return self.position.copy()
        else:
            self.static_count = 0
            self.is_static = False

        # ====== 8. 速度积分（只在确认运动时） ======
        if not self.is_static and is_moving:
            delta_v = linear_acc * dt * self.sensitivity * 1000
            self.velocity += delta_v

        # ====== 9. 智能速度衰减 ======
        if self.is_static or acc_mag < self.static_threshold:
            # 静止状态：快速衰减
            self.velocity *= self.velocity_decay_static
        else:
            # 运动状态：保持惯性
            self.velocity *= self.velocity_decay_moving

        if np.linalg.norm(self.velocity) < 3.0:
            self.velocity = np.zeros(3)
            self.is_static = True

        # ====== 10. 速度限制 ======
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > self.max_velocity:
            self.velocity = self.velocity / vel_mag * self.max_velocity

        # ====== 11. 位置积分 ======
        self.position += self.velocity * dt

        # ====== 12. 边界约束 ======
        self.position = np.clip(self.position, -self.boundary, self.boundary)

        return self.position.copy()

    def _remove_gravity(self, acceleration, quaternion):
        """去除重力"""
        # 使用四元数将重力旋转到传感器坐标系
        q0, q1, q2, q3 = quaternion  # w, x, y, z

        # 重力在世界坐标系中 [0, 0, -9.81]
        # 旋转到传感器坐标系
        gx = 2 * (q1 * q3 - q0 * q2) * 9.81
        gy = 2 * (q2 * q3 + q0 * q1) * 9.81
        gz = (q0*q0 - q1*q1 - q2*q2 + q3*q3) * 9.81

        gravity_body = np.array([gx, gy, gz])

        return acceleration - gravity_body

    def _highpass_filter(self, acc):
        """高通滤波 - 去除缓慢漂移（重力残差）"""
        if self.last_acc is None:
            self.last_acc = acc.copy()
            return np.zeros(3)

        # 高通滤波: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        self.filtered_acc = self.hp_alpha * (
            self.filtered_acc + acc - self.last_acc
        )
        self.last_acc = acc.copy()

        return self.filtered_acc.copy()

    def _detect_motion(self, acc):
        """检测是否真正在运动（基于加速度变化率）"""
        self.acc_history.append(acc.copy())

        if len(self.acc_history) < 3:
            return False

        # 计算相邻帧加速度变化率
        acc_diff = np.linalg.norm(
            self.acc_history[-1] - self.acc_history[-2]
        )

        return acc_diff > self.motion_threshold

    def _finish_calibration(self):
        """完成校准"""
        samples = np.array(self.calibration_samples)
        self.acc_bias = np.mean(samples, axis=0)
        self.calibrating = False
        print(f"[Tracker] 校准完成！偏置: [{self.acc_bias[0]:.4f}, "
              f"{self.acc_bias[1]:.4f}, {self.acc_bias[2]:.4f}]")
        print("[Tracker] 现在可以移动了")

    def get_state(self):
        """获取状态"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'is_static': self.is_static,
            'calibrating': self.calibrating,
            'vel_magnitude': np.linalg.norm(self.velocity)
        }
