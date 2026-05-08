"""穿刺会话管理模块 - 管理穿刺流程状态"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer


class PunctureSession(QObject):
    """穿刺会话管理器

    管理一次穿刺训练的完整流程，包括：
    - 阶段状态机
    - 深度追踪
    - 结果判定
    """

    # 信号定义
    phase_changed = pyqtSignal(str)  # 阶段变化
    depth_changed = pyqtSignal(float, float)  # (当前深度, 目标深度)
    alignment_changed = pyqtSignal(float, bool)  # (角度, 是否对齐)
    deviation_warning = pyqtSignal(float, bool)  # (偏离角度, 是否严重)
    result_determined = pyqtSignal(str, dict)  # ('success'/'failed', 详情)
    simulated_tip_moved = pyqtSignal(object)  # 模拟针尖位置更新

    # 阶段常量
    PHASE_IDLE = 'idle'  # 空闲
    PHASE_ALIGNING = 'aligning'  # 对齐中
    PHASE_LOCKED = 'locked'  # 已锁定，准备进针
    PHASE_ADVANCING = 'advancing'  # 进针中
    PHASE_COMPLETED = 'completed'  # 完成（成功）
    PHASE_FAILED = 'failed'  # 失败

    def __init__(self, puncture_monitor):
        """初始化

        Args:
            puncture_monitor: PunctureMonitor 实例
        """
        super().__init__()

        self.monitor = puncture_monitor

        # 状态
        self._phase = self.PHASE_IDLE

        # 目标信息
        self.target_position = None  # 目标点位置
        self.sphere_center = None  # 外层球体中心
        self.outer_radius = 40.0  # 外层球体半径
        self.entry_point = None  # 入针点
        self.target_depth = 0.0  # 目标深度

        # 当前状态
        self.current_depth = 0.0  # 当前模拟深度
        self.locked_direction = None  # 锁定的进针方向
        self.lock_position = None  # 锁定时的IMU位置
        self.simulated_tip = None  # 模拟针尖位置

        # 参数
        self.alignment_threshold = 3.0  # 对齐阈值（度）
        self.success_tolerance = 3.0  # 成功判定容差（mm）
        self.advance_speed = 2.0  # 基础进针速度（mm/次）

        # 进针定时器
        self._advance_timer = QTimer()
        self._advance_timer.setInterval(50)  # 50ms
        self._advance_timer.timeout.connect(self._on_advance_tick)
        self._advancing_direction = 0  # 1=进针, -1=退针, 0=停止
        self._speed_factor = 1.0

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if self._phase != value:
            self._phase = value
            self.phase_changed.emit(value)
            print(f"📍 阶段变更: {value}")

    def start(self, target_position, sphere_center, outer_radius=40.0):
        """开始穿刺会话

        Args:
            target_position: 目标点位置 [x, y, z]
            sphere_center: 球体中心位置 [x, y, z]
            outer_radius: 外层球体半径
        """
        self.target_position = np.array(target_position, dtype=float)
        self.sphere_center = np.array(sphere_center, dtype=float)
        self.outer_radius = outer_radius

        # 重置状态
        self.current_depth = 0.0
        self.locked_direction = None
        self.lock_position = None
        self.simulated_tip = None
        self.entry_point = None
        self.target_depth = 0.0

        self.phase = self.PHASE_ALIGNING
        print(f"✓ 会话开始: 目标={target_position}, 球心={sphere_center}")

    def end(self):
        """结束会话"""
        self._advance_timer.stop()
        self._advancing_direction = 0

        if self.monitor.is_locked:
            self.monitor.unlock()

        self.phase = self.PHASE_IDLE
        print("✓ 会话结束")

    def check_alignment(self, needle_direction, imu_position):
        """检查方向对齐

        Args:
            needle_direction: 当前针尖方向向量
            imu_position: 当前IMU位置

        Returns:
            angle: 对齐角度误差（度）
            is_aligned: 是否已对齐
        """
        if self.phase != self.PHASE_ALIGNING or self.sphere_center is None:
            return 0.0, False

        # 计算从IMU指向目标的方向
        target_direction = self.sphere_center - np.array(imu_position)
        dist = np.linalg.norm(target_direction)
        if dist < 1e-6:
            return 0.0, False
        target_direction = target_direction / dist

        # 计算角度误差
        dot = np.clip(np.dot(needle_direction, target_direction), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        is_aligned = (angle <= self.alignment_threshold)

        self.alignment_changed.emit(angle, is_aligned)

        return angle, is_aligned

    def lock_attitude(self, quaternion, euler, imu_position):
        """锁定姿态，进入准备进针状态

        Args:
            quaternion: 当前四元数
            euler: 当前欧拉角
            imu_position: 当前IMU位置
        """
        if self.phase != self.PHASE_ALIGNING:
            print("⚠ 只能在对齐阶段锁定姿态")
            return False

        # 调用底层监测器锁定
        self.monitor.lock_attitude(quaternion, euler)

        # 保存锁定状态
        self.locked_direction = self.monitor.get_locked_direction()
        self.lock_position = np.array(imu_position, dtype=float)
        self.simulated_tip = self.lock_position.copy()

        # 计算入针点和目标深度
        self._calculate_entry_and_depth()

        self.phase = self.PHASE_LOCKED

        return True

    def unlock_attitude(self):
        """解锁姿态，返回对齐阶段"""
        self._advance_timer.stop()
        self._advancing_direction = 0
        self.current_depth = 0.0

        self.monitor.unlock()

        self.phase = self.PHASE_ALIGNING

    def start_advance(self, speed_factor=1.0):
        """开始进针"""
        if self.phase not in [self.PHASE_LOCKED, self.PHASE_ADVANCING]:
            return

        self._advancing_direction = 1
        self._speed_factor = speed_factor
        self.phase = self.PHASE_ADVANCING

        if not self._advance_timer.isActive():
            self._advance_timer.start()

    def start_retract(self, speed_factor=1.0):
        """开始退针"""
        if self.phase not in [self.PHASE_LOCKED, self.PHASE_ADVANCING]:
            return

        self._advancing_direction = -1
        self._speed_factor = speed_factor

        if not self._advance_timer.isActive():
            self._advance_timer.start()

    def stop_movement(self):
        """停止进针/退针"""
        self._advancing_direction = 0
        self._advance_timer.stop()

        # 如果有深度，保持在进针阶段；否则回到锁定阶段
        if self.current_depth > 0:
            self.phase = self.PHASE_ADVANCING
        else:
            self.phase = self.PHASE_LOCKED

    def update(self, current_quaternion, imu_position):
        """更新会话状态（由主循环调用）

        Args:
            current_quaternion: 当前四元数
            imu_position: 当前IMU位置

        Returns:
            dict: 状态信息
        """
        result = {
            'phase': self.phase,
            'deviation_angle': 0.0,
            'is_deviated': False,
            'suggestions': []
        }

        # 如果已锁定，检查姿态偏离
        if self.phase in [self.PHASE_LOCKED, self.PHASE_ADVANCING]:
            angle, is_deviated = self.monitor.check_deviation(current_quaternion)
            result['deviation_angle'] = angle
            result['is_deviated'] = is_deviated

            if is_deviated:
                _, _, suggestions = self.monitor.calculate_correction(current_quaternion)
                result['suggestions'] = suggestions

                # 检查是否严重偏离（针尖会穿出球体）
                is_critical = self._check_critical_deviation(angle)
                self.deviation_warning.emit(angle, is_critical)

        return result

    def _on_advance_tick(self):
        """进针/退针定时器回调"""
        if self._advancing_direction == 0:
            self._advance_timer.stop()
            return

        # 计算位移
        delta = self.advance_speed * self._speed_factor * self._advancing_direction
        new_depth = self.current_depth + delta

        # 限制最小深度为0
        if new_depth < 0:
            new_depth = 0
            self._advancing_direction = 0
            self._advance_timer.stop()

        self.current_depth = new_depth

        # 更新模拟针尖位置
        if self.locked_direction is not None and self.lock_position is not None:
            self.simulated_tip = self.lock_position + self.locked_direction * self.current_depth
            self.simulated_tip_moved.emit(self.simulated_tip)

        # 发送深度变化信号
        self.depth_changed.emit(self.current_depth, self.target_depth)

        # 检查结果
        self._check_result()

    def _calculate_entry_and_depth(self):
        """计算入针点和目标深度"""
        if self.locked_direction is None or self.lock_position is None:
            return

        if self.sphere_center is None or self.target_position is None:
            return

        # 射线与球体求交：从lock_position沿locked_direction方向
        # 射线方程: P = lock_position + t * locked_direction
        # 球体方程: |P - sphere_center|² = outer_radius²

        oc = self.lock_position - self.sphere_center
        a = np.dot(self.locked_direction, self.locked_direction)
        b = 2.0 * np.dot(oc, self.locked_direction)
        c = np.dot(oc, oc) - self.outer_radius ** 2

        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)

            # 取较近的正交点作为入针点
            if t1 > 0:
                t_entry = t1
            elif t2 > 0:
                t_entry = t2
            else:
                t_entry = 0

            if t_entry > 0:
                self.entry_point = self.lock_position + self.locked_direction * t_entry

        # 计算目标深度（从lock_position到target_position沿方向的投影）
        to_target = self.target_position - self.lock_position
        self.target_depth = np.dot(to_target, self.locked_direction)

        # 确保目标深度为正
        if self.target_depth < 0:
            self.target_depth = np.linalg.norm(to_target)

        print(f"✓ 入针点: {self.entry_point}")
        print(f"✓ 目标深度: {self.target_depth:.1f}mm")

    def _check_critical_deviation(self, angle):
        """检查是否严重偏离（可能导致穿出球体）"""
        # 简化判断：偏离超过阈值的2倍算严重
        return angle > self.monitor.threshold * 2

    def _check_result(self):
        """检查穿刺结果"""
        if self.simulated_tip is None or self.target_position is None:
            return

        # 计算针尖到目标点距离
        dist_to_target = np.linalg.norm(self.simulated_tip - self.target_position)

        # 成功：到达目标点
        if dist_to_target <= self.success_tolerance:
            self._advance_timer.stop()
            self._advancing_direction = 0
            self.phase = self.PHASE_COMPLETED
            self.result_determined.emit('success', {
                'target_position': self.target_position.tolist(),
                'final_depth': self.current_depth,
                'accuracy': dist_to_target
            })
            return

        # 检查是否穿出球体
        if self.sphere_center is not None:
            dist_to_center = np.linalg.norm(self.simulated_tip - self.sphere_center)

            # 失败：穿出球体但没到目标
            if dist_to_center > self.outer_radius:
                self._advance_timer.stop()
                self._advancing_direction = 0
                self.phase = self.PHASE_FAILED
                self.result_determined.emit('failed', {
                    'reason': '穿出目标区域',
                    'target_position': self.target_position.tolist(),
                    'final_position': self.simulated_tip.tolist(),
                    'miss_distance': dist_to_target
                })
                return

        # 失败：过深（超过目标深度太多）
        if self.current_depth > self.target_depth + self.outer_radius:
            self._advance_timer.stop()
            self._advancing_direction = 0
            self.phase = self.PHASE_FAILED
            self.result_determined.emit('failed', {
                'reason': '穿刺过深',
                'target_depth': self.target_depth,
                'final_depth': self.current_depth,
                'overshoot': self.current_depth - self.target_depth
            })

    def get_status(self):
        """获取当前状态摘要"""
        return {
            'phase': self.phase,
            'target_depth': self.target_depth,
            'current_depth': self.current_depth,
            'progress': (self.current_depth / self.target_depth * 100) if self.target_depth > 0 else 0,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'simulated_tip': self.simulated_tip.tolist() if self.simulated_tip is not None else None,
            'entry_point': self.entry_point.tolist() if self.entry_point is not None else None
        }
