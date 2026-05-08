"""穿刺姿态监测模块 - 只负责姿态相关计算"""
import numpy as np


class PunctureMonitor:
    """穿刺姿态监测器"""

    def __init__(self, threshold=3.0):
        """初始化

        Args:
            threshold: 偏离警告阈值（度）
        """
        self.reference_quaternion = None
        self.reference_euler = None
        self.threshold = threshold
        self.is_locked = False

    def lock_attitude(self, quaternion, euler=None):
        """锁定参考姿态

        Args:
            quaternion: 参考四元数 [w, x, y, z]
            euler: 参考欧拉角 [roll, pitch, yaw]（可选）
        """
        self.reference_quaternion = np.array(quaternion, dtype=float)
        self.reference_quaternion /= np.linalg.norm(self.reference_quaternion)

        if euler is not None:
            self.reference_euler = np.array(euler, dtype=float)

        self.is_locked = True
        print(f"✓ 姿态已锁定: q={self.reference_quaternion}")

    def unlock(self):
        """解锁姿态"""
        self.reference_quaternion = None
        self.reference_euler = None
        self.is_locked = False
        print("✓ 姿态已解锁")

    def set_threshold(self, threshold):
        """设置偏离阈值"""
        self.threshold = float(threshold)

    def check_deviation(self, current_quaternion):
        """检查姿态偏离（基于针尖方向向量）

        Args:
            current_quaternion: 当前四元数 [w, x, y, z]

        Returns:
            angle: 针尖方向偏离角度（度）
            is_deviated: 是否超出阈值
        """
        if not self.is_locked or self.reference_quaternion is None:
            return 0.0, False

        # 归一化当前四元数
        q_cur = np.array(current_quaternion, dtype=float)
        q_cur = q_cur / np.linalg.norm(q_cur)

        # 防止符号跳变
        if np.dot(self.reference_quaternion, q_cur) < 0:
            q_cur = -q_cur

        # 使用方向向量计算偏离角度
        ref_direction = self._quat_to_direction(self.reference_quaternion)
        cur_direction = self._quat_to_direction(q_cur)

        # 计算两个方向向量的夹角
        dot = np.clip(np.dot(ref_direction, cur_direction), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        is_deviated = (angle > self.threshold)

        return angle, is_deviated

    def calculate_correction(self, current_quaternion):
        """计算纠偏建议

        Returns:
            roll_err: Roll误差（度）
            pitch_err: Pitch误差（度）
            suggestions: 纠偏建议列表
        """
        if not self.is_locked or self.reference_quaternion is None:
            return 0.0, 0.0, []

        q_cur = np.array(current_quaternion, dtype=float)
        q_cur = q_cur / np.linalg.norm(q_cur)

        if np.dot(self.reference_quaternion, q_cur) < 0:
            q_cur = -q_cur

        # 计算误差四元数
        q_ref = self.reference_quaternion
        q_ref_inv = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])
        q_error = self._quaternion_multiply(q_ref_inv, q_cur)

        # 转换为欧拉角
        roll_err, pitch_err, yaw_err = self._quat_to_euler(q_error)

        # 生成建议
        suggestions = []

        if abs(roll_err) > 1.0:
            if roll_err > 0:
                suggestions.append(f"↶ 向左转 {abs(roll_err):.1f}°")
            else:
                suggestions.append(f"↷ 向右转 {abs(roll_err):.1f}°")

        if abs(pitch_err) > 1.0:
            if pitch_err > 0:
                suggestions.append(f"↑ 抬高 {abs(pitch_err):.1f}°")
            else:
                suggestions.append(f"↓ 降低 {abs(pitch_err):.1f}°")

        return roll_err, pitch_err, suggestions

    def get_locked_direction(self):
        """获取锁定时的针尖方向向量

        Returns:
            direction: 单位方向向量 [x, y, z]，未锁定时返回 None
        """
        if not self.is_locked or self.reference_quaternion is None:
            return None
        return self._quat_to_direction(self.reference_quaternion)

    def _quat_to_direction(self, q):
        """四元数转换为针尖方向向量"""
        w, x, y, z = q

        # 四元数转旋转矩阵
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        # 针尖初始方向（沿-Z轴）
        needle_vec = np.array([0.0, 0.0, -1.0])

        # 旋转后的方向
        direction = R @ needle_vec

        return direction / np.linalg.norm(direction)

    def _quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _quat_to_euler(self, q):
        """四元数转欧拉角（度）"""
        w, x, y, z = q

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
