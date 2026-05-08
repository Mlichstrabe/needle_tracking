"""姿态处理工具 - 修复版"""
import numpy as np


def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角转旋转矩阵（ZYX顺序，符合JY901S）

    Args:
        roll, pitch, yaw: 欧拉角（度）

    Returns:
        3x3旋转矩阵
    """
    # 转为弧度
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # 计算三角函数
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    # ZYX旋转顺序（先Yaw绕Z，再Pitch绕Y，最后Roll绕X）
    # 这是标准的航空航天顺序
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    return R


class OrientationProcessor:
    """姿态处理器 - 极简稳定版"""

    def __init__(self):
        self._euler = np.array([0.0, 0.0, 0.0])
        self._yaw_offset = 0.0

        # ====== 动态平滑缓冲 ======
        from collections import deque
        self._euler_buffer = deque(maxlen=5)  # 🔥 增加到5帧（最大容量）

        # ====== 新增：运动检测 ======
        self._last_euler = None
        self._motion_threshold = 2.0# 只保留最近3帧

    def reset(self):
        """重置"""
        self._euler = np.array([0.0, 0.0, 0.0])
        self._yaw_offset = 0.0
        self._euler_buffer.clear()
        self._last_euler = None  # 🔥 新增：重置运动检测
        print("✓ 姿态处理器已重置")

    def process_euler(self, roll, pitch, yaw):
        """处理欧拉角（动态平滑）

        Args:
            roll, pitch, yaw: 欧拉角（度）

        Returns:
            平滑后的欧拉角
        """
        # ====== 1. 应用 Yaw 偏移 ======
        corrected_yaw = yaw - self._yaw_offset

        # 规范化 Yaw 到 [-180, 180]
        corrected_yaw = ((corrected_yaw + 180) % 360) - 180

        # 当前帧的欧拉角
        current_euler = np.array([roll, pitch, corrected_yaw])

        # ====== 2. 运动检测（动态调整平滑强度） ======
        if self._last_euler is not None:
            # 计算角度变化量
            delta = np.abs(current_euler - self._last_euler)

            # 处理 Yaw 跨越 ±180° 的情况
            if delta[2] > 180:
                delta[2] = 360 - delta[2]

            max_delta = np.max(delta)

            # ====== 动态窗口大小 ======
            if max_delta > self._motion_threshold:
                # 🔥 快速运动：缩小窗口（减少延迟）
                target_buffer_size = 2
            elif max_delta > 0.5:
                # 🔥 中速运动：中等窗口
                target_buffer_size = 3
            else:
                # 🔥 静止/微动：大窗口（更平滑）
                target_buffer_size = 5

            # 动态调整缓冲区大小
            if len(self._euler_buffer) > target_buffer_size:
                # 缩小窗口：移除旧数据
                while len(self._euler_buffer) > target_buffer_size:
                    self._euler_buffer.popleft()

        # ====== 3. 添加到缓冲区 ======
        self._euler_buffer.append(current_euler)

        # ====== 4. 加权平均（优先新数据） ======
        if len(self._euler_buffer) >= 2:
            # 🔥 指数加权平均：权重从旧到新递增
            weights = np.array([0.6 ** i for i in range(len(self._euler_buffer) - 1, -1, -1)])
            weights /= weights.sum()  # 归一化

            buffer_array = np.array(self._euler_buffer)
            self._euler = np.average(buffer_array, axis=0, weights=weights)
        else:
            self._euler = current_euler

        # ====== 5. 更新历史 ======
        self._last_euler = current_euler.copy()

        return self._euler

    def get_euler_angles(self):
        """获取当前欧拉角"""
        return self._euler.copy()

    def get_rotation_matrix(self):
        """获取旋转矩阵"""
        return euler_to_rotation_matrix(*self._euler)

    def reset_yaw(self):
        """将当前Yaw设为零点"""
        self._yaw_offset = self._euler[2] + self._yaw_offset
        print(f"✓ Yaw已重置，新偏移: {self._yaw_offset:.1f}°")

    def get_smoothing_info(self):
        """获取平滑器状态（用于调试）"""
        return {
            'buffer_size': len(self._euler_buffer),
            'max_buffer': self._euler_buffer.maxlen,
            'current_euler': self._euler.tolist(),
            'yaw_offset': self._yaw_offset,
            'motion_detected': self._last_euler is not None and
                               np.max(np.abs(self._euler - self._last_euler)) > self._motion_threshold
        }

# 向后兼容别名
OrientationEKF = OrientationProcessor
