"""模拟场景管理模块 - 管理目标生成和3D场景元素"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class SimulationManager(QObject):
    """模拟场景管理器

    负责：
    - 生成随机目标
    - 管理3D场景元素（球体、目标点等）
    - 透明度控制
    """

    # 信号
    target_generated = pyqtSignal(dict)  # 目标信息
    opacity_changed = pyqtSignal(int)  # 透明度级别变化

    # 透明度级别
    OPACITY_TRANSPARENT = 0  # 全透明（完全可见内部）
    OPACITY_TRANSLUCENT = 1  # 半透明（隐约可见）
    OPACITY_OPAQUE = 2  # 不透明（完全不可见内部）

    def __init__(self):
        super().__init__()

        # 目标配置
        self.min_distance = 80.0  # 最小距离（mm）
        self.max_distance = 150.0  # 最大距离（mm）
        self.outer_radius = 40.0  # 外层球体半径（mm）
        self.inner_radius = 3.0  # 目标点半径（mm）
        self.angle_range = 60.0  # 生成角度范围（度）

        # 当前目标
        self.target_position = None  # 目标点位置
        self.sphere_center = None  # 外层球体中心（与目标点相同）

        # 显示状态
        self._opacity_level = self.OPACITY_TRANSLUCENT
        self._show_target_point = False  # 初始隐藏目标点
        self._show_guide_line = True  # 显示引导线

    @property
    def opacity_level(self):
        return self._opacity_level

    @opacity_level.setter
    def opacity_level(self, value):
        if value != self._opacity_level:
            self._opacity_level = value
            self.opacity_changed.emit(value)

    @property
    def show_target_point(self):
        return self._show_target_point

    @show_target_point.setter
    def show_target_point(self, value):
        self._show_target_point = bool(value)

    @property
    def show_guide_line(self):
        return self._show_guide_line

    @show_guide_line.setter
    def show_guide_line(self, value):
        self._show_guide_line = bool(value)

    def get_opacity_value(self):
        """获取实际透明度值（0.0-1.0）

        Returns:
            float: 透明度值，0为完全透明，1为完全不透明
        """
        if self._opacity_level == self.OPACITY_TRANSPARENT:
            return 0.1  # 几乎全透明
        elif self._opacity_level == self.OPACITY_TRANSLUCENT:
            return 0.35  # 半透明
        else:
            return 0.85  # 不透明

    def generate_target(self, imu_position, needle_direction=None):
        """生成随机目标

        Args:
            imu_position: 当前IMU位置 [x, y, z]
            needle_direction: 当前针尖方向（可选，用于在前方生成）

        Returns:
            dict: 目标信息
        """
        imu_pos = np.array(imu_position, dtype=float)

        # 生成随机方向
        if needle_direction is not None:
            # 在针尖方向附近生成（±angle_range度范围内）
            base_direction = np.array(needle_direction, dtype=float)
            norm = np.linalg.norm(base_direction)
            if norm > 1e-6:
                base_direction = base_direction / norm
            else:
                base_direction = np.array([0, 0, -1])

            # 添加随机偏移
            theta = np.radians(np.random.uniform(0, self.angle_range / 2))
            phi = np.radians(np.random.uniform(0, 360))

            # 创建偏移方向
            random_direction = self._rotate_direction(base_direction, theta, phi)
        else:
            # 完全随机方向（但偏向前下方）
            theta = np.radians(np.random.uniform(20, self.angle_range))
            phi = np.radians(np.random.uniform(0, 360))

            random_direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                -np.cos(theta)  # 主要朝-Z方向（前方）
            ])
            random_direction = random_direction / np.linalg.norm(random_direction)

        # 随机距离
        distance = np.random.uniform(self.min_distance, self.max_distance)

        # 计算目标位置
        self.target_position = imu_pos + random_direction * distance
        self.sphere_center = self.target_position.copy()

        # 初始隐藏目标点
        self._show_target_point = False

        # 构建目标信息
        target_info = {
            'target_position': self.target_position.tolist(),
            'sphere_center': self.sphere_center.tolist(),
            'outer_radius': self.outer_radius,
            'inner_radius': self.inner_radius,
            'distance': distance,
            'direction': random_direction.tolist()
        }

        # 发送信号
        self.target_generated.emit(target_info)

        print(f"✓ 生成目标: 位置={self.target_position}, 距离={distance:.1f}mm")

        return target_info

    def _rotate_direction(self, base_dir, theta, phi):
        """将基础方向旋转指定角度

        Args:
            base_dir: 基础方向单位向量
            theta: 偏离角度（弧度）
            phi: 旋转角度（弧度）

        Returns:
            rotated_dir: 旋转后的方向向量
        """
        # 找到一个与base_dir垂直的向量
        if abs(base_dir[0]) < 0.9:
            perp1 = np.cross(base_dir, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(base_dir, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)

        # 第二个垂直向量
        perp2 = np.cross(base_dir, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # 在锥形范围内生成方向
        rotated = (
                base_dir * np.cos(theta) +
                perp1 * np.sin(theta) * np.cos(phi) +
                perp2 * np.sin(theta) * np.sin(phi)
        )

        return rotated / np.linalg.norm(rotated)

    def get_target_direction(self, from_position):
        """获取从指定位置到目标的方向向量

        Args:
            from_position: 起始位置 [x, y, z]

        Returns:
            direction: 单位方向向量，无目标时返回None
        """
        if self.sphere_center is None:
            return None

        direction = self.sphere_center - np.array(from_position)
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return None

        return direction / norm

    def get_distance_to_target(self, from_position):
        """获取从指定位置到目标中心的距离

        Args:
            from_position: 起始位置 [x, y, z]

        Returns:
            distance: 距离（mm），无目标时返回None
        """
        if self.sphere_center is None:
            return None

        return np.linalg.norm(self.sphere_center - np.array(from_position))

    def check_point_in_sphere(self, point):
        """检查点是否在外层球体内

        Args:
            point: 点位置 [x, y, z]

        Returns:
            is_inside: 是否在球体内
            distance_to_surface: 到球体表面的距离（负=内部，正=外部）
        """
        if self.sphere_center is None:
            return False, 0.0

        dist_to_center = np.linalg.norm(np.array(point) - self.sphere_center)
        distance_to_surface = dist_to_center - self.outer_radius

        return distance_to_surface <= 0, distance_to_surface

    def reveal_target(self):
        """显示目标点（穿刺成功或结束时调用）"""
        self._show_target_point = True

    def hide_target(self):
        """隐藏目标点"""
        self._show_target_point = False

    def clear(self):
        """清除当前目标"""
        self.target_position = None
        self.sphere_center = None
        self._show_target_point = False

    def get_display_info(self):
        """获取显示信息（供GLWidget使用）

        Returns:
            dict: 显示相关信息
        """
        return {
            'sphere_center': self.sphere_center.tolist() if self.sphere_center is not None else None,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'outer_radius': self.outer_radius,
            'inner_radius': self.inner_radius,
            'opacity': self.get_opacity_value(),
            'show_target_point': self._show_target_point,
            'show_guide_line': self._show_guide_line
        }
