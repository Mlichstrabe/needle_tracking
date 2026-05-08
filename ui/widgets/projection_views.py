"""三视图投影组件 - 轻量级2D实现"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
import numpy as np


class ProjectionView(QWidget):
    """单个投影视图（2D绘制）"""

    def __init__(self, title, axis1, axis2, parent=None):
        """
        Args:
            title: 视图标题（如"正视图"）
            axis1: 水平轴名称（如"X"）
            axis2: 垂直轴名称（如"Z"）
        """
        super().__init__(parent)
        self.title = title
        self.axis1 = axis1
        self.axis2 = axis2

        # 数据
        self.needle_start = [0, 0]  # 针体起点（投影后）
        self.needle_end = [0, 0]  # 针体终点（投影后）
        self.target_line = None  # 目标方向虚线（锁定后）
        self.preset_line = None  # 预设路径虚线（选择路径后）
        self.angle_deviation = 0.0  # 角度偏差
        self.is_perpendicular = False  # 针体是否垂直于视图

        # 样式
        self.setMinimumSize(120, 120)
        self.setStyleSheet("background: #1a1a2e; border-radius: 4px;")

    def update_projection(self, imu_pos, tip_pos, target_direction=None):
        """更新投影数据（针尖优先模式）"""
        # 根据视图类型提取对应轴
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        a1 = axis_map[self.axis1]
        a2 = axis_map[self.axis2]

        # 反转：针尖作为起点，IMU作为终点
        self.needle_start = [tip_pos[a1], tip_pos[a2]]  # 针尖（固定）
        self.needle_end = [imu_pos[a1], imu_pos[a2]]  # IMU（可动）

        # ====== NaN 检测 ======
        if any(np.isnan(x) or np.isinf(x) for x in self.needle_start + self.needle_end):
            return

        #  计算当前方向角度（从IMU指向针尖）
        dx = self.needle_start[0] - self.needle_end[0]  # 针尖 - IMU
        dy = self.needle_start[1] - self.needle_end[1]

        # ====== 避免除以零 ======
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 0.001:
            self.angle_deviation = 0.0
            self.target_line = None
            self.update()
            return

        self.is_perpendicular = False
        current_angle = np.degrees(np.arctan2(dy, dx))

        # 如果有目标方向，计算偏差
        if target_direction is not None:
            target_dx = target_direction[a1]
            target_dy = target_direction[a2]

            # ====== 检查目标方向在此视图中是否有效 ======
            target_length = (target_dx ** 2 + target_dy ** 2) ** 0.5
            if target_length < 0.001:
                self.angle_deviation = 0.0
                self.target_line = None
            else:
                target_angle = np.degrees(np.arctan2(target_dy, target_dx))
                self.angle_deviation = current_angle - target_angle

                # 归一化到 -180 ~ 180
                while self.angle_deviation > 180:
                    self.angle_deviation -= 360
                while self.angle_deviation < -180:
                    self.angle_deviation += 360

                # 目标线（橙色虚线）
                line_length = 50
                self.target_line = [
                    [0, 0],
                    [line_length * target_dx / target_length,
                     line_length * target_dy / target_length]
                ]
        else:
            self.angle_deviation = 0.0
            self.target_line = None

        self.update()

    def set_preset_line(self, preset_direction):
        """新增：设置预设路径虚线（蓝色）

        Args:
            preset_direction: [dx, dy, dz] 预设路径方向向量
        """
        if preset_direction is None:
            self.preset_line = None
            self.update()
            return

        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        a1 = axis_map[self.axis1]
        a2 = axis_map[self.axis2]

        preset_dx = preset_direction[a1]
        preset_dy = preset_direction[a2]

        # 检查是否垂直于此视图
        length = (preset_dx ** 2 + preset_dy ** 2) ** 0.5
        if length < 0.001:
            self.preset_line = None
        else:
            line_length = 60  # 比目标线稍长一点
            self.preset_line = [
                [0, 0],
                [line_length * preset_dx / length,
                 line_length * preset_dy / length]
            ]

        self.update()

    def paintEvent(self, event):
        """绘制视图"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        scale = min(w, h) / 300

        # ====== 背景 ======
        painter.fillRect(0, 0, w, h, QColor(26, 26, 46))

        # ====== 标题 ======
        painter.setPen(QPen(QColor(150, 150, 150)))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(5, 15, self.title)

        # ====== 坐标轴 ======
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(10, cy, w - 10, cy)  # 水平轴
        painter.drawLine(cx, 10, cx, h - 10)  # 垂直轴

        # 轴标签
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.drawText(w - 15, cy - 5, self.axis1)
        painter.drawText(cx + 5, 15, self.axis2)

        #  如果针体垂直于此视图，显示提示
        if self.is_perpendicular:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(self.rect(), Qt.AlignCenter, "针体垂直于此视图")
            return

        # ===== 绘制预设路径虚线（青蓝色，最底层） ======
        if self.preset_line is not None:
            try:
                painter.setPen(QPen(QColor(100, 200, 255, 180), 3, Qt.DashLine))
                x1 = cx + self.preset_line[0][0] * scale
                y1 = cy - self.preset_line[0][1] * scale
                x2 = cx + self.preset_line[1][0] * scale
                y2 = cy - self.preset_line[1][1] * scale

                if all(np.isfinite(c) for c in [x1, y1, x2, y2]):
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            except Exception as e:
                print(f"绘制预设虚线失败: {e}")

        # ====== 绘制目标方向虚线（橙色，中层） ======
        if self.target_line is not None:
            try:
                painter.setPen(QPen(QColor(255, 150, 50, 200), 2, Qt.DashLine))
                x1 = cx + self.target_line[0][0] * scale
                y1 = cy - self.target_line[0][1] * scale
                x2 = cx + self.target_line[1][0] * scale
                y2 = cy - self.target_line[1][1] * scale

                coords = [x1, y1, x2, y2]
                if all(np.isfinite(c) for c in coords):
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            except Exception as e:
                print(f"绘制目标线失败: {e}")

        # ====== 绘制针体线（最上层） ======
        try:
            # 根据偏差选择颜色
            if abs(self.angle_deviation) < 3:
                color = QColor(50, 255, 50)
            elif abs(self.angle_deviation) < 10:
                color = QColor(255, 255, 50)
            else:
                color = QColor(255, 80, 80)

            painter.setPen(QPen(color, 3))

            x1 = cx + self.needle_start[0] * scale
            y1 = cy - self.needle_start[1] * scale
            x2 = cx + self.needle_end[0] * scale
            y2 = cy - self.needle_end[1] * scale

            coords = [x1, y1, x2, y2]
            if all(np.isfinite(c) for c in coords):
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # 针尖点（红色大点）
                painter.setBrush(QBrush(QColor(255, 100, 100)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x1) - 5, int(y1) - 5, 10, 10)

                # IMU点（绿色小点）
                painter.setBrush(QBrush(QColor(100, 255, 100)))
                painter.drawEllipse(int(x2) - 3, int(y2) - 3, 6, 6)
        except Exception as e:
            print(f"绘制针体失败: {e}")

        # ====== 偏差显示 ======
        if self.target_line is not None:
            try:
                painter.setPen(QPen(color))
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                text = f"{abs(self.angle_deviation):.1f}°"
                painter.drawText(5, h - 25, text)

                painter.setFont(QFont("Arial", 8))
                if abs(self.angle_deviation) > 1:
                    if self.angle_deviation > 0:
                        hint = f"↓ 向下 {abs(self.angle_deviation):.0f}°"
                    else:
                        hint = f"↑ 向上 {abs(self.angle_deviation):.0f}°"
                    painter.drawText(5, h - 10, hint)
                else:
                    painter.setPen(QPen(QColor(100, 255, 100)))
                    painter.drawText(5, h - 10, "✓ OK")
            except Exception as e:
                print(f"绘制偏差文字失败: {e}")


class AngleIndicatorPanel(QWidget):
    """角度指示面板 - 汇总显示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #1a1a2e; border-radius: 4px;")
        self.setMinimumWidth(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # 标题
        title = QLabel("角度偏差")
        title.setStyleSheet("color: #aaa; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 模式说明
        mode_hint = QLabel("(针尖固定模式)")
        mode_hint.setStyleSheet("color: #666; font-size: 8px;")
        mode_hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(mode_hint)

        # 三个角度指示器
        self.pitch_label = QLabel("俯仰: --")
        self.yaw_label = QLabel("偏航: --")
        self.roll_label = QLabel("横滚: --")

        for label in [self.pitch_label, self.yaw_label, self.roll_label]:
            label.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(label)

        layout.addStretch()

        # 总偏差
        self.total_label = QLabel("总偏差: --")
        self.total_label.setStyleSheet("color: #fff; font-size: 12px; font-weight: bold;")
        self.total_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.total_label)

        # 修正建议
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("color: #5af; font-size: 10px;")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

    def update_angles(self, pitch_dev, yaw_dev, roll_dev):
        """更新角度偏差显示"""
        # 更新各轴
        self._update_label(self.pitch_label, "俯仰", pitch_dev)
        self._update_label(self.yaw_label, "偏航", yaw_dev)
        self._update_label(self.roll_label, "横滚", roll_dev)

        # 总偏差
        total = (pitch_dev ** 2 + yaw_dev ** 2 + roll_dev ** 2) ** 0.5
        if total < 3:
            color = "#5f5"
        elif total < 8:
            color = "#ff5"
        else:
            color = "#f55"

        self.total_label.setText(f"总偏差: {total:.1f}°")
        self.total_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")

        # 修正建议
        hints = []
        if abs(pitch_dev) > 2:
            direction = "向下" if pitch_dev > 0 else "向上"
            hints.append(f"{direction} {abs(pitch_dev):.0f}°")
        if abs(yaw_dev) > 2:
            direction = "向右" if yaw_dev > 0 else "向左"
            hints.append(f"{direction} {abs(yaw_dev):.0f}°")

        if hints:
            self.hint_label.setText("建议: " + ", ".join(hints))
        else:
            self.hint_label.setText("✓ 角度已对齐")

    def _update_label(self, label, name, deviation):
        """更新单个标签"""
        if abs(deviation) < 2:
            color = "#5f5"
            symbol = "✓"
        elif abs(deviation) < 5:
            color = "#ff5"
            symbol = "△"
        else:
            color = "#f55"
            symbol = "✗"

        label.setText(f"{name}: {deviation:+.1f}° {symbol}")
        label.setStyleSheet(f"color: {color}; font-size: 11px;")


class ProjectionPanel(QFrame):
    """三视图面板 - 集成所有投影视图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #12121a;
                border: 1px solid #333;
                border-radius: 6px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 三个投影视图
        self.front_view = ProjectionView("正视图", "X", "Z")
        self.side_view = ProjectionView("侧视图", "Y", "Z")
        self.top_view = ProjectionView("俯视图", "X", "Y")

        # 角度指示面板
        self.angle_panel = AngleIndicatorPanel()

        layout.addWidget(self.front_view, 1)
        layout.addWidget(self.side_view, 1)
        layout.addWidget(self.top_view, 1)
        layout.addWidget(self.angle_panel, 0)

        # 目标方向（锁定后）
        self.target_direction = None
        # 新增：预设路径方向（选择路径后）
        self.preset_path = None

    def set_target_direction(self, direction):
        """设置目标穿刺方向（锁定姿态后调用）

        Args:
            direction: [dx, dy, dz] 归一化方向向量
        """
        self.target_direction = direction

    def set_preset_path(self, path_direction):
        """新增：设置预设路径方向（选择路径后调用）

        Args:
            path_direction: [dx, dy, dz] 预设路径方向向量
        """
        self.preset_path = path_direction
        print(f"[ProjectionPanel] 预设路径已设置: {path_direction}")

        # 立即更新三视图，显示预设虚线
        self.front_view.set_preset_line(path_direction)
        self.side_view.set_preset_line(path_direction)
        self.top_view.set_preset_line(path_direction)

    def clear_preset_path(self):
        """新增：清除预设路径虚线"""
        self.preset_path = None
        self.front_view.set_preset_line(None)
        self.side_view.set_preset_line(None)
        self.top_view.set_preset_line(None)
        print("[ProjectionPanel] 预设路径已清除")

    def update_data(self, imu_pos, tip_pos, current_direction=None):
        """更新所有视图"""
        # 更新三个投影视图
        self.front_view.update_projection(imu_pos, tip_pos, self.target_direction)
        self.side_view.update_projection(imu_pos, tip_pos, self.target_direction)
        self.top_view.update_projection(imu_pos, tip_pos, self.target_direction)

        # 如果有目标方向，计算角度偏差
        if self.target_direction is not None and current_direction is not None:
            pitch_dev = self._calc_pitch_deviation(current_direction, self.target_direction)
            yaw_dev = self._calc_yaw_deviation(current_direction, self.target_direction)
            roll_dev = 0
            self.angle_panel.update_angles(pitch_dev, yaw_dev, roll_dev)

    def _calc_pitch_deviation(self, current, target):
        """计算俯仰偏差（度）"""
        cur_xy_mag = np.sqrt(current[0] ** 2 + current[1] ** 2)
        tgt_xy_mag = np.sqrt(target[0] ** 2 + target[1] ** 2)

        if cur_xy_mag < 0.01 and tgt_xy_mag < 0.01:
            return 0.0

        cur_pitch = np.degrees(np.arctan2(current[2], cur_xy_mag))
        tgt_pitch = np.degrees(np.arctan2(target[2], tgt_xy_mag))
        return cur_pitch - tgt_pitch

    def _calc_yaw_deviation(self, current, target):
        """计算偏航偏差（度）"""
        cur_xy_mag = np.sqrt(current[0] ** 2 + current[1] ** 2)
        tgt_xy_mag = np.sqrt(target[0] ** 2 + target[1] ** 2)

        if cur_xy_mag < 0.01 and tgt_xy_mag < 0.01:
            return 0.0
        elif cur_xy_mag < 0.01:
            return 0.0
        elif tgt_xy_mag < 0.01:
            return np.degrees(np.arctan2(current[1], current[0]))

        cur_yaw = np.degrees(np.arctan2(current[1], current[0]))
        tgt_yaw = np.degrees(np.arctan2(target[1], target[0]))
        diff = cur_yaw - tgt_yaw

        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def clear_target(self):
        """清除目标方向（锁定的目标虚线）"""
        self.target_direction = None
