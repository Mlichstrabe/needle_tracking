"""穿刺模拟面板 - 重构版（路径引导模式）"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFrame, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import numpy as np





class OrientationLockWidget(QGroupBox):
    """姿态锁定控件"""

    lock_toggled = pyqtSignal(bool)  # True=锁定, False=解锁

    def __init__(self, parent=None):
        super().__init__("姿态锁定", parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #5af;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        self.is_locked = False
        self.locked_direction = None

        layout = QVBoxLayout(self)

        # ====== 锁定按钮 ======
        self.lock_btn = QPushButton("🔒 锁定当前姿态")
        self.lock_btn.setShortcut(Qt.Key_Space)
        self.lock_btn.setStyleSheet("""
            QPushButton {
                background: #2a5;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #3b6;
            }
            QPushButton:pressed {
                background: #1a4;
            }
        """)
        self.lock_btn.clicked.connect(self._toggle_lock)
        layout.addWidget(self.lock_btn)

        # ====== 状态显示 ======
        self.status_label = QLabel("当前状态: 未锁定")
        self.status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.status_label)

        # ====== 提示文字 ======
        hint = QLabel(
            "💡 提示: 调整到满意的姿态后，\n"
            "   点击按钮（或按空格键）锁定"
        )
        hint.setStyleSheet("color: #888; font-size: 9px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _toggle_lock(self):
        """切换锁定状态"""
        self.is_locked = not self.is_locked

        if self.is_locked:
            self.lock_btn.setText("🔓 解除锁定")
            self.lock_btn.setStyleSheet("""
                QPushButton {
                    background: #d55;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background: #e66;
                }
            """)
            self.status_label.setText("当前状态: 已锁定 ✓")
            self.status_label.setStyleSheet("color: #5f5; font-size: 10px;")
        else:
            self.lock_btn.setText("🔒 锁定当前姿态")
            self.lock_btn.setStyleSheet("""
                QPushButton {
                    background: #2a5;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background: #3b6;
                }
            """)
            self.status_label.setText("当前状态: 未锁定")
            self.status_label.setStyleSheet("color: #aaa; font-size: 10px;")
            self.locked_direction = None

        self.lock_toggled.emit(self.is_locked)

    def lock_current_direction(self, direction):
        """锁定当前方向"""
        self.locked_direction = np.array(direction).copy()
        self.is_locked = True
        self._toggle_lock()  # 更新UI状态


class DeviationDisplay(QFrame):
    """偏差显示面板（复用并增强 AngleIndicatorPanel）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #1a1a2e; border-radius: 6px; padding: 8px;")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ====== 标题 ======
        title = QLabel("📊 角度偏差监控")
        title.setStyleSheet("color: #5af; font-weight: bold; font-size: 11px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ====== 总偏差显示 ======
        self.total_label = QLabel("总偏差: --")
        self.total_label.setStyleSheet(
            "color: #fff; font-size: 14px; font-weight: bold;"
        )
        self.total_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.total_label)

        # ====== 分项偏差 ======
        self.pitch_label = QLabel("俯仰偏差: --")
        self.yaw_label = QLabel("偏航偏差: --")

        for label in [self.pitch_label, self.yaw_label]:
            label.setStyleSheet("color: #aaa; font-size: 10px;")
            layout.addWidget(label)

        layout.addStretch()

        # ====== 调整建议 ======
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet(
            "color: #5af; font-size: 10px; background: #222; "
            "padding: 6px; border-radius: 4px;"
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.hint_label)

    def update_deviation(self, total, pitch_dev, yaw_dev):
        """更新偏差显示

        Args:
            total: 总偏差角度（度）
            pitch_dev: 俯仰偏差（度，正=需向下调整）
            yaw_dev: 偏航偏差（度，正=需向右调整）
        """
        # ====== 总偏差 ======
        if total < 3:
            color = "#5f5"
            status = "✓ 优秀"
        elif total < 8:
            color = "#ff5"
            status = "△ 良好"
        else:
            color = "#f55"
            status = "✗ 需调整"

        self.total_label.setText(f"总偏差: {total:.1f}° {status}")
        self.total_label.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: bold;"
        )

        # ====== 分项偏差 ======
        self._update_axis_label(self.pitch_label, "俯仰", pitch_dev)
        self._update_axis_label(self.yaw_label, "偏航", yaw_dev)

        # ====== 调整建议 ======
        hints = []
        if abs(pitch_dev) > 2:
            direction = "向下" if pitch_dev > 0 else "向上"
            hints.append(f"• {direction}调整 {abs(pitch_dev):.1f}°")

        if abs(yaw_dev) > 2:
            direction = "向右" if yaw_dev > 0 else "向左"
            hints.append(f"• {direction}调整 {abs(yaw_dev):.1f}°")

        if hints:
            self.hint_label.setText("🎯 调整建议:\n" + "\n".join(hints))
        else:
            self.hint_label.setText("✓ 角度已对齐，可以开始穿刺")

    def _update_axis_label(self, label, name, deviation):
        """更新单个轴标签"""
        if abs(deviation) < 2:
            color = "#5f5"
            symbol = "✓"
        elif abs(deviation) < 5:
            color = "#ff5"
            symbol = "△"
        else:
            color = "#f55"
            symbol = "✗"

        label.setText(f"{name}偏差: {deviation:+.1f}° {symbol}")
        label.setStyleSheet(f"color: {color}; font-size: 10px;")


class SimulationPanel(QFrame):
    """穿刺模拟主面板（重构版）"""

    # ====== 信号定义 ======
    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()
    target_direction_changed = pyqtSignal(np.ndarray)  # 目标方向改变
    orientation_locked = pyqtSignal(np.ndarray)        # 姿态锁定

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

        # ====== 状态变量 ======
        self.is_simulation_active = False
        self.current_target_direction = None
        self.current_needle_direction = None

        # ====== 初始化UI ======
        self._init_ui()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ====== 标题 ======
        title = QLabel("🎯 穿刺路径引导")
        title.setStyleSheet(
            "color: #5af; font-size: 14px; font-weight: bold;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ====== 模拟开关 ======
        self.toggle_btn = QPushButton("▶ 启动引导模式")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: #2a5;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #3b6;
            }
            QPushButton:pressed {
                background: #1a4;
            }
        """)
        self.toggle_btn.clicked.connect(self._toggle_simulation)
        layout.addWidget(self.toggle_btn)


        # ====== 姿态锁定控件 ======
        self.lock_widget = OrientationLockWidget()
        self.lock_widget.lock_toggled.connect(self._on_lock_toggled)
        layout.addWidget(self.lock_widget)

        # ====== 偏差显示 ======
        self.deviation_display = DeviationDisplay()
        layout.addWidget(self.deviation_display)

        layout.addStretch()

    def _toggle_simulation(self):
        """切换模拟状态"""
        self.is_simulation_active = not self.is_simulation_active

        if self.is_simulation_active:
            # ====== 启动引导 ======
            self.toggle_btn.setText("■ 停止引导模式")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background: #d55;
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 10px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background: #e66;
                }
            """)

            # 🔥 简化：只发射启动信号
            self.simulation_started.emit()
            print("[Simulation] 引导已启动")

        else:
            # ====== 停止引导 ======
            self.toggle_btn.setText("▶ 启动引导模式")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background: #2a5;
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 10px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background: #3b6;
                }
            """)

            # 🔥 停止时解锁（如果已锁定）
            if self.lock_widget.is_locked:
                self.lock_widget._toggle_lock()

            self.simulation_stopped.emit()
            print("[Simulation] 引导已停止")


    def _on_lock_toggled(self, is_locked):
        """姿态锁定状态改变"""
        if is_locked and self.current_needle_direction is not None:
            # 锁定当前姿态
            self.current_target_direction = self.current_needle_direction.copy()
            self.orientation_locked.emit(self.current_target_direction)
            print(f"[Simulation] 姿态已锁定: {self.current_target_direction}")

    def update_current_direction(self, direction):
        """更新当前针体方向（由主窗口调用）"""
        self.current_needle_direction = np.array(direction)

        if not self.is_simulation_active or self.current_target_direction is None:
            return

        # ====== 计算偏差 ======
        total, pitch, yaw = self._calculate_deviation(
            self.current_needle_direction,
            self.current_target_direction
        )

        # 更新显示
        self.deviation_display.update_deviation(total, pitch, yaw)

    def _calculate_deviation(self, current, target):
        """计算偏差角度

        Returns:
            (total, pitch_dev, yaw_dev): 总偏差、俯仰偏差、偏航偏差
        """
        # ====== 总偏差（三维夹角） ======
        cos_angle = np.dot(current, target)
        cos_angle = np.clip(cos_angle, -1, 1)
        total = np.degrees(np.arccos(cos_angle))

        # ====== 俯仰偏差 ======
        cur_pitch = np.degrees(np.arctan2(
            current[2],
            np.sqrt(current[0]**2 + current[1]**2)
        ))
        tgt_pitch = np.degrees(np.arctan2(
            target[2],
            np.sqrt(target[0]**2 + target[1]**2)
        ))
        pitch_dev = cur_pitch - tgt_pitch

        # ====== 偏航偏差 ======
        cur_yaw = np.degrees(np.arctan2(current[1], current[0]))
        tgt_yaw = np.degrees(np.arctan2(target[1], target[0]))
        yaw_dev = cur_yaw - tgt_yaw

        # 归一化到 -180~180
        while yaw_dev > 180:
            yaw_dev -= 360
        while yaw_dev < -180:
            yaw_dev += 360

        return total, pitch_dev, yaw_dev
