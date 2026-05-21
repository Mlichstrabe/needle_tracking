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

            #  简化：只发射启动信号
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

            #  停止时解锁（如果已锁定）
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
