"""穿刺模拟面板 - 重构版（路径引导模式）"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np


def _set_button_variant(button, variant):
    button.setProperty("variant", variant)
    button.style().unpolish(button)
    button.style().polish(button)


class OrientationLockWidget(QGroupBox):
    """姿态锁定控件"""

    lock_toggled = pyqtSignal(bool)  # True=锁定, False=解锁

    def __init__(self, parent=None):
        super().__init__("姿态锁定", parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                background: #151d2a;
                border: 1px solid #2a3a50;
                border-radius: 7px;
                margin-top: 10px;
                padding: 12px 8px 8px 8px;
            }
            QGroupBox::title {
                color: #70d6ff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #111824;
            }
        """)

        self.is_locked = False
        self.locked_direction = None

        layout = QVBoxLayout(self)

        # ====== 锁定按钮 ======
        self.lock_btn = QPushButton("锁定当前姿态")
        self.lock_btn.setShortcut(Qt.Key_Space)
        _set_button_variant(self.lock_btn, "primary")
        self.lock_btn.clicked.connect(self._toggle_lock)
        layout.addWidget(self.lock_btn)

        # ====== 状态显示 ======
        self.status_label = QLabel("当前状态: 未锁定")
        self.status_label.setStyleSheet("color: #8090a3; font-size: 11px;")
        layout.addWidget(self.status_label)

        # ====== 提示文字 ======
        hint = QLabel(
            "调整到满意的姿态后，点击按钮（或按空格键）锁定。"
        )
        hint.setStyleSheet("color: #8090a3; font-size: 10px; line-height: 140%;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _toggle_lock(self):
        """切换锁定状态"""
        self.is_locked = not self.is_locked

        if self.is_locked:
            self.lock_btn.setText("解除锁定")
            _set_button_variant(self.lock_btn, "danger")
            self.status_label.setText("当前状态: 已锁定")
            self.status_label.setStyleSheet("color: #58d68d; font-size: 11px; font-weight: 700;")
        else:
            self.lock_btn.setText("锁定当前姿态")
            _set_button_variant(self.lock_btn, "primary")
            self.status_label.setText("当前状态: 未锁定")
            self.status_label.setStyleSheet("color: #8090a3; font-size: 11px;")
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
    orientation_locked = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #151d2a;
                border: 1px solid #2a3a50;
                border-radius: 7px;
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ====== 标题 ======
        title = QLabel("穿刺路径引导")
        title.setStyleSheet(
            "color: #70d6ff; font-size: 13px; font-weight: bold;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ====== 模拟开关 ======
        self.toggle_btn = QPushButton("启动引导模式")
        _set_button_variant(self.toggle_btn, "primary")
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
            self.toggle_btn.setText("停止引导模式")
            _set_button_variant(self.toggle_btn, "danger")

            #  简化：只发射启动信号
            self.simulation_started.emit()
            print("[Simulation] 引导已启动")

        else:
            # ====== 停止引导 ======
            self.toggle_btn.setText("启动引导模式")
            _set_button_variant(self.toggle_btn, "primary")

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
