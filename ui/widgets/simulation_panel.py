"""穿刺路径引导 — 引导轨第 3 步。"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

from ui.widgets.ui_helpers import set_button_variant, set_label_role, make_hint_label


class OrientationLockWidget(QWidget):
    lock_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_locked = False
        self.locked_direction = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lock_btn = QPushButton("锁定当前姿态")
        self.lock_btn.setShortcut(Qt.Key_Space)
        set_button_variant(self.lock_btn, "secondary")
        self.lock_btn.clicked.connect(self._toggle_lock)
        layout.addWidget(self.lock_btn)

        self.status_label = QLabel("未锁定")
        set_label_role(self.status_label, "muted")
        layout.addWidget(self.status_label)

        layout.addWidget(make_hint_label("满意后按空格或点击按钮锁定为目标路径。"))

    def _toggle_lock(self):
        self.is_locked = not self.is_locked
        if self.is_locked:
            self.lock_btn.setText("解除锁定")
            set_button_variant(self.lock_btn, "danger")
            self.status_label.setText("已锁定")
            set_label_role(self.status_label, "ok")
        else:
            self.lock_btn.setText("锁定当前姿态")
            set_button_variant(self.lock_btn, "secondary")
            self.status_label.setText("未锁定")
            set_label_role(self.status_label, "muted")
            self.locked_direction = None
        self.lock_toggled.emit(self.is_locked)


class SimulationPanel(QFrame):
    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()
    orientation_locked = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowCard")
        self.is_simulation_active = False
        self.current_target_direction = None
        self.current_needle_direction = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("③ 路径引导")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.toggle_btn = QPushButton("启动引导模式")
        set_button_variant(self.toggle_btn, "primary")
        self.toggle_btn.clicked.connect(self._toggle_simulation)
        layout.addWidget(self.toggle_btn)

        self.lock_widget = OrientationLockWidget()
        self.lock_widget.lock_toggled.connect(self._on_lock_toggled)
        layout.addWidget(self.lock_widget)

    def _toggle_simulation(self):
        self.is_simulation_active = not self.is_simulation_active
        if self.is_simulation_active:
            self.toggle_btn.setText("停止引导模式")
            set_button_variant(self.toggle_btn, "danger")
            self.simulation_started.emit()
        else:
            self.toggle_btn.setText("启动引导模式")
            set_button_variant(self.toggle_btn, "primary")
            if self.lock_widget.is_locked:
                self.lock_widget._toggle_lock()
            self.simulation_stopped.emit()

    def _on_lock_toggled(self, is_locked):
        if is_locked and self.current_needle_direction is not None:
            self.current_target_direction = self.current_needle_direction.copy()
            self.orientation_locked.emit(self.current_target_direction)

    def update_current_direction(self, direction):
        self.current_needle_direction = np.array(direction)
