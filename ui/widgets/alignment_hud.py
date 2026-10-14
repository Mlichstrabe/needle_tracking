"""右侧对准 HUD：紧凑圆盘 + 状态文案。"""
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt

from ui.widgets.panels import GuidanceArrowWidget


class AlignmentHudPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HudPanel")
        # The HUD shares the fixed real-time area with the range card.  Do not
        # let it consume the extra height needed by the scrollable controls.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(200)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("② 对准 Target")
        title.setObjectName("SectionTitle")
        self.title_label = title
        layout.addWidget(title)

        guide_row = QHBoxLayout()
        guide_row.setSpacing(12)
        self.guidance_widget = GuidanceArrowWidget()
        self.guidance_widget.setFixedSize(104, 104)
        guide_row.addWidget(self.guidance_widget)

        value_column = QVBoxLayout()
        value_column.setSpacing(2)
        value_column.addStretch()
        self.angle_label = QLabel("—°")
        self.angle_label.setObjectName("HudAngle")
        self.angle_label.setProperty("hudTone", "idle")
        self.angle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value_column.addWidget(self.angle_label)

        angle_caption = QLabel("方向偏差")
        angle_caption.setProperty("role", "muted")
        value_column.addWidget(angle_caption)
        value_column.addStretch()
        guide_row.addLayout(value_column, 1)
        layout.addLayout(guide_row)

        self.status_label = QLabel("连接设备并开始对准后显示偏差")
        self.status_label.setObjectName("AlignmentStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_status(self, status):
        self.status_label.setText(status)
        if "完美" in status:
            state = "ok"
        elif "已对齐" in status:
            state = "ok"
        elif "需调整" in status:
            state = "warn"
        else:
            state = ""

        self.status_label.setProperty("status", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def hide_guidance(self):
        self.guidance_widget.hide_guidance()
        self._set_angle(None)
        if getattr(self, "_observe_mode", False):
            self.status_label.setText("观察模式 — 针尖固定在原点，仅显示姿态")
        else:
            self.status_label.setText("连接设备并开始对准后显示偏差")
        self.status_label.setProperty("status", "")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_observe_mode(self, enabled: bool, connected: bool = False):
        """观察模式：隐藏对准罗盘，显示姿态观察提示。"""
        self._observe_mode = enabled
        if enabled:
            self.title_label.setText("姿态观察")
            self.guidance_widget.hide_guidance()
            if connected:
                self.status_label.setText("已连接 · 针尖在原点，转动探针观察姿态")
            else:
                self.status_label.setText("观察模式 — 连接 IMU 后针尖固定在原点")
            self.status_label.setProperty("status", "ok" if connected else "")
        else:
            self.title_label.setText("② 对准 Target")
            self.hide_guidance()
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_guidance(self, correction_3d, angle_deg):
        self.guidance_widget.set_guidance(correction_3d, angle_deg)
        self._set_angle(angle_deg)

    def _set_angle(self, angle_deg):
        if angle_deg is None:
            self.angle_label.setText("—°")
            tone = "idle"
        else:
            self.angle_label.setText(f"{float(angle_deg):.1f}°")
            if angle_deg <= 5.0:
                tone = "ok"
            elif angle_deg <= 8.0:
                tone = "warn"
            else:
                tone = "danger"
        self.angle_label.setProperty("hudTone", tone)
        self.angle_label.style().unpolish(self.angle_label)
        self.angle_label.style().polish(self.angle_label)
