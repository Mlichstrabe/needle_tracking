"""右侧对准 HUD：圆盘 + 大角度读数 + 状态。"""
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

from ui.widgets.panels import GuidanceArrowWidget


class AlignmentHudPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HudPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("② 对准 Target")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.guidance_widget = GuidanceArrowWidget()
        self.guidance_widget.setMinimumSize(168, 168)
        self.guidance_widget.setMaximumSize(220, 220)
        layout.addWidget(self.guidance_widget, alignment=Qt.AlignHCenter)

        self.angle_label = QLabel("--")
        self.angle_label.setObjectName("HudAngle")
        self.angle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.angle_label)

        self.status_label = QLabel("连接设备并开始对准后显示偏差")
        self.status_label.setObjectName("AlignmentStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_angle(self, angle_deg):
        self.angle_label.setText(f"{angle_deg:.1f}°")
        if angle_deg < 2.0:
            tone = "ok"
        elif angle_deg < 5.0:
            tone = "warn"
        else:
            tone = "danger"
        self.angle_label.setProperty("hudTone", tone)
        self.angle_label.style().unpolish(self.angle_label)
        self.angle_label.style().polish(self.angle_label)

    def set_status(self, status):
        self.status_label.setText(status)
        if "完美" in status or "已对齐" in status:
            state = "ok"
        elif "偏离" in status:
            state = "warn" if "°" in status else "danger"
        else:
            state = "danger"

        self.status_label.setProperty("status", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def hide_guidance(self):
        self.guidance_widget.hide_guidance()
        self.angle_label.setText("--")
        self.angle_label.setProperty("hudTone", "idle")
        self.angle_label.style().unpolish(self.angle_label)
        self.angle_label.style().polish(self.angle_label)

    def set_guidance(self, correction_3d, angle_deg):
        self.guidance_widget.set_guidance(correction_3d, angle_deg)
        self.set_angle(angle_deg)
