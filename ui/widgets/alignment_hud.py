"""右侧对准 HUD：紧凑圆盘 + 状态文案。"""
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

from ui.widgets.panels import GuidanceArrowWidget


class AlignmentHudPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HudPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("② 对准 Target")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.guidance_widget = GuidanceArrowWidget()
        self.guidance_widget.setMinimumSize(104, 104)
        self.guidance_widget.setMaximumSize(152, 152)
        layout.addWidget(self.guidance_widget, alignment=Qt.AlignHCenter)

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
        self.status_label.setText("连接设备并开始对准后显示偏差")
        self.status_label.setProperty("status", "")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_guidance(self, correction_3d, angle_deg):
        self.guidance_widget.set_guidance(correction_3d, angle_deg)
