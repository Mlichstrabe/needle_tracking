"""左侧准备坞：深色可折叠分区（替代 QToolBox，避免白底）。"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt

from ui.widgets.ui_helpers import apply_panel_chrome


class PrepSection(QFrame):
    """单块可折叠准备区。"""

    def __init__(self, title, body: QWidget, expanded=True, parent=None):
        super().__init__(parent)
        self.setObjectName("PrepSection")
        apply_panel_chrome(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._base_title = title
        self._header = QPushButton()
        self._header.setObjectName("PrepSectionHeader")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.toggled.connect(self._on_toggle)
        layout.addWidget(self._header)

        self._body_wrap = QFrame()
        self._body_wrap.setObjectName("PrepSectionBody")
        apply_panel_chrome(self._body_wrap)
        body_layout = QVBoxLayout(self._body_wrap)
        body_layout.setContentsMargins(10, 8, 10, 10)
        body_layout.setSpacing(8)
        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        body_layout.addWidget(body)
        layout.addWidget(self._body_wrap)
        self._body_wrap.setVisible(expanded)
        self._update_header_text(expanded)

    def _on_toggle(self, checked):
        self._body_wrap.setVisible(checked)
        self._update_header_text(checked)

    def _update_header_text(self, expanded):
        arrow = "▾" if expanded else "▴"
        self._header.setText(f"{self._base_title}  {arrow}")

    def set_expanded(self, expanded):
        self._header.setChecked(expanded)


class PrepSidebar(QWidget):
    """设备 / 影像 / 遥测 三块手风琴（CT 居中）。"""

    SEC_DEVICE = 0
    SEC_CT = 1
    SEC_IMU = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PrepSidebar")
        apply_panel_chrome(self)

        self._sections = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)

    def add_section(self, title, widget, expanded=False):
        section = PrepSection(title, widget, expanded=expanded, parent=self)
        self._layout = self.layout()
        self._layout.addWidget(section)
        self._sections.append(section)
        return section

    def set_active_section(self, index):
        for i, section in enumerate(self._sections):
            section.set_expanded(i == index)
