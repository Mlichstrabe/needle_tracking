"""Shared UI helpers for PyQt panels."""
from PyQt5.QtWidgets import QLabel, QScrollArea, QWidget, QFrame, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor


def set_button_variant(button, variant):
    button.setProperty("variant", variant)
    button.style().unpolish(button)
    button.style().polish(button)


def set_label_role(label, role):
    label.setProperty("role", role)
    label.style().unpolish(label)
    label.style().polish(label)


def make_step_label(text):
    label = QLabel(text)
    label.setObjectName("StepLabel")
    return label


def make_section_title(text):
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def make_hint_label(text):
    label = QLabel(text)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return label


def apply_panel_chrome(widget: QWidget, bg="#101722") -> None:
    """统一控件/子控件窗口底色，避免 Fusion 默认白底。"""
    widget.setAutoFillBackground(True)
    pal = widget.palette()
    pal.setColor(QPalette.Window, QColor(bg))
    pal.setColor(QPalette.Base, QColor("#0c121c"))
    widget.setPalette(pal)
    if isinstance(widget, QFrame):
        widget.setAttribute(Qt.WA_StyledBackground, True)


def configure_side_scroll(scroll: QScrollArea, content: QWidget) -> None:
    """侧栏滚动区：顶对齐，避免内容少时沉底。"""
    scroll.setObjectName("SideScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    scroll.setAutoFillBackground(False)
    apply_panel_chrome(scroll, "#101722")

    viewport = scroll.viewport()
    viewport.setObjectName("SideScrollViewport")
    apply_panel_chrome(viewport, "#101722")

    content.setObjectName("SidePanelContent")
    content.setSizePolicy(content.sizePolicy().horizontalPolicy(), QSizePolicy.Minimum)
    apply_panel_chrome(content, "#101722")
    scroll.setWidget(content)
