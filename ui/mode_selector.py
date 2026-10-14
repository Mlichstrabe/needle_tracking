"""启动模式选择窗口。"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ModeSelector(QDialog):
    OBSERVE = "needle_observe"
    PUNCTURE = "puncture_training"
    mode_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择工作模式")
        self.setModal(False)
        self.setMinimumWidth(460)
        self.setObjectName("ModeSelector")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)
        title = QLabel("手术探针定位系统")
        title.setObjectName("AppTitle")
        layout.addWidget(title)
        subtitle = QLabel("请选择本次训练的工作模式")
        subtitle.setProperty("role", "muted")
        layout.addWidget(subtitle)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        observe = QPushButton("姿态观察\n查看针体姿态与锁轴平移")
        puncture = QPushButton("穿刺模拟\n固定训练 CT 与进针引导")
        for button in (observe, puncture):
            button.setMinimumHeight(84)
            button.setProperty("variant", "secondary")
        observe.clicked.connect(lambda: self._select(self.OBSERVE))
        puncture.clicked.connect(lambda: self._select(self.PUNCTURE))
        buttons.addWidget(observe)
        buttons.addWidget(puncture)
        layout.addLayout(buttons)

        hint = QLabel("两种模式共享 IMU、测距和三维针体显示；进入工作页后可返回重新选择。")
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")
        layout.addWidget(hint)

    def _select(self, mode):
        self.mode_selected.emit(mode)
        self.hide()
