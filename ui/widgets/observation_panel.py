"""姿态观察模式的常显操作区。"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout


class ObservationPanel(QFrame):
    toggle_translation = pyqtSignal()
    zero_requested = pyqtSignal()
    recording_toggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("锁轴平移")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.state_label = QLabel("等待连接 IMU；按 Space 后启用测距")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.measured_label = QLabel("— mm")
        self.applied_label = QLabel("— mm")
        self.angle_label = QLabel("—°")
        for label in (self.measured_label, self.applied_label, self.angle_label):
            label.setProperty("role", "value-sm")
        grid.addWidget(QLabel("测得位移"), 0, 0)
        grid.addWidget(self.measured_label, 0, 1)
        grid.addWidget(QLabel("已应用位移"), 1, 0)
        grid.addWidget(self.applied_label, 1, 1)
        grid.addWidget(QLabel("锁定轴夹角"), 2, 0)
        grid.addWidget(self.angle_label, 2, 1)
        layout.addLayout(grid)

        self.zero_button = QPushButton("记录测距零点")
        self.zero_button.clicked.connect(self.zero_requested.emit)
        layout.addWidget(self.zero_button)
        self.toggle_button = QPushButton("按 Space 进入平移")
        self.toggle_button.setProperty("variant", "primary")
        self.toggle_button.clicked.connect(self.toggle_translation.emit)
        layout.addWidget(self.toggle_button)
        self.action_feedback_label = QLabel("按空格后会显示是否已锁定及当前阻断原因")
        self.action_feedback_label.setWordWrap(True)
        self.action_feedback_label.setProperty("role", "muted")
        layout.addWidget(self.action_feedback_label)

        self.record_button = QPushButton("开始记录姿态异常")
        self.record_button.setFocusPolicy(Qt.NoFocus)
        self.record_button.setToolTip("保存按下前 10 秒及之后的 IMU 与界面针轴数据")
        self.record_button.clicked.connect(self.recording_toggled.emit)
        layout.addWidget(self.record_button)
        self.record_status_label = QLabel("出现异常时点击；再次点击结束并保存，最长 120 秒")
        self.record_status_label.setWordWrap(True)
        self.record_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.record_status_label.setProperty("role", "muted")
        layout.addWidget(self.record_status_label)

    def update_state(self, state, reason, measured, applied, angle, can_zero=False, locked=False):
        self.state_label.setText(reason or state)
        self.measured_label.setText("— mm" if measured is None else f"{measured:.1f} mm")
        self.applied_label.setText("— mm" if applied is None else f"{applied:.1f} mm")
        self.angle_label.setText("—°" if angle is None else f"{angle:.1f}°")
        self.zero_button.setEnabled(bool(can_zero))
        self.toggle_button.setText("按 Space 退出平移" if locked else "按 Space 进入平移")
        # 始终允许点击，让用户看到当前阻断原因；是否能真正进入由控制器判断。
        self.toggle_button.setEnabled(True)

    def report_action(self, message):
        self.action_feedback_label.setText(message)

    def set_recording_state(self, active, message):
        self.record_button.setText("结束并保存姿态记录" if active else "开始记录姿态异常")
        self.record_status_label.setText(message)
