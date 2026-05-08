"""穿刺控制面板"""
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QTextEdit)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class PuncturePanel(QGroupBox):
    """穿刺控制面板"""

    # 信号
    start_puncture = pyqtSignal()
    lock_attitude = pyqtSignal()
    end_puncture = pyqtSignal()
    threshold_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__("穿刺监测")
        self._init_ui()
        self._set_state('idle')

    def _init_ui(self):
        """初始化UI"""
        # 🔥 改为 self.layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(8, 12, 8, 8)

        # === 控制按钮 ===
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("开始穿刺")
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self.btn_start)

        self.btn_lock = QPushButton("锁定姿态")
        self.btn_lock.clicked.connect(self._on_lock_clicked)
        self.btn_lock.setEnabled(False)
        btn_layout.addWidget(self.btn_lock)

        self.btn_end = QPushButton("结束穿刺")
        self.btn_end.clicked.connect(self._on_end_clicked)
        self.btn_end.setEnabled(False)
        btn_layout.addWidget(self.btn_end)

        # 🔥 改为 self.layout
        self.layout.addLayout(btn_layout)

        # === 状态显示 ===
        self.lbl_status = QLabel("状态: 待机")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.lbl_status.setFont(font)
        # 🔥 改为 self.layout
        self.layout.addWidget(self.lbl_status)

        # === 参考姿态显示 ===
        ref_layout = QHBoxLayout()
        ref_title = QLabel("参考:")
        ref_title.setStyleSheet("font-weight: bold; color: #4a90e2;")
        ref_layout.addWidget(ref_title)

        self.lbl_ref_attitude = QLabel("R:-- P:-- Y:--")
        ref_layout.addWidget(self.lbl_ref_attitude)
        ref_layout.addStretch()
        # 🔥 改为 self.layout
        self.layout.addLayout(ref_layout)

        # === 偏离显示 ===
        self.lbl_deviation = QLabel("当前偏离: 0.0°")
        self.lbl_deviation.setStyleSheet("color: green; font-size: 12pt;")
        font_dev = QFont()
        font_dev.setBold(True)
        self.lbl_deviation.setFont(font_dev)
        # 🔥 改为 self.layout
        self.layout.addWidget(self.lbl_deviation)

        # === 纠偏建议 ===
        correction_title = QLabel("【纠偏建议】")
        correction_title.setStyleSheet("font-weight: bold; color: #e74c3c;")
        # 🔥 改为 self.layout
        self.layout.addWidget(correction_title)

        self.txt_correction = QTextEdit()
        self.txt_correction.setReadOnly(True)
        self.txt_correction.setMaximumHeight(80)
        self.txt_correction.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 6px;
                font-size: 11pt;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
            }
        """)
        self.txt_correction.setText("等待锁定...")
        # 🔥 改为 self.layout
        self.layout.addWidget(self.txt_correction)

        # === 阈值设置 ===
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("偏离阈值:"))

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 10)
        self.spin_threshold.setValue(3)
        self.spin_threshold.setSuffix("°")
        self.spin_threshold.valueChanged.connect(
            lambda v: self.threshold_changed.emit(float(v))
        )
        threshold_layout.addWidget(self.spin_threshold)
        threshold_layout.addStretch()

        # 🔥 改为 self.layout
        self.layout.addLayout(threshold_layout)

    def _on_start_clicked(self):
        """开始穿刺按钮"""
        self.start_puncture.emit()
        self._set_state('puncturing_unlocked')

    def _on_lock_clicked(self):
        """锁定姿态按钮"""
        self.lock_attitude.emit()

    def _on_end_clicked(self):
        """结束穿刺按钮"""
        self.end_puncture.emit()
        self._reset_all()

    def _set_state(self, state):
        """设置界面状态"""
        if state == 'idle':
            self.btn_start.setEnabled(True)
            self.btn_lock.setEnabled(False)
            self.btn_end.setEnabled(False)
            self.lbl_status.setText("状态: 待机")
            self.lbl_status.setStyleSheet("color: gray;")

        elif state == 'puncturing_unlocked':
            self.btn_start.setEnabled(False)
            self.btn_lock.setEnabled(True)
            self.btn_end.setEnabled(True)
            self.lbl_status.setText("状态: 等待锁定")
            self.lbl_status.setStyleSheet("color: orange;")

        elif state == 'puncturing_locked':
            self.btn_start.setEnabled(False)
            self.btn_lock.setEnabled(False)
            self.btn_end.setEnabled(True)
            self.lbl_status.setText("状态: 监测中")
            self.lbl_status.setStyleSheet("color: blue;")

    def _reset_all(self):
        """重置所有状态"""
        self.btn_lock.setText("锁定姿态")
        self.btn_lock.setStyleSheet("")
        self.lbl_ref_attitude.setText("R:-- P:-- Y:--")
        self.lbl_deviation.setText("当前偏离: 0.0°")
        self.lbl_deviation.setStyleSheet("color: green; font-size: 12pt;")
        self.txt_correction.setText("等待锁定...")
        self._set_state('idle')

    def set_locked(self, locked):
        """设置锁定状态"""
        if locked:
            self._set_state('puncturing_locked')
            self.btn_lock.setText("✓ 已锁定")
            self.btn_lock.setStyleSheet("background-color: #4CAF50; color: white;")

    def update_reference_attitude(self, roll, pitch, yaw):
        """更新参考姿态显示"""
        self.lbl_ref_attitude.setText(
            f"R:{roll:5.1f}° P:{pitch:5.1f}° Y:{yaw:5.1f}°"
        )

    def update_deviation(self, angle, is_deviated):
        """更新偏离显示"""
        self.lbl_deviation.setText(f"当前偏离: {angle:.2f}°")

        if is_deviated:
            self.lbl_deviation.setStyleSheet(
                "color: red; font-weight: bold; font-size: 12pt;"
            )
        else:
            self.lbl_deviation.setStyleSheet(
                "color: green; font-size: 12pt;"
            )

    def update_correction(self, suggestions):
        """更新纠偏建议

        Args:
            suggestions: 建议列表，例如 ["↶ 向左转 5.2°", "↑ 抬高 3.3°"]
        """
        if not suggestions:
            self.txt_correction.setText("✓ 姿态良好，保持当前方向")
            self.txt_correction.setStyleSheet("""
                QTextEdit {
                    background-color: #27ae60;
                    color: white;
                    border: 1px solid #229954;
                    border-radius: 4px;
                    padding: 6px;
                    font-size: 11pt;
                    font-family: "Microsoft YaHei", "SimHei", sans-serif;
                }
            """)
        else:
            text = "\n".join(suggestions)
            self.txt_correction.setText(text)
            self.txt_correction.setStyleSheet("""
                QTextEdit {
                    background-color: #e74c3c;
                    color: white;
                    border: 1px solid #c0392b;
                    border-radius: 4px;
                    padding: 6px;
                    font-size: 11pt;
                    font-weight: bold;
                    font-family: "Microsoft YaHei", "SimHei", sans-serif;
                }
            """)

    def set_alignment_error(self, angle_deg):
        """设置偏离角度"""
        # 🔥 如果没有对应的Label，先创建
        if not hasattr(self, 'alignment_error_label'):
            self.alignment_error_label = QLabel("偏离角度: --")
            self.alignment_error_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 5px;
                    background: #34495e;
                    color: white;
                    border-radius: 3px;
                }
            """)
            # 🔥 添加到布局中（假设你有一个主布局）
            if hasattr(self, 'layout'):
                self.layout.addWidget(self.alignment_error_label)

        # 更新文本
        self.alignment_error_label.setText(f"偏离角度: {angle_deg:.1f}°")

    def set_alignment_status(self, status):
        """设置对齐状态"""
        # 🔥 如果没有对应的Label，先创建
        if not hasattr(self, 'alignment_status_label'):
            self.alignment_status_label = QLabel("等待连接...")
            self.alignment_status_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    background: #95a5a6;
                    color: white;
                }
            """)
            # 🔥 添加到布局中
            if hasattr(self, 'layout'):
                self.layout.addWidget(self.alignment_status_label)

        # 更新文本
        self.alignment_status_label.setText(status)

        # 🔥 根据状态改变颜色
        if "已对齐" in status:
            self.alignment_status_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    background: #2ecc71;
                    color: white;
                }
            """)
        elif "接近对齐" in status:
            self.alignment_status_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    background: #f39c12;
                    color: white;
                }
            """)
        else:
            self.alignment_status_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    background: #e74c3c;
                    color: white;
                }
            """)
