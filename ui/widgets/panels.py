"""侧边栏 UI 组件"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox,
    QFrame, QSizePolicy, QProgressBar, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
import math
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QRect, QPointF


_IMU_CARD_STYLE = """
    QFrame {
        background: #151f2d;
        border: 1px solid #2a3a50;
        border-radius: 6px;
    }
"""

_STATUS_CARD_STYLE = """
    QFrame {
        background: #101824;
        border: 1px solid #26374d;
        border-radius: 6px;
    }
"""

_VALUE_FONT = "font-family: 'Cascadia Mono', 'Consolas', monospace;"


def _set_button_variant(button, variant):
    button.setProperty("variant", variant)
    button.style().unpolish(button)
    button.style().polish(button)


class IMUDataPanel(QGroupBox):
    """IMU 数据面板（紧凑卡片布局，适配高 DPI）"""

    reset_view_clicked = pyqtSignal()
    clear_trajectory_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("IMU 数据", parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(9)
        layout.setContentsMargins(6, 14, 6, 6)
        layout.setContentsMargins(6, 14, 6, 6)

        # 顶部状态条
        status_frame = QFrame()
        status_frame.setStyleSheet(_STATUS_CARD_STYLE)
        status_row = QHBoxLayout(status_frame)
        status_row.setContentsMargins(10, 7, 10, 7)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #e57373; font-size: 11px;")
        status_row.addWidget(self.status_dot)

        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: #cfd8dc; font-size: 11px; font-weight: bold;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.fps_label = QLabel("-- Hz")
        self.fps_label.setStyleSheet(
            f"color: #8090a3; font-size: 10px; {_VALUE_FONT}"
        )
        status_row.addWidget(self.fps_label)
        layout.addWidget(status_frame)

        # 欧拉角（主读数）
        layout.addWidget(self._build_metric_card(
            "欧拉角",
            ["Roll", "Pitch", "Yaw"],
            "euler_labels",
            suffix="°",
            value_style=f"color: #45d7ff; font-size: 16px; font-weight: bold; {_VALUE_FONT}",
        ))

        # 四元数（次要，单行紧凑）
        quat_card = QFrame()
        quat_card.setStyleSheet(_IMU_CARD_STYLE)
        quat_outer = QVBoxLayout(quat_card)
        quat_outer.setContentsMargins(8, 6, 8, 6)
        quat_outer.setSpacing(4)
        quat_title = QLabel("四元数")
        quat_title.setStyleSheet("color: #70d6ff; font-size: 10px; font-weight: bold;")
        quat_outer.addWidget(quat_title)

        quat_row = QHBoxLayout()
        quat_row.setSpacing(6)
        self.quat_labels = []
        for name in ("W", "X", "Y", "Z"):
            cell = QVBoxLayout()
            cell.setSpacing(0)
            n = QLabel(name)
            n.setStyleSheet("color: #8090a3; font-size: 9px;")
            n.setAlignment(Qt.AlignCenter)
            v = QLabel("0.000")
            v.setStyleSheet(
                f"color: #b7d8ff; font-size: 11px; {_VALUE_FONT}"
            )
            v.setAlignment(Qt.AlignCenter)
            cell.addWidget(n)
            cell.addWidget(v)
            quat_row.addLayout(cell)
            self.quat_labels.append(v)
        quat_outer.addLayout(quat_row)
        layout.addWidget(quat_card)

        # 视图快捷操作
        tools = QHBoxLayout()
        tools.setSpacing(6)
        self.btn_reset_view = QPushButton("重置视角")
        _set_button_variant(self.btn_reset_view, "secondary")
        self.btn_reset_view.setToolTip("重置 3D 相机")
        self.btn_reset_view.clicked.connect(self.reset_view_clicked.emit)
        tools.addWidget(self.btn_reset_view)

        self.btn_clear = QPushButton("清除轨迹")
        _set_button_variant(self.btn_clear, "secondary")
        self.btn_clear.setToolTip("清除 3D 轨迹线")
        self.btn_clear.clicked.connect(self.clear_trajectory_clicked.emit)
        tools.addWidget(self.btn_clear)
        layout.addLayout(tools)

    def _build_metric_card(self, title, names, attr_name, suffix, value_style):
        card = QFrame()
        card.setStyleSheet(_IMU_CARD_STYLE)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        t = QLabel(title)
        t.setStyleSheet("color: #70d6ff; font-size: 10px; font-weight: bold;")
        outer.addWidget(t)

        grid = QGridLayout()
        grid.setSpacing(4)
        labels = []
        for i, name in enumerate(names):
            n = QLabel(name)
            n.setStyleSheet("color: #8090a3; font-size: 9px;")
            n.setAlignment(Qt.AlignCenter)
            grid.addWidget(n, 0, i)

            val = QLabel(f"0{suffix}")
            val.setStyleSheet(value_style)
            val.setAlignment(Qt.AlignCenter)
            grid.addWidget(val, 1, i)
            labels.append(val)

        setattr(self, attr_name, labels)
        outer.addLayout(grid)
        return card

    def update_quaternion(self, quaternion):
        """更新四元数显示

        Args:
            quaternion: [w, x, y, z] 四元数
        """
        if len(quaternion) >= 4:
            for i, val in enumerate(quaternion[:4]):
                self.quat_labels[i].setText(f"{val:.3f}")

    def update_euler(self, euler):
        if len(euler) >= 3:
            for i, val in enumerate(euler[:3]):
                self.euler_labels[i].setText(f"{val:.1f}°")

    def set_status(self, connected, fps=0):
        if connected:
            self.status_dot.setStyleSheet("color: #66bb6a; font-size: 11px;")
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet(
                "color: #a5d6a7; font-size: 11px; font-weight: bold;"
            )
        else:
            self.status_dot.setStyleSheet("color: #e57373; font-size: 11px;")
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet(
                "color: #cfd8dc; font-size: 11px; font-weight: bold;"
            )

        if fps > 0:
            self.fps_label.setText(f"{fps:.0f} Hz")
        else:
            self.fps_label.setText("-- Hz")


class DeviceConnectionPanel(QGroupBox):
    """设备连接面板"""

    connect_clicked = pyqtSignal(str, int)
    disconnect_clicked = pyqtSignal()
    calibration_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("设备连接", parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 连接状态
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        status_layout.addWidget(self.status_indicator)
        self.status_text = QLabel("未连接")
        self.status_text.setStyleSheet("color: #ff6b6b; font-weight: 700;")
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # 串口选择
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(120)
        port_layout.addWidget(self.port_combo)

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setToolTip("刷新串口列表")
        self.btn_refresh.setFixedWidth(32)
        self.btn_refresh.clicked.connect(self._refresh_ports)
        port_layout.addWidget(self.btn_refresh)

        layout.addLayout(port_layout)

        # 连接/断开按钮
        btn_layout = QHBoxLayout()
        self.btn_connect = QPushButton("连接设备")
        _set_button_variant(self.btn_connect, "primary")
        btn_layout.addWidget(self.btn_connect)

        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)
        _set_button_variant(self.btn_disconnect, "danger")
        btn_layout.addWidget(self.btn_disconnect)

        layout.addLayout(btn_layout)

        self.btn_calibrate = QPushButton("校准传感器")
        self.btn_calibrate.setToolTip("静止约 3 秒完成陀螺仪零偏与磁力计校准")
        _set_button_variant(self.btn_calibrate, "secondary")
        self.btn_calibrate.clicked.connect(self.calibration_clicked.emit)
        layout.addWidget(self.btn_calibrate)

        # 初始扫描串口
        self._refresh_ports()

    def _refresh_ports(self):
        """扫描可用串口"""
        self.port_combo.clear()
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in sorted(ports):
                label = f"{p.device} - {p.description}" if p.description else p.device
                self.port_combo.addItem(p.device, p.device)
            if self.port_combo.count() == 0:
                self.port_combo.addItem("(无可用串口)", "")
            print(f"✓ 扫描到 {self.port_combo.count()} 个串口")
        except ImportError:
            self.port_combo.addItem("COM3", "COM3")  # 降级默认
            print("⚠ serial.tools.list_ports 不可用，使用默认 COM3")

    def _connect_signals(self):
        """连接内部信号"""
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self.disconnect_clicked.emit)

    def _on_connect(self):
        """点击连接按钮"""
        port = self.port_combo.currentData()
        if port:
            self.connect_clicked.emit(port, 115200)

    def set_connected(self, connected):
        """设置连接状态

        Args:
            connected: 是否已连接
        """
        if connected:
            self.status_indicator.setStyleSheet("color: #58d68d; font-size: 15px;")
            self.status_text.setText("已连接")
            self.status_text.setStyleSheet("color: #58d68d; font-weight: 700;")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
        else:
            self.status_indicator.setStyleSheet("color: #ff6b6b; font-size: 15px;")
            self.status_text.setText("未连接")
            self.status_text.setStyleSheet("color: #ff6b6b; font-weight: 700;")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)


class CTModelPanel(QGroupBox):
    """CT模型导入面板"""

    # 信号定义
    load_clicked = pyqtSignal(str)  # 用户点击加载按钮
    clear_clicked = pyqtSignal()  # 清除模型
    visibility_changed = pyqtSignal(bool)  # 可见性切换

    def __init__(self):
        super().__init__("CT模型导入")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(9)
        layout.setContentsMargins(6, 14, 6, 6)

        # ===== 加载按钮 =====
        self.load_btn = QPushButton("选择 DICOM 文件夹")
        _set_button_variant(self.load_btn, "secondary")
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ===== 状态标签 =====
        self.status_label = QLabel("未加载模型")
        self.status_label.setStyleSheet("color: #8090a3; font-size: 11px;")
        layout.addWidget(self.status_label)

        # ===== 显示/隐藏 =====
        self.visibility_checkbox = QCheckBox("显示头部模型")
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.toggled.connect(self.visibility_changed.emit)
        self.visibility_checkbox.setEnabled(False)
        layout.addWidget(self.visibility_checkbox)

        # ===== 清除按钮 =====
        self.clear_btn = QPushButton("清除模型")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        self.clear_btn.setEnabled(False)
        _set_button_variant(self.clear_btn, "secondary")
        layout.addWidget(self.clear_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _on_load_clicked(self):
        """选择文件夹"""
        from PyQt5.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if folder:
            self.load_clicked.emit(folder)

    def set_loading(self, is_loading):
        """设置加载状态"""
        self.load_btn.setEnabled(not is_loading)
        self.progress_bar.setVisible(is_loading)

    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def set_model_loaded(self, loaded, info=None):
        """设置模型加载状态"""
        self.visibility_checkbox.setEnabled(loaded)
        self.clear_btn.setEnabled(loaded)

        if loaded and info:
            self.status_label.setText(
                f"✓ 已加载: {info['num_vertices']} 顶点, {info['num_faces']} 面"
            )
            self.status_label.setStyleSheet("color: #58d68d; font-size: 11px; font-weight: 700;")
        else:
            self.status_label.setText("未加载模型")
            self.status_label.setStyleSheet("color: #8090a3; font-size: 11px;")


class PuncturePointPanel(QGroupBox):
    """穿刺点选择面板"""

    #  新增信号
    start_selection_clicked = pyqtSignal()
    reselect_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("穿刺点选择", parent)
        self.setStyleSheet("""
            QGroupBox {
                color: #f4c430;
                background: #171f2a;
                border: 1px solid #8a6d1f;
                border-radius: 7px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                background: #111824;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(9)
        layout.setContentsMargins(6, 14, 6, 6)

        #  提示文字
        self.hint_label = QLabel("请先导入CT模型")
        self.hint_label.setStyleSheet("""
            color: #d6e2ea;
            font-size: 12px; 
            padding: 8px;
            background: #101824;
            border: 1px solid #26374d;
            border-radius: 5px;
        """)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        #  开始选择按钮
        self.start_btn = QPushButton("开始选择穿刺点")
        _set_button_variant(self.start_btn, "primary")
        self.start_btn.clicked.connect(self.start_selection_clicked.emit)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)

        # 坐标显示（初始隐藏）
        self.coord_widget = QWidget()
        coord_layout = QVBoxLayout(self.coord_widget)
        coord_layout.setContentsMargins(0, 0, 0, 0)
        coord_layout.setSpacing(5)

        coord_grid = QGridLayout()
        coord_grid.setSpacing(5)

        self.coord_labels = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            label = QLabel(f"{axis}:")
            label.setStyleSheet("color: #f4c430; font-weight: bold;")
            value = QLabel("--")
            value.setStyleSheet(f"color: #f7fbff; {_VALUE_FONT}")

            coord_grid.addWidget(label, i, 0)
            coord_grid.addWidget(value, i, 1)
            self.coord_labels[axis] = value

        coord_layout.addLayout(coord_grid)

        # 法线方向显示
        normal_title = QLabel("法线方向:")
        normal_title.setStyleSheet("color: #f4c430; font-weight: bold;")
        coord_layout.addWidget(normal_title)

        self.normal_label = QLabel("--")
        self.normal_label.setStyleSheet(f"color: #f7fbff; {_VALUE_FONT}")
        self.normal_label.setWordWrap(True)
        coord_layout.addWidget(self.normal_label)

        self.coord_widget.setVisible(False)
        layout.addWidget(self.coord_widget)

        # 重选按钮
        self.reselect_btn = QPushButton("重新选择穿刺点")
        _set_button_variant(self.reselect_btn, "danger")
        self.reselect_btn.clicked.connect(self.reselect_clicked.emit)
        self.reselect_btn.setVisible(False)
        layout.addWidget(self.reselect_btn)

        # ===== 对齐监控UI（预创建，初始隐藏）=====
        self._init_alignment_ui(layout)

        self.setLayout(layout)

    def set_model_loaded(self, loaded):
        """设置模型加载状态"""
        if loaded:
            self.hint_label.setText("CT 模型已加载，可以开始选择穿刺点")
            self.hint_label.setStyleSheet("""
                color: #58d68d;
                font-size: 12px; 
                padding: 8px;
                background: rgba(31, 157, 99, 0.16);
                border: 1px solid rgba(88, 214, 141, 0.35);
                border-radius: 5px;
            """)
            self.start_btn.setEnabled(True)
        else:
            self.hint_label.setText("请先导入CT模型")
            self.hint_label.setStyleSheet("""
                color: #d6e2ea;
                font-size: 12px; 
                padding: 8px;
                background: #101824;
                border: 1px solid #26374d;
                border-radius: 5px;
            """)
            self.start_btn.setEnabled(False)

    def set_selecting_mode(self, selecting):
        """设置选择模式"""
        if selecting:
            self.hint_label.setText("请在 CT 模型上点击选择穿刺点")
            self.hint_label.setStyleSheet("""
                color: #f4c430;
                font-size: 12px; 
                font-weight: bold;
                padding: 8px;
                background: rgba(244, 196, 48, 0.12);
                border: 1px solid rgba(244, 196, 48, 0.42);
                border-radius: 5px;
            """)
            self.start_btn.setEnabled(False)
        else:
            self.start_btn.setEnabled(True)

    def set_puncture_point(self, point, normal):
        """显示穿刺点信息"""
        self.coord_labels['X'].setText(f"{point[0]:7.2f} mm")
        self.coord_labels['Y'].setText(f"{point[1]:7.2f} mm")
        self.coord_labels['Z'].setText(f"{point[2]:7.2f} mm")

        self.normal_label.setText(
            f"[{normal[0]:6.3f}, {normal[1]:6.3f}, {normal[2]:6.3f}]"
        )

        self.hint_label.setText("穿刺点已选择")
        self.hint_label.setStyleSheet("""
            color: #58d68d;
            font-size: 12px; 
            font-weight: bold;
            padding: 8px;
            background: rgba(31, 157, 99, 0.18);
            border: 1px solid rgba(88, 214, 141, 0.42);
            border-radius: 5px;
        """)

        self.start_btn.setVisible(False)
        self.coord_widget.setVisible(True)
        self.reselect_btn.setVisible(True)

    def clear(self):
        """清除显示"""
        for label in self.coord_labels.values():
            label.setText("--")
        self.normal_label.setText("--")

        self.hint_label.setText("CT 模型已加载，可以开始选择穿刺点")
        self.hint_label.setStyleSheet("""
               color: #58d68d;
               font-size: 12px; 
               padding: 8px;
               background: rgba(31, 157, 99, 0.16);
               border: 1px solid rgba(88, 214, 141, 0.35);
               border-radius: 5px;
           """)

        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.coord_widget.setVisible(False)
        self.reselect_btn.setVisible(False)

    def _init_alignment_ui(self, layout):
        """预创建对齐监控UI（初始隐藏）"""
        self.alignment_group = QWidget()
        self.alignment_group.setVisible(False)
        alignment_layout = QVBoxLayout(self.alignment_group)
        alignment_layout.setContentsMargins(0, 8, 0, 0)
        alignment_layout.setSpacing(5)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background: rgba(244, 196, 48, 0.3);")
        alignment_layout.addWidget(separator)

        title = QLabel("角度偏差监控")
        title.setStyleSheet("color: #f4c430; font-weight: bold; font-size: 13px; padding: 5px 0;")
        alignment_layout.addWidget(title)

        self.alignment_status_label = QLabel("等待连接...")
        self.alignment_status_label.setAlignment(Qt.AlignCenter)
        self.alignment_status_label.setStyleSheet("""
            QLabel {
                font-size: 13px; font-weight: bold; padding: 8px;
                border-radius: 5px; background: #34495e; color: white;
            }
        """)
        alignment_layout.addWidget(self.alignment_status_label)

        self.alignment_error_label = QLabel("偏离角度: --")
        self.alignment_error_label.setAlignment(Qt.AlignCenter)
        self.alignment_error_label.setStyleSheet("""
            QLabel {
                font-size: 28px; font-weight: bold; padding: 12px;
                background: rgba(16, 24, 36, 0.95); color: #45d7ff;
                border: 1px solid #26374d;
                border-radius: 6px; margin-top: 5px;
                font-family: 'Cascadia Mono', 'Consolas', monospace;
            }
        """)
        alignment_layout.addWidget(self.alignment_error_label)

        layout.addWidget(self.alignment_group)

    def _show_alignment_ui(self):
        """显示对齐监控UI"""
        self.alignment_group.setVisible(True)

    def set_alignment_error(self, angle_deg):
        """设置偏离角度"""
        self._show_alignment_ui()
        self.alignment_error_label.setText(f"{angle_deg:.1f}°")

    def set_alignment_status(self, status):
        """设置对齐状态"""
        self._show_alignment_ui()
        self.alignment_status_label.setText(status)

        if "已对齐" in status:
            bg = "#1f9d63"
        elif "接近对齐" in status:
            bg = "#c58b12"
        else:
            bg = "#c94a5a"

        self.alignment_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                background: {bg};
                color: white;
            }}
        """)


class GuidanceArrowWidget(QWidget):
    """对准引导瞄准镜 — 十字线+偏移点，直观显示偏离方向"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle_deg = 0.0
        self._dot_x = 0.0
        self._dot_y = 0.0
        self._visible = False
        self.setMinimumSize(100, 100)
        self.setMaximumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_guidance(self, correction_3d, angle_deg):
        """设置引导方向"""
        self._angle_deg = angle_deg
        cx, cy, _ = correction_3d
        mag = (cx*cx + cy*cy + 1e-12) ** 0.5
        self._dot_x = cx / mag if mag > 0.01 else 0.0
        self._dot_y = -cy / mag if mag > 0.01 else 0.0
        self._visible = True
        self.update()

    def hide_guidance(self):
        self._visible = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx_f, cy_f = w / 2.0, h / 2.0
        radius = min(w, h) / 2.0 - 8

        # === 背景（半透明暗色圆盘）===
        bg = QColor(10, 17, 27)
        painter.setPen(QPen(QColor(55, 78, 105), 1.5))
        painter.setBrush(bg)
        painter.drawEllipse(QPointF(cx_f, cy_f), radius, radius)

        if not self._visible:
            painter.setPen(QColor(95, 111, 130))
            fnt = QFont("Consolas", 9)
            painter.setFont(fnt)
            painter.drawText(QRect(int(cx_f - 30), int(cy_f + radius - 20),
                                   60, 20), Qt.AlignCenter, "--")
            return

        # === 刻度圈 ===
        painter.setPen(QPen(QColor(42, 58, 80), 1))
        for i in range(12):
            a = i * 30 * math.pi / 180
            r1 = radius * 0.85 if i % 3 == 0 else radius * 0.92
            r2 = radius * 0.95
            painter.drawLine(QPointF(cx_f + r1*math.cos(a), cy_f + r1*math.sin(a)),
                             QPointF(cx_f + r2*math.cos(a), cy_f + r2*math.sin(a)))

        # === 十字线 ===
        painter.setPen(QPen(QColor(64, 88, 116), 1))
        inner = radius * 0.1
        outer = radius * 0.85
        painter.drawLine(QPointF(cx_f - outer, cy_f), QPointF(cx_f - inner, cy_f))
        painter.drawLine(QPointF(cx_f + inner, cy_f), QPointF(cx_f + outer, cy_f))
        painter.drawLine(QPointF(cx_f, cy_f - outer), QPointF(cx_f, cy_f - inner))
        painter.drawLine(QPointF(cx_f, cy_f + inner), QPointF(cx_f, cy_f + outer))

        # === 中心点（目标标记）===
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(69, 215, 255))
        painter.drawEllipse(QPointF(cx_f, cy_f), 3, 3)

        # === 颜色（角度→色相）===
        a = self._angle_deg
        if a < 2:
            color = QColor(0, 230, 70)
        elif a < 5:
            t = (a - 2) / 3
            color = QColor(int(255 * t), 255, int(70 - 50 * t))
        elif a < 10:
            t = (a - 5) / 5
            color = QColor(255, int(255 - 90 * t), 20)
        else:
            color = QColor(255, 50, 50)

        # === 偏移点（当前方向相对目标的位置）===
        dot_max = radius * 0.75
        dist = min(self._angle_deg / 30.0, 1.0) * dot_max
        dx = self._dot_x * dist
        dy = self._dot_y * dist
        px = cx_f + dx
        py = cy_f + dy

        # 连线（从中心到偏移点）
        painter.setPen(QPen(color, 2))
        painter.drawLine(QPointF(cx_f, cy_f), QPointF(px, py))

        # 偏移点（半透明外晕 + 实心点）
        painter.setPen(Qt.NoPen)
        glow = QColor(color.red(), color.green(), color.blue(), 60)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(px, py), 12, 12)

        painter.setBrush(color)
        painter.drawEllipse(QPointF(px, py), 6, 6)

        # 外圈高亮
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(px, py), 9, 9)

        # === 角度大数字 ===
        painter.setPen(QColor(200, 210, 230))
        fnt = QFont("Consolas", 14, QFont.Bold)
        painter.setFont(fnt)
        painter.drawText(QRect(int(cx_f - 50), int(cy_f + radius - 28),
                               100, 28), Qt.AlignCenter, f"{self._angle_deg:.1f}°")
