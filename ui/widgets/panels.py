"""左侧面板组件 - IMU数据显示和针具配置"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QFrame, QSizePolicy, QSlider, QProgressBar, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class IMUDataPanel(QGroupBox):
    """IMU数据显示面板"""

    def __init__(self, parent=None):
        super().__init__("IMU 数据", parent)
        self._init_ui()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title_color = "#81d4fa"
        value_font = "font-family: Consolas, monospace; font-size: 13px;"
        label_dim = "color: #888; font-size: 10px;"

        # 四元数显示
        quat_group = QFrame()
        quat_layout = QGridLayout(quat_group)
        quat_layout.setSpacing(4)

        quat_label = QLabel("四元数:")
        quat_label.setStyleSheet(f"color: {title_color}; font-weight: bold; font-size: 13px;")
        quat_layout.addWidget(quat_label, 0, 0, 1, 4)

        self.quat_labels = []
        quat_names = ['W', 'X', 'Y', 'Z']
        for i, name in enumerate(quat_names):
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet(label_dim)
            quat_layout.addWidget(name_label, 1, i)

            value_label = QLabel("0.000")
            value_label.setStyleSheet(f"color: #4fc3f7; {value_font}")
            value_label.setMinimumWidth(60)
            quat_layout.addWidget(value_label, 2, i)
            self.quat_labels.append(value_label)

        layout.addWidget(quat_group)

        # 欧拉角显示
        euler_group = QFrame()
        euler_layout = QGridLayout(euler_group)
        euler_layout.setSpacing(4)

        euler_label = QLabel("欧拉角:")
        euler_label.setStyleSheet(f"color: {title_color}; font-weight: bold; font-size: 13px;")
        euler_layout.addWidget(euler_label, 0, 0, 1, 3)

        self.euler_labels = []
        euler_names = ['Roll', 'Pitch', 'Yaw']
        for i, name in enumerate(euler_names):
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet(label_dim)
            euler_layout.addWidget(name_label, 1, i)

            value_label = QLabel("0.0°")
            value_label.setStyleSheet(f"color: #4fc3f7; {value_font}")
            value_label.setMinimumWidth(70)
            euler_layout.addWidget(value_label, 2, i)
            self.euler_labels.append(value_label)

        layout.addWidget(euler_group)

        # 位置显示
        pos_group = QFrame()
        pos_layout = QGridLayout(pos_group)
        pos_layout.setSpacing(4)

        pos_label = QLabel("位置 (mm):")
        pos_label.setStyleSheet(f"color: {title_color}; font-weight: bold; font-size: 13px;")
        pos_layout.addWidget(pos_label, 0, 0, 1, 3)

        self.pos_labels = []
        pos_names = ['X', 'Y', 'Z']
        for i, name in enumerate(pos_names):
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet(label_dim)
            pos_layout.addWidget(name_label, 1, i)

            value_label = QLabel("0.0")
            value_label.setStyleSheet(f"color: #4fc3f7; {value_font}")
            value_label.setMinimumWidth(70)
            pos_layout.addWidget(value_label, 2, i)
            self.pos_labels.append(value_label)

        layout.addWidget(pos_group)

        # 连接状态
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: #f44336;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        self.fps_label = QLabel("0 Hz")
        self.fps_label.setStyleSheet("color: #aaa;")
        status_layout.addWidget(self.fps_label)

        layout.addLayout(status_layout)

    def update_quaternion(self, quaternion):
        """更新四元数显示

        Args:
            quaternion: [w, x, y, z] 四元数
        """
        if len(quaternion) >= 4:
            for i, val in enumerate(quaternion[:4]):
                self.quat_labels[i].setText(f"{val:.3f}")

    def update_euler(self, euler):
        """更新欧拉角显示

        Args:
            euler: [roll, pitch, yaw] 欧拉角（度）
        """
        if len(euler) >= 3:
            for i, val in enumerate(euler[:3]):
                self.euler_labels[i].setText(f"{val:.1f}°")

    def update_position(self, position):
        """更新位置显示

        Args:
            position: [x, y, z] 位置（mm）
        """
        if len(position) >= 3:
            for i, val in enumerate(position[:3]):
                self.pos_labels[i].setText(f"{val:.1f}")

    def set_status(self, connected, fps=0):
        """设置连接状态

        Args:
            connected: 是否已连接
            fps: 数据刷新率
        """
        if connected:
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("color: #4caf50;")
        else:
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("color: #f44336;")

        self.fps_label.setText(f"{fps} Hz")


class NeedleConfigPanel(QGroupBox):
    """针具配置面板"""

    # 原有信号
    needle_length_changed = pyqtSignal(float)
    zero_position_clicked = pyqtSignal()
    clear_trajectory_clicked = pyqtSignal()
    reset_view_clicked = pyqtSignal()

    #  新增：校准信号
    calibration_clicked = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__("针具配置", parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 仅保留QSlider定制（全局QSS不包含此项）
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a4a6a;
                height: 6px;
                background: #2a2a4a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #81d4fa;
                border: 1px solid #5c5c7c;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)

        # 针具长度设置
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("针具长度:"))

        self.length_spinbox = QDoubleSpinBox()
        self.length_spinbox.setRange(10, 300)
        self.length_spinbox.setValue(162)
        self.length_spinbox.setSuffix(" mm")
        self.length_spinbox.setSingleStep(5)
        self.length_spinbox.setMinimumWidth(100)
        length_layout.addWidget(self.length_spinbox)

        length_layout.addStretch()
        layout.addLayout(length_layout)

        # 快捷按钮
        btn_layout = QGridLayout()

        self.btn_zero = QPushButton("位置归零")
        self.btn_zero.setToolTip("将当前位置设为原点")
        btn_layout.addWidget(self.btn_zero, 0, 0)

        self.btn_clear = QPushButton("清除轨迹")
        self.btn_clear.setToolTip("清除3D视图中的轨迹")
        btn_layout.addWidget(self.btn_clear, 0, 1)

        self.btn_reset_view = QPushButton("重置视角")
        self.btn_reset_view.setToolTip("重置3D视图相机位置")
        btn_layout.addWidget(self.btn_reset_view, 1, 0)

        # 预留按钮位置
        self.btn_calibrate = QPushButton("校准传感器")
        self.btn_calibrate.setToolTip("启动IMU校准：保持静止3秒完成陀螺仪零偏校准 + 磁力计校准")
        btn_layout.addWidget(self.btn_calibrate, 1, 1)

        layout.addLayout(btn_layout)

        # 当前针尖位置显示
        tip_layout = QGridLayout()
        tip_label = QLabel("针尖位置:")
        tip_label.setStyleSheet("color: #81d4fa;")
        tip_layout.addWidget(tip_label, 0, 0, 1, 3)

        self.tip_labels = []
        for i, name in enumerate(['X', 'Y', 'Z']):
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet("color: #aaa;")
            tip_layout.addWidget(name_label, 1, i)

            value_label = QLabel("0.0")
            value_label.setStyleSheet("color: #ff9800; font-family: Consolas;")
            tip_layout.addWidget(value_label, 2, i)
            self.tip_labels.append(value_label)

        layout.addLayout(tip_layout)


        layout.addStretch()

    def _connect_signals(self):
        """连接内部信号"""
        # 原有信号
        self.length_spinbox.valueChanged.connect(
            lambda v: self.needle_length_changed.emit(v)
        )
        self.btn_zero.clicked.connect(self.zero_position_clicked.emit)
        self.btn_clear.clicked.connect(self.clear_trajectory_clicked.emit)
        self.btn_reset_view.clicked.connect(self.reset_view_clicked.emit)
        self.btn_calibrate.clicked.connect(self.calibration_clicked.emit)


    def get_needle_length(self):
        """获取当前针具长度

        Returns:
            float: 针具长度（mm）
        """
        return self.length_spinbox.value()

    def set_needle_length(self, length):
        """设置针具长度

        Args:
            length: 针具长度（mm）
        """
        self.length_spinbox.setValue(length)

    def update_tip_position(self, position):
        """更新针尖位置显示

        Args:
            position: [x, y, z] 针尖位置（mm）
        """
        if len(position) >= 3:
            for i, val in enumerate(position[:3]):
                self.tip_labels[i].setText(f"{val:.1f}")


class DeviceConnectionPanel(QGroupBox):
    """设备连接面板"""

    # 信号定义
    connect_clicked = pyqtSignal(str, int)  # port, baudrate
    disconnect_clicked = pyqtSignal()

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
        self.status_indicator.setStyleSheet("color: #f44336; font-size: 20px;")
        status_layout.addWidget(self.status_indicator)
        self.status_text = QLabel("未连接")
        self.status_text.setStyleSheet("color: #f44336;")
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
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background: #388e3c;
            }
            QPushButton:hover {
                background: #43a047;
            }
        """)
        btn_layout.addWidget(self.btn_connect)

        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.setStyleSheet("""
            QPushButton {
                background: #c62828;
            }
            QPushButton:hover {
                background: #d32f2f;
            }
            QPushButton:disabled {
                background: #1a1a2a;
                color: #666;
            }
        """)
        btn_layout.addWidget(self.btn_disconnect)

        layout.addLayout(btn_layout)

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
            self.status_indicator.setStyleSheet("color: #4caf50; font-size: 16px;")
            self.status_text.setText("已连接")
            self.status_text.setStyleSheet("color: #4caf50;")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
        else:
            self.status_indicator.setStyleSheet("color: #f44336; font-size: 16px;")
            self.status_text.setText("未连接")
            self.status_text.setStyleSheet("color: #f44336;")
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
        layout.setSpacing(8)

        # ===== 加载按钮 =====
        self.load_btn = QPushButton("📁 选择DICOM文件夹")
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ===== 状态标签 =====
        self.status_label = QLabel("未加载模型")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)

        # ===== 显示/隐藏 =====
        self.visibility_checkbox = QCheckBox("显示头部模型")
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.toggled.connect(self.visibility_changed.emit)
        self.visibility_checkbox.setEnabled(False)
        layout.addWidget(self.visibility_checkbox)

        # ===== 清除按钮 =====
        self.clear_btn = QPushButton("🗑️ 清除模型")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        self.clear_btn.setEnabled(False)
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
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            self.status_label.setText("未加载模型")
            self.status_label.setStyleSheet("color: #888; font-size: 11px;")


class PuncturePointPanel(QGroupBox):
    """穿刺点选择面板"""

    #  新增信号
    start_selection_clicked = pyqtSignal()
    reselect_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("📍 穿刺点选择", parent)
        self.setStyleSheet("""
            QGroupBox {
                color: #ffcc00;
                border: 2px solid #ffcc00;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                font-size: 14px;
                background: rgba(255, 204, 0, 0.1);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)

        #  提示文字
        self.hint_label = QLabel("请先导入CT模型")
        self.hint_label.setStyleSheet("""
            color: #aaaaaa; 
            font-size: 12px; 
            padding: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        """)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        #  开始选择按钮
        self.start_btn = QPushButton("🎯 开始选择穿刺点")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QPushButton:disabled {
                background: #555555;
                color: #888888;
            }
        """)
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
            label.setStyleSheet("color: #ffcc00; font-weight: bold;")
            value = QLabel("--")
            value.setStyleSheet("color: white; font-family: 'Consolas', monospace;")

            coord_grid.addWidget(label, i, 0)
            coord_grid.addWidget(value, i, 1)
            self.coord_labels[axis] = value

        coord_layout.addLayout(coord_grid)

        # 法线方向显示
        normal_title = QLabel("法线方向:")
        normal_title.setStyleSheet("color: #ffcc00; font-weight: bold;")
        coord_layout.addWidget(normal_title)

        self.normal_label = QLabel("--")
        self.normal_label.setStyleSheet("color: white; font-family: 'Consolas', monospace;")
        self.normal_label.setWordWrap(True)
        coord_layout.addWidget(self.normal_label)

        self.coord_widget.setVisible(False)
        layout.addWidget(self.coord_widget)

        # 重选按钮
        self.reselect_btn = QPushButton("🔄 重新选择穿刺点")
        self.reselect_btn.setStyleSheet("""
            QPushButton {
                background: #ff6b6b;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff5252;
            }
            QPushButton:disabled {
                background: #555555;
                color: #888888;
            }
        """)
        self.reselect_btn.clicked.connect(self.reselect_clicked.emit)
        self.reselect_btn.setVisible(False)
        layout.addWidget(self.reselect_btn)

        # ===== 对齐监控UI（预创建，初始隐藏）=====
        self._init_alignment_ui(layout)

        self.setLayout(layout)

    def set_model_loaded(self, loaded):
        """设置模型加载状态"""
        if loaded:
            self.hint_label.setText("✓ CT模型已加载，可以开始选择穿刺点")
            self.hint_label.setStyleSheet("""
                color: #00ff00; 
                font-size: 12px; 
                padding: 8px;
                background: rgba(0, 255, 0, 0.1);
                border-radius: 4px;
            """)
            self.start_btn.setEnabled(True)
        else:
            self.hint_label.setText("请先导入CT模型")
            self.hint_label.setStyleSheet("""
                color: #aaaaaa; 
                font-size: 12px; 
                padding: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            """)
            self.start_btn.setEnabled(False)

    def set_selecting_mode(self, selecting):
        """设置选择模式"""
        if selecting:
            self.hint_label.setText("👆 请在CT模型上点击选择穿刺点")
            self.hint_label.setStyleSheet("""
                color: #ffcc00; 
                font-size: 12px; 
                font-weight: bold;
                padding: 8px;
                background: rgba(255, 204, 0, 0.15);
                border-radius: 4px;
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

        self.hint_label.setText("✅ 穿刺点已选择")
        self.hint_label.setStyleSheet("""
            color: #00ff00; 
            font-size: 12px; 
            font-weight: bold;
            padding: 8px;
            background: rgba(0, 255, 0, 0.15);
            border-radius: 4px;
        """)

        self.start_btn.setVisible(False)
        self.coord_widget.setVisible(True)
        self.reselect_btn.setVisible(True)

    def clear(self):
        """清除显示"""
        for label in self.coord_labels.values():
            label.setText("--")
        self.normal_label.setText("--")

        self.hint_label.setText("✓ CT模型已加载，可以开始选择穿刺点")
        self.hint_label.setStyleSheet("""
               color: #00ff00; 
               font-size: 12px; 
               padding: 8px;
               background: rgba(0, 255, 0, 0.1);
               border-radius: 4px;
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
        separator.setStyleSheet("background: rgba(255, 204, 0, 0.3);")
        alignment_layout.addWidget(separator)

        title = QLabel("📐 角度偏差监控")
        title.setStyleSheet("color: #ffcc00; font-weight: bold; font-size: 13px; padding: 5px 0;")
        alignment_layout.addWidget(title)

        self.alignment_status_label = QLabel("等待连接...")
        self.alignment_status_label.setAlignment(Qt.AlignCenter)
        self.alignment_status_label.setStyleSheet("""
            QLabel {
                font-size: 13px; font-weight: bold; padding: 8px;
                border-radius: 4px; background: #95a5a6; color: white;
            }
        """)
        alignment_layout.addWidget(self.alignment_status_label)

        self.alignment_error_label = QLabel("偏离角度: --")
        self.alignment_error_label.setAlignment(Qt.AlignCenter)
        self.alignment_error_label.setStyleSheet("""
            QLabel {
                font-size: 28px; font-weight: bold; padding: 12px;
                background: rgba(52, 73, 94, 0.8); color: #3498db;
                border-radius: 4px; margin-top: 5px;
                font-family: 'Consolas', monospace;
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
            bg = "#2ecc71"
        elif "接近对齐" in status:
            bg = "#f39c12"
        else:
            bg = "#e74c3c"

        self.alignment_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
                background: {bg};
                color: white;
            }}
        """)
