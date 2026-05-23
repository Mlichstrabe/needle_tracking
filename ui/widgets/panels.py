"""侧边栏 UI 组件"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton,
    QFrame, QSizePolicy, QProgressBar, QCheckBox, QComboBox,
    QToolButton,
)
from PyQt5.QtCore import Qt, pyqtSignal
import math
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QRect, QPointF

from ui.widgets.ui_helpers import set_button_variant, set_label_role, apply_panel_chrome


def _hint_role_for_state(state):
    if state == "success":
        return "hint-success"
    if state == "warn":
        return "hint-warn"
    return "hint"


def _apply_hint_state(label, text, state="default"):
    label.setText(text)
    set_label_role(label, _hint_role_for_state(state))


class IMUDataPanel(QWidget):
    """IMU 遥测（手风琴内嵌，无 GroupBox 外框）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_panel_chrome(self)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 2, 2, 2)

        status_frame = QFrame()
        status_frame.setObjectName("StatusCard")
        status_row = QHBoxLayout(status_frame)
        status_row.setContentsMargins(12, 8, 12, 8)

        self.status_dot = QLabel("●")
        set_label_role(self.status_dot, "danger")
        status_row.addWidget(self.status_dot)

        self.status_label = QLabel("未连接")
        set_label_role(self.status_label, "muted")
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.fps_label = QLabel("-- Hz")
        set_label_role(self.fps_label, "value-sm")
        status_row.addWidget(self.fps_label)
        layout.addWidget(status_frame)

        # 欧拉角（主读数）
        layout.addWidget(self._build_metric_card(
            "欧拉角",
            ["Roll", "Pitch", "Yaw"],
            "euler_labels",
            suffix="°",
        ))  # value role from QSS

        quat_card = QFrame()
        quat_card.setObjectName("MetricCard")
        quat_outer = QVBoxLayout(quat_card)
        quat_outer.setContentsMargins(10, 8, 10, 8)
        quat_outer.setSpacing(4)
        quat_title = QLabel("四元数")
        set_label_role(quat_title, "muted")
        quat_outer.addWidget(quat_title)

        quat_row = QHBoxLayout()
        quat_row.setSpacing(6)
        self.quat_labels = []
        for name in ("W", "X", "Y", "Z"):
            cell = QVBoxLayout()
            cell.setSpacing(0)
            n = QLabel(name)
            set_label_role(n, "muted")
            n.setAlignment(Qt.AlignCenter)
            v = QLabel("0.000")
            set_label_role(v, "value-sm")
            v.setAlignment(Qt.AlignCenter)
            cell.addWidget(n)
            cell.addWidget(v)
            quat_row.addLayout(cell)
            self.quat_labels.append(v)
        quat_outer.addLayout(quat_row)
        self._quat_card = quat_card
        self._quat_card.setVisible(False)

        self._quat_toggle = QToolButton()
        self._quat_toggle.setText("显示四元数 ▾")
        self._quat_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._quat_toggle.setAutoRaise(True)
        self._quat_toggle.clicked.connect(self._toggle_quaternion)
        layout.addWidget(self._quat_toggle)
        layout.addWidget(self._quat_card)

    def _toggle_quaternion(self):
        show = not self._quat_card.isVisible()
        self._quat_card.setVisible(show)
        self._quat_toggle.setText("隐藏四元数 ▴" if show else "显示四元数 ▾")

    def _build_metric_card(self, title, names, attr_name, suffix, value_style=None):
        card = QFrame()
        card.setObjectName("MetricCard")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        t = QLabel(title)
        set_label_role(t, "muted")
        outer.addWidget(t)

        grid = QGridLayout()
        grid.setSpacing(6)
        labels = []
        for i, name in enumerate(names):
            n = QLabel(name)
            set_label_role(n, "muted")
            n.setAlignment(Qt.AlignCenter)
            grid.addWidget(n, 0, i)

            val = QLabel(f"0{suffix}")
            set_label_role(val, "value")
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
            set_label_role(self.status_dot, "ok")
            self.status_label.setText("已连接")
            set_label_role(self.status_label, "ok")
        else:
            set_label_role(self.status_dot, "danger")
            self.status_label.setText("未连接")
            set_label_role(self.status_label, "muted")

        if fps > 0:
            self.fps_label.setText(f"{fps:.0f} Hz")
        else:
            self.fps_label.setText("-- Hz")


class DeviceConnectionPanel(QWidget):
    """设备连接（手风琴页内容）"""

    connect_clicked = pyqtSignal(str, int)
    disconnect_clicked = pyqtSignal()
    calibration_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_panel_chrome(self)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 2, 2, 2)

        status_frame = QFrame()
        status_frame.setObjectName("StatusCard")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        status_layout.addWidget(QLabel("状态"))
        self.status_indicator = QLabel("●")
        set_label_role(self.status_indicator, "danger")
        status_layout.addWidget(self.status_indicator)
        self.status_text = QLabel("未连接")
        set_label_role(self.status_text, "danger")
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        layout.addWidget(status_frame)

        # 串口选择
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(140)
        self.port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        port_layout.addWidget(self.port_combo, 1)

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setToolTip("刷新串口列表")
        self.btn_refresh.setFixedWidth(32)
        self.btn_refresh.clicked.connect(self._refresh_ports)
        port_layout.addWidget(self.btn_refresh)

        layout.addLayout(port_layout)

        # 连接/断开按钮
        btn_layout = QHBoxLayout()
        self.btn_connect = QPushButton("连接设备")
        set_button_variant(self.btn_connect, "primary")
        btn_layout.addWidget(self.btn_connect)

        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)
        set_button_variant(self.btn_disconnect, "danger")
        btn_layout.addWidget(self.btn_disconnect)

        layout.addLayout(btn_layout)

        self.btn_calibrate = QPushButton("校准传感器")
        self.btn_calibrate.setToolTip("静止约 3 秒完成陀螺仪零偏与磁力计校准")
        set_button_variant(self.btn_calibrate, "ghost")
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
            set_label_role(self.status_indicator, "ok")
            self.status_text.setText("已连接")
            set_label_role(self.status_text, "ok")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
        else:
            set_label_role(self.status_indicator, "danger")
            self.status_text.setText("未连接")
            set_label_role(self.status_text, "danger")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)


class CTModelPanel(QWidget):
    """CT 影像导入（手风琴页内容）"""

    load_clicked = pyqtSignal(str)
    clear_clicked = pyqtSignal()
    visibility_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_panel_chrome(self)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 2, 2, 2)

        self.load_btn = QPushButton("选择 DICOM 文件夹")
        set_button_variant(self.load_btn, "primary")
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ===== 状态标签 =====
        self.status_label = QLabel("未加载模型")
        set_label_role(self.status_label, "muted")
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
        set_button_variant(self.clear_btn, "ghost")
        layout.addWidget(self.clear_btn)

        layout.addStretch()

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
            set_label_role(self.status_label, "ok")
        else:
            self.status_label.setText("未加载模型")
            set_label_role(self.status_label, "muted")


class PuncturePointPanel(QFrame):
    """穿刺点选择（引导轨紧凑卡片）"""

    start_selection_clicked = pyqtSignal()
    reselect_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowCard")
        apply_panel_chrome(self)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        self.hint_label = QLabel("请先导入 CT 模型")
        self.hint_label.setWordWrap(True)
        _apply_hint_state(self.hint_label, "请先导入 CT 模型", "default")
        layout.addWidget(self.hint_label)

        self.start_btn = QPushButton("开始选择穿刺点")
        set_button_variant(self.start_btn, "primary")
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
            set_label_role(label, "warn")
            value = QLabel("--")
            set_label_role(value, "value-sm")

            coord_grid.addWidget(label, i, 0)
            coord_grid.addWidget(value, i, 1)
            self.coord_labels[axis] = value

        coord_layout.addLayout(coord_grid)

        # 法线方向显示
        normal_title = QLabel("法线方向:")
        set_label_role(normal_title, "warn")
        coord_layout.addWidget(normal_title)

        self.normal_label = QLabel("--")
        set_label_role(self.normal_label, "value-sm")
        self.normal_label.setWordWrap(True)
        coord_layout.addWidget(self.normal_label)

        self.coord_widget.setVisible(False)
        layout.addWidget(self.coord_widget)

        # 重选按钮
        self.reselect_btn = QPushButton("重新选择穿刺点")
        set_button_variant(self.reselect_btn, "danger")
        self.reselect_btn.clicked.connect(self.reselect_clicked.emit)
        self.reselect_btn.setVisible(False)
        layout.addWidget(self.reselect_btn)

        step_title = QLabel("① 选择 Entry")
        step_title.setObjectName("SectionTitle")
        layout.insertWidget(0, step_title)

    def set_model_loaded(self, loaded):
        """设置模型加载状态"""
        if loaded:
            _apply_hint_state(
                self.hint_label,
                "CT 模型已加载，可以开始选择穿刺点",
                "success",
            )
            self.start_btn.setEnabled(True)
        else:
            _apply_hint_state(self.hint_label, "请先导入 CT 模型", "default")
            self.start_btn.setEnabled(False)

    def set_selecting_mode(self, selecting):
        """设置选择模式"""
        if selecting:
            _apply_hint_state(
                self.hint_label,
                "请在 3D 视图中点击头部表面，选择穿刺 Entry 点",
                "warn",
            )
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

        _apply_hint_state(self.hint_label, "穿刺点已选择", "success")

        self.start_btn.setVisible(False)
        self.coord_widget.setVisible(True)
        self.reselect_btn.setVisible(True)

    def clear(self):
        """清除显示"""
        for label in self.coord_labels.values():
            label.setText("--")
        self.normal_label.setText("--")

        _apply_hint_state(
            self.hint_label,
            "CT 模型已加载，可以开始选择穿刺点",
            "success",
        )

        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.coord_widget.setVisible(False)
        self.reselect_btn.setVisible(False)

class GuidanceArrowWidget(QWidget):
    """对准引导瞄准镜 — 十字线+偏移点，直观显示偏离方向"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle_deg = 0.0
        self._dot_x = 0.0
        self._dot_y = 0.0
        self._visible = False
        self.setMinimumSize(140, 140)
        self.setMaximumSize(200, 200)
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
        bg = QColor(14, 20, 30, 240)
        painter.setPen(QPen(QColor(70, 100, 135), 2))
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

