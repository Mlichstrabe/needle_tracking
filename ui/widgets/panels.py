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

    axis_mapping_changed = pyqtSignal(bool, float, float, float, float)  # swap_xy, sx, sy, sz, yaw_offset_deg
    smoothing_changed = pyqtSignal(bool, float)  # enabled, alpha

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

        self.chk_mirror_lr = QCheckBox("左右镜像修正（3D 针体与实物同向）")
        self.chk_mirror_lr.setChecked(True)
        self.chk_mirror_lr.setToolTip(
            "若左右仍反向，取消勾选；若前后反向或上下反向，请用下方开关继续调。"
        )
        self.chk_mirror_lr.toggled.connect(self._emit_axis_mapping)
        layout.addWidget(self.chk_mirror_lr)

        self.chk_mirror_fb = QCheckBox("前后翻转（有时是这个反向）")
        self.chk_mirror_fb.setChecked(True)
        self.chk_mirror_fb.toggled.connect(self._emit_axis_mapping)
        layout.addWidget(self.chk_mirror_fb)

        self.chk_mirror_ud = QCheckBox("上下翻转（仅在上下颠倒时打开）")
        self.chk_mirror_ud.setChecked(True)
        self.chk_mirror_ud.toggled.connect(self._emit_axis_mapping)
        layout.addWidget(self.chk_mirror_ud)

        self.chk_swap_xy = QCheckBox("交换 X/Y（转动方向怪、或 45°像 90° 时尝试）")
        self.chk_swap_xy.setChecked(False)
        self.chk_swap_xy.toggled.connect(self._emit_axis_mapping)
        layout.addWidget(self.chk_swap_xy)

        yaw_row = QHBoxLayout()
        yaw_row.addWidget(QLabel("水平偏置"))
        self.combo_yaw_off = QComboBox()
        self.combo_yaw_off.addItem("0°", 0.0)
        self.combo_yaw_off.addItem("+90°", 90.0)
        self.combo_yaw_off.addItem("-90°", -90.0)
        self.combo_yaw_off.addItem("180°", 180.0)
        self.combo_yaw_off.setToolTip("交换XY后仍固定偏 90° 时，用这里一键消掉偏差。")
        self.combo_yaw_off.currentIndexChanged.connect(lambda *_: self._emit_axis_mapping())
        yaw_row.addWidget(self.combo_yaw_off, 1)
        layout.addLayout(yaw_row)

        self.chk_smooth = QCheckBox("方向平滑（抑制乱飘/跳变）")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(self._emit_smoothing)
        layout.addWidget(self.chk_smooth)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("平滑强度"))
        self.combo_smooth = QComboBox()
        self.combo_smooth.addItem("弱（更跟手）", 0.45)
        self.combo_smooth.addItem("中（推荐）", 0.25)
        self.combo_smooth.addItem("强（更稳）", 0.12)
        self.combo_smooth.setCurrentIndex(1)
        self.combo_smooth.currentIndexChanged.connect(lambda *_: self._emit_smoothing())
        smooth_row.addWidget(self.combo_smooth, 1)
        layout.addLayout(smooth_row)

        hint = QLabel("映射与平滑会自动保存到 config/imu_geometry.json")
        set_label_role(hint, "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _set_combo_by_data(self, combo: QComboBox, value: float) -> None:
        target = float(value)
        for i in range(combo.count()):
            if float(combo.itemData(i)) == target:
                combo.setCurrentIndex(i)
                return

    def scene_mapping_dict(self) -> dict:
        return {
            "mirror_lr": self.chk_mirror_lr.isChecked(),
            "mirror_fb": self.chk_mirror_fb.isChecked(),
            "mirror_ud": self.chk_mirror_ud.isChecked(),
            "swap_xy": self.chk_swap_xy.isChecked(),
            "yaw_offset_deg": float(self.combo_yaw_off.currentData()),
        }

    def smoothing_dict(self) -> dict:
        return {
            "enabled": self.chk_smooth.isChecked(),
            "alpha": float(self.combo_smooth.currentData()),
        }

    def apply_settings(self, cfg: dict) -> None:
        """从 config/imu_geometry.json 恢复 UI，并同步到主窗口。"""
        sm = cfg.get("scene_mapping", {})
        smooth = cfg.get("smoothing", {})

        widgets = (
            self.chk_mirror_lr,
            self.chk_mirror_fb,
            self.chk_mirror_ud,
            self.chk_swap_xy,
            self.chk_smooth,
            self.combo_yaw_off,
            self.combo_smooth,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.chk_mirror_lr.setChecked(bool(sm.get("mirror_lr", True)))
            self.chk_mirror_fb.setChecked(bool(sm.get("mirror_fb", True)))
            self.chk_mirror_ud.setChecked(bool(sm.get("mirror_ud", True)))
            self.chk_swap_xy.setChecked(bool(sm.get("swap_xy", False)))
            self._set_combo_by_data(self.combo_yaw_off, float(sm.get("yaw_offset_deg", 0.0)))
            self.chk_smooth.setChecked(bool(smooth.get("enabled", True)))
            self._set_combo_by_data(self.combo_smooth, float(smooth.get("alpha", 0.25)))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        self._emit_axis_mapping()
        self._emit_smoothing()

    def _emit_axis_mapping(self, *_):
        # sx: 左右，sy: 前后，sz: 上下
        sx = -1.0 if self.chk_mirror_lr.isChecked() else 1.0
        sy = -1.0 if self.chk_mirror_fb.isChecked() else 1.0
        sz = -1.0 if self.chk_mirror_ud.isChecked() else 1.0
        swap_xy = self.chk_swap_xy.isChecked()
        yaw_off = float(self.combo_yaw_off.currentData())
        self.axis_mapping_changed.emit(swap_xy, sx, sy, sz, yaw_off)

    def _emit_smoothing(self, *_):
        enabled = self.chk_smooth.isChecked()
        alpha = float(self.combo_smooth.currentData())
        self.smoothing_changed.emit(enabled, alpha)

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

    MODE_PUNCTURE_TRAINING = "puncture_training"
    MODE_NEEDLE_OBSERVE = "needle_observe"

    connect_clicked = pyqtSignal(str, int)
    disconnect_clicked = pyqtSignal()
    calibration_clicked = pyqtSignal()
    operation_mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_panel_chrome(self)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 2, 2, 2)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("工作模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("穿刺训练（需 CT + Entry）", self.MODE_PUNCTURE_TRAINING)
        self.mode_combo.addItem("姿态观察（仅看针体）", self.MODE_NEEDLE_OBSERVE)
        self.mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mode_layout.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_layout)

        self.observe_hint = QLabel("无需加载 CT，针尖固定在坐标原点")
        self.observe_hint.setWordWrap(True)
        set_label_role(self.observe_hint, "muted")
        self.observe_hint.setVisible(False)
        layout.addWidget(self.observe_hint)

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
        self.mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)

    def _on_mode_combo_changed(self, _index):
        self._update_mode_hint()
        self.operation_mode_changed.emit(self.get_operation_mode())

    def _update_mode_hint(self):
        observe = self.get_operation_mode() == self.MODE_NEEDLE_OBSERVE
        self.observe_hint.setVisible(observe)

    def get_operation_mode(self):
        data = self.mode_combo.currentData()
        return data if data else self.MODE_PUNCTURE_TRAINING

    def set_mode_switch_enabled(self, enabled: bool):
        self.mode_combo.setEnabled(enabled)

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
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

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
        if getattr(self, "_observe_mode", False):
            return
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

    def set_observe_mode(self, observe: bool, ct_loaded: bool = False):
        """观察模式下禁用 Entry 选择。"""
        self._observe_mode = observe
        if observe:
            self.start_btn.setEnabled(False)
            self.reselect_btn.setEnabled(False)
            _apply_hint_state(
                self.hint_label,
                "观察模式：切换到「穿刺训练」后可选择 Entry",
                "default",
            )
        else:
            self.set_model_loaded(ct_loaded)

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
        self.setMinimumSize(100, 100)
        self.setMaximumSize(152, 152)
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

        painter.setPen(color)
        fnt = QFont("Consolas", 10, QFont.Bold)
        painter.setFont(fnt)
        painter.drawText(
            QRect(int(cx_f - 44), int(cy_f + radius - 22), 88, 18),
            Qt.AlignCenter,
            f"{self._angle_deg:.1f}°",
        )

