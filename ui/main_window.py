"""主窗口 - 整合所有模块"""
import logging
import os
import time
import traceback
import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QVector3D
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QFrame,
    QMessageBox,
    QLabel,
    QSizePolicy,
    QScrollArea,
    QPushButton,
    QShortcut,
)

from core.device_manager import DeviceManager
from core.imu_diagnostic_recorder import ImuDiagnosticRecorder
from core.dicom_loader import DicomModelLoader
from core.imu_geometry_config import apply_kinematics, load_config, save_config
from core.puncture_depth_controller import PunctureDepthController
from core.imu_kinematics import (
    imu_position_from_tip,
    needle_axis_for_position,
    needle_axis_scene_normalized,
    needle_body_angle_deg,
    needle_body_bias_deg,
    scene_z_ccw_deg,
    scene_x_deg,
    scene_y_mirror,
    needle_tilt_from_scene_down_deg,
    capture_vertical_display_offset,
    clear_display_offset,
    display_offset_dict,
    display_offset_enabled,
    quat_slerp,
    tip_position_from_fixed,
)
from ui.widgets.gl_widget import GLVisualizationWidget
from ui.widgets.range_depth_panel import RangeDepthPanel
from ui.widgets.panels import (
    CTModelPanel,
    DeviceConnectionPanel,
    IMUDataPanel,
    PuncturePointPanel,
)
from ui.widgets.puncture_point_selector import PuncturePointSelector
from ui.widgets.simulation_panel import SimulationPanel
from ui.widgets.workflow_stepper import WorkflowStepBar
from ui.widgets.alignment_hud import AlignmentHudPanel
from ui.widgets.prep_sidebar import PrepSidebar, PrepSection
from ui.widgets.ui_helpers import configure_side_scroll
from ui.widgets.observation_panel import ObservationPanel
from core.observation_translation_controller import ObservationTranslationController
from core.serial_auto_connector import SerialAutoConnector

logger = logging.getLogger(__name__)


def default_intracerebral_hemorrhage_target_mm(bbox):
    """
    默认出血靶点：右侧豆状核/基底节区（高血压脑出血最常见部位之一）。

    加载后场景系方向（见 core/dicom_loader.py 的两次旋转链）：
        X = 前后（+ 前方）
        Y = 上下（+ 上方）
        Z = 左右（+ 右方）

    临床常用偏移（mm）：前 ~15 mm、上 ~25 mm、右 ~30 mm。
    最终 clip 到 bbox 半轴的 85%，防止不同头围下落到模型外。
    """
    offset = np.array(
        [
            -15.0,   # 略向额侧（前方为正 → 取负）
            25.0,    # 基底节层面（中央偏上）
            30.0,    # 右侧外侧（豆状核区）
        ],
        dtype=float,
    )
    size = np.asarray(bbox["size"], dtype=float).reshape(3)
    half = size / 2.0
    return np.clip(offset, -half * 0.85, half * 0.85)


class MainWindow(QMainWindow):
    """主窗口"""

    return_to_mode_selector = pyqtSignal()

    def __init__(self, operation_mode=DeviceConnectionPanel.MODE_PUNCTURE_TRAINING):
        super().__init__()
        self._operation_mode = operation_mode
        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle(
            "手术探针定位系统 - 姿态观察模式"
            if operation_mode == DeviceConnectionPanel.MODE_NEEDLE_OBSERVE
            else "手术探针定位系统 - 穿刺训练模式"
        )

        # 几何参数需在 UI 创建前可用
        # 你当前实测：IMU 中心到针尖约 20cm
        self.needle_length = 200.0
        self._range_state = "pending"
        self._range_state_text = "未连接"

        self._init_core_components()
        self._init_ui()

        self.space_shortcut = QShortcut(Qt.Key_Space, self)
        self.space_shortcut.setContext(Qt.WindowShortcut)
        self.space_shortcut.activated.connect(self._on_space_pressed)

        self._ui_ready = False
        self._closing = False
        self.puncture_selector = None
        self.puncture_point = None
        self.puncture_normal = None
        self.target_point = None
        self.target_direction_world = None
        self._default_target_point = None
        self._ct_model_center = None
        self._selecting_target = False
        self._ct_model_loaded = False
        self._ct_loading = False
        self.ct_loader_thread = None
        # 姿态观察模式用于展示测距产生的沿轴位移，不要求先静止 0.3 秒，
        # 也不因姿态小幅变化冻结位移；穿刺模式仍由独立状态机负责严格准入。
        self.observation_controller = ObservationTranslationController(
            require_stationary=False,
            enforce_angle_pause=False,
        )
        self._imu_recorder = ImuDiagnosticRecorder(
            Path(__file__).resolve().parents[1] / "imu_diagnostic_logs"
        )
        self._imu_record_timer = QTimer(self)
        self._imu_record_timer.setSingleShot(True)
        self._imu_record_timer.timeout.connect(
            lambda: self._stop_imu_diagnostic_recording("120 秒自动结束")
        )
        self._last_gyro = None
        self._auto_connector_thread = None
        self._auto_connector = None
        self._range_auto_port = None
        self._observation_range_enabled = False
        self._fixed_training_case = self._load_fixed_training_case()
        self._fixed_case_active = False

        self._current_quaternion = [1, 0, 0, 0]
        self._current_euler = [0, 0, 0]
        self._last_needle_direction = np.array([0, 0, -1])
        self._needle_direction = [0, 0, -1]
        self._smooth_enabled = False
        self._smooth_alpha = 0.25
        self._smooth_quat = None

        self._cached_imu_pos = np.zeros(3)
        self._cached_tip_pos = np.zeros(3)

        self._connect_signals()
        self._load_imu_geometry()
        self._sync_needle_length_to_gl()
        self._init_timers()
        self._apply_window_geometry()

        self.alignment_timer = QTimer(self)
        self.alignment_timer.setInterval(10)
        self.alignment_timer.timeout.connect(self._update_alignment)

        self._ALIGN_THRESHOLD_DEG = PunctureDepthController.ALIGN_THRESHOLD_DEG
        self._current_angle_error_deg = 0.0
        self._puncture_plan_depth = 0.0
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        # 进针完成标志：避免完成后 tick 反复 enable_puncture_mode 导致进度条闪烁
        self._puncture_finished = False
        # 进针期间用 override tip 接管针尖位置（避免每帧 IMU 数据退回原点）
        self._puncture_override_tip = None
        self.puncture_controller = PunctureDepthController()

        self._refresh_workflow_steps()
        self._apply_operation_mode_ui()
        self._ui_ready = True
        if not self._is_observe_mode():
            QTimer.singleShot(0, self._auto_load_fixed_training_case)
        else:
            QTimer.singleShot(300, self._start_auto_connection)
        logger.info("✓ 主窗口初始化完成")

    def _load_fixed_training_case(self):
        path = Path(__file__).resolve().parents[1] / "config" / "fixed_training_case.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("固定训练 CT 配置读取失败: %s", exc)
            return None

    def _auto_load_fixed_training_case(self):
        if self._closing or self._is_observe_mode() or not self._fixed_training_case:
            return
        root = Path(__file__).resolve().parents[1]
        folder = (root / self._fixed_training_case["relative_dicom_path"]).resolve()
        if not folder.is_dir():
            self.ct_panel.set_model_loaded(False, {"error": f"固定训练 CT 不存在：{folder}"})
            self._update_header_status()
            return
        ok, message = DicomModelLoader.validate_folder_identity(
            str(folder), self._fixed_training_case
        )
        if not ok:
            self.ct_panel.set_model_loaded(False, {"error": message})
            QMessageBox.critical(self, "固定训练 CT 校验失败", message)
            return
        self._fixed_case_active = True
        self.puncture_panel.set_fixed_case_mode(True)
        self._on_ct_load(str(folder))

    def _start_auto_connection(self, probe_range=None):
        if self._closing or (not self._is_observe_mode() and self.puncture_point is None):
            return
        if self._auto_connector_thread is not None:
            return
        self._auto_connector_thread = QThread(self)
        if probe_range is None:
            probe_range = not self._is_observe_mode() or self._observation_range_enabled
        self._auto_connector = SerialAutoConnector(
            probe_range=bool(probe_range), parent=None
        )
        self._auto_connector.moveToThread(self._auto_connector_thread)
        self._auto_connector_thread.started.connect(self._auto_connector.run)
        self._auto_connector.status.connect(self._on_auto_connection_status)
        self._auto_connector.finished.connect(self._on_auto_connection_finished)
        self._auto_connector.finished.connect(self._auto_connector_thread.quit)
        self._auto_connector_thread.finished.connect(self._on_auto_connection_thread_finished)
        self._auto_connector_thread.start()

    def _on_auto_connection_status(self, text):
        if self._is_observe_mode() and hasattr(self, "observation_panel"):
            self.observation_panel.state_label.setText(text)

    def _on_auto_connection_finished(self, result):
        if self._closing:
            return
        connection_issues = []
        imu_matches = result.get("imu_matches", [])
        range_matches = result.get("range_matches", [])
        imu_port = result.get("imu")
        range_port = result.get("range")
        if imu_port:
            self._select_combo_port(self.connection_panel.port_combo, imu_port)
            # Space 后的二次探测同时会看到已经工作的 IMU；不要重连它，
            # 否则会在锁轴平移过程中短暂断开并清掉控制器状态。
            if not (
                self.device_manager.is_connected
                and self.device_manager.port == imu_port
            ):
                self._on_serial_connect(imu_port, 115200, self._operation_mode)
        elif len(imu_matches) != 1:
            logger.warning("IMU 自动识别未得到唯一结果: %s", imu_matches)
            connection_issues.append("IMU 自动识别失败，请展开设备详情选择端口")
            self.connection_panel.status_text.setText(
                "自动识别失败，请人工选择 IMU 端口"
                if not imu_matches else "检测到多个 IMU 端口，请人工选择"
            )
        if range_port and self._is_observe_mode():
            # 姿态观察在按 Space 前不启动测距线程。只记住自动识别到的端口，
            # 进入锁轴平移后再打开它，避免启动阶段测距噪声影响零点。
            self._range_auto_port = range_port
            self._select_combo_port(self.range_depth_panel.ports, range_port)
            if self._observation_range_enabled:
                self.range_depth_panel.start()
            else:
                self.observation_panel.report_action("测距端口已识别，按 Space 后启用测距")
        elif range_port:
            self._select_combo_port(self.range_depth_panel.ports, range_port)
            self.range_depth_panel.start()
        elif self._is_observe_mode() and not self._observation_range_enabled:
            # 初始观察模式只探测 IMU；测距端口在 Space 后才探测。
            pass
        elif len(range_matches) != 1:
            logger.warning("测距自动识别未得到唯一结果: %s", range_matches)
            connection_issues.append("测距自动识别失败，请展开测距详情选择端口")
        if connection_issues and self._is_observe_mode():
            self.observation_panel.report_action("；".join(connection_issues))
        self._update_observation_panel()

    def _on_auto_connection_thread_finished(self):
        if self._auto_connector is not None:
            self._auto_connector.deleteLater()
        if self._auto_connector_thread is not None:
            self._auto_connector_thread.deleteLater()
        self._auto_connector = None
        self._auto_connector_thread = None

    @staticmethod
    def _select_combo_port(combo, port):
        index = combo.findData(port)
        if index < 0:
            combo.addItem(port, port)
            index = combo.findData(port)
        combo.setCurrentIndex(index)

    def _is_observe_mode(self):
        return self._operation_mode == DeviceConnectionPanel.MODE_NEEDLE_OBSERVE

    def _apply_operation_mode_ui(self):
        mode = self._operation_mode
        observe = self._is_observe_mode()
        self.puncture_panel.set_observe_mode(observe, self._ct_model_loaded)
        self.sim_panel.set_training_enabled(not observe)
        connected = self.device_manager.is_connected
        self.alignment_hud.set_observe_mode(observe, connected=connected)
        self.connection_panel.set_mode_switch_enabled(not connected)
        self.connection_panel.set_fixed_mode(mode)
        if hasattr(self, "workflow_bar"):
            # 两种模式都使用右侧工作区承载当前状态，隐藏旧六步条，避免重复信息。
            self.workflow_bar.setVisible(False)
        if hasattr(self, "observation_panel"):
            self.observation_panel.setVisible(observe)
        if hasattr(self, "prep_sidebar") and observe and self.prep_sidebar._sections:
            self.prep_sidebar._sections[PrepSidebar.SEC_CT].setVisible(False)
        elif hasattr(self, "prep_sidebar") and self.prep_sidebar._sections:
            self.prep_sidebar._sections[PrepSidebar.SEC_CT].setVisible(True)
        if hasattr(self, "puncture_panel"):
            self.puncture_panel.setVisible(not observe)
        if hasattr(self, "sim_panel"):
            self.sim_panel.setVisible(not observe)
        if hasattr(self, "sim_section"):
            self.sim_section.setVisible(not observe)
        if hasattr(self, "alignment_hud"):
            self.alignment_hud.setVisible(not observe)
        if hasattr(self, "live_guidance_card"):
            self.live_guidance_card.setVisible(not observe)
        if hasattr(self, "live_title"):
            self.live_title.setText("姿态观察" if observe else "实时引导")
        if observe:
            self.setWindowTitle("手术探针定位系统 - 姿态观察模式")
        else:
            self.setWindowTitle("手术探针定位系统 - 穿刺训练模式")

    def _init_core_components(self):
        self.device_manager = DeviceManager()
        self.dicom_loader = DicomModelLoader()

    def _init_ui(self):
        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._create_app_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._create_center_panel())
        controls = QFrame()
        controls.setObjectName("SidePanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)
        # 一个滚动区保留完整侧栏高度；实时工作区和测距放在设备详情之前。
        prep_panel = self._create_left_panel()
        content_layout.addWidget(self._create_right_panel())
        content_layout.addWidget(prep_panel)
        content_layout.addStretch()
        configure_side_scroll(scroll, content)
        controls_layout.addWidget(scroll)
        self._splitter.addWidget(controls)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)

        body.addWidget(self._splitter)
        root.addLayout(body, stretch=1)

    def _create_app_header(self):
        header = QFrame()
        header.setObjectName("AppHeader")
        outer = QVBoxLayout(header)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        top = QHBoxLayout()
        title = QLabel("手术探针定位系统")
        title.setObjectName("AppTitle")
        top.addWidget(title)
        top.addStretch()

        self.back_mode_button = QPushButton("返回模式选择")
        self.back_mode_button.setProperty("variant", "secondary")
        self.back_mode_button.clicked.connect(self._request_mode_selector)
        top.addWidget(self.back_mode_button)

        # 4 个全局状态 Chips
        self.header_chips = {}
        for key, label in [
            ("ct", "CT"),
            ("entry", "Entry"),
            ("target", "Target"),
            ("imu", "IMU"),
            ("range", "测距"),
        ]:
            chip = QLabel(f"{label} ?")
            chip.setObjectName("StatusChip")
            chip.setProperty("chipState", "pending")
            top.addWidget(chip)
            self.header_chips[key] = chip

        outer.addLayout(top)

        self.workflow_bar = WorkflowStepBar()
        outer.addWidget(self.workflow_bar)
        return header

    def _create_center_panel(self):
        panel = QFrame()
        panel.setObjectName("ViewportPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(0)

        self.gl_widget = GLVisualizationWidget()
        self.gl_widget.needle_length = self.needle_length
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.gl_widget)
        return panel

    def _create_left_panel(self):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.connection_panel = DeviceConnectionPanel()
        self.ct_panel = CTModelPanel()
        self.imu_panel = IMUDataPanel()
        self.range_depth_panel = RangeDepthPanel(
            lambda: self.device_manager.port if self.device_manager.is_connected else None
        )

        self.prep_sidebar = PrepSidebar()
        self.prep_sidebar.add_section("1  影像 · DICOM", self.ct_panel)
        self.prep_sidebar.add_section("2  设备 · IMU 串口", self.connection_panel)
        self.prep_sidebar.add_section("3  遥测 · IMU", self.imu_panel)
        outer.addWidget(self.prep_sidebar)
        return panel

    def _create_right_panel(self):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(7)

        live_title = QLabel("实时引导")
        live_title.setObjectName("SectionTitle")
        self.live_title = live_title
        outer_layout.addWidget(live_title)

        # 测距是实时工作区的第二个栏位，避免用户在右侧底部来回寻找深度。
        self.range_section = PrepSection(
            "2  实时穿刺深度 · 100m 测距",
            self.range_depth_panel,
            expanded=True,
        )
        outer_layout.addWidget(self.range_section)

        # 角度和进针深度属于同一个实时决策区，固定病例下无需在两张卡片之间切换视线。
        self.live_guidance_card = QFrame()
        self.live_guidance_card.setObjectName("LiveGuidanceCard")
        live_layout = QVBoxLayout(self.live_guidance_card)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(6)

        self.alignment_hud = AlignmentHudPanel()
        live_layout.addWidget(self.alignment_hud)

        self.observation_panel = ObservationPanel()
        self.observation_panel.toggle_translation.connect(self._toggle_observation_translation)
        self.observation_panel.zero_requested.connect(self.range_depth_panel.set_zero)
        self.observation_panel.recording_toggled.connect(self._toggle_imu_diagnostic_recording)
        outer_layout.addWidget(self.observation_panel)

        self.puncture_panel = PuncturePointPanel()
        self.puncture_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        live_layout.addWidget(self.puncture_panel)
        outer_layout.addWidget(self.live_guidance_card)

        self.sim_panel = SimulationPanel()
        self.sim_section = PrepSection("路径引导 · 可选", self.sim_panel, expanded=False)
        outer_layout.addWidget(self.sim_section)
        outer_layout.addStretch()
        return panel

    def _refresh_workflow_steps(self):
        if self._is_observe_mode():
            states = ["pending"] * 6
            if not self.device_manager.is_connected:
                states[2] = "active"
            else:
                states[2] = "done"
                states[3] = "active"
            self.workflow_bar.set_states(states)
            self._update_header_status()
            self._sync_prep_sidebar()
            return

        states = ["pending"] * 6
        if not self._ct_model_loaded:
            states[0] = "active"
        else:
            states[0] = "done"
            if self.puncture_point is None or self.target_point is None:
                states[1] = "active"
            else:
                states[1] = "done"
                devices_ready = (
                    self.device_manager.is_connected
                    and self._range_state == "done"
                )
                if not devices_ready:
                    states[2] = "active"
                else:
                    states[2] = "done"
                    puncture_state = self.puncture_controller.state
                    if puncture_state == PunctureDepthController.COMPLETE:
                        states[3] = "done"
                        states[4] = "done"
                        states[5] = "done"
                    elif puncture_state in (
                        PunctureDepthController.PUNCTURING,
                        PunctureDepthController.ANGLE_PAUSED,
                    ):
                        states[3] = "done"
                        states[4] = "active"
                    else:
                        states[3] = "active"
        self.workflow_bar.set_states(states)
        self._update_header_status()
        self._sync_prep_sidebar()

    def _sync_prep_sidebar(self):
        # 设备、影像和遥测属于详情，保持用户选择的折叠状态。
        return

    def _update_header_status(self):
        def _set_chip(key, state, text=None):
            chip = self.header_chips.get(key)
            if chip is None:
                return
            labels = {"ct": "CT", "entry": "Entry", "target": "Target", "imu": "IMU", "range": "测距"}
            label = labels.get(key, key)
            display = text if text is not None else (f"{label} ✓" if state == "done" else f"{label} ?")
            chip.setText(display)
            chip.setProperty("chipState", state)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

        _set_chip("ct", "done" if self._ct_model_loaded else "pending")
        _set_chip("entry", "done" if self.puncture_point is not None else "pending")
        _set_chip("target", "done" if self.target_point is not None else "pending")

        if self.device_manager.is_connected:
            fps = getattr(self, "_display_fps", 0)
            text = f"IMU {fps:.0f}Hz" if fps > 0 else "IMU ✓"
            _set_chip("imu", "done", text)
        else:
            _set_chip("imu", "pending")

        range_text = {
            "pending": "测距 ?",
            "active": "测距连接中",
            "done": "测距 ✓",
            "error": "测距异常",
        }.get(self._range_state, "测距 ?")
        _set_chip("range", self._range_state, range_text)

    def _apply_window_geometry(self):
        """按当前屏幕可用区域设置窗口大小，避免超出显示器导致面板被裁切。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.setMinimumSize(900, 600)
            self.resize(1280, 800)
            return

        avail = screen.availableGeometry()
        margin = 16
        max_w = max(900, avail.width() - margin)
        max_h = max(600, avail.height() - margin)

        min_w = min(900, max_w)
        min_h = min(600, max_h)
        self.setMinimumSize(min_w, min_h)

        target_w = int(min(max_w, max(min_w, avail.width() * 0.88)))
        target_h = int(min(max_h, max(min_h, avail.height() * 0.90)))
        self.resize(target_w, target_h)

        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

        controls_w = int(max(360, target_w * 0.35))
        viewport_w = max(420, target_w - controls_w)
        self._splitter.setSizes([viewport_w, controls_w])

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_geometry_applied_on_show", False):
            self._apply_window_geometry()
            self._geometry_applied_on_show = True
            self._refresh_workflow_steps()

    def _connect_signals(self):
        self.device_manager.data_received.connect(self._on_device_data)
        self.device_manager.connected.connect(self._on_device_connected_wrapper)
        self.device_manager.disconnected.connect(self._on_device_disconnected)
        self.device_manager.error_occurred.connect(self._on_device_error)

        self.connection_panel.connect_clicked.connect(self._on_serial_connect)
        self.connection_panel.disconnect_clicked.connect(self._on_serial_disconnect)

        self.connection_panel.calibration_clicked.connect(self._on_calibration_requested)
        self.connection_panel.operation_mode_changed.connect(self._on_operation_mode_changed)

        # IMU 平滑 / 场景安装标定
        self.imu_panel.smoothing_changed.connect(self._on_smoothing_changed)
        self.imu_panel.vertical_calibrate_clicked.connect(self._on_vertical_calibrate)
        self.imu_panel.vertical_recalibrate_clicked.connect(self._on_vertical_recalibrate)
        self.imu_panel.reset_view_clicked.connect(self._on_reset_view)
        self.imu_panel.save_viewport_clicked.connect(self._on_save_viewport)

        self.sim_panel.simulation_started.connect(self._on_simulation_started)
        self.sim_panel.simulation_stopped.connect(self._on_simulation_stopped)
        self.sim_panel.orientation_locked.connect(self._on_orientation_locked)

        self.ct_panel.load_clicked.connect(self._on_ct_load)
        self.ct_panel.clear_clicked.connect(self._on_ct_clear)
        self.ct_panel.visibility_changed.connect(self.gl_widget.set_head_model_visible)

        self.dicom_loader.progress_updated.connect(self._on_ct_progress)
        self.dicom_loader.loading_finished.connect(self._on_ct_loaded)
        self.dicom_loader.loading_failed.connect(self._on_ct_failed)

        self.puncture_panel.start_selection_clicked.connect(self._on_start_selection)
        self.puncture_panel.reselect_clicked.connect(self._on_reselect_puncture_point)
        self.puncture_panel.start_target_selection_clicked.connect(
            self._on_start_target_selection
        )
        self.puncture_panel.reselect_target_clicked.connect(self._on_reselect_target)
        self.puncture_panel.use_default_target_clicked.connect(
            self._on_use_default_target
        )
        self.puncture_panel.puncture_reset_clicked.connect(self._on_puncture_reset)
        self.range_depth_panel.sensor_state_changed.connect(
            self._on_range_sensor_state_changed
        )
        self.range_depth_panel.range_sample.connect(self._on_range_sample)
        self.range_depth_panel.range_validity_changed.connect(
            self._on_range_validity_changed
        )
        self.range_depth_panel.baseline_changed.connect(
            self._on_range_baseline_changed
        )

    def _on_range_sensor_state_changed(self, state, text):
        self._range_state = state
        self._range_state_text = text
        if state == "done":
            events = self.puncture_controller.set_range_status(
                connected=True, valid=False, reason="等待有效测距读数"
            )
        elif state == "error":
            events = self.puncture_controller.set_range_status(
                connected=False, valid=False, reason=text, error=True
            )
        else:
            events = self.puncture_controller.set_range_status(
                connected=False, valid=False, reason=text
            )
        self._consume_puncture_events(events)
        self.observation_controller.set_range_status(
            connected=state == "done", valid=False, reason=text
        )
        self._update_observation_panel()
        self._update_header_status()
        self._refresh_workflow_steps()

    def _on_range_validity_changed(self, valid, reason):
        events = self.puncture_controller.set_range_status(
            connected=self._range_state == "done",
            valid=bool(valid),
            reason=reason or "测距数据无效",
            error=not bool(valid),
        )
        self._consume_puncture_events(events)
        self.observation_controller.set_range_status(
            connected=self._range_state == "done",
            valid=bool(valid),
            reason=reason or "测距数据无效",
        )
        self._update_observation_panel()
        self._refresh_workflow_steps()

    def _on_range_baseline_changed(self, valid, baseline_mm):
        events = self.puncture_controller.set_baseline(valid, baseline_mm)
        self._consume_puncture_events(events)
        was_locked = self.observation_controller.locked
        self.observation_controller.set_baseline(valid, baseline_mm)
        if was_locked and not self.observation_controller.locked:
            self.observation_panel.report_action("测距零点丢失，已退出平移")
        self._update_observation_panel()
        self._refresh_workflow_steps()

    def _on_range_sample(self, raw_mm, filtered_mm, measured_depth_mm, age_s):
        if self._is_observe_mode():
            self.observation_controller.update_range(
                measured_depth_mm, measured_depth_mm is not None
            )
            self.observation_controller.tick()
            self._update_observation_panel()
            # 测距帧不是 IMU 帧，不能等下一帧姿态数据才刷新针尖，否则界面
            # 会显示“已应用位移”但针体仍停在原处。
            self._refresh_3d_from_current_state()
            return
        if measured_depth_mm is None:
            self._refresh_workflow_steps()
            return
        events = self.puncture_controller.update_range(
            raw_mm,
            filtered_mm,
            measured_depth_mm,
            age_s=age_s,
            valid=True,
        )
        self._consume_puncture_events(events)

    def _consume_puncture_events(self, events):
        for event in events or []:
            event_type = event.get("type")
            if event_type == "state_changed":
                state = event["state"]
                reason = event.get("reason", "")
                self.puncture_panel.set_puncture_state(state, reason)
                if state == PunctureDepthController.PUNCTURING:
                    self.gl_widget.clear_fixed_tip()
                self._refresh_workflow_steps()
                logger.info("穿刺状态: %s (%s)", state, reason)
            elif event_type == "depth_applied":
                applied = float(event["applied_depth_mm"])
                plan = float(event["plan_depth_mm"])
                self._puncture_current_depth = applied
                self.puncture_panel.update_puncture_depth(applied, plan)
                if self.puncture_controller.state == PunctureDepthController.PUNCTURING:
                    self._apply_tip_at_depth(applied)
            elif event_type == "completed":
                self._puncture_finished = True
                final_depth = float(event["final_depth_mm"])
                self._puncture_current_depth = final_depth
                self.puncture_panel.set_puncture_done()
                logger.info(
                    "穿刺完成 — 规划深度 %.1f mm，应用深度 %.1f mm",
                    self._puncture_plan_depth,
                    final_depth,
                )

    def _init_timers(self):
        self.panel_update_timer = QTimer()
        self.panel_update_timer.setInterval(100)  # 10 Hz，数据显示无需高频刷新
        self.panel_update_timer.timeout.connect(self._update_panels)
        self.panel_update_timer.start()

        self._data_frame_count = 0
        self._fps_last_time = time.perf_counter()
        self._display_fps = 0

    def _on_operation_mode_changed(self, mode):
        if not getattr(self, "_ui_ready", False):
            return
        self._operation_mode = mode
        if self.device_manager.is_connected:
            self.device_manager.disconnect()
        self._apply_operation_mode_ui()
        self._refresh_workflow_steps()
        if not self._is_observe_mode():
            QTimer.singleShot(300, self._start_auto_connection)
        label = "姿态观察" if mode == DeviceConnectionPanel.MODE_NEEDLE_OBSERVE else "穿刺训练"
        print(f"[Main] 工作模式已切换: {label}")

    def _request_mode_selector(self):
        self.shutdown_for_mode_switch()
        self.hide()
        self.return_to_mode_selector.emit()

    def shutdown_for_mode_switch(self):
        """停止本工作页拥有的计时器和串口，供返回模式选择时复用。"""
        if getattr(self, "_closing", False):
            return
        self._closing = True
        self._stop_imu_diagnostic_recording("窗口关闭或切换模式")
        if self.device_manager.is_connected:
            self.device_manager.disconnect()
        if hasattr(self, "range_depth_panel"):
            self.range_depth_panel.shutdown()
        if hasattr(self, "alignment_timer"):
            self.alignment_timer.stop()
        if hasattr(self, "panel_update_timer"):
            self.panel_update_timer.stop()
        self.observation_controller.reset()
        if self._auto_connector is not None:
            self._auto_connector.stop()
        if self._auto_connector_thread is not None and self._auto_connector_thread.isRunning():
            self._auto_connector_thread.quit()
            self._auto_connector_thread.wait(2000)

    def _on_serial_connect(self, port, baudrate, mode):
        rp = self.range_depth_panel
        if rp.worker and rp.worker.is_alive() and port == rp.ports.currentData():
            QMessageBox.warning(self, '串口已占用', '该串口正在用于 100m 测距，请为 IMU 选择另一个串口。')
            return
        self._operation_mode = mode or DeviceConnectionPanel.MODE_PUNCTURE_TRAINING
        if self._operation_mode == DeviceConnectionPanel.MODE_PUNCTURE_TRAINING:
            if self.puncture_point is None:
                QMessageBox.warning(
                    self,
                    "请先选择 Entry",
                    "穿刺训练模式下需先加载 CT 并选择 Entry 与 Target，再连接 IMU。\n\n"
                    "若只想查看针体姿态，请将工作模式切换为「姿态观察（仅看针体）」。",
                )
                return
            if self.target_point is None:
                QMessageBox.warning(
                    self,
                    "请先选择 Target",
                    "请在右栏「② 选择 Target」中在 3D 上选点，或点击「使用默认」。",
                )
                return
        if not self.device_manager.connect(port, baudrate):
            QMessageBox.warning(self, "连接失败", f"无法连接到 {port}")

    def _on_serial_disconnect(self):
        self.device_manager.disconnect()

    def _on_device_disconnected(self):
        self._on_connection_changed(False)
        self._stop_alignment_monitoring()

    def _on_connection_changed(self, connected):
        if not connected:
            self._stop_imu_diagnostic_recording("IMU 断开")
            self._imu_recorder.clear_buffer()
        self.connection_panel.set_connected(connected)
        self.connection_panel.set_mode_switch_enabled(not connected)
        self.imu_panel.set_status(connected, self._display_fps if connected else 0)
        if not connected:
            self._display_fps = 0
        events = self.puncture_controller.set_imu_connected(connected)
        self._consume_puncture_events(events)
        was_locked = self.observation_controller.locked
        self.observation_controller.set_imu_connected(connected)
        if was_locked and not connected:
            self.observation_panel.report_action("IMU 已断开，已退出平移")
        self._update_observation_panel()
        self._apply_operation_mode_ui()
        self._update_header_status()
        self._refresh_workflow_steps()
        print("✓ 串口已连接" if connected else "✓ 串口已断开")

    def _on_device_error(self, error_msg):
        print(f"✗ 设备错误: {error_msg}")
        self._stop_imu_diagnostic_recording("IMU 连接错误")
        events = self.puncture_controller.set_imu_connected(False)
        self._consume_puncture_events(events)
        self.observation_controller.set_imu_connected(False)
        self._update_observation_panel()

    def _on_device_data(self, data):
        ui_receive_ns = time.monotonic_ns()
        try:
            quaternion = data.get("quaternion")
            euler = data.get("euler")

            if quaternion is None or euler is None:
                return

            self._current_quaternion = list(quaternion)
            self._current_euler = list(euler)

            gyro = data.get("gyro")
            if gyro is not None:
                self._last_gyro = np.asarray(gyro, dtype=float)
                calibrated = self.device_manager.get_gyro_bias() is not None
                calibrated_gyro = self.device_manager.get_calibrated_gyro(self._last_gyro)
                self.observation_controller.update_gyro(
                    calibrated_gyro, calibrated=calibrated
                )

            self._update_needle_direction_fast(quaternion)
            self.observation_controller.update_axis(self._needle_direction)
            self.observation_controller.tick()
            imu_pos, tip_pos = self._calculate_positions_fast(quaternion)
            self._cached_imu_pos = imu_pos
            self._cached_tip_pos = tip_pos
            self.gl_widget.update_data(imu_pos, tip_pos)
            self._update_observation_panel()
            if self._is_observe_mode():
                self._capture_imu_diagnostic_sample(data, ui_receive_ns)

            self._data_frame_count += 1
            now = time.perf_counter()
            if now - self._fps_last_time >= 1.0:
                self._display_fps = self._data_frame_count / (now - self._fps_last_time)
                self._data_frame_count = 0
                self._fps_last_time = now

        except Exception as e:
            print(f"✗ 数据处理错误: {e}")
            traceback.print_exc()

    def _capture_imu_diagnostic_sample(self, data, ui_receive_ns):
        """在正常绘制后记录传感器输入与实际显示位置。"""
        def vec(key, count):
            value = data.get(key)
            if value is None:
                return [None] * count
            arr = np.asarray(value, dtype=float).reshape(-1)
            return [float(arr[i]) if i < arr.size else None for i in range(count)]

        device_emit_ns = int(data.get("device_emit_ns", ui_receive_ns))
        ui_done_ns = time.monotonic_ns()
        q = vec("quaternion", 4)
        used_q = np.asarray(self._smooth_quat, dtype=float).reshape(4)
        gyro = vec("gyro", 3)
        acc = vec("acc", 3)
        euler = vec("euler", 3)
        axis = np.asarray(self.gl_widget._needle_direction, dtype=float).reshape(3)
        tip = np.asarray(self.gl_widget.tip_position, dtype=float).reshape(3)
        imu = np.asarray(self.gl_widget.imu_position, dtype=float).reshape(3)
        sample = dict(zip(
            ("q_w", "q_x", "q_y", "q_z",
             "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
             "acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2",
             "roll_deg", "pitch_deg", "yaw_deg"),
            q + gyro + acc + euler,
        ))
        sample.update({
            "used_q_w": float(used_q[0]), "used_q_x": float(used_q[1]),
            "used_q_y": float(used_q[2]), "used_q_z": float(used_q[3]),
            "device_emit_ns": device_emit_ns,
            "ui_receive_ns": ui_receive_ns,
            "ui_done_ns": ui_done_ns,
            "ui_queue_delay_ms": round((ui_receive_ns - device_emit_ns) / 1e6, 3),
            "ui_processing_ms": round((ui_done_ns - ui_receive_ns) / 1e6, 3),
            "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
            "tip_x_mm": float(tip[0]), "tip_y_mm": float(tip[1]), "tip_z_mm": float(tip[2]),
            "imu_x_mm": float(imu[0]), "imu_y_mm": float(imu[1]), "imu_z_mm": float(imu[2]),
            "smoothing_enabled": int(self._smooth_enabled),
            "translation_locked": int(self.observation_controller.locked),
            "range_raw_mm": self.range_depth_panel.latest,
            "range_filtered_mm": self.range_depth_panel.filtered_distance,
            "range_baseline_mm": self.range_depth_panel.baseline,
            "range_valid": int(self.observation_controller.range_valid),
            "measured_displacement_mm": float(self.observation_controller.measured_displacement_mm),
            "applied_displacement_mm": float(self.observation_controller.applied_displacement_mm),
        })
        if self._imu_recorder.push(sample):
            self._stop_imu_diagnostic_recording("样本上限自动结束")

    def _toggle_imu_diagnostic_recording(self):
        if self._imu_recorder.active:
            self._stop_imu_diagnostic_recording("手动结束")
            return
        if not self._is_observe_mode() or not self.device_manager.is_connected:
            self.observation_panel.set_recording_state(False, "请先在姿态观察模式连接 IMU")
            return
        metadata = {
            "mode": "姿态观察",
            "port": self.device_manager.port,
            "baudrate": self.device_manager.baudrate,
            "needle_length_mm": self.needle_length,
            "imu_geometry": load_config(),
        }
        self._imu_recorder.start(metadata)
        self._imu_record_timer.start(120_000)
        self.observation_panel.set_recording_state(
            True, f"记录中；已包含触发前 {self._imu_recorder.sample_count} 帧，点击结束并保存"
        )

    def _stop_imu_diagnostic_recording(self, reason):
        if not self._imu_recorder.active:
            return
        try:
            path = self._imu_recorder.stop(reason)
        except Exception as exc:
            logger.exception("姿态诊断记录保存失败")
            self.observation_panel.set_recording_state(True, f"保存失败：{exc}；可再次点击重试")
            return
        self._imu_record_timer.stop()
        self.observation_panel.set_recording_state(
            False, f"已保存：{path.name}\n位置：imu_diagnostic_logs（{reason}，同名 JSON 含配置）"
        )
        self.observation_panel.record_status_label.setToolTip(str(path))

    def _toggle_observation_translation(self):
        if not self._is_observe_mode():
            return
        was_locked = self.observation_controller.locked
        self.observation_controller.toggle_lock()
        if was_locked:
            message = "已退出平移，针尖回到原点"
            if self._observation_range_enabled:
                self.range_depth_panel.stop()
                self._observation_range_enabled = False
            self.range_depth_panel.clear_baseline()
        elif self.observation_controller.locked:
            message = self.observation_controller.reason
            if not self._observation_range_enabled:
                self._observation_range_enabled = True
                if self._range_auto_port:
                    self._select_combo_port(
                        self.range_depth_panel.ports, self._range_auto_port
                    )
                    self.range_depth_panel.start()
                else:
                    self.observation_panel.report_action("正在按 Space 后识别测距端口")
                    self._start_auto_connection(probe_range=True)
        else:
            message = self.observation_controller.reason
        self.observation_panel.report_action(message)
        self._update_observation_panel()
        self._refresh_3d_from_current_state()

    def _refresh_3d_from_current_state(self):
        """用当前姿态和控制器位置立即刷新 3D 针体。

        测距线程和键盘事件都可能在两帧 IMU 数据之间改变针尖位置；
        这里集中复用主线程中的 OpenGL 更新路径，保证数值与画面同步。
        """
        quaternion = getattr(self, "_current_quaternion", None)
        if quaternion is None:
            return
        try:
            imu_pos, tip_pos = self._calculate_positions_fast(quaternion)
            self._cached_imu_pos = np.asarray(imu_pos, dtype=float)
            self._cached_tip_pos = np.asarray(tip_pos, dtype=float)
            self.gl_widget.update_data(imu_pos, tip_pos)
        except Exception:
            logger.exception("测距位移刷新 3D 失败")

    def _on_space_pressed(self):
        if self._is_observe_mode():
            self._toggle_observation_translation()
        elif self.sim_section.is_expanded():
            self.sim_panel.lock_widget._toggle_lock()

    def _update_observation_panel(self):
        if not hasattr(self, "observation_panel"):
            return
        controller = self.observation_controller
        reason = controller.reason
        if self._is_observe_mode() and self.device_manager.is_connected and not controller.locked:
            if not self._observation_range_enabled:
                reason = "IMU 已连接，可按 Space 进入平移并启用测距"
        self.observation_panel.update_state(
            controller.state,
            reason,
            controller.measured_displacement_mm,
            controller.applied_displacement_mm,
            controller.angle_deg,
            can_zero=bool(getattr(self.range_depth_panel, "zero", None)
                          and self.range_depth_panel.zero.isEnabled()),
            locked=controller.locked,
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._on_space_pressed()
            event.accept()
            return
        super().keyPressEvent(event)

    def _calculate_positions_fast(self, quaternion):
        if self._is_observe_mode():
            direction = np.asarray(self._needle_direction, dtype=float).reshape(3)
            tip_pos = self.observation_controller.tip_position
            imu_pos = imu_position_from_tip(tip_pos, direction, float(self.needle_length))
            return np.asarray(imu_pos, dtype=float), tip_pos

        direction = needle_axis_for_position(quaternion)

        # 进针 override 优先：避免 clear_fixed_tip 后针尖漂移到原点
        if self._puncture_override_tip is not None:
            tip_pos = list(self._puncture_override_tip)
            imu_pos = imu_position_from_tip(
                tip_pos, direction, float(self.needle_length)
            )
            return np.array(imu_pos), np.array(tip_pos)

        fixed_tip = self.gl_widget.get_fixed_tip_position()
        tip_pos, skip = tip_position_from_fixed(fixed_tip)
        if skip:
            print("[Calculate] ⚠️ fixed_tip 无效，跳过本次计算")
            return np.zeros(3), np.zeros(3)

        imu_pos = imu_position_from_tip(
            tip_pos, direction, float(self.needle_length)
        )
        return imu_pos, tip_pos

    def _update_needle_direction_fast(self, quaternion):
        q_raw = np.asarray(quaternion, dtype=float).reshape(4)
        if self._smooth_enabled:
            if self._smooth_quat is None:
                self._smooth_quat = q_raw.copy()
            else:
                self._smooth_quat = quat_slerp(self._smooth_quat, q_raw, self._smooth_alpha)
            q_use = self._smooth_quat
        else:
            q_use = q_raw
            self._smooth_quat = q_raw.copy()

        direction = needle_axis_scene_normalized(q_use.tolist())
        if direction is not None:
            new_d = np.asarray(direction, dtype=float)
            n = float(np.linalg.norm(new_d))
            if n > 1e-9:
                new_d = new_d / n
            self._needle_direction = new_d.tolist()
            self._last_needle_direction = new_d.copy()
        self.gl_widget.update_needle_direction(self._needle_direction)

    def _load_imu_geometry(self):
        """启动时从 config/imu_geometry.json 恢复针轴几何与场景安装标定。"""
        self._imu_geometry_loading = True
        try:
            cfg = load_config()
            apply_kinematics(cfg)
            self.imu_panel.apply_settings(cfg)
            self.imu_panel.set_vertical_calibrate_status(display_offset_enabled())
            if cfg.get("needle_length_mm"):
                self.needle_length = float(cfg["needle_length_mm"])
            self._sync_needle_length_to_gl()
            self.gl_widget.apply_scene_orientation()
            smooth = cfg.get("smoothing", {})
            self._smooth_enabled = bool(smooth.get("enabled", False))
            self._smooth_alpha = float(smooth.get("alpha", 0.25))
            self._smooth_quat = None
            print(
                "[Main] IMU 几何已加载: "
                f"针轴={needle_body_angle_deg():.1f}°(顺时针), "
                f"针长={self.needle_length:.0f}mm, "
                f"竖直标定={'已设置' if display_offset_enabled() else '未设置'}"
            )
        except Exception as exc:
            print(f"[Main] ⚠ IMU 几何配置未加载，使用默认值: {exc}")
        finally:
            self._imu_geometry_loading = False

    def _sync_needle_length_to_gl(self):
        """针长唯一来源：MainWindow.needle_length（由 imu_geometry.json 加载）。"""
        nl = float(self.needle_length)
        self.gl_widget.needle_length = nl

    def _persist_imu_geometry(self):
        if getattr(self, "_imu_geometry_loading", False):
            return
        cfg = load_config()
        cfg["needle_body_angle_deg"] = needle_body_angle_deg()
        cfg["needle_body_bias_deg"] = needle_body_bias_deg()
        cfg["scene_z_ccw_deg"] = scene_z_ccw_deg()
        cfg["scene_x_deg"] = scene_x_deg()
        cfg["scene_y_mirror"] = scene_y_mirror()
        cfg["needle_length_mm"] = float(self.needle_length)
        cfg["display_offset"] = display_offset_dict()
        cfg["smoothing"] = self.imu_panel.smoothing_dict()
        save_config(cfg)

    def _on_vertical_calibrate(self):
        if not self.device_manager.is_connected:
            logger.warning("竖直校准失败：IMU 未连接")
            return
        capture_vertical_display_offset(self._current_quaternion)
        self._smooth_quat = None
        self._update_needle_direction_fast(self._current_quaternion)
        imu_pos, tip_pos = self._calculate_positions_fast(self._current_quaternion)
        self.gl_widget.update_data(imu_pos, tip_pos)
        self._persist_imu_geometry()
        self.imu_panel.set_vertical_calibrate_status(True)
        tilt = needle_tilt_from_scene_down_deg(self._needle_direction)
        logger.info(
            "竖直标定完成 — 偏竖直角 %.1f°, 配置已写入 config/imu_geometry.json",
            tilt,
        )

    def _on_vertical_recalibrate(self):
        clear_display_offset()
        self._smooth_quat = None
        if self.device_manager.is_connected:
            self._update_needle_direction_fast(self._current_quaternion)
        self._persist_imu_geometry()

    def _on_smoothing_changed(self, enabled: bool, alpha: float):
        self._smooth_enabled = bool(enabled)
        self._smooth_alpha = float(alpha)
        self._smooth_quat = None
        self._persist_imu_geometry()

    def _stop_alignment_monitoring(self):
        if self.alignment_timer.isActive():
            self.alignment_timer.stop()
        if self._is_observe_mode():
            self.alignment_hud.set_observe_mode(
                True, connected=self.device_manager.is_connected
            )
        else:
            self.alignment_hud.hide_guidance()
        self._refresh_workflow_steps()

    def _on_device_connected_wrapper(self):
        self._on_connection_changed(True)
        self._on_device_connected()

    def _on_clear_trajectory(self):
        self.gl_widget.clear_trajectory()

    def _on_reset_view(self):
        self.gl_widget.reset_view(clear_trajectory=False)

    def _on_save_viewport(self):
        self.gl_widget.capture_viewport_to_config()
        logger.info("默认视角已保存到 config/viewport.json")

    def _on_calibration_requested(self):
        """执行传感器校准序列：磁力计 + 陀螺仪零偏"""
        if not self.device_manager.is_connected:
            logger.warning("传感器校准：IMU 未连接，跳过")
            return

        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "传感器校准",
            "校准步骤：\n\n"
            "1. 将传感器水平静止放置\n"
            "2. 点击「确定」后保持静止3秒\n"
            "3. 等待校准完成提示\n\n"
            "开始校准？",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return

        print("=== 传感器校准开始 ===")

        # 1. 陀螺仪零偏校准
        self.device_manager.start_gyro_bias_calibration()

        # 2. 磁力计校准
        self.device_manager.calibrate_magnetic_start()

        # 3. 3秒后结束磁力计校准
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(3000, self._finish_calibration)

        self.connection_panel.btn_calibrate.setEnabled(False)
        self.connection_panel.btn_calibrate.setText("校准中...")
        print("  保持传感器静止... (3秒)")

    def _finish_calibration(self):
        """完成校准"""
        if self._closing:
            return
        self.device_manager.calibrate_magnetic_end()
        self.connection_panel.btn_calibrate.setEnabled(True)
        self.connection_panel.btn_calibrate.setText("校准传感器")
        logger.info("传感器校准完成：陀螺仪零偏已记录，磁力计校准已保存")

    def _on_simulation_started(self):
        if self._is_observe_mode():
            return
        print("[Main] 穿刺路径引导模式已启动")
        default_path = [0, 0, 1]
        self.gl_widget.set_preset_path(default_path)

        # 设置对齐目标方向并启动监控
        self.target_direction_world = np.array(default_path, dtype=float)
        self._start_alignment_monitoring()
        print(f"[Main] ✓ 已设置预设路径: {default_path}")

    def _on_simulation_stopped(self):
        print("[Main] 穿刺路径引导模式已停止")
        self._stop_alignment_monitoring()
        self.gl_widget.clear_path_lines()

    def _on_orientation_locked(self, direction):
        print(f"[Main] 姿态已锁定: {direction}")
        d = np.asarray(direction).tolist()
        self.gl_widget.set_target_path(d)

    def _update_panels(self):
        self.imu_panel.update_quaternion(self._current_quaternion)
        self.imu_panel.update_euler(self._current_euler)
        tilt = needle_tilt_from_scene_down_deg(self._needle_direction)
        self.imu_panel.update_needle_scene(self._needle_direction, tilt)

        if self.device_manager.is_connected:
            self.imu_panel.set_status(True, self._display_fps)
        self._update_header_status()

        inverted_direction = [
            -self._needle_direction[0],
            -self._needle_direction[1],
            -self._needle_direction[2],
        ]
        self.sim_panel.update_current_direction(inverted_direction)

    def closeEvent(self, event):
        self.shutdown_for_mode_switch()
        # 清理 CT 加载线程
        if self.ct_loader_thread is not None:
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait(3000)
            self.ct_loader_thread = None
        logger.info("✓ 程序已退出")
        event.accept()

    def _on_ct_load(self, folder_path):
        # 防止重复加载导致线程泄漏
        if self._ct_loading:
            print("[Main] ⚠ CT 正在加载中，忽略重复请求")
            return
        print(f"[Main] 开始加载DICOM文件夹: {folder_path}")
        self.ct_panel.set_loading(True)
        self._ct_loading = True

        # 清理旧线程
        if self.ct_loader_thread is not None:
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait(3000)

        self.ct_loader_thread = QThread()
        self.ct_loader_thread.setObjectName("CTLoader")
        self.dicom_loader.moveToThread(self.ct_loader_thread)
        self.ct_loader_thread.started.connect(
            lambda: self.dicom_loader.load_dicom_folder(folder_path)
        )
        self.ct_loader_thread.start()

    def _on_ct_progress(self, value, message):
        self.ct_panel.update_progress(value, message)

    def _on_ct_loaded(self, model_data):
        self._ct_loading = False
        logger.info("CT模型加载成功")
        print(f"  顶点数: {model_data['num_vertices']}")
        print(f"  面数: {model_data['num_faces']}")

        self.gl_widget.load_head_model(model_data["vertices"], model_data["faces"])
        self.ct_panel.set_loading(False)
        self.ct_panel.set_model_loaded(True, model_data)

        if self.puncture_selector is None:
            self.puncture_selector = PuncturePointSelector(self.gl_widget)
            self.puncture_selector.point_selected.connect(self._on_puncture_point_selected)
            self.puncture_selector.target_selected.connect(self._on_target_point_selected)

        self.puncture_selector.set_model(model_data["vertices"], model_data["faces"])

        self.puncture_panel.set_model_loaded(True)
        self._ct_model_loaded = True
        self._refresh_workflow_steps()

        center = np.asarray(model_data["center"], dtype=float)
        self._ct_model_center = center
        bbox = model_data.get("bbox") or {}
        if self._fixed_case_active and self._fixed_training_case:
            self._default_target_point = np.asarray(
                self._fixed_training_case["target_point_mm"], dtype=float
            )
        else:
            self._default_target_point = default_intracerebral_hemorrhage_target_mm(bbox)
        self.target_point = self._default_target_point.copy()
        self.gl_widget.set_bleeding_point(self.target_point)
        self.puncture_panel.set_target_point(
            self.target_point, entry_point=None, from_default=True
        )

        if self._fixed_case_active and self._fixed_training_case:
            entry = np.asarray(self._fixed_training_case["entry_point_mm"], dtype=float)
            vertices = np.asarray(model_data["vertices"], dtype=float)
            nearest_distance = float(np.min(np.linalg.norm(vertices - entry, axis=1)))
            plan_depth = float(np.linalg.norm(self.target_point - entry))
            max_surface = float(self._fixed_training_case.get("entry_surface_tolerance_mm", 3.0))
            expected_plan = float(self._fixed_training_case.get("expected_plan_depth_mm", plan_depth))
            plan_tol = float(self._fixed_training_case.get("plan_depth_tolerance_mm", 3.0))
            if nearest_distance > max_surface or abs(plan_depth - expected_plan) > plan_tol:
                self._fixed_case_active = False
                self._ct_model_loaded = False
                self.ct_panel.set_model_loaded(False, {
                    "error": (
                        f"固定点几何核验失败：Entry 到网格最近点 {nearest_distance:.1f} mm，"
                        f"规划深度 {plan_depth:.1f} mm"
                    )
                })
                QMessageBox.critical(self, "固定点校验失败", "当前 CT 网格与固定训练点不匹配。")
                return
            center = np.asarray(model_data.get("center", [0, 0, 0]), dtype=float)
            normal = entry - center
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
            self._on_puncture_point_selected(entry, normal)
        print(
            f"[Main] 默认出血靶点（右豆状核区）: "
            f"[{self.target_point[0]:.1f}, {self.target_point[1]:.1f}, {self.target_point[2]:.1f}] mm"
        )

        if hasattr(self, "ct_loader_thread"):
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait()

        self._apply_operation_mode_ui()
        self._refresh_workflow_steps()
        if not self._is_observe_mode():
            QTimer.singleShot(300, self._start_auto_connection)

    def _on_ct_failed(self, error_msg):
        self._ct_loading = False
        print(f"[Main] ✗ CT模型加载失败: {error_msg}")
        self.ct_panel.set_loading(False)
        QMessageBox.critical(self, "加载失败", error_msg)
        if hasattr(self, "ct_loader_thread"):
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait()

    def _on_ct_clear(self):
        self.gl_widget.clear_head_model()
        self.ct_panel.set_model_loaded(False)
        self._ct_model_loaded = False
        self.puncture_point = None
        self.puncture_normal = None
        self.target_point = None
        self._default_target_point = None
        self._ct_model_center = None
        self._selecting_target = False
        self._puncture_plan_depth = 0.0
        self._puncture_current_depth = 0.0
        self._puncture_finished = False
        self._puncture_override_tip = None
        self._stop_alignment_monitoring()
        self._consume_puncture_events(
            self.puncture_controller.reset(clear_baseline=True)
        )
        self.range_depth_panel.clear_baseline()
        self.puncture_panel.clear()
        self.puncture_panel.clear_target_display()
        self._apply_operation_mode_ui()
        self._refresh_workflow_steps()
        print("[Main] ✓ CT模型已清除")

    def _effective_target(self):
        if self.target_point is not None:
            return np.asarray(self.target_point, dtype=float).reshape(3)
        return None

    def _entry_target_distance_mm(self):
        tgt = self._effective_target()
        if tgt is None or self.puncture_point is None:
            return None
        entry = np.asarray(self.puncture_point, dtype=float).reshape(3)
        return float(np.linalg.norm(tgt - entry))

    def _update_target_direction_world(self):
        if self.puncture_point is None:
            return
        tgt = self._effective_target()
        if tgt is None:
            return
        entry = np.asarray(self.puncture_point, dtype=float).reshape(3)
        vec = tgt - entry
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            return
        self.target_direction_world = vec / n

    def _apply_target_to_scene(self, from_default: bool = False):
        tgt = self._effective_target()
        if tgt is None:
            return
        self.gl_widget.set_bleeding_point(tgt)
        entry = self.puncture_point
        if entry is not None:
            self.gl_widget.set_entry_target_line(entry, tgt)
        self.puncture_panel.set_target_point(
            tgt, entry_point=entry, from_default=from_default
        )
        self._update_target_direction_world()
        if (
            self.device_manager.is_connected
            and not self._is_observe_mode()
            and hasattr(self, "target_direction_world")
        ):
            self._start_alignment_monitoring()

    def _on_target_point_selected(self, point):
        logger.info("Target 手动选择: %s", point)
        self._selecting_target = False
        self.target_point = np.asarray(point, dtype=float).reshape(3)
        if self.puncture_selector:
            self.puncture_selector.setEnabled(False)
            self.puncture_selector.set_selection_mode(
                PuncturePointSelector.MODE_ENTRY
            )
        self.puncture_panel.set_target_selecting_mode(False)
        self._on_target_set()
        logger.info(
            "Target 已设置: [%.1f, %.1f, %.1f] mm, 规划深度 %.1f mm",
            self.target_point[0], self.target_point[1], self.target_point[2],
            self._puncture_plan_depth,
        )

    def _on_start_target_selection(self):
        logger.info("Target 使用 CT 加载后的默认点，跳过手动选择")

    def _on_reselect_target(self):
        print("[Main] 重新选择 Target")
        self.target_point = None
        self.gl_widget.set_bleeding_point(None)
        if self.puncture_point is not None:
            self.gl_widget.set_entry_target_line(self.puncture_point, None)
        self.puncture_panel.clear_target_display()
        self._on_start_target_selection()

    def _on_use_default_target(self):
        if self._default_target_point is None:
            QMessageBox.warning(self, "无默认 Target", "CT 模型未提供默认 Target。")
            return
        self.target_point = np.asarray(self._default_target_point, dtype=float).reshape(
            3
        )
        self._selecting_target = False
        if self.puncture_selector:
            self.puncture_selector.setEnabled(False)
        self.puncture_panel.set_target_selecting_mode(False)
        self._on_target_set()
        logger.info(
            "续接 — 使用默认 Target: [%.1f, %.1f, %.1f] mm, 规划深度 %.1f mm",
            self.target_point[0], self.target_point[1], self.target_point[2],
            self._puncture_plan_depth,
        )

    def _on_puncture_point_selected(self, point, normal):
        print(f"[Main] 📍 穿刺点已选择: {point}")
        print(f"[Main] 📐 法线方向: {normal}")

        self.puncture_point = point
        self.puncture_normal = normal

        self._selecting_target = False
        if self.target_point is None and self._default_target_point is not None:
            self.target_point = np.asarray(self._default_target_point, dtype=float).copy()

        self.puncture_panel.set_puncture_point(point, normal)
        self.gl_widget.set_puncture_point(point, normal)
        if self.target_point is not None:
            self.gl_widget.set_bleeding_point(self.target_point)
            self.gl_widget.set_entry_target_line(point, self.target_point)
            self.puncture_panel.set_target_point(
                self.target_point, entry_point=point, from_default=True
            )
        else:
            self.gl_widget.set_bleeding_point(None)
            self.gl_widget.set_entry_target_line(point, None)
            self.puncture_panel.clear_target_display()
        self.puncture_panel.set_target_selecting_mode(False)

        if self.puncture_selector:
            self.puncture_selector.setEnabled(False)
            self.puncture_selector.set_selection_mode(
                PuncturePointSelector.MODE_ENTRY
            )
        self.puncture_panel.set_selecting_mode(False)

        self._on_target_set()

        logger.info(
            "Entry 已选择: [%.1f, %.1f, %.1f] mm, 规划深度 %.1f mm",
            point[0], point[1], point[2], self._puncture_plan_depth,
        )

    def _on_start_selection(self):
        if self._is_observe_mode():
            logger.info("观察模式下跳过 Entry 选择")
            return
        print("[Main] 🎯 开始选择穿刺点模式")
        if not self.puncture_selector:
            print("[Main] ✗ 选择器未初始化")
            return
        self._selecting_target = False
        self.puncture_selector.set_selection_mode(PuncturePointSelector.MODE_ENTRY)
        self.puncture_selector.setEnabled(True)
        self.puncture_panel.set_selecting_mode(True)
        self.puncture_panel.set_target_selecting_mode(False)
        self._refresh_workflow_steps()

    def _on_reselect_puncture_point(self):
        print("[Main] 🔄 重新选择穿刺点")
        self.puncture_point = None
        self.puncture_normal = None
        self._selecting_target = False
        # 清除进针 override，避免重选时残留旧 tip 位置
        self._puncture_override_tip = None
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        self._puncture_finished = False
        self._consume_puncture_events(
            self.puncture_controller.reset(clear_baseline=True)
        )
        self.range_depth_panel.clear_baseline()
        self.gl_widget.clear_entry_markers()
        self.gl_widget.set_entry_target_line(None, None)
        self.puncture_panel.clear()
        if self._default_target_point is not None:
            self.target_point = np.asarray(self._default_target_point, dtype=float).copy()
            self.gl_widget.set_bleeding_point(self.target_point)
            self.puncture_panel.set_target_point(
                self.target_point, entry_point=None, from_default=True
            )
        else:
            self.target_point = None
            self.gl_widget.set_bleeding_point(None)
            self.puncture_panel.clear_target_display()
        self._stop_alignment_monitoring()
        self._on_start_selection()
        self._refresh_workflow_steps()

    def _on_device_connected(self):
        print("[Main] ✓ 设备已连接")

        if self._is_observe_mode():
            self.gl_widget._camera_adjusted = False
            # 观察模式的针尖由平移控制器提供，不使用穿刺模式的固定针尖覆盖。
            self.gl_widget.clear_fixed_tip_position()
            self.gl_widget.tip_position = np.zeros(3, dtype=float)
            self._update_needle_direction_fast(self._current_quaternion)
            imu_pos, tip_pos = self._calculate_positions_fast(self._current_quaternion)
            self.gl_widget.update_data(imu_pos, tip_pos)
            logger.info("观察模式：针尖固定世界原点，IMU 端随针轴方向移动")
            self.alignment_hud.set_observe_mode(True, connected=True)
            self._refresh_workflow_steps()
            return

        if self.puncture_point is None:
            QMessageBox.warning(
                self,
                "⚠️ 未选择穿刺点",
                "请先选择穿刺点，再连接设备！",
                QMessageBox.Ok,
            )
            self.device_manager.disconnect()
            return

        self._apply_target_to_scene(from_default=False)
        self.gl_widget.set_needle_tip_position(self.puncture_point)
        print(f"[Main] ✓ 针尖位置已设置为Entry点: {self.puncture_point}")

        QTimer.singleShot(1000, self._calibrate_initial_pose)

    def _calibrate_initial_pose(self):
        if self._closing:
            return
        print("[Main] 正在校准初始姿态...")

        if self._current_quaternion is None or len(self._current_quaternion) < 4:
            print("[Main] ⚠️ IMU数据尚未接收，延迟校准")
            QTimer.singleShot(500, self._calibrate_initial_pose)
            return

        self._initial_quaternion = list(self._current_quaternion)
        print(f"[Main] ✓ 初始姿态已记录: {self._initial_quaternion}")

        self._update_target_direction_world()
        if not hasattr(self, "target_direction_world"):
            print("[Main] ⚠ 无 Target，跳过对准方向")
            return
        print(f"[Main] 目标方向（Entry→Target）: {self.target_direction_world}")

        self._start_alignment_monitoring()
        self._refresh_workflow_steps()

        logger.info(
            "初始姿态已校准 — 针尖=%s, 针体=竖直向下, 目标方向=%s",
            self.puncture_point, self.target_direction_world,
        )

    def _start_alignment_monitoring(self):
        self._stop_alignment_monitoring()
        self.alignment_timer.start()
        print("[Main] ✓ 对齐监控已启动")

    def _update_alignment(self):
        if self._needle_direction is None:
            return
        if self.target_direction_world is None:
            return

        current_direction = self._needle_direction
        target_direction = self.target_direction_world

        dot_product = float(np.dot(current_direction, target_direction))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error_deg = np.degrees(np.arccos(dot_product))
        self._current_angle_error_deg = angle_error_deg

        # 罗盘显示当前针轴相对目标的屏幕方向，圆点随针体在 3D 视图中同向移动。
        curr_u = np.asarray(current_direction, dtype=float)
        curr_u = curr_u / np.linalg.norm(curr_u)
        targ_u = np.asarray(target_direction, dtype=float)
        targ_u = targ_u / np.linalg.norm(targ_u)
        # 3D 针体从固定针尖沿 -needle_direction 延伸；罗盘显示可见针杆
        # 相对目标针杆的偏离，因此要使用针轴偏差的相反方向。
        deviation = targ_u * np.dot(curr_u, targ_u) - curr_u
        projected = self.gl_widget.view.viewMatrix().mapVector(
            QVector3D(float(deviation[0]), float(deviation[1]), float(deviation[2]))
        )

        thr = self._ALIGN_THRESHOLD_DEG
        if angle_error_deg <= thr:
            if not self.puncture_controller.range_connected:
                self.alignment_hud.set_status("请连接测距模块")
            elif not self.puncture_controller.range_valid:
                self.alignment_hud.set_status("等待有效测距")
            elif not self.puncture_controller.baseline_valid:
                self.alignment_hud.set_status("请记录测距零点")
            elif self.puncture_controller.state == PunctureDepthController.PUNCTURING:
                self.alignment_hud.set_status("★ 对准，测距进针")
            else:
                self.alignment_hud.set_status("保持对准 0.3 秒")
        elif angle_error_deg <= PunctureDepthController.PAUSE_THRESHOLD_DEG:
            self.alignment_hud.set_status("接近目标")
        else:
            self.alignment_hud.set_status("需调整姿态")

        self.alignment_hud.set_guidance((projected.x(), projected.y()), angle_error_deg)

        self._puncture_tick(angle_error_deg)

    def _puncture_tick(self, angle_error_deg: float):
        """只把姿态角送入状态机；深度不由定时器累加。"""
        if self._is_observe_mode():
            return
        events = self.puncture_controller.update_angle(angle_error_deg)
        self._consume_puncture_events(events)

    def _apply_tip_at_depth(self, depth_mm: float):
        """更新 Entry→Target 方向上的针尖，并立即刷新 3D。"""
        if self.puncture_point is None or not hasattr(self, "target_direction_world"):
            return
        entry = np.asarray(self.puncture_point, dtype=float)
        dir_w = np.asarray(self.target_direction_world, dtype=float)
        self._puncture_override_tip = entry + dir_w * depth_mm
        self._refresh_3d_from_current_state()

    def _on_puncture_reset(self):
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        self._puncture_finished = False
        self._puncture_override_tip = None
        self._consume_puncture_events(
            self.puncture_controller.reset(clear_baseline=True)
        )
        self._consume_puncture_events(
            self.puncture_controller.set_plan_depth(self._puncture_plan_depth)
        )
        self.puncture_panel.reset_puncture_ui()
        if self._puncture_plan_depth > 0.0:
            self.puncture_panel.update_puncture_depth(
                0.0, self._puncture_plan_depth
            )
            self.puncture_panel.set_puncture_state(
                self.puncture_controller.state,
                self.puncture_controller.reason,
            )
        self.range_depth_panel.clear_baseline()
        # 重置时重新把 fixed_tip 设回 Entry，让 update_data 拦截正常（非进针态）
        if self.puncture_point is not None:
            self.gl_widget.set_needle_tip_position(self.puncture_point)
        self._refresh_workflow_steps()

    def _on_target_set(self):
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        self._puncture_finished = False
        self._puncture_override_tip = None
        self.puncture_controller.reset(clear_baseline=True)
        self.range_depth_panel.clear_baseline()
        if self.puncture_point is None:
            self._puncture_plan_depth = 0.0
            self._consume_puncture_events(
                self.puncture_controller.set_plan_depth(0.0)
            )
            return
        tgt = self._effective_target()
        if tgt is None:
            self._puncture_plan_depth = 0.0
            self._consume_puncture_events(
                self.puncture_controller.set_plan_depth(0.0)
            )
            return
        entry = np.asarray(self.puncture_point, dtype=float)
        vec = tgt - entry
        self._puncture_plan_depth = float(np.linalg.norm(vec))
        self._consume_puncture_events(
            self.puncture_controller.set_plan_depth(self._puncture_plan_depth)
        )
        self.puncture_panel.update_puncture_depth(
            0.0, self._puncture_plan_depth
        )
        self.puncture_panel.set_puncture_state(
            self.puncture_controller.state,
            self.puncture_controller.reason,
        )
        self._update_target_direction_world()
        self.gl_widget.set_entry_target_line(entry, tgt)
        self._apply_tip_at_depth(0.0)
        if not self.device_manager.is_connected and self.target_direction_world is not None:
            self.gl_widget.update_needle_direction(self.target_direction_world)
        self.gl_widget.set_needle_tip_position(entry)
        if self._fixed_case_active:
            self.puncture_panel.set_fixed_case_mode(True)
            self.gl_widget.frame_training_needle()
        self._refresh_workflow_steps()
