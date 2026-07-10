"""主窗口 - 整合所有模块"""
import logging
import os
import time
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread
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
)

from core.device_manager import DeviceManager
from core.dicom_loader import DicomModelLoader
from core.imu_geometry_config import apply_kinematics, load_config, save_config
from core.imu_kinematics import (
    imu_position_from_tip,
    needle_axis_for_position,
    needle_axis_scene_normalized,
    needle_body_angle_deg,
    needle_body_bias_deg,
    scene_z_ccw_deg,
    needle_tilt_from_scene_down_deg,
    capture_vertical_display_offset,
    clear_display_offset,
    display_offset_dict,
    display_offset_enabled,
    quat_slerp,
    tip_position_from_fixed,
)
from ui.widgets.gl_widget import GLVisualizationWidget
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
from ui.widgets.ui_helpers import configure_side_scroll, set_label_role, apply_panel_chrome

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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手术探针定位系统 - 穿刺训练模式")

        # 几何参数需在 UI 创建前可用
        # 你当前实测：IMU 中心到针尖约 20cm
        self.needle_length = 200.0

        self._init_core_components()
        self._init_ui()

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
        self._operation_mode = DeviceConnectionPanel.MODE_PUNCTURE_TRAINING

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

        self._ALIGN_THRESHOLD_DEG = 3.0
        self._current_angle_error_deg = 0.0
        self._puncture_plan_depth = 0.0
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        # 进针完成标志：避免完成后 tick 反复 enable_puncture_mode 导致进度条闪烁
        self._puncture_finished = False
        # 进针期间用 override tip 接管针尖位置（避免每帧 IMU 数据退回原点）
        self._puncture_override_tip = None

        self._refresh_workflow_steps()
        self._apply_operation_mode_ui()
        self._ui_ready = True
        logger.info("✓ 主窗口初始化完成")

    def _is_observe_mode(self):
        return self._operation_mode == DeviceConnectionPanel.MODE_NEEDLE_OBSERVE

    def _apply_operation_mode_ui(self):
        mode = self.connection_panel.get_operation_mode()
        self._operation_mode = mode
        observe = self._is_observe_mode()
        self.puncture_panel.set_observe_mode(observe, self._ct_model_loaded)
        self.sim_panel.set_training_enabled(not observe)
        connected = self.device_manager.is_connected
        self.alignment_hud.set_observe_mode(observe, connected=connected)
        self.connection_panel.set_mode_switch_enabled(not connected)
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
        self._splitter.addWidget(self._create_left_panel())
        self._splitter.addWidget(self._create_center_panel())
        self._splitter.addWidget(self._create_right_panel())
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setStretchFactor(2, 1)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(2, False)

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

        # 4 个全局状态 Chips
        self.header_chips = {}
        for key, label in [("ct", "CT"), ("entry", "Entry"), ("target", "Target"), ("imu", "IMU")]:
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

    def _make_left_block(self, title: str, body: QWidget) -> QFrame:
        """左栏分区：标题 + 内容。"""
        block = QFrame()
        block.setObjectName("PrepSection")
        apply_panel_chrome(block)
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(0)

        hdr = QLabel(title)
        hdr.setObjectName("LeftBlockTitle")
        block_layout.addWidget(hdr)

        body_wrap = QFrame()
        body_wrap.setObjectName("PrepSectionBody")
        apply_panel_chrome(body_wrap, "#0e141e")
        body_layout = QVBoxLayout(body_wrap)
        body_layout.setContentsMargins(8, 6, 8, 8)
        body_layout.setSpacing(4)

        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        body_layout.addWidget(body)

        block_layout.addWidget(body_wrap)
        return block

    def _create_left_panel(self):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(320)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        prep_title = QLabel("准备")
        prep_title.setObjectName("SectionTitle")
        outer.addWidget(prep_title)

        self.connection_panel = DeviceConnectionPanel()
        self.ct_panel = CTModelPanel()
        self.imu_panel = IMUDataPanel()

        scroll = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        scroll_layout.addWidget(
            self._make_left_block("设备 · 串口", self.connection_panel)
        )
        scroll_layout.addWidget(
            self._make_left_block("影像 · DICOM", self.ct_panel)
        )
        scroll_layout.addWidget(
            self._make_left_block("遥测 · IMU", self.imu_panel)
        )

        configure_side_scroll(scroll, scroll_content)

        outer.addWidget(scroll, 1)
        return panel

    def _create_right_panel(self):
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(260)

        scroll = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.alignment_hud = AlignmentHudPanel()
        layout.addWidget(self.alignment_hud)

        self.puncture_panel = PuncturePointPanel()
        layout.addWidget(self.puncture_panel)

        self.sim_panel = SimulationPanel()
        layout.addWidget(self.sim_panel)

        configure_side_scroll(scroll, content)

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
        return panel

    def _refresh_workflow_steps(self):
        if self._is_observe_mode():
            states = ["pending", "pending", "pending", "pending"]
            if not self.device_manager.is_connected:
                states[2] = "active"
            else:
                states[2] = "done"
                states[3] = "active"
            self.workflow_bar.set_states(states)
            self._update_header_status()
            return

        states = ["pending", "pending", "pending", "pending"]
        if not self._ct_model_loaded:
            states[0] = "active"
        else:
            states[0] = "done"
            if self.puncture_point is None or self.target_point is None:
                states[1] = "active"
            elif not self.device_manager.is_connected:
                states[1] = "done"
                states[2] = "active"
            else:
                states[1] = "done"
                states[2] = "done"
                states[3] = "active"
        self.workflow_bar.set_states(states)
        self._update_header_status()

    def _update_header_status(self):
        def _set_chip(key, state, text=None):
            chip = self.header_chips.get(key)
            if chip is None:
                return
            labels = {"ct": "CT", "entry": "Entry", "target": "Target", "imu": "IMU"}
            label = labels.get(key, key)
            display = text or f"{label} ✓" if state == "done" else f"{label} ?"
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
        target_h = int(min(max_h, max(min_h, avail.height() * 0.78)))
        self.resize(target_w, target_h)

        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

        left_w = int(max(320, target_w * 0.28))
        right_w = int(max(260, target_w * 0.24))
        center_w = max(420, target_w - left_w - right_w)
        self._splitter.setSizes([left_w, center_w, right_w])

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
        label = "姿态观察" if mode == DeviceConnectionPanel.MODE_NEEDLE_OBSERVE else "穿刺训练"
        print(f"[Main] 工作模式已切换: {label}")

    def _on_serial_connect(self, port, baudrate, mode):
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
        self.connection_panel.set_connected(connected)
        self.connection_panel.set_mode_switch_enabled(not connected)
        self.imu_panel.set_status(connected, self._display_fps if connected else 0)
        if not connected:
            self._display_fps = 0
        self._apply_operation_mode_ui()
        self._update_header_status()
        self._refresh_workflow_steps()
        print("✓ 串口已连接" if connected else "✓ 串口已断开")

    def _on_device_error(self, error_msg):
        print(f"✗ 设备错误: {error_msg}")

    def _on_device_data(self, data):
        try:
            quaternion = data.get("quaternion")
            euler = data.get("euler")

            if quaternion is None or euler is None:
                return

            self._current_quaternion = list(quaternion)
            self._current_euler = list(euler)

            imu_pos, tip_pos = self._calculate_positions_fast(quaternion)
            self._cached_imu_pos = imu_pos
            self._cached_tip_pos = tip_pos
            self._update_needle_direction_fast(quaternion)
            self.gl_widget.update_data(imu_pos, tip_pos)

            self._data_frame_count += 1
            now = time.perf_counter()
            if now - self._fps_last_time >= 1.0:
                self._display_fps = self._data_frame_count / (now - self._fps_last_time)
                self._data_frame_count = 0
                self._fps_last_time = now

        except Exception as e:
            print(f"✗ 数据处理错误: {e}")
            traceback.print_exc()

    def _calculate_positions_fast(self, quaternion):
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
        self._closing = True
        self._stop_alignment_monitoring()
        self.panel_update_timer.stop()
        self.device_manager.disconnect()
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
        print("[Main] ✓ CT模型加载成功")
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
        self._default_target_point = default_intracerebral_hemorrhage_target_mm(bbox)
        self.target_point = self._default_target_point.copy()
        self.gl_widget.set_bleeding_point(self.target_point)
        self.puncture_panel.set_target_point(
            self.target_point, entry_point=None, from_default=True
        )
        print(
            f"[Main] 默认出血靶点（右豆状核区）: "
            f"[{self.target_point[0]:.1f}, {self.target_point[1]:.1f}, {self.target_point[2]:.1f}] mm"
        )

        if hasattr(self, "ct_loader_thread"):
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait()

        self._apply_operation_mode_ui()
        self._refresh_workflow_steps()

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
        if self._is_observe_mode():
            return
        if self.puncture_point is None:
            logger.warning("Target 选择：Entry 未设置，跳过")
            return
        if not self.puncture_selector:
            return
        print("[Main] 开始选择 Target")
        self._selecting_target = True
        self.puncture_selector.set_selection_mode(PuncturePointSelector.MODE_TARGET)
        self.puncture_selector.setEnabled(True)
        self.puncture_panel.set_target_selecting_mode(True)
        self.puncture_panel.set_selecting_mode(False)

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
            self.gl_widget.set_needle_tip_position([0.0, 0.0, 0.0])
            logger.info("观察模式：针尖已锚定世界原点 [0, 0, 0]")
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

        # 计算 correction 用于罗盘显示
        curr_u = np.asarray(current_direction, dtype=float)
        curr_u = curr_u / np.linalg.norm(curr_u)
        targ_u = np.asarray(target_direction, dtype=float)
        targ_u = targ_u / np.linalg.norm(targ_u)
        correction = targ_u - curr_u * np.dot(targ_u, curr_u)

        thr = self._ALIGN_THRESHOLD_DEG
        if angle_error_deg < thr:
            self.alignment_hud.set_status("★ 对准，可进针")
        elif angle_error_deg < thr * 2:
            self.alignment_hud.set_status("接近目标")
        else:
            self.alignment_hud.set_status("需调整姿态")

        self.alignment_hud.set_guidance(correction, angle_error_deg)

        self._puncture_tick(angle_error_deg)

    # ── 进针逻辑 ───────────────────────────────────────────────────────────────
    PUNCTURE_DURATION_S = 10.0

    def _puncture_tick(self, angle_error_deg: float):
        """每 alignment_timer 触发一次：判角、推进、更新 UI + 针尖 override。"""
        if not self._puncture_ready():
            return
        # 已完成进针：不再 tick，避免 UI 闪烁
        if self._puncture_finished:
            return

        thr = self._ALIGN_THRESHOLD_DEG
        aligned = angle_error_deg < thr

        if not aligned:
            # 偏离：暂停进针，但保留当前深度（不退回 0），override tip 仍按当前深度
            if self._puncture_in_progress:
                self._puncture_in_progress = False
                self.puncture_panel.update_puncture_depth(
                    self._puncture_current_depth, self._puncture_plan_depth
                )
            return

        if not self._puncture_in_progress:
            # 首次推进：解除 GL widget 的 fixed_tip 拦截，让 override tip 接管针尖位置
            self.gl_widget.clear_fixed_tip()
            self._puncture_in_progress = True
            self._puncture_last_tick = time.perf_counter()
            self.puncture_panel.enable_puncture_mode(self._puncture_plan_depth)
            # 进度条/剩余时间显示对齐
            self.puncture_panel.update_puncture_depth(
                self._puncture_current_depth, self._puncture_plan_depth
            )

        now = time.perf_counter()
        dt = now - getattr(self, "_puncture_last_tick", now)
        self._puncture_last_tick = now
        # 防止首次或异常帧 dt 过大
        dt = min(dt, 0.1)
        self._puncture_current_depth += (
            self._puncture_plan_depth / self.PUNCTURE_DURATION_S
        ) * dt

        if self._puncture_current_depth >= self._puncture_plan_depth:
            self._puncture_current_depth = self._puncture_plan_depth
            self._puncture_in_progress = False
            self._puncture_finished = True
            self.puncture_panel.set_puncture_done()
            # ── 入针完成弹窗 ──
            if not self._closing:
                QMessageBox.information(
                    self,
                    "穿刺完成",
                    f"✅ 已到达 Target！\n\n"
                    f"  Entry→Target 距离: {self._puncture_plan_depth:.1f} mm\n"
                    f"  实际进针深度: {self._puncture_current_depth:.1f} mm",
                )
            return
        else:
            self.puncture_panel.update_puncture_depth(
                self._puncture_current_depth, self._puncture_plan_depth
            )

        # 每 tick 更新 override tip，让每帧 IMU 数据都用这个 tip 算 imu_pos
        entry = np.asarray(self.puncture_point, dtype=float)
        dir_w = np.asarray(self.target_direction_world, dtype=float)
        self._puncture_override_tip = entry + dir_w * self._puncture_current_depth

    def _puncture_ready(self) -> bool:
        return (
            self._puncture_plan_depth > 0
            and self.device_manager.is_connected
            and not self._is_observe_mode()
        )

    def _apply_tip_at_depth(self, depth_mm: float):
        """仅更新 override tip；下一帧 IMU 数据会自动用 override 算 imu_pos 并刷新 3D。"""
        if self.puncture_point is None or not hasattr(self, "target_direction_world"):
            return
        entry = np.asarray(self.puncture_point, dtype=float)
        dir_w = np.asarray(self.target_direction_world, dtype=float)
        self._puncture_override_tip = entry + dir_w * depth_mm

    def _on_puncture_reset(self):
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        self._puncture_finished = False
        self._puncture_override_tip = None
        # 重置时重新把 fixed_tip 设回 Entry，让 update_data 拦截正常（非进针态）
        if self.puncture_point is not None:
            self.gl_widget.set_needle_tip_position(self.puncture_point)
        self.puncture_panel.reset_puncture_ui()

    def _on_target_set(self):
        self._puncture_current_depth = 0.0
        self._puncture_in_progress = False
        if self.puncture_point is None:
            self._puncture_plan_depth = 0.0
            return
        tgt = self._effective_target()
        if tgt is None:
            self._puncture_plan_depth = 0.0
            return
        entry = np.asarray(self.puncture_point, dtype=float)
        vec = tgt - entry
        self._puncture_plan_depth = float(np.linalg.norm(vec))
        self._update_target_direction_world()
        self.gl_widget.set_entry_target_line(entry, tgt)
        self._apply_tip_at_depth(0.0)
        self._refresh_workflow_steps()
