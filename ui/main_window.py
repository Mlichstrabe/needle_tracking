"""主窗口 - 整合所有模块"""
import os
import time
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtWidgets import (
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
)

from core.device_manager import DeviceManager
from core.dicom_loader import DicomModelLoader
from core.imu_kinematics import (
    imu_position_from_tip,
    needle_axis_for_position,
    needle_axis_scene_normalized,
    tip_position_from_fixed,
)
from core.puncture_monitor import PunctureMonitor
from core.puncture_session import PunctureSession
from core.simulation_manager import SimulationManager
from ui.widgets.gl_widget import GLVisualizationWidget
from ui.widgets.panels import (
    CTModelPanel,
    DeviceConnectionPanel,
    GuidanceArrowWidget,
    IMUDataPanel,
    NeedleConfigPanel,
    PuncturePointPanel,
)
from ui.widgets.puncture_point_selector import PuncturePointSelector
from ui.widgets.simulation_panel import SimulationPanel


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手术探针定位系统 - 穿刺训练模式")
        self.setMinimumSize(1280, 820)

        self._init_core_components()
        self._init_ui()
        self._connect_signals()
        self._init_timers()

        self._current_quaternion = [1, 0, 0, 0]
        self._current_euler = [0, 0, 0]
        self._last_needle_direction = np.array([0, 0, -1])
        self._needle_direction = [0, 0, -1]

        self._cached_imu_pos = np.zeros(3)
        self._cached_tip_pos = np.zeros(3)

        # 直接使用IMU原始数据（依赖IMU自带滤波）
        self._filter_mode = "normal"

        self.needle_length = 162.0

        self.preset_paths = [
            {"name": "路径1 - 沿Z轴", "direction": [0, 0, 1]},
            {"name": "路径2 - 45°倾斜", "direction": [0.707, 0, 0.707]},
            {"name": "路径3 - 水平", "direction": [1, 0, 0]},
            {"name": "路径4 - 复杂角度", "direction": [0.577, 0.577, 0.577]},
        ]

        self.puncture_selector = None
        self.puncture_point = None
        self.puncture_normal = None

        self.alignment_timer = QTimer(self)
        self.alignment_timer.setInterval(100)
        self.alignment_timer.timeout.connect(self._update_alignment)

        print("✓ 主窗口初始化完成")

    def _init_core_components(self):
        self.device_manager = DeviceManager()
        self.puncture_monitor = PunctureMonitor(threshold=3.0)
        self.puncture_session = PunctureSession(self.puncture_monitor)
        self.simulation_manager = SimulationManager()
        self.dicom_loader = DicomModelLoader()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 720, 360])

        main_layout.addWidget(splitter)

    def _create_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.gl_widget = GLVisualizationWidget()
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.gl_widget)

        return panel

    def _create_left_panel(self):
        panel = QFrame()
        panel.setStyleSheet(
            """
            QFrame { background: #12121f; border-radius: 8px; }
            """
        )

        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        self.ct_panel = CTModelPanel()
        layout.addWidget(self.ct_panel)

        self.connection_panel = DeviceConnectionPanel()
        layout.addWidget(self.connection_panel)

        self.needle_panel = NeedleConfigPanel()
        layout.addWidget(self.needle_panel)

        self.imu_panel = IMUDataPanel()
        layout.addWidget(self.imu_panel)

        layout.addStretch()
        return panel

    def _create_right_panel(self):
        panel = QFrame()
        panel.setStyleSheet(
            """
            QFrame { background: #12121f; border-radius: 8px; }
            """
        )

        # 滚动区域，防止内容被裁
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }"
                             "QScrollBar:vertical { width: 6px; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        guide_title = QLabel("对准引导")
        guide_title.setStyleSheet(
            "color: #81d4fa; font-size: 13px; font-weight: bold; padding: 2px 0;"
        )
        layout.addWidget(guide_title)

        self.guidance_widget = GuidanceArrowWidget()
        layout.addWidget(self.guidance_widget, alignment=Qt.AlignHCenter)

        self.puncture_panel = PuncturePointPanel()
        self.puncture_panel.setVisible(False)
        layout.addWidget(self.puncture_panel)

        self.sim_panel = SimulationPanel()
        layout.addWidget(self.sim_panel)

        layout.addStretch()
        scroll.setWidget(content)

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
        return panel

    def _connect_signals(self):
        self.device_manager.data_received.connect(self._on_device_data)
        self.device_manager.connected.connect(self._on_device_connected_wrapper)
        self.device_manager.disconnected.connect(self._on_device_disconnected)
        self.device_manager.error_occurred.connect(self._on_device_error)

        self.connection_panel.connect_clicked.connect(self._on_serial_connect)
        self.connection_panel.disconnect_clicked.connect(self._on_serial_disconnect)

        self.needle_panel.needle_length_changed.connect(self._on_needle_length_changed)
        self.needle_panel.zero_position_clicked.connect(self._on_zero_position)
        self.needle_panel.clear_trajectory_clicked.connect(self._on_clear_trajectory)
        self.needle_panel.reset_view_clicked.connect(self._on_reset_view)
        self.needle_panel.calibration_clicked.connect(self._on_calibration_requested)

        self.sim_panel.simulation_started.connect(self._on_simulation_started)
        self.sim_panel.simulation_stopped.connect(self._on_simulation_stopped)
        self.sim_panel.target_direction_changed.connect(self._on_target_direction_changed)
        self.sim_panel.orientation_locked.connect(self._on_orientation_locked)

        self.puncture_session.phase_changed.connect(self._on_phase_changed)
        self.puncture_session.depth_changed.connect(self._on_depth_changed)
        self.puncture_session.deviation_warning.connect(self._on_deviation_warning)
        self.puncture_session.result_determined.connect(self._on_result_determined)
        self.puncture_session.simulated_tip_moved.connect(self._on_simulated_tip_moved)

        self.ct_panel.load_clicked.connect(self._on_ct_load)
        self.ct_panel.clear_clicked.connect(self._on_ct_clear)
        self.ct_panel.visibility_changed.connect(self.gl_widget.set_head_model_visible)

        self.dicom_loader.progress_updated.connect(self._on_ct_progress)
        self.dicom_loader.loading_finished.connect(self._on_ct_loaded)
        self.dicom_loader.loading_failed.connect(self._on_ct_failed)

        if hasattr(self.sim_panel, "path_selected"):
            self.sim_panel.path_selected.connect(self._on_path_selected)

        self.puncture_panel.start_selection_clicked.connect(self._on_start_selection)
        self.puncture_panel.reselect_clicked.connect(self._on_reselect_puncture_point)

    def _init_timers(self):
        self.update_timer = QTimer()
        self.update_timer.setInterval(16)
        self.update_timer.timeout.connect(self._on_update_tick)
        self.update_timer.start()

        self.panel_update_timer = QTimer()
        self.panel_update_timer.setInterval(33)
        self.panel_update_timer.timeout.connect(self._update_panels)
        self.panel_update_timer.start()

    def _on_serial_connect(self, port, baudrate):
        if not self.device_manager.connect(port, baudrate):
            QMessageBox.warning(self, "连接失败", f"无法连接到 {port}")

    def _on_serial_disconnect(self):
        self.device_manager.disconnect()

    def _on_device_disconnected(self):
        self._on_connection_changed(False)
        self._stop_alignment_monitoring()

    def _on_connection_changed(self, connected):
        self.connection_panel.set_connected(connected)
        self.imu_panel.set_status(connected)
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

        except Exception as e:
            print(f"✗ 数据处理错误: {e}")
            traceback.print_exc()

    def _calculate_positions_fast(self, quaternion):
        direction = needle_axis_for_position(quaternion)
        fixed_tip = self.gl_widget.get_fixed_tip_position()
        tip_pos, skip = tip_position_from_fixed(fixed_tip)
        if skip:
            print("[Calculate] ⚠️ fixed_tip 无效，跳过本次计算")
            return np.zeros(3), np.zeros(3)

        needle_length = getattr(self, "needle_length", 100.0)
        imu_pos = imu_position_from_tip(tip_pos, direction, needle_length)
        return imu_pos, tip_pos

    def _update_needle_direction_fast(self, quaternion):
        direction = needle_axis_scene_normalized(quaternion)
        if direction is not None:
            self._needle_direction = direction
            self._last_needle_direction = np.array(self._needle_direction)
        self.gl_widget.update_needle_direction(self._needle_direction)

    def _stop_alignment_monitoring(self):
        if self.alignment_timer.isActive():
            self.alignment_timer.stop()
        self.guidance_widget.hide_guidance()

    def _on_device_connected_wrapper(self):
        self._on_connection_changed(True)
        self._on_device_connected()

    def _on_needle_length_changed(self, length):
        self.needle_length = float(length)
        print(f"针具长度: {length}mm")

    def _on_path_selected(self, path_index):
        if 0 <= path_index < len(self.preset_paths):
            path = self.preset_paths[path_index]
            direction = path["direction"]
            self.gl_widget.set_preset_path(direction)
            print(f"[Main] 已选择路径: {path['name']} -> {direction}")
        else:
            print(f"[Main] ⚠️ 无效的路径索引: {path_index}")

    def _on_zero_position(self):
        print("✓ 位置已归零")

    def _on_clear_trajectory(self):
        self.gl_widget.clear_trajectory()

    def _on_reset_view(self):
        self.gl_widget.reset_view(clear_trajectory=False)

    def _on_calibration_requested(self):
        """执行传感器校准序列：磁力计 + 陀螺仪零偏"""
        if not self.device_manager.is_connected:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "未连接", "请先连接设备后再校准")
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

        self.needle_panel.btn_calibrate.setEnabled(False)
        self.needle_panel.btn_calibrate.setText("校准中...")
        print("  保持传感器静止... (3秒)")

    def _finish_calibration(self):
        """完成校准"""
        self.device_manager.calibrate_magnetic_end()
        self.needle_panel.btn_calibrate.setEnabled(True)
        self.needle_panel.btn_calibrate.setText("校准传感器")
        print("=== 传感器校准完成 ===")
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "校准完成", "传感器校准已完成！\n\n• 陀螺仪零偏已记录\n• 磁力计校准已保存")

    def _on_simulation_started(self):
        print("[Main] 穿刺路径引导模式已启动")
        default_path = [0, 0, 1]
        self.gl_widget.set_preset_path(default_path)

        # 设置对齐目标方向并启动监控
        self.target_direction_world = np.array(default_path, dtype=float)
        self._start_alignment_monitoring()
        self.puncture_panel.setVisible(True)
        print(f"[Main] ✓ 已设置预设路径: {default_path}")

    def _on_simulation_stopped(self):
        print("[Main] 穿刺路径引导模式已停止")
        self._stop_alignment_monitoring()
        self.gl_widget.clear_path_lines()

    def _on_target_direction_changed(self, direction):
        print(f"[Main] 目标路径方向: {direction}")
        d = direction.tolist()
        self.target_direction_world = np.array(d, dtype=float)
        self.gl_widget.set_preset_path(d)

    def _on_orientation_locked(self, direction):
        print(f"[Main] 姿态已锁定: {direction}")
        d = np.asarray(direction).tolist()
        self.gl_widget.set_target_path(d)

    def _update_panels(self):
        self.imu_panel.update_quaternion(self._current_quaternion)
        self.imu_panel.update_euler(self._current_euler)

        imu_pos = self._cached_imu_pos
        tip_pos = self._cached_tip_pos

        self.imu_panel.update_position(imu_pos)
        self.needle_panel.update_tip_position(tip_pos)

        inverted_direction = [
            -self._needle_direction[0],
            -self._needle_direction[1],
            -self._needle_direction[2],
        ]
        self.sim_panel.update_current_direction(inverted_direction)

    def _on_phase_changed(self, phase):
        self.sim_panel.set_phase(phase)

        if phase == "aligning":
            self.gl_widget.set_target_visible(False)
        elif phase in ("completed", "failed"):
            self.simulation_manager.reveal_target()
            self.gl_widget.set_target_visible(True)
            if phase == "completed":
                self.gl_widget.show_success_effect()
            else:
                self.gl_widget.show_failure_effect()

    def _on_depth_changed(self, current, target):
        self.sim_panel.update_depth(current, target)

    def _on_deviation_warning(self, angle, is_critical):
        if is_critical:
            print(f"⚠ 严重偏离: {angle:.1f}°")

    def _on_result_determined(self, result, details):
        self.sim_panel.show_result(result, details)
        if result == "success":
            print(f"✓ 穿刺成功！精度: {details.get('accuracy', 0):.1f}mm")
        else:
            print(f"✗ 穿刺失败: {details.get('reason', '未知')}")

    def _on_simulated_tip_moved(self, sim_tip):
        self.gl_widget.update_simulated_tip(sim_tip, self.puncture_session.lock_position)

    def _on_update_tick(self):
        start = time.perf_counter()
        phase = self.puncture_session.phase

        if phase == "idle":
            return

        imu_pos = self._cached_imu_pos

        if phase == "aligning":
            angle, is_aligned = self.puncture_session.check_alignment(
                self._last_needle_direction, imu_pos
            )
            self.sim_panel.update_alignment(angle, is_aligned)

        elif phase in ("locked", "advancing"):
            status = self.puncture_session.update(self._current_quaternion, imu_pos)
            self.sim_panel.update_deviation(
                status["deviation_angle"],
                status["suggestions"],
            )

            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 20:
                print(f"⚠️ _on_update 耗时: {elapsed:.1f}ms")

    def closeEvent(self, event):
        self._stop_alignment_monitoring()
        self.update_timer.stop()
        self.panel_update_timer.stop()
        self.device_manager.disconnect()
        self.puncture_session.end()
        print("✓ 程序已退出")
        event.accept()

    def _on_ct_load(self, folder_path):
        print(f"[Main] 开始加载DICOM文件夹: {folder_path}")
        self.ct_panel.set_loading(True)

        self.ct_loader_thread = QThread()
        self.dicom_loader.moveToThread(self.ct_loader_thread)
        self.ct_loader_thread.started.connect(
            lambda: self.dicom_loader.load_dicom_folder(folder_path)
        )
        self.ct_loader_thread.start()

    def _on_ct_progress(self, value, message):
        self.ct_panel.update_progress(value, message)

    def _on_ct_loaded(self, model_data):
        print("[Main] ✓ CT模型加载成功")
        print(f"  顶点数: {model_data['num_vertices']}")
        print(f"  面数: {model_data['num_faces']}")

        self.gl_widget.load_head_model(model_data["vertices"], model_data["faces"])
        self.ct_panel.set_loading(False)
        self.ct_panel.set_model_loaded(True, model_data)

        if self.puncture_selector is None:
            self.puncture_selector = PuncturePointSelector(self.gl_widget)
            self.puncture_selector.point_selected.connect(self._on_puncture_point_selected)

        self.puncture_selector.set_model(model_data["vertices"], model_data["faces"])

        self.puncture_panel.setVisible(True)
        self.puncture_panel.set_model_loaded(True)

        center = model_data["center"]
        self.target_point = center + np.array([30, -25, 60])
        self.gl_widget.set_bleeding_point(self.target_point)

        if hasattr(self, "ct_loader_thread"):
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait()

    def _on_ct_failed(self, error_msg):
        print(f"[Main] ✗ CT模型加载失败: {error_msg}")
        self.ct_panel.set_loading(False)
        QMessageBox.critical(self, "加载失败", error_msg)
        if hasattr(self, "ct_loader_thread"):
            self.ct_loader_thread.quit()
            self.ct_loader_thread.wait()

    def _on_ct_clear(self):
        self.gl_widget.clear_head_model()
        self.ct_panel.set_model_loaded(False)
        print("[Main] ✓ CT模型已清除")

    def _on_puncture_point_selected(self, point, normal):
        print(f"[Main] 📍 穿刺点已选择: {point}")
        print(f"[Main] 📐 法线方向: {normal}")

        self.puncture_point = point
        self.puncture_normal = normal

        self.gl_widget.set_bleeding_point(self.target_point)
        self.puncture_panel.set_puncture_point(point, normal)
        self.gl_widget.set_puncture_point(point, normal)
        self.gl_widget.set_entry_target_line(point, self.target_point)

        self.puncture_selector.setEnabled(False)

        QMessageBox.information(
            self,
            "✓ 穿刺点已选择",
            f"穿刺点已成功选择！\n\n"
            f"📍 Entry点: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}] mm\n"
            f"🎯 Target点: [{self.target_point[0]:.1f}, {self.target_point[1]:.1f}, {self.target_point[2]:.1f}] mm\n"
            f"📏 穿刺深度: 80.0 mm\n\n"
            f"下一步：点击左侧的连接设备按钮",
            QMessageBox.Ok,
        )

    def _on_start_selection(self):
        print("[Main] 🎯 开始选择穿刺点模式")
        if not self.puncture_selector:
            print("[Main] ✗ 选择器未初始化")
            return
        self.puncture_selector.setEnabled(True)
        self.puncture_panel.set_selecting_mode(True)

    def _on_reselect_puncture_point(self):
        print("[Main] 🔄 重新选择穿刺点")
        self.puncture_point = None
        self.puncture_normal = None
        self.gl_widget.clear_puncture_point()
        self.puncture_panel.clear()
        self._on_start_selection()

    def _on_device_connected(self):
        print("[Main] ✓ 设备已连接")

        if self.puncture_point is None:
            QMessageBox.warning(
                self,
                "⚠️ 未选择穿刺点",
                "请先选择穿刺点，再连接设备！",
                QMessageBox.Ok,
            )
            self.device_manager.disconnect()
            return

        self.gl_widget.set_needle_tip_position(self.puncture_point)
        print(f"[Main] ✓ 针尖位置已设置为Entry点: {self.puncture_point}")

        QTimer.singleShot(1000, self._calibrate_initial_pose)

    def _calibrate_initial_pose(self):
        print("[Main] 正在校准初始姿态...")

        if not hasattr(self, "_current_quaternion") or self._current_quaternion is None:
            print("[Main] ⚠️ IMU数据尚未接收，延迟校准")
            QTimer.singleShot(500, self._calibrate_initial_pose)
            return

        self._initial_quaternion = list(self._current_quaternion)
        print(f"[Main] ✓ 初始姿态已记录: {self._initial_quaternion}")

        entry_to_target = self.target_point - self.puncture_point
        self.target_direction_world = entry_to_target / np.linalg.norm(entry_to_target)
        print(f"[Main] 目标方向（Entry→Target）: {self.target_direction_world}")

        self.puncture_panel.setVisible(True)
        self._start_alignment_monitoring()

        QMessageBox.information(
            self,
            "✓ 初始姿态已校准",
            "初始姿态已成功校准！\n\n"
            "当前假设：\n"
            "• 针尖位置 = Entry点\n"
            "• 针体方向 = 竖直向下（重力方向）\n\n"
            "下一步：\n"
            "调整针体姿态，使其对准Target点（红色球）",
            QMessageBox.Ok,
        )

    def _start_alignment_monitoring(self):
        self._stop_alignment_monitoring()
        self.alignment_timer.start()
        print("[Main] ✓ 对齐监控已启动")

    def _update_alignment(self):
        if self._needle_direction is None:
            return

        current_direction = self._needle_direction
        target_direction = self.target_direction_world

        dot_product = float(np.dot(current_direction, target_direction))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error_deg = np.degrees(np.arccos(dot_product))

        self.puncture_panel.set_alignment_error(angle_error_deg)
        if angle_error_deg < 5.0:
            self.puncture_panel.set_alignment_status("✓ 已对齐")
            if angle_error_deg < 2.0:
                self.puncture_panel.set_alignment_status("★ 完美对齐")
        else:
            self.puncture_panel.set_alignment_status(f"偏离 {angle_error_deg:.1f}°")

        # 方向指示器：显示"针尖该往哪个方向摆"（target在垂直于current平面上的投影）
        curr_u = np.asarray(current_direction, dtype=float)
        curr_u = curr_u / np.linalg.norm(curr_u)
        targ_u = np.asarray(target_direction, dtype=float)
        targ_u = targ_u / np.linalg.norm(targ_u)
        correction = targ_u - curr_u * np.dot(targ_u, curr_u)
        self.guidance_widget.set_guidance(correction, angle_error_deg)

        if not hasattr(self, "_last_log_time"):
            self._last_log_time = 0.0
        now = time.time()
        if now - self._last_log_time > 1.0:
            print(f"[Main] 目标方向: {target_direction}")
            print(f"[Main] 偏离角度: {angle_error_deg:.1f}°")
            self._last_log_time = now
