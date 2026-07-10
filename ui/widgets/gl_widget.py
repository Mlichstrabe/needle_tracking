"""3D可视化组件 - 支持模拟穿刺显示"""
import json
import os

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtGui import QVector3D, QMatrix4x4


class PunctureGLViewWidget(gl.GLViewWidget):
    """在 pyqtgraph 默认鼠标处理之前处理选点左键。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._puncture_selector = None

    def set_puncture_selector(self, selector):
        self._puncture_selector = selector

    def mousePressEvent(self, ev):
        if (
            self._puncture_selector is not None
            and ev.button() == Qt.LeftButton
            and self._puncture_selector.try_pick_from_view_event(ev)
        ):
            ev.accept()
            return
        super().mousePressEvent(ev)


class GLVisualizationWidget(QFrame):
    """基于PyQtGraph的3D可视化组件"""

    # 等轴测：视线与 X/Y/Z 夹角均为 45° → elev≈35.26°, azim=45°
    DEFAULT_CAMERA_DISTANCE = 240
    DEFAULT_CAMERA_ELEVATION = 35.26
  # 在 azimuth=135 基础上再绕竖直轴顺时针 90°
    DEFAULT_CAMERA_AZIMUTH = 45

    @staticmethod
    def _viewport_config_path():
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        return os.path.join(root, "config", "viewport.json")

    def _load_viewport_config(self):
        path = self._viewport_config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            print(f"[GL] 视角配置未加载，使用内置默认: {exc}")
            return {}
        return cfg

    def _apply_viewport_preset(self, cfg=None):
        """应用 config/viewport.json 或内置默认（与床旁对齐的方位）。"""
        if cfg is None:
            cfg = self._load_viewport_config()
        dist = float(cfg.get("distance", self.DEFAULT_CAMERA_DISTANCE))
        elev = float(cfg.get("elevation", self.DEFAULT_CAMERA_ELEVATION))
        azim = float(cfg.get("azimuth", self.DEFAULT_CAMERA_AZIMUTH))
        center = cfg.get("center", [0.0, 0.0, 0.0])
        if len(center) >= 3:
            self.view.opts["center"] = QVector3D(
                float(center[0]), float(center[1]), float(center[2])
            )
        self.view.setCameraPosition(distance=dist, elevation=elev, azimuth=azim)
        self.view.opts["distance"] = dist
        self._viewport_cfg = cfg

    def capture_viewport_to_config(self):
        """将当前相机写入 config/viewport.json（调好后点「设为默认视角」）。"""
        opts = self.view.opts
        center = opts.get("center")
        if center is not None:
            if hasattr(center, "x"):
                c = [float(center.x()), float(center.y()), float(center.z())]
            else:
                c = [float(center[0]), float(center[1]), float(center[2])]
        else:
            c = [0.0, 0.0, 0.0]
        dist = float(opts.get("distance", self.DEFAULT_CAMERA_DISTANCE))
        elev = float(opts.get("elevation", self.DEFAULT_CAMERA_ELEVATION))
        azim = float(opts.get("azimuth", self.DEFAULT_CAMERA_AZIMUTH))
        try:
            cam = self.view.cameraParams()
            if cam:
                dist = float(cam.get("distance", dist))
                elev = float(cam.get("elevation", elev))
                azim = float(cam.get("azimuth", azim))
        except Exception:
            pass
        cfg = {
            "comment": "与床旁观察对齐的默认视角（capture_viewport_to_config 写入）",
            "distance": dist,
            "elevation": elev,
            "azimuth": azim,
            "center": c,
            "fit_ct_distance_only": True,
        }
        path = self._viewport_config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self._viewport_cfg = cfg
        print(f"[GL] 已保存默认视角 → {path}: dist={dist:.0f} elev={elev:.1f} azim={azim:.1f}")
        return cfg

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("background: #05070a; border-radius: 6px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # 3D视图
        self.view = PunctureGLViewWidget()
        self._viewport_cfg = {}
        self._apply_viewport_preset()
        fmt = self.view.format()
        fmt.setSamples(0)
        self.view.setFormat(fmt)

        layout.addWidget(self.view)

        # 数据存储
        self.tip_positions = []
        self.imu_position = np.array([0, 0, 0], dtype=float)
        self.tip_position = np.array([0, 0, 0], dtype=float)
        # 由 MainWindow 覆盖（IMU 中心→针尖距离）
        self.needle_length = 200.0

        # 虚线路径
        self.preset_path_direction = None
        self.target_path_direction = None
        self.path_line_length = 400.0
        self.preset_path_line = None
        self.target_path_line = None

        # 性能优化计数器
        self._traj_counter = 0
        self._gl_initialized = False
        self._camera_adjusted = False
        self._path_lines_dirty = True
        self._needle_update_confirmed = False
        self._traj_error_shown = False
        self._box_error_shown = False
        self._box_update_confirmed = False
        self._traj_threshold_sq = 16.0

        self._tip_box_half = np.array([2.5, 2.5, 2.5], dtype=float)
        self._imu_box_half = np.array([2.0, 2.0, 2.0], dtype=float)

        # 初始化场景
        self.axes = None
        self._init_scene()
        self._init_objects()
        self._init_simulation_objects()

        # ====== CT头部模型 ======
        self.head_mesh = None
        self.head_model_visible = True
        self.head_model_color = (0.9, 0.9, 0.9, 0.6)

        print("[GL] 性能优化已启用")

    def _init_scene(self):
        """网格与 GLAxisItem 同处默认 XZ 平面（含蓝/黄轴方向），原点对齐世界系。"""
        self.axes = gl.GLAxisItem()
        self.axes.setSize(150, 150, 150)
        self.view.addItem(self.axes)

        self.grid = gl.GLGridItem()
        self.grid.setSize(x=400, y=400, z=1)
        self.grid.setSpacing(x=40, y=40, z=40)
        self.view.addItem(self.grid)
        self._sync_grid_axes_transform((0.0, 0.0, 0.0))

    def apply_scene_orientation(self):
        """scene_z_ccw_deg 只作用于 IMU 针向（kinematics），不旋转 3D 网格/坐标轴。"""
        pass

    def _init_objects(self):
        """初始化基础绘制对象"""
        self.needle_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, -200]], dtype=np.float32),
            color=(1, 1, 1, 1),
            width=5,
            antialias=True,
            mode='lines'
        )
        self.view.addItem(self.needle_line)

        self.trajectory_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=(0.3, 0.6, 1, 0.7),
            width=2,
            antialias=True
        )
        self.view.addItem(self.trajectory_line)

        self.tip_box = gl.GLBoxItem(
            size=QVector3D(5, 5, 5),
            color=(255, 0, 0, 255),
            glOptions='opaque'
        )
        self.view.addItem(self.tip_box)

        self.imu_box = gl.GLBoxItem(
            size=QVector3D(4, 4, 4),
            color=(0, 255, 0, 255),
            glOptions='opaque'
        )
        self.view.addItem(self.imu_box)

        print("[GL] 针杆线条已创建（白色，5px）")
        print("[GL] 立方体标记已创建（针尖5mm红色，IMU4mm绿色）")

    def _set_box_center(self, box, center, half_size):
        """GLBoxItem 顶点从原点延伸，translate 使方块几何中心落在 center。"""
        c = np.asarray(center, dtype=float).reshape(3)
        h = np.asarray(half_size, dtype=float).reshape(3)
        transform = QMatrix4x4()
        transform.translate(float(c[0] - h[0]), float(c[1] - h[1]), float(c[2] - h[2]))
        box.setTransform(transform)

    def _needle_endpoints(self):
        """针体直线两端：针尖（tip）与针尾（IMU）。"""
        if not hasattr(self, '_needle_direction') or self._needle_direction is None:
            direction = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            direction = np.asarray(self._needle_direction, dtype=float).reshape(3)
            n = float(np.linalg.norm(direction))
            if n > 1e-9:
                direction = direction / n
        tip = np.asarray(self.tip_position, dtype=float).reshape(3)
        nl = float(getattr(self, "needle_length", 200.0))
        tail = tip - direction * nl
        return tip, tail

    def _init_simulation_objects(self):
        """初始化模拟穿刺相关的3D对象"""
        # 1. Entry点标记（黄色高亮球体）
        md_entry = gl.MeshData.sphere(rows=10, cols=20, radius=4.0)  # 8mm半径
        self.puncture_point_marker = gl.GLMeshItem(
            meshdata=md_entry,
            smooth=True,
            color=(1, 1, 0, 0.5),  # 黄色，完全不透明
            shader='shaded',
            glOptions='opaque'
        )
        self.puncture_point_marker.setVisible(False)
        self.view.addItem(self.puncture_point_marker)

        # 2. Entry点外圈（脉冲效果，可选）
        self.puncture_point_glow = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1, 1, 0, 0.3),  # 半透明黄色
            size=14,  # 更大
            pxMode=False  # 随距离缩放
        )
        self.puncture_point_glow.setVisible(False)
        self.view.addItem(self.puncture_point_glow)

        # 3. Target点标记（红色高亮球体）
        md_target = gl.MeshData.sphere(rows=10, cols=20, radius=6.0)  # 6mm半径
        self.bleeding_marker = gl.GLMeshItem(
            meshdata=md_target,
            smooth=True,
            color=(1, 0.15, 0.15, 1.0),
            shader='shaded',
            glOptions='opaque',
        )
        self.bleeding_marker.setVisible(False)
        self.view.addItem(self.bleeding_marker)

        # 4. Entry-Target连线（蓝色虚线）
        self.entry_target_line = None  # 稍后创建

        print("[GL] ✓ 穿刺点标记系统已初始化")

    def update_data(self, imu_pos, tip_pos, max_points=500):
        """更新可视化数据"""

        #  如果设置了固定针尖位置，覆盖传入的 tip_pos
        if hasattr(self, '_use_fixed_tip') and self._use_fixed_tip:
            tip_pos = list(self._fixed_tip_position)
            # 重新计算 imu_pos（保持针体长度不变）
            if hasattr(self, '_needle_direction') and self._needle_direction is not None:
                direction = np.array(self._needle_direction)
                nl = float(getattr(self, "needle_length", 200.0))
                imu_pos = [
                    tip_pos[0] - direction[0] * nl,
                    tip_pos[1] - direction[1] * nl,
                    tip_pos[2] - direction[2] * nl,
                ]

        # ====== 以下是原有代码，保持不变 ======

        if not self._gl_initialized:
            self._gl_initialized = True
            try:
                from OpenGL import GL
                GL.glEnable(GL.GL_POINT_SMOOTH)
                GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            except:
                pass

        # 不在首帧 IMU 上改相机，避免画面跳变、与手柄转动不同步的错觉
        self.imu_position = np.array(imu_pos, dtype=np.float32)
        self.tip_position = np.array(tip_pos, dtype=np.float32)

        self._update_needle_visualization()

        #  固定针尖模式下，不记录轨迹
        if not (hasattr(self, '_use_fixed_tip') and self._use_fixed_tip):
            # 更新轨迹
            self._traj_counter += 1
            if self._traj_counter >= 5:
                self._traj_counter = 0

                should_add = len(self.tip_positions) == 0
                if not should_add and len(self.tip_positions) > 0:
                    last_pos = self.tip_positions[-1]
                    dist_sq = np.sum((self.tip_position - last_pos) ** 2)
                    should_add = dist_sq > self._traj_threshold_sq

                if should_add:
                    self.tip_positions.append(self.tip_position.copy())
                    if len(self.tip_positions) > max_points:
                        self.tip_positions = self.tip_positions[-max_points:]

                if len(self.tip_positions) >= 2:
                    try:
                        traj = np.array(self.tip_positions, dtype=np.float32)
                        self.trajectory_line.setData(pos=traj)
                    except Exception as e:
                        if not self._traj_error_shown:
                            self._traj_error_shown = True
                            print(f"[GL 错误] 轨迹线更新失败: {e}")

        # 更新虚线
        if self._path_lines_dirty:
            self._update_path_lines()

    def _update_path_lines(self):
        """更新路径虚线 - 使用贯穿式起点/终点"""

        # ========== 预设路径虚线（蓝色）==========
        if self.preset_path_direction is not None:
            #  使用 set_preset_path() 中设置的起点和终点
            if hasattr(self, 'preset_path_start') and hasattr(self, 'preset_path_end'):
                start_point = self.preset_path_start
                end_point = self.preset_path_end
            else:
                # 降级方案：从原点延伸
                start_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                end_point = start_point + self.preset_path_direction * self.path_line_length

            dashed_line = self._generate_dashed_line(start_point, end_point, 10.0, 5.0)

            if self.preset_path_line is None:
                self.preset_path_line = gl.GLLinePlotItem(
                    pos=dashed_line,
                    color=(0.4, 0.8, 1.0, 0.8),  # 蓝色
                    width=4.0,
                    antialias=True,
                    mode='lines'
                )
                self.view.addItem(self.preset_path_line)
                print(f"[GL] 预设路径虚线已创建（从{start_point}到{end_point}）")
            else:
                self.preset_path_line.setData(pos=dashed_line)
        else:
            if self.preset_path_line is not None:
                self.view.removeItem(self.preset_path_line)
                self.preset_path_line = None

        # ========== 锁定目标虚线（橙色）==========
        if self.target_path_direction is not None:
            #  使用 set_target_path() 中设置的起点和终点
            if hasattr(self, 'target_path_start') and hasattr(self, 'target_path_end'):
                start_point = self.target_path_start
                end_point = self.target_path_end
            else:
                # 降级方案
                start_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                end_point = start_point + self.target_path_direction * self.path_line_length

            dashed_line = self._generate_dashed_line(start_point, end_point, 8.0, 4.0)

            if self.target_path_line is None:
                self.target_path_line = gl.GLLinePlotItem(
                    pos=dashed_line,
                    color=(1.0, 0.6, 0.2, 0.9),  # 橙色
                    width=3.0,
                    antialias=True,
                    mode='lines'
                )
                self.view.addItem(self.target_path_line)
                print(f"[GL] 锁定目标虚线已创建（从{start_point}到{end_point}）")
            else:
                self.target_path_line.setData(pos=dashed_line)
        else:
            if self.target_path_line is not None:
                self.view.removeItem(self.target_path_line)
                self.target_path_line = None

        self._path_lines_dirty = False

    def _generate_dashed_line(self, start, end, dash_length, gap_length):
        """生成虚线顶点"""
        direction = end - start
        total_length = np.linalg.norm(direction)

        if total_length < 0.001:
            return np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)

        direction = direction / total_length
        vertices = []
        current_pos = 0.0
        segment_length = dash_length + gap_length

        while current_pos < total_length:
            seg_start = start + direction * current_pos
            seg_end_pos = min(current_pos + dash_length, total_length)
            seg_end = start + direction * seg_end_pos

            vertices.append(seg_start)
            vertices.append(seg_end)
            current_pos += segment_length

        return np.array(vertices, dtype=np.float32)

    def set_preset_path(self, direction):
        """设置预设路径（蓝色虚线）- 贯穿整个空间"""
        self.preset_path_direction = np.array(direction)

        #  延长到整个可视化空间
        extension_length = 500.0  # 延伸500mm（根据你的空间大小调整）

        # 从原点向两个方向延伸
        self.preset_path_start = -np.array(direction) * extension_length
        self.preset_path_end = np.array(direction) * extension_length

        self._path_lines_dirty = True

        print(f"[GL] 预设路径已设置（贯穿）: {direction}")

    def set_target_path(self, direction):
        """设置锁定路径（橙色虚线）- 贯穿整个空间"""
        self.target_path_direction = np.array(direction)

        #  延长到整个可视化空间
        extension_length = 500.0

        self.target_path_start = -np.array(direction) * extension_length
        self.target_path_end = np.array(direction) * extension_length

        self._path_lines_dirty = True

        print(f"[GL] 目标路径已设置（贯穿）: {direction}")

    def clear_path_lines(self):
        """清除所有路径虚线"""
        self.preset_path_direction = None
        self.target_path_direction = None

        if self.preset_path_line is not None:
            self.view.removeItem(self.preset_path_line)
            self.preset_path_line = None

        if self.target_path_line is not None:
            self.view.removeItem(self.target_path_line)
            self.target_path_line = None

        self._path_lines_dirty = True

        print("[GL] 所有路径虚线已清除")

    def clear_trajectory(self):
        """清除轨迹"""
        self.tip_positions = []
        self.trajectory_line.setData(pos=np.zeros((1, 3)))

    def reset_view(self, clear_trajectory=False):
        """重置视角为 config/viewport.json（与床旁对齐的默认方位）。"""
        if clear_trajectory:
            self.tip_positions = []
            self.trajectory_line.setData(pos=np.zeros((1, 3)))
            if hasattr(self, "_camera_adjusted"):
                delattr(self, "_camera_adjusted")
        self._apply_viewport_preset()

    def reset_camera(self):
        self.reset_view(clear_trajectory=False)

    def _sync_grid_axes_transform(self, translate_xyz=(0.0, 0.0, 0.0)):
        """网格与坐标轴同变换，始终共面（默认 XZ = 蓝/黄轴平面）。"""
        tx, ty, tz = (float(translate_xyz[0]), float(translate_xyz[1]), float(translate_xyz[2]))
        self.grid.resetTransform()
        self.grid.translate(tx, ty, tz)
        self.axes.resetTransform()
        self.axes.translate(tx, ty, tz)

    def fit_view_to_center(self, center, extent=None):
        """加载 CT 后：默认只调距离，方位/俯仰用 viewport.json。"""
        c = np.asarray(center, dtype=float).flatten()[:3]
        cfg = getattr(self, "_viewport_cfg", None) or self._load_viewport_config()
        fit_dist_only = bool(cfg.get("fit_ct_distance_only", True))

        if fit_dist_only:
            self.view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
            if extent is not None:
                ext = np.asarray(extent, dtype=float).flatten()[:3]
                radius = float(np.linalg.norm(ext) / 2.0)
            else:
                radius = 100.0
            distance = float(np.clip(radius * 2.2, 160.0, 420.0))
            elev = float(cfg.get("elevation", self.DEFAULT_CAMERA_ELEVATION))
            azim = float(cfg.get("azimuth", self.DEFAULT_CAMERA_AZIMUTH))
            self.view.setCameraPosition(distance=distance, elevation=elev, azimuth=azim)
            self.view.opts["distance"] = distance
        else:
            self._apply_viewport_preset(cfg)
            if extent is not None:
                ext = np.asarray(extent, dtype=float).flatten()[:3]
                radius = float(np.linalg.norm(ext) / 2.0)
                distance = float(np.clip(radius * 2.2, 160.0, 420.0))
                self.view.setCameraPosition(distance=distance)
                self.view.opts["distance"] = distance
        self._camera_adjusted = True

    def load_head_model(self, vertices, faces):
        """加载头部模型并添加到视图"""
        # 1. 如果已有模型，先移除
        if self.head_mesh is not None:
            self.view.removeItem(self.head_mesh)

        # 2. 创建 pyqtgraph 的 MeshItem
        # shader='shaded' 会自动根据法线计算光照，产生立体感
        self.head_mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            color=self.head_model_color,
            shader='shaded',
            glOptions='translucent',  # 支持半透明
            smooth=True
        )

        # 3. 添加到视图
        self.view.addItem(self.head_mesh)
        self.head_mesh.setVisible(self.head_model_visible)

        print(f"[GL] 头部模型已渲染: {len(vertices)} 顶点")

        verts = np.asarray(vertices, dtype=float)
        if len(verts) > 0:
            mn = verts.min(axis=0)
            mx = verts.max(axis=0)
            extent = mx - mn
            origin = np.zeros(3, dtype=float)
            self._sync_grid_axes_transform((0.0, 0.0, 0.0))
            self.fit_view_to_center(origin, extent)

        self.update()

    def clear_head_model(self):
        """清除头部模型"""
        if self.head_mesh is not None:
            self.view.removeItem(self.head_mesh)
            self.head_mesh = None
        self.update()

    def set_head_model_visible(self, visible):
        """控制头部模型显示/隐藏"""
        self.head_model_visible = visible
        if self.head_mesh is not None:
            self.head_mesh.setVisible(visible)
        self.update()

    def set_puncture_point(self, point, normal):
        """显示穿刺点标记和法线"""
        #print(f"[GL] 正在标记穿刺点: {point}")

        #  1. 显示Entry点球体标记
        self.puncture_point_marker.resetTransform()
        self.puncture_point_marker.translate(*point)
        self.puncture_point_marker.setVisible(True)

        #  2. 显示Entry点外圈（脉冲效果）
        self.puncture_point_glow.setData(pos=np.array([point]))
        self.puncture_point_glow.setVisible(True)

        #  3. 显示法线虚线（蓝色，400mm）
        #self.set_preset_path(normal.tolist())

        print(f"[GL] ✓ 穿刺点已标记")
        #print(f"[GL] ✓ Entry-Target连线已显示，距离: {distance:.1f} mm")

    def clear_entry_markers(self):
        """仅清除 Entry 标记与 Entry–Target 连线，保留 Target。"""
        self.puncture_point_marker.setVisible(False)
        self.puncture_point_glow.setVisible(False)

        if self.entry_target_line is not None:
            self.view.removeItem(self.entry_target_line)
            self.entry_target_line = None

        print("[GL] Entry 标记已清除")

    def clear_puncture_point(self):
        """清除 Entry 与 Target 及连线。"""
        self.clear_entry_markers()
        self.bleeding_marker.setVisible(False)
        if hasattr(self, "bleeding_point"):
            self.bleeding_point = None

        self.clear_path_lines()

        print("[GL] 穿刺点与 Target 标记已清除")

    def set_bleeding_point(self, point):
        """设置 Target 点位置；point 为 None 时隐藏。"""
        if point is None:
            self.bleeding_marker.setVisible(False)
            if hasattr(self, "bleeding_point"):
                self.bleeding_point = None
            return

        print(f"[GL] 正在标记 Target: {point}")
        self.bleeding_point = np.array(point, dtype=float)
        self.bleeding_marker.resetTransform()
        self.bleeding_marker.translate(*point)
        self.bleeding_marker.setVisible(True)
        print(f"[GL] ✓ Target 已标记: {point}")

    def set_entry_target_line(self, entry_point, target_point):
        """显示 Entry–Target 连线；任一端为 None 时仅清除连线。"""
        if self.entry_target_line is not None:
            self.view.removeItem(self.entry_target_line)
            self.entry_target_line = None

        if entry_point is None or target_point is None:
            return

        print(f"[GL] 正在创建 Entry-Target 连线")
        entry = np.array(entry_point, dtype=float)
        target = np.array(target_point, dtype=float)

        #  计算从Target到Entry的方向（反向）
        direction = entry - target  # 注意：这里是 entry - target，而不是 target - entry
        distance = np.linalg.norm(direction)
        direction = direction / distance  # 归一化

        #  从Target点开始，向Entry方向延伸
        extension_length = 200.0  # 延长200mm

        # 起点：Target点
        start_point = target

        # 终点：Target + 方向 * (原始距离 + 延长长度)
        end_point = target + direction * (distance + extension_length)

        # 创建虚线数据（从Target到延长后的终点）
        dashed_line = self._generate_dashed_line(start_point, end_point, 10.0, 5.0)

        # 创建虚线
        self.entry_target_line = gl.GLLinePlotItem(
            pos=dashed_line,
            color=(0.2, 0.6, 1.0, 0.8),  # 蓝色，半透明
            width=3.0,
            antialias=True,
            mode='lines'
        )

        self.view.addItem(self.entry_target_line)

        print(f"[GL] ✓ Entry-Target连线已显示（反向延长）")
        print(f"[GL]   起点（Target）: {target}")
        print(f"[GL]   终点（Entry外）: {end_point}")
        print(f"[GL]   原始距离: {distance:.1f} mm")
        print(f"[GL]   总长度: {distance + extension_length:.1f} mm")

    def set_needle_tip_position(self, position):
        """设置针尖固定位置（初始化+穿刺模式）"""
        self.tip_position = np.array(position, dtype=float)
        self._fixed_tip_position = self.tip_position.copy()
        self._use_fixed_tip = True
        self._update_needle_visualization()
        print(f"[GL] ✓ 针尖固定位置已设置: {self.tip_position}")

    def clear_fixed_tip(self):
        """解除针尖固定，允许 update_data 传入的动态位置生效"""
        self._use_fixed_tip = False
        if hasattr(self, '_fixed_tip_position'):
            delattr(self, '_fixed_tip_position')
        print("[GL] 针尖固定已解除，进入动态跟踪模式")

    def _update_needle_visualization(self):
        """更新针体：白线两端 + 方块中心落在端点。"""
        tip, tail = self._needle_endpoints()
        self.imu_position = tail.astype(np.float32)

        if self.needle_line is not None:
            self.needle_line.setData(pos=np.array([tip, tail], dtype=np.float32))

        try:
            self._set_box_center(self.tip_box, tip, self._tip_box_half)
            self._set_box_center(self.imu_box, tail, self._imu_box_half)
        except Exception as e:
            if not self._box_error_shown:
                self._box_error_shown = True
                print(f"[GL 错误] 立方体更新失败: {e}")

    def update_needle_direction(self, direction):
        """更新针体方向（保持针尖位置不变）"""
        self._needle_direction = np.array(direction, dtype=float)

        #  更新针体可视化
        self._update_needle_visualization()

    def clear_fixed_tip_position(self):
        """清除针尖固定位置（恢复原点模式）"""
        self._use_fixed_tip = False
        self._fixed_tip_position = np.array([0.0, 0.0, 0.0])
        print("[GL] ✓ 针尖固定位置已清除，恢复原点模式")

    def get_fixed_tip_position(self):
        """获取当前固定的针尖位置"""
        if hasattr(self, '_use_fixed_tip') and self._use_fixed_tip:
            return self._fixed_tip_position
        return None

    def set_marker_replay_mode(self, enabled: bool = True):
        """JetArm marker V1 回放：允许针尖随视觉估计移动，不记录 Entry 固定轨迹。"""
        self._use_fixed_tip = not enabled
        if enabled:
            self._marker_replay_mode = True
        elif hasattr(self, '_marker_replay_mode'):
            delattr(self, '_marker_replay_mode')

    def set_marker_needle_pose(self, tip_scene_mm, axis_scene_unit, confidence: float = 1.0):
        """
        JetArm 4-marker V1：设置针尖位置与针轴（scene 系，mm）。
        内部复用 set_needle_tip_position + update_needle_direction。
        """
        tip = np.asarray(tip_scene_mm, dtype=float).reshape(3)
        axis = np.asarray(axis_scene_unit, dtype=float).reshape(3)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return
        axis = axis / n

        self.tip_position = tip.copy()
        self._needle_direction = axis
        if hasattr(self, '_marker_replay_mode') and self._marker_replay_mode:
            self._use_fixed_tip = False

        self._update_needle_visualization()

        self._last_marker_confidence = float(confidence)
