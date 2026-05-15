"""3D可视化组件 - 支持模拟穿刺显示"""
from PyQt5 import QtGui
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtGui import QVector3D, QMatrix4x4


class GLVisualizationWidget(QFrame):
    """基于PyQtGraph的3D可视化组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("background: #1a1a2e; border-radius: 8px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 3D视图
        self.view = gl.GLViewWidget()
        #  调整相机以适应CT模型（更大的距离和更低的仰角）
        self.view.setCameraPosition(distance=350, elevation=20, azimuth=45)

        # 性能优化设置
        self.view.opts['distance'] = 250
        fmt = self.view.format()
        fmt.setSamples(0)
        self.view.setFormat(fmt)

        layout.addWidget(self.view)

        # 数据存储
        self.tip_positions = []
        self.imu_position = np.array([0, 0, 0], dtype=float)
        self.tip_position = np.array([0, 0, 0], dtype=float)

        # 模拟相关
        self.simulated_tip_position = None
        self.target_position = None
        self.sphere_center = None
        self.outer_radius = 40.0
        self.entry_point = None

        # 虚线路径
        self.preset_path_direction = None
        self.target_path_direction = None
        self.path_line_length = 400.0
        self.preset_path_line = None
        self.target_path_line = None

        # 性能优化计数器
        self._traj_counter = 0
        self._guide_counter = 0
        self._gl_initialized = False
        self._camera_adjusted = False
        self._path_lines_dirty = True
        self._needle_update_confirmed = False
        self._traj_error_shown = False
        self._guide_error_shown = False
        self._box_error_shown = False
        self._box_update_confirmed = False
        self._traj_threshold_sq = 16.0

        # 初始化场景
        self._init_scene()
        self._init_objects()
        self._init_simulation_objects()

        # ====== CT头部模型 ======
        self.head_mesh = None
        self.head_model_visible = True
        self.head_model_color = (0.9, 0.9, 0.9, 0.6)

        print("[GL] 性能优化已启用")

    def _init_scene(self):
        """初始化场景"""
        axes = gl.GLAxisItem()
        axes.setSize(150, 150, 150)
        self.view.addItem(axes)

        grid_lines = self._create_grid(400, 40)
        self.grid = gl.GLLinePlotItem(
            pos=grid_lines,
            color=(1, 1, 1, 0.3),
            width=1,
            mode='lines'
        )
        self.view.addItem(self.grid)

    def _create_grid(self, size, spacing):
        """创建网格线数据"""
        lines = []
        half = size / 2
        for i in np.arange(-half, half + spacing, spacing):
            lines.append([[-half, i, 0], [half, i, 0]])
            lines.append([[i, -half, 0], [i, half, 0]])
        return np.array(lines).reshape(-1, 3)

    def _init_objects(self):
        """初始化基础绘制对象"""
        self.needle_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, -100]], dtype=np.float32),
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
            color=(1, 0, 0, 0.6),  # 红色，90%不透明
            shader='shaded',
            glOptions='additive'
        )
        self.bleeding_marker.setVisible(False)
        self.view.addItem(self.bleeding_marker)

        # 4. Entry-Target连线（蓝色虚线）
        self.entry_target_line = None  # 稍后创建

        print("[GL] ✓ 穿刺点标记系统已初始化")

    def _create_sphere_mesh(self, center, radius, color, rows=15, cols=15):
        """创建球体网格"""
        if not hasattr(self, '_sphere_cache'):
            self._sphere_cache = {}

        cache_key = (rows, cols)

        if cache_key in self._sphere_cache:
            verts_template, faces = self._sphere_cache[cache_key]
            verts = verts_template * radius + center
        else:
            phi = np.linspace(0, np.pi, rows)
            theta = np.linspace(0, 2 * np.pi, cols)
            phi, theta = np.meshgrid(phi, theta)

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            verts_template = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

            faces = []
            for i in range(rows - 1):
                for j in range(cols - 1):
                    idx = i * cols + j
                    faces.append([idx, idx + 1, idx + cols])
                    faces.append([idx + 1, idx + cols + 1, idx + cols])
            faces = np.array(faces)

            self._sphere_cache[cache_key] = (verts_template, faces)
            verts = verts_template * radius + center

        return gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            smooth=False,
            color=color,
            shader='shaded',
            glOptions='translucent'
        )

    def update_data(self, imu_pos, tip_pos, max_points=500):
        """更新可视化数据"""

        #  如果设置了固定针尖位置，覆盖传入的 tip_pos
        if hasattr(self, '_use_fixed_tip') and self._use_fixed_tip:
            tip_pos = list(self._fixed_tip_position)
            # 重新计算 imu_pos（保持针体长度不变）
            if hasattr(self, '_needle_direction') and self._needle_direction is not None:
                needle_length = 100.0  # 默认针体长度
                direction = np.array(self._needle_direction)
                imu_pos = [
                    tip_pos[0] - direction[0] * needle_length,
                    tip_pos[1] - direction[1] * needle_length,
                    tip_pos[2] - direction[2] * needle_length
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

        if not self._camera_adjusted:
            self._camera_adjusted = True
            imu_array = np.array(imu_pos, dtype=np.float32)
            tip_array = np.array(tip_pos, dtype=np.float32)
            center = (imu_array + tip_array) / 2

            self.view.opts['center'] = QVector3D(
                float(center[0]),
                float(center[1]),
                float(center[2])
            )

            distance_from_origin = np.linalg.norm(center)
            if distance_from_origin > 100:
                self.view.setCameraPosition(distance=distance_from_origin * 2.5)

        self.imu_position = np.array(imu_pos, dtype=np.float32)
        self.tip_position = np.array(tip_pos, dtype=np.float32)

        # 更新立方体位置
        try:
            tip_transform = QMatrix4x4()
            tip_transform.translate(
                float(self.tip_position[0]),
                float(self.tip_position[1]),
                float(self.tip_position[2]) + 2.5
            )
            self.tip_box.setTransform(tip_transform)

            imu_transform = QMatrix4x4()
            imu_transform.translate(
                float(self.imu_position[0]),
                float(self.imu_position[1]),
                float(self.imu_position[2])
            )
            self.imu_box.setTransform(imu_transform)
        except Exception as e:
            if not self._box_error_shown:
                self._box_error_shown = True
                print(f"[GL 错误] 立方体更新失败: {e}")

        # 更新针杆线条
        if self.needle_line is not None:
            try:
                needle_data = np.array([
                    self.tip_position,
                    self.imu_position
                ], dtype=np.float32)
                self.needle_line.setData(pos=needle_data)
            except Exception as e:
                print(f"[GL 错误] 针杆线条更新失败: {e}")

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

        # 更新引导线
        if hasattr(self, 'guide_line') and self.guide_line.visible():
            self._guide_counter += 1
            if self._guide_counter >= 10:
                self._guide_counter = 0
                if self.sphere_center is not None:
                    try:
                        guide_data = np.array([
                            self.imu_position,
                            self.sphere_center
                        ], dtype=np.float32)
                        self.guide_line.setData(pos=guide_data)
                    except Exception as e:
                        if not self._guide_error_shown:
                            self._guide_error_shown = True
                            print(f"[GL 错误] 引导线更新失败: {e}")

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

    def update_needle_direction(self, direction):
        """更新针尖方向"""
        self._needle_direction = direction if direction else [0, 0, 1]

    def set_simulation_target(self, target_position, sphere_center, outer_radius=40.0):
        """设置模拟目标"""
        self.target_position = np.array(target_position, dtype=float)
        self.sphere_center = np.array(sphere_center, dtype=float)
        self.outer_radius = outer_radius
        self.target_scatter.setData(pos=np.array([self.target_position]))
        self._update_outer_sphere()
        self.guide_line.setVisible(True)

    def _update_outer_sphere(self):
        """更新外层球体"""
        if self.sphere_center is None:
            if self.outer_sphere is not None:
                self.view.removeItem(self.outer_sphere)
                self.outer_sphere = None
            return

        if self.outer_sphere is not None:
            self.view.removeItem(self.outer_sphere)

        color = (0.2, 0.5, 1.0, self._sphere_opacity)
        self.outer_sphere = self._create_sphere_mesh(
            self.sphere_center,
            self.outer_radius,
            color
        )
        self.view.addItem(self.outer_sphere)

    def set_sphere_opacity(self, opacity_level):
        """设置球体透明度"""
        if opacity_level == 0:
            self._sphere_opacity = 0.1
        elif opacity_level == 1:
            self._sphere_opacity = 0.35
        else:
            self._sphere_opacity = 0.85
        self._update_outer_sphere()

    def clear_simulation(self):
        """清除模拟显示"""
        if self.outer_sphere is not None:
            self.view.removeItem(self.outer_sphere)
            self.outer_sphere = None

        self.target_scatter.setVisible(False)
        self.entry_scatter.setVisible(False)
        self.guide_line.setVisible(False)
        self.sim_tip_scatter.setVisible(False)
        self.sim_trajectory_line.setVisible(False)

    def clear_trajectory(self):
        """清除轨迹"""
        self.tip_positions = []
        self.trajectory_line.setData(pos=np.zeros((1, 3)))

    def reset_view(self, clear_trajectory=False):
        """重置视角"""
        if clear_trajectory:
            self.view.opts['center'] = QVector3D(0, 0, 0)
            self.view.setCameraPosition(distance=200, elevation=30, azimuth=45)
            self.tip_positions = []
            self.trajectory_line.setData(pos=np.zeros((1, 3)))
            if hasattr(self, '_camera_adjusted'):
                delattr(self, '_camera_adjusted')
        else:
            self.view.setCameraPosition(distance=250, elevation=30, azimuth=45)

    @property
    def trajectory_count(self):
        """获取轨迹点数量"""
        return len(self.tip_positions)

    # 兼容方法
    def set_correction_arrows(self, *args, **kwargs):
        pass

    def set_target_visible(self, visible):
        self.target_scatter.setVisible(visible)

    def set_entry_point(self, entry_point):
        if entry_point is None:
            self.entry_scatter.setVisible(False)
        else:
            self.entry_point = np.array(entry_point, dtype=float)
            self.entry_scatter.setData(pos=np.array([self.entry_point]))
            self.entry_scatter.setVisible(True)

    def update_simulated_tip(self, sim_tip_position, lock_position=None):
        if sim_tip_position is None:
            self.sim_tip_scatter.setVisible(False)
            self.sim_trajectory_line.setVisible(False)
        else:
            self.sim_tip_scatter.setData(pos=np.array([sim_tip_position]))
            self.sim_tip_scatter.setVisible(True)

    def set_guide_line_visible(self, visible):
        self.guide_line.setVisible(visible)

    def show_success_effect(self):
        if self.target_position is not None:
            self.target_scatter.setData(
                pos=np.array([self.target_position]),
                color=(0, 1, 0, 1),
                size=20
            )

    def show_failure_effect(self):
        if self.target_position is not None:
            self.target_scatter.setData(
                pos=np.array([self.target_position]),
                color=(1, 0, 0, 1),
                size=20
            )

    def reset_camera(self):
        self.view.setCameraPosition(distance=250, elevation=30, azimuth=45)

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

        # 4. 自动调整视角看向模型
        #self.view.setCameraPosition(distance=300)
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

    def clear_puncture_point(self):
        """清除穿刺点标记"""
        self.puncture_point_marker.setVisible(False)
        self.puncture_point_glow.setVisible(False)
        self.bleeding_marker.setVisible(False)

        if self.entry_target_line is not None:
            self.view.removeItem(self.entry_target_line)
            self.entry_target_line = None

        self.clear_path_lines()

        print("[GL] 穿刺点标记已清除")

    def set_bleeding_point(self, point):
        """设置出血点（Target点）位置"""
        print(f"[GL] 正在标记出血点: {point}")

        self.bleeding_point = np.array(point)
        self.bleeding_marker.resetTransform()
        self.bleeding_marker.translate(*point)
        self.bleeding_marker.setVisible(True)

        print(f"[GL] ✓ 出血点已标记: {point}")

    def set_entry_target_line(self, entry_point, target_point):
        """显示Entry-Target连线（蓝色虚线，从Target向Entry方向反向延长）"""
        print(f"[GL] 正在创建Entry-Target连线（反向延长）")

        # 清除旧的连线
        if self.entry_target_line is not None:
            self.view.removeItem(self.entry_target_line)

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

    def _update_needle_visualization(self):
        """更新针体可视化（根据当前位置和方向）"""
        # 如果还没有方向数据，使用默认方向（竖直向下）
        if not hasattr(self, '_needle_direction') or self._needle_direction is None:
            direction = np.array([0, 0, -1], dtype=float)
        else:
            direction = self._needle_direction

        # 针体长度
        needle_length = 100.0  # mm

        # 计算针体末端位置
        needle_end = self.tip_position - direction * needle_length

        # 更新针体线条
        self.needle_line.setData(pos=np.array([
            self.tip_position,
            needle_end
        ]))

        # 更新针尖立方体位置
        self.tip_box.resetTransform()
        self.tip_box.translate(*self.tip_position)

        # 更新IMU立方体位置（在针体中部）
        imu_position = self.tip_position - direction * 50.0
        self.imu_box.resetTransform()
        self.imu_box.translate(*imu_position)

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
