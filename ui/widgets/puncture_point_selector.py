"""穿刺点选择器 - 基于射线投射的交互式选择"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QEvent


class PuncturePointSelector(QObject):
    """穿刺点选择器"""

    # 信号：选择了穿刺点（点坐标，法线向量）
    point_selected = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        self.enabled = False

        # CT模型数据（需要外部设置）
        self.vertices = None
        self.faces = None

        # 🔥 尝试找到真正的 GLViewWidget
        actual_gl_view = None

        # 方法1：检查是否有 view 属性
        if hasattr(gl_widget, 'view'):
            actual_gl_view = gl_widget.view
            print("[Selector] 找到内部 GLViewWidget: gl_widget.view")

        # 方法2：检查是否有 gl_view 属性
        elif hasattr(gl_widget, 'gl_view'):
            actual_gl_view = gl_widget.gl_view
            print("[Selector] 找到内部 GLViewWidget: gl_widget.gl_view")

        # 方法3：遍历子控件查找
        else:
            for child in gl_widget.children():
                if 'GLViewWidget' in type(child).__name__:
                    actual_gl_view = child
                    print(f"[Selector] 找到内部 GLViewWidget: {type(child).__name__}")
                    break

        # 如果找到了真正的 GLViewWidget，安装事件过滤器
        if actual_gl_view:
            self.actual_gl_view = actual_gl_view
            actual_gl_view.installEventFilter(self)
            print("[Selector] ✓ 事件过滤器已安装到真正的 GLViewWidget")
        else:
            # 降级方案：安装到外层容器
            self.actual_gl_view = gl_widget
            gl_widget.installEventFilter(self)
            print("[Selector] ⚠️ 未找到内部 GLViewWidget，安装到外层容器")

        print("[Selector] 穿刺点选择器已初始化（使用事件过滤器）")

    def set_model(self, vertices, faces):
        """设置CT模型数据（用于射线投射）"""
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        print(f"[Selector] 已加载模型: {len(vertices)} 顶点, {len(faces)} 面")

    def setEnabled(self, enabled):
        """启用/禁用选择器"""
        self.enabled = enabled
        if enabled:
            print("[Selector] ✓ 穿刺点选择器已启用（点击CT表面选择穿刺点）")
        else:
            print("[Selector] ✗ 穿刺点选择器已禁用")

    def eventFilter(self, obj, event):
        """事件过滤器 - 拦截鼠标点击事件"""
        # 只处理gl_widget的鼠标按下事件
        if obj != self.actual_gl_view:  # 🔥 改为 actual_gl_view
            return super().eventFilter(obj, event)

        if event.type() != QEvent.MouseButtonPress:
            return super().eventFilter(obj, event)


        # 如果选择器未启用，放行事件（让视角旋转正常工作）
        if not self.enabled or self.vertices is None:
            print("[Selector] 选择器未启用或模型未加载，放行事件")
            return False  # 不拦截，继续传播

        # 只处理左键点击
        if event.button() != Qt.LeftButton:
            print("[Selector] 非左键点击，放行事件")
            return False

        # 🔥 执行射线投射
        print("[Selector] 开始执行射线投射...")
        try:
            point, normal = self._perform_ray_cast(event.pos())

            if point is not None:
                print(f"[Selector] ✓ 选中穿刺点: {point}")
                print(f"[Selector]   法线方向: {normal}")

                # 发射信号
                self.point_selected.emit(point, normal)

                # 🔥 拦截事件（防止视角旋转）
                return True
            else:
                print("[Selector] ✗ 未击中CT模型")
                return False  # 未击中，放行事件

        except Exception as e:
            print(f"[Selector] ✗ 射线投射失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _perform_ray_cast(self, screen_pos):
        """执行射线投射"""
        # 1. 屏幕坐标转换为OpenGL坐标
        width = self.gl_widget.width()
        height = self.gl_widget.height()

        x = screen_pos.x()
        y = screen_pos.y()

        # 归一化设备坐标（NDC）
        ndc_x = (2.0 * x) / width - 1.0
        ndc_y = 1.0 - (2.0 * y) / height

        print(f"[Selector] 屏幕坐标: ({x}, {y}) → NDC: ({ndc_x:.3f}, {ndc_y:.3f})")

        # 2. 重建射线（世界坐标）
        ray_origin, ray_direction = self._screen_to_world_ray(ndc_x, ndc_y)
        print(f"[Selector] 射线原点: {ray_origin}")
        print(f"[Selector] 射线方向: {ray_direction}")

        # 3. 遍历所有三角形，找到最近的交点
        closest_point = None
        closest_distance = float('inf')
        closest_triangle_idx = None

        hit_count = 0  # 统计击中的三角形数量

        for i, face in enumerate(self.faces):
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Möller-Trumbore算法（射线-三角形相交）
            intersection = self._ray_triangle_intersect(
                ray_origin, ray_direction, v0, v1, v2
            )

            if intersection is not None:
                hit_count += 1
                distance = np.linalg.norm(intersection - ray_origin)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = intersection
                    closest_triangle_idx = i

        print(f"[Selector] 击中 {hit_count} 个三角形")

        if closest_point is None:
            return None, None

        print(f"[Selector] 最近交点距离: {closest_distance:.2f}mm")

        # 4. 计算法线
        face = self.faces[closest_triangle_idx]
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]

        # 三角形边向量
        edge1 = v1 - v0
        edge2 = v2 - v0

        # 法线 = edge1 × edge2
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)

        # 🔥 确保法线朝外（指向相机）
        to_camera = ray_origin - closest_point
        if np.dot(normal, to_camera) < 0:
            normal = -normal

        return closest_point, normal

    def _screen_to_world_ray(self, ndc_x, ndc_y):
        """将屏幕坐标转换为世界坐标射线"""
        # PyQtGraph的GLViewWidget使用透视投影
        # 我们需要从相机参数重建射线

        # 获取相机参数
        opts = self.actual_gl_view.opts
        distance = opts['distance']
        elevation = opts['elevation']  # 度
        azimuth = opts['azimuth']  # 度
        fov = opts.get('fov', 60)  # 视场角（默认60度）

        # 中心点
        center = opts.get('center', np.array([0, 0, 0]))
        if isinstance(center, (list, tuple)):
            center = np.array(center)
        elif hasattr(center, 'x'):  # QVector3D
            center = np.array([center.x(), center.y(), center.z()])

        # 1. 计算相机位置（球坐标系）
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)

        cam_x = distance * np.cos(elev_rad) * np.cos(azim_rad)
        cam_y = distance * np.cos(elev_rad) * np.sin(azim_rad)
        cam_z = distance * np.sin(elev_rad)

        camera_pos = center + np.array([cam_x, cam_y, cam_z])

        # 2. 计算相机的局部坐标系
        # 前向向量（指向中心）
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)

        # 上向量（世界Z轴）
        world_up = np.array([0, 0, 1])

        # 右向量
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # 重新计算上向量（确保正交）
        up = np.cross(right, forward)

        # 3. 计算射线方向（透视投影）
        aspect_ratio = self.actual_gl_view.width() / self.actual_gl_view.height()
        tan_fov = np.tan(np.radians(fov / 2))

        # NDC转相机空间
        cam_x = ndc_x * tan_fov * aspect_ratio
        cam_y = ndc_y * tan_fov

        # 相机空间 → 世界空间
        ray_direction = forward + cam_x * right + cam_y * up
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return camera_pos, ray_direction

    def _ray_triangle_intersect(self, ray_origin, ray_dir, v0, v1, v2):
        """Möller-Trumbore射线-三角形相交算法"""
        epsilon = 1e-6

        edge1 = v1 - v0
        edge2 = v2 - v0

        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)

        # 射线平行于三角形
        if abs(a) < epsilon:
            return None

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)

        if v < 0.0 or u + v > 1.0:
            return None

        # 计算交点距离
        t = f * np.dot(edge2, q)

        if t > epsilon:  # 射线相交
            intersection = ray_origin + ray_dir * t
            return intersection

        return None
