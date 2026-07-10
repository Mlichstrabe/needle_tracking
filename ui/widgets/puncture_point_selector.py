"""穿刺点选择器 - 基于trimesh BVH加速的交互式选择"""
import numpy as np
import trimesh
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QEvent


class PuncturePointSelector(QObject):
    """穿刺点 / Target 选择器（同一套射线投射）"""

    point_selected = pyqtSignal(np.ndarray, np.ndarray)
    target_selected = pyqtSignal(np.ndarray)

    MODE_ENTRY = "entry"
    MODE_TARGET = "target"

    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        self.enabled = False
        self._mode = self.MODE_ENTRY

        self.vertices = None
        self.faces = None
        self._mesh = None

        # 找到真正的 GLViewWidget
        actual_gl_view = None
        if hasattr(gl_widget, 'view'):
            actual_gl_view = gl_widget.view
        elif hasattr(gl_widget, 'gl_view'):
            actual_gl_view = gl_widget.gl_view
        else:
            for child in gl_widget.children():
                if 'GLViewWidget' in type(child).__name__:
                    actual_gl_view = child
                    break

        if actual_gl_view:
            self.actual_gl_view = actual_gl_view
            actual_gl_view.installEventFilter(self)
        else:
            self.actual_gl_view = gl_widget
            gl_widget.installEventFilter(self)

    def set_model(self, vertices, faces):
        """设置CT模型数据并构建BVH加速结构"""
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self._mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False
        )
        print(f"[Selector] 已加载模型: {len(vertices)} 顶点, {len(faces)} 面, BVH加速就绪")

    def setEnabled(self, enabled):
        self.enabled = enabled

    def set_selection_mode(self, mode: str):
        if mode not in (self.MODE_ENTRY, self.MODE_TARGET):
            mode = self.MODE_ENTRY
        self._mode = mode

    def eventFilter(self, obj, event):
        if obj != self.actual_gl_view:
            return super().eventFilter(obj, event)
        if event.type() != QEvent.MouseButtonPress:
            return super().eventFilter(obj, event)
        if not self.enabled or self.vertices is None:
            return False
        if event.button() != Qt.LeftButton:
            return False

        try:
            point, normal = self._perform_ray_cast(event.pos())
            if point is not None:
                if self._mode == self.MODE_TARGET:
                    self.target_selected.emit(point)
                else:
                    self.point_selected.emit(point, normal)
                return True
            return False
        except Exception as e:
            print(f"[Selector] ✗ 射线投射失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _perform_ray_cast(self, screen_pos):
        """执行BVH加速的射线投射"""
        width = self.gl_widget.width()
        height = self.gl_widget.height()
        ndc_x = (2.0 * screen_pos.x()) / width - 1.0
        ndc_y = 1.0 - (2.0 * screen_pos.y()) / height

        ray_origin, ray_direction = self._screen_to_world_ray(ndc_x, ndc_y)

        # BVH加速的射线-网格相交
        locations, _, index_tri = self._mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            multiple_hits=True
        )

        if len(locations) == 0:
            return None, None

        # 找到距离相机最近的交点
        dists = np.linalg.norm(locations - ray_origin, axis=1)
        closest = np.argmin(dists)
        return self._point_and_normal(locations[closest], index_tri[closest], ray_origin)

    def _point_and_normal(self, point, tri_idx, ray_origin):
        """从交点+三角形索引计算法线"""
        face = self.faces[tri_idx]
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]

        normal = np.cross(v1 - v0, v2 - v0)
        n = np.linalg.norm(normal)
        if n > 1e-10:
            normal = normal / n
        else:
            normal = np.array([0.0, 0.0, 1.0])

        if np.dot(normal, ray_origin - point) < 0:
            normal = -normal

        return point, normal

    def _screen_to_world_ray(self, ndc_x, ndc_y):
        """屏幕NDC坐标 → 世界空间射线"""
        opts = self.actual_gl_view.opts
        distance = opts['distance']
        elevation = opts['elevation']
        azimuth = opts['azimuth']
        fov = opts.get('fov', 60)

        center = opts.get('center', np.array([0, 0, 0]))
        if isinstance(center, (list, tuple)):
            center = np.array(center, dtype=float)
        elif hasattr(center, 'x'):
            center = np.array([center.x(), center.y(), center.z()], dtype=float)

        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)

        camera_pos = center + np.array([
            distance * np.cos(elev_rad) * np.cos(azim_rad),
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.sin(elev_rad),
        ])

        forward = center - camera_pos
        fn = np.linalg.norm(forward)
        if fn < 1e-10:
            forward = np.array([0, 0, -1])
        else:
            forward = forward / fn

        right = np.cross(forward, np.array([0, 0, 1]))
        rn = np.linalg.norm(right)
        if rn < 1e-10:
            right = np.array([1, 0, 0])
        else:
            right = right / rn

        up = np.cross(right, forward)

        aspect = self.actual_gl_view.width() / max(self.actual_gl_view.height(), 1)
        tan_fov = np.tan(np.radians(fov / 2))

        ray_dir = forward + ndc_x * tan_fov * aspect * right + ndc_y * tan_fov * up
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return camera_pos, ray_dir
