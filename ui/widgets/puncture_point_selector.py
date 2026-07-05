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

        actual_gl_view = None
        if hasattr(gl_widget, "view"):
            actual_gl_view = gl_widget.view
        elif hasattr(gl_widget, "gl_view"):
            actual_gl_view = gl_widget.gl_view
        else:
            for child in gl_widget.children():
                if "GLViewWidget" in type(child).__name__:
                    actual_gl_view = child
                    break

        if actual_gl_view:
            self.actual_gl_view = actual_gl_view
            if hasattr(actual_gl_view, "set_puncture_selector"):
                actual_gl_view.set_puncture_selector(self)
            actual_gl_view.installEventFilter(self)
            gl_widget.installEventFilter(self)
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
            process=False,
        )
        print(
            f"[Selector] 已加载模型: {len(vertices)} 顶点, {len(faces)} 面, BVH加速就绪"
        )

    def setEnabled(self, enabled):
        self.enabled = enabled
        print(f"[Selector] 选点模式: {'开启' if enabled else '关闭'} ({self._mode})")

    def set_selection_mode(self, mode: str):
        if mode not in (self.MODE_ENTRY, self.MODE_TARGET):
            mode = self.MODE_ENTRY
        self._mode = mode

    def try_pick_from_view_event(self, ev) -> bool:
        """由 PunctureGLViewWidget.mousePressEvent 调用。"""
        if not self.enabled or self.vertices is None or self._mesh is None:
            return False
        return self._pick_and_emit(ev)

    def eventFilter(self, obj, event):
        if event.type() != QEvent.MouseButtonPress:
            return super().eventFilter(obj, event)
        if not self.enabled or self.vertices is None:
            return False
        if event.button() != Qt.LeftButton:
            return False
        if self._pick_and_emit(event, obj):
            return True
        return False

    def _pick_and_emit(self, event, obj=None) -> bool:
        try:
            point, normal = self._perform_ray_cast(event, obj)
            if point is None:
                print("[Selector] 未命中模型表面（请对准头部网格点击）")
                return False
            if self._mode == self.MODE_TARGET:
                self.target_selected.emit(point)
            else:
                self.point_selected.emit(point, normal)
            print(f"[Selector] ✓ 已选点: {point}")
            return True
        except Exception as e:
            print(f"[Selector] ✗ 射线投射失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _event_pos_in_view(self, event, obj):
        view = self.actual_gl_view
        w = max(view.width(), 1)
        h = max(view.height(), 1)
        if hasattr(event, "position"):
            lx, ly = float(event.position().x()), float(event.position().y())
        else:
            lx, ly = float(event.pos().x()), float(event.pos().y())
        if obj is not None and obj is not view:
            gpos = event.globalPos()
            local = view.mapFromGlobal(gpos)
            lx, ly = float(local.x()), float(local.y())
        return w, h, lx, ly

    def _perform_ray_cast(self, event, obj=None):
        width, height, lx, ly = self._event_pos_in_view(event, obj)
        ndc_x = (2.0 * lx) / width - 1.0
        ndc_y = 1.0 - (2.0 * ly) / height

        ray_origin, ray_direction = self._screen_to_world_ray(ndc_x, ndc_y)

        locations, _, index_tri = self._mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            multiple_hits=True,
        )

        if len(locations) == 0:
            return None, None

        dists = np.linalg.norm(locations - ray_origin, axis=1)
        closest = np.argmin(dists)
        return self._point_and_normal(locations[closest], index_tri[closest], ray_origin)

    def _point_and_normal(self, point, tri_idx, ray_origin):
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

    @staticmethod
    def _qmatrix4x4_to_numpy_col_major(m):
        """Qt QMatrix4x4 → OpenGL 列主序 4×4（与 shader 一致）。"""
        flat = np.array(m.data(), dtype=float)
        return flat.reshape(4, 4, order="F")

    def _screen_to_world_ray(self, ndc_x, ndc_y):
        view = self.actual_gl_view
        viewport = view.getViewport()
        proj = self._qmatrix4x4_to_numpy_col_major(
            view.projectionMatrix(viewport, viewport)
        )
        mv = self._qmatrix4x4_to_numpy_col_major(view.viewMatrix())
        mvp = proj @ mv
        try:
            inv_mvp = np.linalg.inv(mvp)
        except np.linalg.LinAlgError:
            return self._screen_to_world_ray_fallback(ndc_x, ndc_y)

        def unproject(ndc):
            clip = np.array([ndc[0], ndc[1], ndc[2], 1.0], dtype=float)
            world = inv_mvp @ clip
            if abs(world[3]) < 1e-12:
                return None
            return world[:3] / world[3]

        near = unproject((ndc_x, ndc_y, -1.0))
        far = unproject((ndc_x, ndc_y, 1.0))
        if near is None or far is None:
            return self._screen_to_world_ray_fallback(ndc_x, ndc_y)
        direction = far - near
        dn = np.linalg.norm(direction)
        if dn < 1e-12:
            return self._screen_to_world_ray_fallback(ndc_x, ndc_y)
        direction = direction / dn
        cam = np.array(view.cameraPosition(), dtype=float).reshape(3)
        return cam, direction

    def _screen_to_world_ray_fallback(self, ndc_x, ndc_y):
        view = self.actual_gl_view
        opts = view.opts
        fov = float(opts.get("fov", 60))
        center = opts.get("center", np.array([0, 0, 0]))
        if hasattr(center, "x"):
            center = np.array([center.x(), center.y(), center.z()], dtype=float)
        else:
            center = np.asarray(center, dtype=float).reshape(3)

        camera_pos = np.array(view.cameraPosition(), dtype=float).reshape(3)
        forward = center - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-10:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up = np.cross(right, forward)
        aspect = view.width() / max(view.height(), 1)
        tan_fov = np.tan(np.radians(fov / 2))
        ray_dir = forward + ndc_x * tan_fov * aspect * right + ndc_y * tan_fov * up
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        return camera_pos, ray_dir