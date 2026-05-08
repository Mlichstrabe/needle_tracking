"""DICOM CT数据加载与3D模型生成"""
import os
import numpy as np
import pydicom
from skimage import measure
from PyQt5.QtCore import QObject, pyqtSignal
import trimesh


class DicomModelLoader(QObject):
    """DICOM模型加载器"""

    progress_updated = pyqtSignal(int, str)
    loading_finished = pyqtSignal(dict)
    loading_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.hu_threshold = -800  # 软组织阈值
        self.simplification_ratio = 0.15
        self.downsample_factor = 2

    def load_dicom_folder(self, folder_path):
        """加载DICOM文件夹"""
        try:
            self.progress_updated.emit(0, "正在扫描DICOM文件...")
            dicom_files = self._scan_dicom_files(folder_path)

            if not dicom_files:
                self.loading_failed.emit("未找到有效的DICOM文件")
                return

            self.progress_updated.emit(10, f"正在读取 {len(dicom_files)} 个切片...")
            slices = self._read_and_sort_slices(dicom_files)

            self.progress_updated.emit(40, "正在构建3D体数据...")
            volume_3d = self._build_volume(slices)

            # 🔥 修改1：调整裁剪范围（保留更多下部）
            self.progress_updated.emit(45, "正在裁剪体数据...")
            volume_3d = self._crop_volume(volume_3d)

            self.progress_updated.emit(50, "正在降采样...")
            volume_3d, actual_spacing = self._downsample_volume(volume_3d, slices[0])

            # 🔥 修改2：修正spacing以消除拉伸
            actual_spacing = self._fix_spacing_ratio(actual_spacing)

            self.progress_updated.emit(60, "正在生成3D模型...")
            vertices, faces = self._extract_surface(volume_3d, actual_spacing)

            self.progress_updated.emit(80, "正在优化网格...")
            vertices, faces = self._simplify_mesh(vertices, faces)

            self.progress_updated.emit(90, "正在调整方向...")
            vertices = self._rotate_to_z_axis(vertices)

            # 🔥 修改3：更激进的后脑勺裁剪
            #vertices, faces = self._remove_back_plate(vertices, faces)

            # 🔥 修改：使用几何中心而不是头顶
            bbox = self._calculate_bbox(vertices)
            ct_center = bbox['center']  # 使用包围盒的中心点

            print(f"[坐标变换] CT几何中心: {ct_center}")
            print(f"[坐标变换] CT尺寸: X={bbox['size'][0]:.1f}, Y={bbox['size'][1]:.1f}, Z={bbox['size'][2]:.1f}mm")

            # 将CT中心移到世界坐标原点
            vertices = vertices - ct_center

            # 验证变换结果
            final_bbox = self._calculate_bbox(vertices)
            print(f"[坐标变换] 变换后中心: {final_bbox['center']} (应接近[0,0,0])")
            print(f"[坐标变换] 头顶位置: Z={final_bbox['max'][2]:.1f}mm (应为正值)")

            vertices = self._verify_scale(vertices)

            self.progress_updated.emit(100, "加载完成!")

            model_data = {
                'vertices': vertices,
                'faces': faces,
                'bbox': self._calculate_bbox(vertices),
                'center': np.array([0, 0, 0]),
                'num_vertices': len(vertices),
                'num_faces': len(faces)
            }

            self.loading_finished.emit(model_data)

        except Exception as e:
            import traceback
            error_msg = f"加载失败: {str(e)}\n{traceback.format_exc()}"
            self.loading_failed.emit(error_msg)

    def _scan_dicom_files(self, folder_path):
        """扫描DICOM文件"""
        dicom_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return sorted(dicom_files)

    def _read_and_sort_slices(self, dicom_files):
        """读取并排序切片"""
        slices = []
        for i, filepath in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(filepath)
                slices.append(ds)

                if i % 10 == 0:
                    progress = 10 + int((i / len(dicom_files)) * 30)
                    self.progress_updated.emit(progress, f"读取切片 {i + 1}/{len(dicom_files)}")
            except Exception as e:
                print(f"跳过文件 {filepath}: {e}")
                continue

        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices

    def _build_volume(self, slices):
        """构建3D体数据"""
        img_shape = slices[0].pixel_array.shape
        volume = np.zeros((len(slices), img_shape[0], img_shape[1]), dtype=np.int16)

        for i, s in enumerate(slices):
            pixel_array = s.pixel_array.astype(np.int16)
            intercept = s.RescaleIntercept if hasattr(s, 'RescaleIntercept') else 0
            slope = s.RescaleSlope if hasattr(s, 'RescaleSlope') else 1
            volume[i] = pixel_array * slope + intercept

        print(f"[体数据] Shape = {volume.shape}, HU范围: {volume.min()} ~ {volume.max()}")
        return volume

    def _crop_volume(self, volume):
        """🔥 修改1：保留上70%（之前是50%，现在保留更多下部）"""
        z_slices = volume.shape[0]
        start_slice = int(z_slices * 0.3)  # 从30%处开始（保留70%）
        cropped = volume[start_slice:, :, :]

        print(f"[体数据裁剪] {volume.shape} → {cropped.shape} (保留上70%)")
        return cropped

    def _downsample_volume(self, volume, reference_slice):
        """降采样体数据"""
        from scipy import ndimage

        original_shape = volume.shape
        downsampled = ndimage.zoom(
            volume,
            (1/self.downsample_factor, 1/self.downsample_factor, 1/self.downsample_factor),
            order=1
        )

        # 计算原始spacing
        pixel_spacing = reference_slice.PixelSpacing
        slice_thickness = float(reference_slice.SliceThickness) if hasattr(reference_slice, 'SliceThickness') else 1.0

        actual_spacing = (
            slice_thickness * self.downsample_factor,
            pixel_spacing[0] * self.downsample_factor,
            pixel_spacing[1] * self.downsample_factor
        )

        print(f"[降采样] {original_shape} → {downsampled.shape}")
        print(f"[原始spacing] Z={slice_thickness:.3f}, XY={pixel_spacing[0]:.3f}mm")
        print(f"[降采样后spacing] {actual_spacing}")

        return downsampled, actual_spacing

    def _fix_spacing_ratio(self, spacing):
        """🔥 修改2：修正spacing比例，消除拉伸"""
        # 使用XY方向的平均值作为参考
        xy_avg = (spacing[1] + spacing[2]) / 2

        # 🔥 关键：强制Z方向使用相同的间距（等比例）
        fixed_spacing = (xy_avg, spacing[1], spacing[2])

        print(f"[拉伸修正] {spacing} → {fixed_spacing}")
        print(f"  Z方向间距从 {spacing[0]:.3f}mm 修正为 {xy_avg:.3f}mm")

        return fixed_spacing

    def _extract_surface(self, volume, spacing):
        """提取等值面"""
        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=self.hu_threshold,
            spacing=spacing
        )

        print(f"[Marching Cubes] HU阈值={self.hu_threshold}, 顶点={len(verts)}, 面={len(faces)}")
        return verts, faces

    def _simplify_mesh(self, vertices, faces):
        """网格简化"""
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            target_faces = int(len(faces) * self.simplification_ratio)
            simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
            print(f"[简化] 面数: {len(faces)} → {len(simplified_mesh.faces)}")
            return simplified_mesh.vertices, simplified_mesh.faces
        except Exception as e:
            print(f"网格简化失败: {e}")
            return vertices, faces

    def _rotate_to_z_axis(self, vertices):
        """旋转模型"""
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        rotated = vertices @ rotation_matrix.T

        bbox = self._calculate_bbox(rotated)
        z_height = bbox['max'][2] - bbox['min'][2]
        print(f"[旋转] Z轴高度: {z_height:.1f}mm")

        return rotated

    def _remove_back_plate(self, vertices, faces):
        """移除后脑勺挡板"""
        # 计算Y轴范围
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        y_center = (y_min + y_max) / 2
        y_range = y_max - y_min

        # 🔥 关键修正：判断头部朝向
        # 如果模型面部朝向+Y，后脑勺在-Y
        # 计算前后两侧的顶点数量，判断哪边是面部
        front_count = np.sum(vertices[:, 1] > y_center)
        back_count = np.sum(vertices[:, 1] < y_center)

        print(f"[朝向检测] 前侧顶点数: {front_count}, 后侧顶点数: {back_count}")

        # 如果前侧顶点更多，说明面部朝向+Y，裁剪-Y侧
        if front_count > back_count:
            # 裁剪后40%（-Y侧）
            y_threshold = y_center - y_range * 0.2
            direction = "后方(-Y)"
        else:
            # 裁剪前40%（+Y侧）
            y_threshold = y_center + y_range * 0.2
            direction = "后方(+Y)"

        print(f"[后脑勺裁剪] 裁剪{direction}, 阈值Y={y_threshold:.1f}mm")

        # 标记要保留的三角形
        valid_faces = []
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            center_y = (v0[1] + v1[1] + v2[1]) / 3.0

            # 根据朝向决定保留条件
            if front_count > back_count:
                keep = center_y > y_threshold  # 保留+Y侧（面部）
            else:
                keep = center_y < y_threshold  # 保留-Y侧（面部）

            if keep:
                valid_faces.append(face)

        # 如果裁剪后剩余<30%，说明裁剪过度
        if len(valid_faces) < len(faces) * 0.3:
            print(f"[警告] 裁剪过度，跳过此步骤")
            return vertices, faces

        valid_faces = np.array(valid_faces)

        # 重建网格
        used_indices = np.unique(valid_faces.flatten())
        new_vertices = vertices[used_indices]
        vertex_map = {old: new for new, old in enumerate(used_indices)}
        new_faces = np.array([[vertex_map[idx] for idx in face] for face in valid_faces])

        print(
            f"[后脑勺裁剪] 顶点: {len(vertices)} → {len(new_vertices)} ({100 * len(new_vertices) / len(vertices):.1f}%)")

        return new_vertices, new_faces

    def _calculate_bbox(self, vertices):
        """计算包围盒"""
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return {
            'min': min_coords,
            'max': max_coords,
            'center': (min_coords + max_coords) / 2,
            'size': max_coords - min_coords
        }

    def _calculate_head_top(self, bbox):
        """计算头顶位置"""
        top_z = bbox['max'][2]
        center_x = (bbox['min'][0] + bbox['max'][0]) / 2
        center_y = (bbox['min'][1] + bbox['max'][1]) / 2
        return np.array([center_x, center_y, top_z])

    def _verify_scale(self, vertices):
        """验证物理尺寸"""
        bbox = self._calculate_bbox(vertices)
        h = bbox['max'][2] - bbox['min'][2]
        w = bbox['max'][0] - bbox['min'][0]
        d = bbox['max'][1] - bbox['min'][1]

        print(f"[尺寸检测] X={w:.1f}mm, Y={d:.1f}mm, Z={h:.1f}mm")

        # 典型头部高度220mm
        if h < 150 or h > 300:
            scale = 220.0 / h
            vertices = vertices * scale
            print(f"[尺寸校正] 缩放因子: {scale:.3f}")

            new_bbox = self._calculate_bbox(vertices)
            new_h = new_bbox['max'][2] - new_bbox['min'][2]
            print(f"[尺寸验证] 校正后: Z={new_h:.1f}mm (针长162mm的{new_h/162:.2f}倍)")
        else:
            print(f"[尺寸验证] 正常 (头高/针长 = {h/162:.2f})")

        return vertices
