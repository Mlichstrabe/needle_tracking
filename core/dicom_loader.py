"""DICOM CT数据加载与3D模型生成"""
import os
import concurrent.futures
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

        self.hu_threshold = -800
        self.simplification_ratio = 0.22
        self.large_mesh_simplification_ratio = 0.13
        self.min_face_count = 16000
        self.downsample_factor = 2
        self._voxel_threshold_high = 50_000_000
        self._voxel_threshold_mid = 14_000_000

    def load_dicom_folder(self, folder_path):
        """加载DICOM文件夹"""
        try:
            self.progress_updated.emit(0, "正在扫描DICOM文件...")
            dicom_files = self._scan_dicom_files(folder_path)

            if not dicom_files:
                self.loading_failed.emit("未找到有效的DICOM文件")
                return

            self.progress_updated.emit(10, f"正在并行读取 {len(dicom_files)} 个切片...")
            slices = self._read_and_sort_slices(dicom_files)

            self.progress_updated.emit(40, "正在构建3D体数据...")
            volume_3d = self._build_volume(slices)

            self.progress_updated.emit(45, "正在裁剪体数据...")
            volume_3d = self._crop_volume(volume_3d)

            effective_downsample = self._pick_downsample_factor(volume_3d.shape)
            voxels = int(np.prod(volume_3d.shape))
            self.progress_updated.emit(
                48,
                f"降采样 ×{effective_downsample}（体素 {voxels:,}）",
            )

            self.progress_updated.emit(50, "正在降采样...")
            volume_3d, actual_spacing = self._downsample_volume(
                volume_3d, slices[0], effective_downsample
            )

            actual_spacing = self._fix_spacing_ratio(actual_spacing)

            self.progress_updated.emit(60, "正在生成3D模型...")
            vertices, faces = self._extract_surface(volume_3d, actual_spacing)

            self.progress_updated.emit(80, "正在优化网格...")
            vertices, faces = self._simplify_mesh(vertices, faces)

            self.progress_updated.emit(90, "正在调整方向...")
            vertices = self._rotate_to_z_axis(vertices)
            vertices = self._rotate_z_clockwise_deg(vertices, 90.0)

            vertices = self._center_vertices_at_origin(vertices)

            self._log_bbox(vertices)
            vertices = self._verify_scale(vertices)
            vertices = self._center_vertices_at_origin(vertices)

            self.progress_updated.emit(100, "加载完成!")

            bbox = self._calculate_bbox(vertices)
            model_data = {
                'vertices': vertices,
                'faces': faces,
                'bbox': bbox,
                'center': bbox['center'],
                'num_vertices': len(vertices),
                'num_faces': len(faces)
            }

            self.loading_finished.emit(model_data)

        except Exception as e:
            import traceback
            error_msg = f"加载失败: {str(e)}\n{traceback.format_exc()}"
            self.loading_failed.emit(error_msg)

    def _pick_downsample_factor(self, volume_shape):
        """根据裁剪后体素数选择各向同性降采样因子（上限 3，不再用 ×4）"""
        voxels = int(np.prod(volume_shape))
        if voxels > self._voxel_threshold_high:
            return 3
        if voxels > self._voxel_threshold_mid:
            return 2
        if voxels <= 8_000_000:
            return 1
        return self.downsample_factor

    def _scan_dicom_files(self, folder_path):
        """扫描DICOM文件"""
        dicom_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return sorted(dicom_files)

    def _read_and_sort_slices(self, dicom_files):
        """并行读取并排序切片"""
        total = len(dicom_files)

        def read_one(path):
            try:
                return pydicom.dcmread(path)
            except Exception as e:
                print(f"跳过文件 {path}: {e}")
                return None

        slices = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(read_one, f): i for i, f in enumerate(dicom_files)}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    slices.append(result)
                done += 1
                if done % 15 == 0:
                    progress = 10 + int((done / total) * 30)
                    self.progress_updated.emit(progress, f"读取切片 {done}/{total}")

        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print(f"[DICOM] 并行读取完成: {len(slices)}/{total} 有效切片")
        return slices

    def _build_volume(self, slices):
        """构建3D体数据（预取元数据避免循环内hasattr）"""
        img_shape = slices[0].pixel_array.shape
        volume = np.zeros((len(slices), img_shape[0], img_shape[1]), dtype=np.int16)

        # 所有切片共用相同的Rescale参数，一次性读取
        intercept = slices[0].RescaleIntercept if hasattr(slices[0], 'RescaleIntercept') else 0
        slope = slices[0].RescaleSlope if hasattr(slices[0], 'RescaleSlope') else 1

        for i, s in enumerate(slices):
            volume[i] = s.pixel_array.astype(np.int16) * slope + intercept

        print(f"[体数据] Shape = {volume.shape}, HU范围: {volume.min()} ~ {volume.max()}")
        return volume

    def _crop_volume(self, volume):
        """保留上70%（自适应保留含组织的切片）"""
        z_slices = volume.shape[0]
        # 从底部向上扫描，找到第一个有组织（HU > -500）的切片
        threshold_hu = -500
        first_meaningful = 0
        for i in range(z_slices // 4):  # 只检查底部1/4
            if np.percentile(volume[i, :, :], 90) > threshold_hu:
                first_meaningful = max(0, i - 5)  # 保留一点余量
                break
        else:
            # 没找到有组织的切片，用旧的30%规则
            first_meaningful = int(z_slices * 0.3)

        cropped = volume[first_meaningful:, :, :]
        print(f"[体数据裁剪] {volume.shape} → {cropped.shape} (跳过底部 {first_meaningful} 切片)")
        return cropped

    def _downsample_volume(self, volume, reference_slice, factor=None):
        """降采样体数据"""
        from scipy import ndimage

        f = factor or self.downsample_factor
        pixel_spacing = reference_slice.PixelSpacing
        slice_thickness = (
            float(reference_slice.SliceThickness)
            if hasattr(reference_slice, "SliceThickness")
            else 1.0
        )
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

        if f <= 1:
            print(f"[降采样] 跳过 (体素 {volume.shape})")
            return volume, spacing

        original_shape = volume.shape
        downsampled = ndimage.zoom(volume, (1 / f, 1 / f, 1 / f), order=1)
        actual_spacing = (spacing[0] * f, spacing[1] * f, spacing[2] * f)

        print(f"[降采样] {original_shape} → {downsampled.shape} (因子={f})")
        return downsampled, actual_spacing

    def _fix_spacing_ratio(self, spacing):
        """修正spacing比例，消除拉伸"""
        xy_avg = (spacing[1] + spacing[2]) / 2
        fixed_spacing = (xy_avg, spacing[1], spacing[2])
        print(f"[拉伸修正] {spacing} → {fixed_spacing}")
        return fixed_spacing

    def _extract_surface(self, volume, spacing):
        """提取等值面（Lewiner，表面更稳定）"""
        kwargs = dict(level=self.hu_threshold, spacing=spacing)
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume, method="lewiner", **kwargs
            )
        except TypeError:
            verts, faces, normals, values = measure.marching_cubes(volume, **kwargs)
        print(
            f"[Marching Cubes] HU={self.hu_threshold}, "
            f"顶点={len(verts)}, 面={len(faces)}"
        )
        return verts, faces

    def _simplify_mesh(self, vertices, faces):
        """网格简化（大网格自适应降级）"""
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            ratio = self.simplification_ratio
            if len(faces) > 350000:
                ratio = min(ratio, self.large_mesh_simplification_ratio)
                print(f"[简化] 大网格({len(faces)}面)，简化比例: {ratio}")

            target_faces = max(int(len(faces) * ratio), self.min_face_count)
            simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
            print(f"[简化] 面数: {len(faces)} → {len(simplified_mesh.faces)} (目标{target_faces})")
            return simplified_mesh.vertices, simplified_mesh.faces

        except Exception as e:
            print(f"网格简化失败: {e}, 返回原始网格")
            return vertices, faces

    def _center_vertices_at_origin(self, vertices):
        """平移使几何中心落在世界原点。"""
        bbox = self._calculate_bbox(vertices)
        center = np.asarray(bbox["center"], dtype=float)
        shifted = vertices - center
        print(f"[坐标变换] 几何中心 {center} → 原点")
        return shifted

    def _rotate_z_clockwise_deg(self, vertices, deg_clockwise):
        """绕场景 Z 轴顺时针旋转（俯视 +X→+Y 为逆时针，故取负角）。"""
        theta = -np.deg2rad(float(deg_clockwise))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        out = vertices @ rot.T
        print(f"[旋转] 绕 Z 顺时针 {deg_clockwise}°")
        return out

    def _rotate_to_z_axis(self, vertices):
        """旋转模型"""
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        rotated = vertices @ rotation_matrix.T
        z_height = rotated[:, 2].max() - rotated[:, 2].min()
        print(f"[旋转] Z轴高度: {z_height:.1f}mm")
        return rotated

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

    def _log_bbox(self, vertices):
        """打印变换后的包围盒信息"""
        bbox = self._calculate_bbox(vertices)
        print(f"[坐标变换] CT几何中心: {bbox['center']}")
        print(f"[坐标变换] CT尺寸: X={bbox['size'][0]:.1f}, Y={bbox['size'][1]:.1f}, "
              f"Z={bbox['size'][2]:.1f}mm")
        print(f"[坐标变换] 变换后中心: {bbox['center']} (应接近[0,0,0])")

    def _verify_scale(self, vertices):
        """验证物理尺寸"""
        bbox = self._calculate_bbox(vertices)
        h = bbox['max'][2] - bbox['min'][2]
        w = bbox['max'][0] - bbox['min'][0]
        d = bbox['max'][1] - bbox['min'][1]

        print(f"[尺寸检测] X={w:.1f}mm, Y={d:.1f}mm, Z={h:.1f}mm")

        if h < 150 or h > 300:
            scale = 220.0 / h
            vertices = vertices * scale
            print(f"[尺寸校正] 缩放因子: {scale:.3f}")
            new_h = vertices[:, 2].max() - vertices[:, 2].min()
            print(f"[尺寸验证] 校正后: Z={new_h:.1f}mm")
        else:
            print(f"[尺寸验证] 正常 (头高/针长 = {h/162:.2f})")

        return vertices
