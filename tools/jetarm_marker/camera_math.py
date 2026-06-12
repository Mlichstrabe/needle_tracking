"""深度反投影与刚体拟合（阶段 4–6 共用）。"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .rosbag_io import FrameRecord


def depth_median_window(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    half_win: int = 3,
    z_min_mm: float = 50.0,
    z_max_mm: float = 2000.0,
    encoding: str = "16UC1",
) -> Tuple[Optional[float], int]:
    """
    在 (u,v) 邻域取有效深度中值。
    返回 (depth_mm 或 None, 有效像素数)。
    """
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    u0 = max(0, ui - half_win)
    u1 = min(w, ui + half_win + 1)
    v0 = max(0, vi - half_win)
    v1 = min(h, vi + half_win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)

    if encoding == "32FC1":
        valid = patch[(patch > z_min_mm / 1000.0) & (patch < z_max_mm / 1000.0)]
        if valid.size == 0:
            return None, 0
        return float(np.median(valid) * 1000.0), int(valid.size)

    valid = patch[(patch > z_min_mm) & (patch < z_max_mm) & np.isfinite(patch)]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def backproject_mm(
    u: float,
    v: float,
    z_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    x = (u - cx) * z_mm / fx
    y = (v - cy) * z_mm / fy
    return float(x), float(y), float(z_mm)


def rgb_uv_to_depth_uv(
    u_rgb: float,
    v_rgb: float,
    rgb_wh: Tuple[int, int],
    depth_wh: Tuple[int, int],
) -> Tuple[float, float]:
    """V1 简化：按分辨率比例映射 RGB 像素到 depth 像素。"""
    rw, rh = rgb_wh
    dw, dh = depth_wh
    return u_rgb * dw / rw, v_rgb * dh / rh


def kabsch_rotation(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    src, dst: (N,3) 对应点。
    返回 R (3,3), t (3,) 使 dst ≈ R @ src + t。
    """
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Kabsch 至少需要 3 对 3D 点")
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    s = src - src_c
    d = dst - dst_c
    h = s.T @ d
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_c - r @ src_c
    return r, t


def depth_encoding_hint(frame: FrameRecord) -> Dict[str, str]:
    enc = frame.encoding
    if enc == "16UC1":
        return {"unit": "mm", "note": "Orbbec 常见 16UC1 毫米"}
    if enc == "32FC1":
        return {"unit": "m", "note": "浮点米，反投影前乘 1000"}
    return {"unit": "unknown", "note": f"encoding={enc}"}
