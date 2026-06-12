"""几何参数与坐标变换加载。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_needle_geometry(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    markers = np.asarray(data["markers_local_mm"], dtype=np.float64)
    tip = np.asarray(data["tip_offset_marker_mm"], dtype=np.float64).reshape(3)
    axis = np.asarray(data["axis_marker_unit"], dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return {
        "markers_local_mm": markers,
        "tip_offset_marker_mm": tip,
        "axis_marker_unit": axis,
        "needle_length_mm": float(data.get("needle_length_mm", 162.0)),
        "measured": bool(data.get("measured", False)),
        "raw": data,
    }


def load_scene_transform(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    data = load_json(path)
    r = np.asarray(data["rotation_row_major"], dtype=np.float64)
    t = np.asarray(data["translation_mm"], dtype=np.float64).reshape(3)
    scale = float(data.get("scale", 1.0))
    return r, t, scale


def camera_to_scene(
    points: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """points: (N,3) or (3,) in camera mm → scene mm."""
    p = np.asarray(points, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p.reshape(1, 3)
    out = (scale * (r @ p.T).T) + t
    return out[0] if single else out


def camera_vec_to_scene(vec: np.ndarray, r: np.ndarray, scale: float = 1.0) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    out = scale * (r @ v)
    n = np.linalg.norm(out)
    if n > 1e-12:
        out = out / n
    return out
