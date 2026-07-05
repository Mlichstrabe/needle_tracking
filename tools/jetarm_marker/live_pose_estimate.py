#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由实时 IR 检测 + depth 估计针尖/针轴（相机系 mm）。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.jetarm_marker.camera_math import backproject_mm, depth_median_window
from tools.jetarm_marker.ir_marker_detect import FrameDetectResult, axis_length_ratio_2d
from tools.jetarm_marker.anchor_geometry import load_anchor_geometry
from tools.jetarm_marker.pose_from_ir_depth import _estimate_axis_and_tip, _unit

_DEFAULT_DEPTH_INFO = (
    Path(__file__).resolve().parents[2] / "data" / "jetarm_marker" / "geometry" / "depth_camera_info.json"
)


@dataclass
class LiveNeedlePose:
    valid: bool
    tip: Optional[np.ndarray]
    axis: Optional[np.ndarray]
    tail: Optional[np.ndarray]
    confidence: float
    markers: List[Tuple[int, np.ndarray]]
    axis_length_ratio_2d: Optional[float]
    rom_rms_mm: Optional[float]


def load_depth_camera_info(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _DEFAULT_DEPTH_INFO
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "encoding": str(data.get("encoding", "16UC1")),
    }


def _markers_3d_from_uv(
    selected: np.ndarray,
    depth: np.ndarray,
    depth_info: Dict[str, Any],
    *,
    half_win: int,
    min_depth_pixels: int,
    z_min_mm: float,
    z_max_mm: float,
) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    encoding = depth_info.get("encoding", "16UC1")
    for i in range(4):
        u, v = float(selected[i, 0]), float(selected[i, 1])
        item: Dict[str, Any] = {
            "u": u,
            "v": v,
            "depth_mm": None,
            "depth_pixels": 0,
            "x": None,
            "y": None,
            "z": None,
            "valid": False,
        }
        z_mm, n_valid = depth_median_window(
            depth,
            u,
            v,
            half_win=half_win,
            z_min_mm=z_min_mm,
            z_max_mm=z_max_mm,
            encoding=encoding,
        )
        item["depth_mm"] = z_mm
        item["depth_pixels"] = n_valid
        if z_mm is not None and n_valid >= min_depth_pixels:
            x, y, z = backproject_mm(
                u,
                v,
                z_mm,
                depth_info["fx"],
                depth_info["fy"],
                depth_info["cx"],
                depth_info["cy"],
            )
            item.update({"x": x, "y": y, "z": z, "valid": True})
        markers.append(item)
    return markers


def anchor_marker_3d_from_uv(
    selected: np.ndarray,
    depth: np.ndarray,
    depth_info: Dict[str, Any],
    anchor_index: int,
    *,
    half_win: int,
    min_depth_pixels: int,
    z_min_mm: float = 50.0,
    z_max_mm: float = 2000.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """只对锚点 marker 取深度（hybrid 位移，不要求 4/4 深度）。"""
    meta: Dict[str, Any] = {
        "u": None,
        "v": None,
        "depth_mm": None,
        "depth_pixels": 0,
        "valid": False,
    }
    if selected is None or selected.shape[0] <= anchor_index:
        return None, meta
    u, v = float(selected[anchor_index, 0]), float(selected[anchor_index, 1])
    meta["u"], meta["v"] = u, v
    encoding = depth_info.get("encoding", "16UC1")
    z_mm, n_valid = depth_median_window(
        depth,
        u,
        v,
        half_win=half_win,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
        encoding=encoding,
    )
    meta["depth_mm"] = z_mm
    meta["depth_pixels"] = n_valid
    if z_mm is None or n_valid < min_depth_pixels:
        return None, meta
    x, y, z = backproject_mm(
        u,
        v,
        z_mm,
        depth_info["fx"],
        depth_info["fy"],
        depth_info["cx"],
        depth_info["cy"],
    )
    meta.update({"x": x, "y": y, "z": z, "valid": True})
    return np.array([x, y, z], dtype=float), meta


def estimate_live_needle_pose(
    detect: FrameDetectResult,
    depth: np.ndarray,
    depth_info: Dict[str, Any],
    *,
    axis_start_marker: int = 3,
    axis_end_marker: int = 1,
    tip_offset_mm: float = 141.0,
    needle_length_mm: float = 162.0,
    depth_half_window: int = 13,
    min_depth_pixels: int = 3,
    min_axis_length_ratio: float = 0.55,
    z_min_mm: float = 50.0,
    z_max_mm: float = 2000.0,
) -> LiveNeedlePose:
    empty = LiveNeedlePose(
        valid=False,
        tip=None,
        axis=None,
        tail=None,
        confidence=0.0,
        markers=[],
        axis_length_ratio_2d=detect.axis_length_ratio_2d,
        rom_rms_mm=detect.rom_rms_mm,
    )
    if detect.selected is None or depth is None:
        return empty

    markers = _markers_3d_from_uv(
        detect.selected,
        depth,
        depth_info,
        half_win=depth_half_window,
        min_depth_pixels=min_depth_pixels,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
    )
    ratio = axis_length_ratio_2d(detect.selected)
    pose = _estimate_axis_and_tip(
        markers,
        axis_start_marker=axis_start_marker,
        axis_end_marker=axis_end_marker,
        tip_offset_mm=tip_offset_mm,
        needle_length_mm=needle_length_mm,
    )
    ratio_ok = ratio is not None and ratio >= min_axis_length_ratio
    geom_ok = detect.geometry_valid
    axis_markers_ok = (
        axis_start_marker < len(markers)
        and axis_end_marker < len(markers)
        and markers[axis_start_marker]["valid"]
        and markers[axis_end_marker]["valid"]
    )
    valid = bool(pose["valid"]) and ratio_ok and geom_ok and axis_markers_ok

    marker_pts: List[Tuple[int, np.ndarray]] = []
    for i, m in enumerate(markers):
        if m["valid"]:
            marker_pts.append((i, np.array([m["x"], m["y"], m["z"]], dtype=np.float64)))

    conf = 0.25 * sum(1 for m in markers if m["valid"])
    if axis_markers_ok:
        conf = max(conf, 0.85)
    return LiveNeedlePose(
        valid=valid,
        tip=pose["tip"],
        axis=pose["axis"],
        tail=pose["tail"],
        confidence=conf,
        markers=marker_pts,
        axis_length_ratio_2d=ratio,
        rom_rms_mm=detect.rom_rms_mm,
    )
