"""JetArm 标记板几何与视觉跟踪参数（data/jetarm_marker/geometry/anchor_imu_needle.json）。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = _ROOT / "data" / "jetarm_marker" / "geometry" / "anchor_imu_needle.json"


def default_geometry() -> Dict[str, Any]:
    return {
        "anchor_marker": "m2",
        "m2_to_tip_mm": 140.0,
        "needle_length_mm": 162.0,
        "imu_to_tip_mm": None,
        "max_jump_px": 80.0,
        "max_hold_frames": 10,
        "depth_half_window": 13,
        "min_depth_pixels": 3,
        "z_min_mm": 50.0,
        "z_max_mm": 2000.0,
        "modes": {
            "observe": {"max_hold_frames": 10},
            "puncture": {"max_hold_frames": 3},
        },
    }


def load_geometry(path: Path | None = None) -> Dict[str, Any]:
    path = Path(path or DEFAULT_PATH)
    base = default_geometry()
    if not path.is_file():
        return deepcopy(base)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return deepcopy(base)
    if not isinstance(data, dict):
        return deepcopy(base)
    out = deepcopy(base)
    for key, value in data.items():
        if key == "modes" and isinstance(value, dict) and isinstance(out.get("modes"), dict):
            out["modes"] = {**out["modes"], **value}
        else:
            out[key] = value
    return out


def effective_imu_to_tip_mm(geometry: Dict[str, Any], imu_geometry_fallback: float) -> float:
    """IMU 中心 → 针尖：优先 JSON 的 imu_to_tip_mm，否则沿用 imu_geometry。"""
    raw = geometry.get("imu_to_tip_mm")
    if raw is not None:
        return float(raw)
    return float(imu_geometry_fallback)


def mode_tracking_params(geometry: Dict[str, Any], mode: str) -> Dict[str, float]:
    """按工作模式覆盖 hold 等参数。"""
    modes = geometry.get("modes") or {}
    block = modes.get(mode) or {}
    hold = int(block.get("max_hold_frames", geometry.get("max_hold_frames", 10)))
    return {
        "max_jump_px": float(geometry.get("max_jump_px", 80.0)),
        "max_hold_frames": float(hold),
        "depth_half_window": float(geometry.get("depth_half_window", 13)),
        "min_depth_pixels": float(geometry.get("min_depth_pixels", 3)),
        "z_min_mm": float(geometry.get("z_min_mm", 50.0)),
        "z_max_mm": float(geometry.get("z_max_mm", 2000.0)),
        "m2_to_tip_mm": float(geometry.get("m2_to_tip_mm", 140.0)),
        "anchor_marker": str(geometry.get("anchor_marker", "m2")),
    }