"""IMU 针轴几何与竖直标定持久化。"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from core.project_paths import CONFIG_DIR

CONFIG_PATH = CONFIG_DIR / "imu_geometry.json"


def default_config() -> Dict[str, Any]:
    return {
        "version": 3,
        "needle_body_angle_deg": 121.0,
        "needle_body_bias_deg": 7.7,
        "scene_z_ccw_deg": 135.0,
        "needle_angle_clockwise_from_x": True,
        "needle_length_mm": 200.0,
        "display_offset": {
            "enabled": False,
            "rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "smoothing": {
            "enabled": False,
            "alpha": 0.25,
        },
        "notes": (
            "needle_body_angle_deg：针轴与 IMU +X 夹角（度），顺时针 121°。"
            "display_offset：竖直持握针体时点「竖直校准」写入 R_offset。"
        ),
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _migrate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """旧版 enu_to_scene / vertical_reference → v3 display_offset（无法可靠转换则清零）。"""
    version = int(data.get("version", 1))
    if version >= 3:
        return data

    base = default_config()
    base["needle_body_angle_deg"] = float(data.get("needle_body_angle_deg", 121.0))
    base["needle_angle_clockwise_from_x"] = bool(
        data.get("needle_angle_clockwise_from_x", True)
    )
    base["needle_length_mm"] = float(data.get("needle_length_mm", 200.0))
    smooth = data.get("smoothing", {})
    base["smoothing"] = {
        "enabled": bool(smooth.get("enabled", False)),
        "alpha": float(smooth.get("alpha", 0.25)),
    }

    # 若旧版已有 3×3 矩阵标定，尝试沿用
    for key in ("display_offset", "vertical_reference", "world_to_scene"):
        block = data.get(key)
        if isinstance(block, dict) and block.get("rotation"):
            base["display_offset"] = {
                "enabled": bool(block.get("enabled", False)),
                "rotation": block["rotation"],
            }
            break

    return base


def load_config(path: Path | None = None) -> Dict[str, Any]:
    path = Path(path or CONFIG_PATH)
    base = default_config()
    if not path.is_file():
        save_config(base, path)
        return deepcopy(base)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return deepcopy(base)
    if not isinstance(data, dict):
        return deepcopy(base)
    data = _migrate_config(data)
    merged = _deep_merge(base, data)
    merged["version"] = 3
    # 丢弃旧字段
    for stale in ("enu_to_scene", "world_to_scene", "vertical_reference"):
        merged.pop(stale, None)
    if merged.get("needle_body_angle_deg", 121.0) < 90:
        merged["needle_body_angle_deg"] = 121.0
    return merged


def save_config(cfg: Dict[str, Any], path: Path | None = None) -> None:
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = deepcopy(cfg)
    out["version"] = 3
    for stale in ("enu_to_scene", "world_to_scene", "vertical_reference"):
        out.pop(stale, None)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_kinematics(cfg: Dict[str, Any]) -> None:
    from core.imu_kinematics import (
        apply_display_offset,
        set_needle_angle_clockwise_from_x,
        set_needle_body_angle_deg,
        set_needle_body_bias_deg,
        set_scene_z_ccw_deg,
    )

    set_needle_body_angle_deg(float(cfg.get("needle_body_angle_deg", 121.0)))
    set_needle_body_bias_deg(float(cfg.get("needle_body_bias_deg", 0.0)))
    set_scene_z_ccw_deg(float(cfg.get("scene_z_ccw_deg", 0.0)))
    set_needle_angle_clockwise_from_x(bool(cfg.get("needle_angle_clockwise_from_x", True)))
    apply_display_offset(cfg.get("display_offset"))
