#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 data/jetarm_marker/geometry/anchor_imu_needle.json 读取针轴/针尖几何。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_DEFAULT = (
    Path(__file__).resolve().parents[2] / "data" / "jetarm_marker" / "geometry" / "anchor_imu_needle.json"
)


def load_anchor_geometry(path: Path | None = None) -> Dict[str, Any]:
    p = path or _DEFAULT
    defaults: Dict[str, Any] = {
        "axis_start_marker": 3,
        "axis_end_marker": 1,
        "tip_anchor_marker": 1,
        "tip_offset_mm": 141.0,
        "needle_length_mm": 162.0,
        "depth_half_window": 13,
        "min_depth_pixels": 3,
        "z_min_mm": 50.0,
        "z_max_mm": 2000.0,
        "notes": "针轴 m3→m1；针尖 = m1 球心 + 单位轴 × tip_offset_mm",
    }
    if not p.is_file():
        return defaults
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return defaults
    out = {**defaults, **data}
    # 兼容旧字段名
    if "m2_to_tip_mm" in data and "tip_offset_mm" not in data:
        out["tip_offset_mm"] = float(data["m2_to_tip_mm"])
    if out.get("axis_end_marker") == 1 and out.get("axis_start_marker") is None:
        out["axis_start_marker"] = 3
    return out