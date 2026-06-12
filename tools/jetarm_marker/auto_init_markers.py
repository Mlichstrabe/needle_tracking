#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2 辅助：在 RGB/IR 图上自动检测 4 个高亮反光球，生成 init JSON。

V1 主流程仍推荐 init_markers.py 手动点选；本脚本用于流水线/bootstrap。

用法:
  python tools/jetarm_marker/auto_init_markers.py \\
      --bag data/jetarm_marker/bags/marker_static_clean_01 \\
      --frame-index 238 \\
      --bag-name marker_static_clean_01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.rosbag_io import TOPIC_RGB, decode_image_msg, read_message_at_index  # noqa: E402


def _find_bright_blobs(gray: np.ndarray, n: int = 4) -> List[Tuple[float, float]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, int(np.percentile(blur, 97)), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8 or area > 8000:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        candidates.append((cx, cy, area))
    candidates.sort(key=lambda x: x[2], reverse=True)
    if len(candidates) < n:
        raise RuntimeError(f"仅检测到 {len(candidates)} 个高亮区域，需要 {n} 个")
    pts = [(c[0], c[1]) for c in candidates[:n]]
    # 编号：先按 y 分两行，每行按 x
    pts.sort(key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return top + bot


def main() -> int:
    parser = argparse.ArgumentParser(description="自动检测 4 marker 并写 init JSON")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=238)
    parser.add_argument("--bag-name", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--save-preview", type=Path, default=None)
    args = parser.parse_args()

    msg = read_message_at_index(args.bag, TOPIC_RGB, args.frame_index)
    frame = decode_image_msg(msg)
    img = frame.array
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)

    points = _find_bright_blobs(gray, 4)
    bag_name = args.bag_name or args.bag.name
    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "inits" / f"{bag_name}_init.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bag": bag_name,
        "frame_rgb_index": args.frame_index,
        "frame_ref": f"bag_index_{args.frame_index}",
        "modality": "rgb",
        "source": "auto_init_markers",
        "markers": [{"id": i, "u": u, "v": v} for i, (u, v) in enumerate(points)],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_preview:
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        for i, (u, v) in enumerate(points):
            cv2.circle(vis, (int(u), int(v)), 8, (0, 255, 0), 2)
            cv2.putText(vis, f"m{i}", (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_preview), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"检测到 4 点，已写入: {out}")
    for i, (u, v) in enumerate(points):
        print(f"  m{i}: ({u:.1f}, {v:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
