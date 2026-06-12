#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1：从 bag 导出 RGB / IR / depth 关键帧，生成对比拼图与帧索引表。

用法:
  python tools/jetarm_marker/legacy/export_modality_compare.py \\
      data/jetarm_marker/bags/marker_static_clean_01 \\
      --samples 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.camera_math import depth_encoding_hint  # noqa: E402
from tools.jetarm_marker.rosbag_io import (  # noqa: E402
    TOPIC_DEPTH,
    TOPIC_IR,
    TOPIC_RGB,
    TOPIC_RGB_INFO,
    TOPIC_DEPTH_INFO,
    camera_info_to_dict,
    collect_timestamps,
    decode_image_msg,
    pick_frame_indices,
    read_message_at_index,
)

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("需要 Pillow: pip install -r requirements-jetarm-marker.txt") from exc


def _save_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def _save_depth_vis(path: Path, depth: np.ndarray, encoding: str) -> None:
    """深度伪彩色，便于肉眼检查 marker 区域空洞。"""
    d = depth.astype(np.float64)
    if encoding == "32FC1":
        d = d * 1000.0
    valid = d[(d > 50) & (d < 2000)]
    if valid.size == 0:
        vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [5, 95])
        norm = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
        g = (norm * 255).astype(np.uint8)
        vis = np.stack([g, g, g], axis=-1)
        vis[depth <= 0] = (255, 0, 255)  # 无效深度标红
    Image.fromarray(vis).save(path)


def _ir_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.dtype == np.uint16:
        hi = max(int(np.percentile(arr, 99)), 1)
        g = np.clip(arr.astype(np.float32) / hi * 255, 0, 255).astype(np.uint8)
    else:
        g = arr.astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _make_triptych(rgb: Optional[np.ndarray], ir: Optional[np.ndarray], depth_vis_path: Path) -> np.ndarray:
    panels: List[np.ndarray] = []
    h_target = 480

    def resize_panel(img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img)
        w, h = pil.size
        scale = h_target / max(h, 1)
        new_w = max(int(w * scale), 1)
        return np.array(pil.resize((new_w, h_target), Image.BILINEAR))

    if rgb is not None:
        panels.append(resize_panel(rgb))
    else:
        panels.append(np.zeros((h_target, 320, 3), dtype=np.uint8))

    if ir is not None:
        panels.append(resize_panel(_ir_to_rgb(ir)))
    else:
        panels.append(np.zeros((h_target, 320, 3), dtype=np.uint8))

    depth_rgb = np.array(Image.open(depth_vis_path).convert("RGB"))
    panels.append(resize_panel(depth_rgb))

    return np.concatenate(panels, axis=1)


def _nearest_index(timestamps: List[int], t: int) -> int:
    if not timestamps:
        return 0
    arr = np.asarray(timestamps, dtype=np.int64)
    return int(np.argmin(np.abs(arr - t)))


def export_bag(
    bag_dir: Path,
    out_dir: Path,
    n_samples: int,
) -> Dict[str, Any]:
    bag_dir = Path(bag_dir)
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    rgb_ts = collect_timestamps(bag_dir, TOPIC_RGB)
    if not rgb_ts:
        raise RuntimeError(f"bag 无 {TOPIC_RGB} 消息")

    indices = pick_frame_indices(len(rgb_ts), n_samples)
    meta: Dict[str, Any] = {
        "bag": str(bag_dir.resolve()),
        "rgb_frame_count": len(rgb_ts),
        "sample_indices": indices,
        "camera_info": {},
        "frames": [],
    }

    try:
        info_rgb = read_message_at_index(bag_dir, TOPIC_RGB_INFO, 0)
        meta["camera_info"]["rgb"] = camera_info_to_dict(info_rgb)
    except (KeyError, IndexError):
        pass
    try:
        info_depth = read_message_at_index(bag_dir, TOPIC_DEPTH_INFO, 0)
        meta["camera_info"]["depth"] = camera_info_to_dict(info_depth)
    except (KeyError, IndexError):
        pass

    try:
        ir_ts = collect_timestamps(bag_dir, TOPIC_IR)
    except KeyError:
        ir_ts = []
    try:
        depth_ts = collect_timestamps(bag_dir, TOPIC_DEPTH)
    except KeyError:
        depth_ts = []

    for rank, idx in enumerate(indices):
        t_rgb = rgb_ts[idx]
        rgb_msg = read_message_at_index(bag_dir, TOPIC_RGB, idx)
        rgb_frame = decode_image_msg(rgb_msg)

        ir_frame = None
        if ir_ts:
            ir_idx = _nearest_index(ir_ts, t_rgb)
            ir_msg = read_message_at_index(bag_dir, TOPIC_IR, ir_idx)
            ir_frame = decode_image_msg(ir_msg)

        depth_frame = None
        depth_idx = 0
        if depth_ts:
            depth_idx = _nearest_index(depth_ts, t_rgb)
            depth_msg = read_message_at_index(bag_dir, TOPIC_DEPTH, depth_idx)
            depth_frame = decode_image_msg(depth_msg)

        prefix = out_frames / f"frame_{rank:03d}_idx{idx:04d}"
        rgb_path = prefix.with_name(prefix.name + "_rgb.png")
        _save_uint8(rgb_path, rgb_frame.array)

        ir_path = None
        if ir_frame is not None:
            ir_path = prefix.with_name(prefix.name + "_ir.png")
            if ir_frame.array.ndim == 2:
                _save_uint8(ir_path, ir_frame.array.astype(np.uint8))
            else:
                _save_uint8(ir_path, ir_frame.array)

        depth_vis_path = prefix.with_name(prefix.name + "_depth_vis.png")
        depth_raw_path = prefix.with_name(prefix.name + "_depth_meta.json")
        if depth_frame is not None:
            _save_depth_vis(depth_vis_path, depth_frame.array, depth_frame.encoding)
            depth_raw_path.write_text(
                json.dumps(
                    {
                        "encoding": depth_frame.encoding,
                        **depth_encoding_hint(depth_frame),
                        "shape": list(depth_frame.array.shape),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        tri_path = prefix.with_name(prefix.name + "_compare.png")
        if depth_frame is not None:
            tri = _make_triptych(rgb_frame.array, ir_frame.array if ir_frame else None, depth_vis_path)
            Image.fromarray(tri).save(tri_path)

        meta["frames"].append(
            {
                "rank": rank,
                "rgb_index": idx,
                "rgb_timestamp_ns": t_rgb,
                "ir_index": depth_idx if ir_frame is None else _nearest_index(ir_ts, t_rgb) if ir_ts else None,
                "depth_index": depth_idx if depth_frame else None,
                "paths": {
                    "rgb": str(rgb_path.relative_to(out_dir)),
                    "ir": str(ir_path.relative_to(out_dir)) if ir_path else None,
                    "depth_vis": str(depth_vis_path.relative_to(out_dir)) if depth_frame else None,
                    "compare": str(tri_path.relative_to(out_dir)) if depth_frame else None,
                },
            }
        )

    (out_dir / "frame_index.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 RGB/IR/depth 关键帧对比")
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="均匀采样帧数（默认 12）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出目录（默认 data/jetarm_marker/exports/<bag名>/）",
    )
    args = parser.parse_args()

    if not args.bag_dir.is_dir():
        print(f"错误: 找不到 bag: {args.bag_dir}", file=sys.stderr)
        return 1

    out_dir = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "exports" / args.bag_dir.name
    )

    try:
        meta = export_bag(args.bag_dir, out_dir, args.samples)
    except Exception as exc:
        print(f"导出失败: {exc}", file=sys.stderr)
        return 1

    print(f"RGB 总帧数: {meta['rgb_frame_count']}")
    print(f"已采样: {len(meta['frames'])} 帧")
    print(f"输出目录: {out_dir}")
    print("请打开 frames/*_compare.png 比较 marker 在 RGB / IR / depth 下的可见性。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
