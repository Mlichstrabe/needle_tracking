#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1 核心：对比 JetArm Gemini 的 RGB / IR / depth 在 marker 区域的可用性。

对每帧自动找高亮候选点（反光球），在 RGB、IR（若有）、depth 上采样并统计：
  - 2D 可见性（亮度、对比度）
  - depth 有效率、中值、邻域标准差、空洞比例

输出:
  data/jetarm_marker/exports/<bag>/modality_report.json
  data/jetarm_marker/exports/<bag>/modality_frames/*.png

用法:
  python tools/jetarm_marker/compare_modality_report.py \\
      data/jetarm_marker/bags/marker_static_clean_01 --samples 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.camera_math import depth_median_window, rgb_uv_to_depth_uv  # noqa: E402
from tools.jetarm_marker.rosbag_io import (  # noqa: E402
    TOPIC_DEPTH,
    TOPIC_DEPTH_INFO,
    TOPIC_IR,
    TOPIC_RGB,
    TOPIC_RGB_INFO,
    camera_info_to_dict,
    collect_timestamps,
    decode_image_msg,
    list_bag_topics,
    pick_frame_indices,
    read_message_at_index,
)

IR_RECORDING_HINT = (
    "当前 bag 无 /depth_cam/ir/image_raw。"
    "JetArm 上 Orbbec 驱动默认 enable_ir=false（peripherals/launch/include/astra.launch.py）。"
    "请用 tools/jetarm_marker/record_modality_bag.sh 重新录制含 IR 的 bag 后再对比。"
)


def _gray_from_frame(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype == np.uint16:
        hi = max(int(np.percentile(arr, 99)), 1)
        return np.clip(arr.astype(np.float32) / hi * 255, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _find_bright_candidates(gray: np.ndarray, n: int = 6) -> List[Tuple[float, float, float]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = int(np.percentile(blur, 96))
    _, mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 6 or area > 12000:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        out.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]), float(area)))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:n]


def _sample_brightness(gray: np.ndarray, u: float, v: float, r: int = 5) -> float:
    h, w = gray.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    patch = gray[max(0, vi - r):min(h, vi + r + 1), max(0, ui - r):min(w, ui + r + 1)]
    return float(np.median(patch)) if patch.size else 0.0


def _depth_stats(
    depth: np.ndarray,
    u_rgb: float,
    v_rgb: float,
    rgb_wh: Tuple[int, int],
    depth_wh: Tuple[int, int],
    encoding: str,
    half_win: int = 4,
) -> Dict[str, Any]:
    du, dv = rgb_uv_to_depth_uv(u_rgb, v_rgb, rgb_wh, depth_wh)
    h, w = depth.shape[:2]
    ui, vi = int(round(du)), int(round(dv))
    u0, u1 = max(0, ui - half_win), min(w, ui + half_win + 1)
    v0, v1 = max(0, vi - half_win), min(h, vi + half_win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    if encoding == "32FC1":
        valid_mask = (patch > 0.05) & (patch < 2.0)
        valid_mm = patch[valid_mask] * 1000.0
    else:
        valid_mask = (patch > 50) & (patch < 2000)
        valid_mm = patch[valid_mask]
    total = patch.size
    valid_n = int(valid_mm.size)
    z_med, _ = depth_median_window(depth, du, dv, half_win=half_win, encoding=encoding)
    return {
        "depth_u": du,
        "depth_v": dv,
        "valid_ratio": valid_n / total if total else 0.0,
        "valid_count": valid_n,
        "depth_mm_median": z_med,
        "depth_mm_std": float(np.std(valid_mm)) if valid_n > 1 else None,
        "hole": valid_n < max(3, total // 4),
    }


def _annotate_panel(
    rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    title: str,
) -> np.ndarray:
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(candidates[:4]):
        u, v = c["u_rgb"], c["v_rgb"]
        color = (0, 255, 0) if c.get("depth_ok") else (0, 0, 255)
        cv2.circle(vis, (int(u), int(v)), 7, color, 2)
        cv2.putText(vis, f"{i}", (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def analyze_bag(bag_dir: Path, out_dir: Path, n_samples: int) -> Dict[str, Any]:
    bag_dir = Path(bag_dir)
    topics = {t.topic for t in list_bag_topics(bag_dir)}
    has_ir = TOPIC_IR in topics

    rgb_info = camera_info_to_dict(read_message_at_index(bag_dir, TOPIC_RGB_INFO, 0))
    depth_info = camera_info_to_dict(read_message_at_index(bag_dir, TOPIC_DEPTH_INFO, 0))
    rgb_wh = (rgb_info["width"], rgb_info["height"])
    depth_wh = (depth_info["width"], depth_info["height"])

    rgb_ts = collect_timestamps(bag_dir, TOPIC_RGB)
    ir_ts = collect_timestamps(bag_dir, TOPIC_IR) if has_ir else []
    depth_ts = collect_timestamps(bag_dir, TOPIC_DEPTH)

    indices = pick_frame_indices(len(rgb_ts), n_samples)
    frames_out = out_dir / "modality_frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    per_frame: List[Dict[str, Any]] = []
    agg_depth_valid: List[float] = []
    agg_rgb_bright: List[float] = []
    agg_ir_bright: List[float] = []
    ir_wins = 0
    rgb_wins = 0

    for rank, idx in enumerate(indices):
        rgb_msg = read_message_at_index(bag_dir, TOPIC_RGB, idx)
        rgb_frame = decode_image_msg(rgb_msg)
        rgb_gray = _gray_from_frame(rgb_frame.array)

        ir_gray: Optional[np.ndarray] = None
        if has_ir and ir_ts:
            ir_idx = int(np.argmin(np.abs(np.asarray(ir_ts) - rgb_ts[idx])))
            ir_msg = read_message_at_index(bag_dir, TOPIC_IR, ir_idx)
            ir_gray = _gray_from_frame(decode_image_msg(ir_msg).array)

        depth_idx = int(np.argmin(np.abs(np.asarray(depth_ts) - rgb_ts[idx])))
        depth_msg = read_message_at_index(bag_dir, TOPIC_DEPTH, depth_idx)
        depth_frame = decode_image_msg(depth_msg)

        cands_raw = _find_bright_candidates(rgb_gray, n=6)
        candidates: List[Dict[str, Any]] = []
        for u, v, area in cands_raw[:4]:
            rgb_b = _sample_brightness(rgb_gray, u, v)
            ir_b = _sample_brightness(ir_gray, u, v) if ir_gray is not None else None
            d_st = _depth_stats(
                depth_frame.array, u, v, rgb_wh, depth_wh, depth_frame.encoding,
            )
            depth_ok = (
                d_st["depth_mm_median"] is not None
                and d_st["valid_ratio"] >= 0.35
                and not d_st["hole"]
            )
            candidates.append({
                "u_rgb": u, "v_rgb": v, "area": area,
                "rgb_brightness": rgb_b,
                "ir_brightness": ir_b,
                **d_st,
                "depth_ok": depth_ok,
            })
            agg_rgb_bright.append(rgb_b)
            if ir_b is not None:
                agg_ir_bright.append(ir_b)
            agg_depth_valid.append(d_st["valid_ratio"])
            if ir_b is not None and ir_b > rgb_b * 1.1:
                ir_wins += 1
            else:
                rgb_wins += 1

        n_ok = sum(1 for c in candidates if c["depth_ok"])
        panel_rgb = _annotate_panel(
            rgb_frame.array,
            candidates,
            f"RGB idx={idx} markers_ok_depth={n_ok}/4",
        )
        panels = [panel_rgb]

        if ir_gray is not None:
            ir_vis = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            for i, c in enumerate(candidates[:4]):
                cv2.circle(ir_vis, (int(c["u_rgb"]), int(c["v_rgb"])), 7, (0, 255, 255), 2)
            cv2.putText(ir_vis, "IR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(ir_vis)

        d = depth_frame.array.astype(np.float64)
        if depth_frame.encoding == "32FC1":
            d = d * 1000.0
        valid = d[(d > 50) & (d < 2000)]
        if valid.size:
            lo, hi = np.percentile(valid, [5, 95])
            norm = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
            d_vis = (norm * 255).astype(np.uint8)
        else:
            d_vis = np.zeros_like(d, dtype=np.uint8)
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
        d_vis[d <= 0] = (255, 0, 255)
        for c in candidates[:4]:
            du, dv = int(c["depth_u"]), int(c["depth_v"])
            col = (0, 255, 0) if c["depth_ok"] else (0, 0, 255)
            cv2.circle(d_vis, (du, dv), 5, col, 2)
        cv2.putText(d_vis, "DEPTH", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        target_h = panel_rgb.shape[0]
        if d_vis.shape[0] != target_h:
            scale = target_h / d_vis.shape[0]
            d_vis = cv2.resize(
                d_vis,
                (int(d_vis.shape[1] * scale), target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        panels.append(d_vis)

        combo = np.concatenate(panels, axis=1)
        out_png = frames_out / f"modality_{rank:03d}_idx{idx:04d}.png"
        cv2.imwrite(str(out_png), combo)

        per_frame.append({
            "rank": rank,
            "rgb_index": idx,
            "marker_candidates": len(candidates),
            "depth_ok_count": n_ok,
            "image": str(out_png.relative_to(out_dir)),
            "candidates": candidates,
        })

    mean_depth_valid = float(np.mean(agg_depth_valid)) if agg_depth_valid else 0.0
    recommendation: Dict[str, Any] = {
        "tracking_2d": "pending_ir_comparison",
        "depth_usable": mean_depth_valid >= 0.3,
        "notes": [],
    }
    if not has_ir:
        recommendation["tracking_2d"] = "rgb_provisional"
        recommendation["notes"].append(IR_RECORDING_HINT)
    elif agg_ir_bright:
        recommendation["tracking_2d"] = "ir" if ir_wins > rgb_wins else "rgb"
        recommendation["notes"].append(
            f"IR 更亮候选点次数 {ir_wins} vs RGB {rgb_wins}（启发式，需目视 modality_frames 确认）"
        )
    else:
        recommendation["tracking_2d"] = "rgb"
    if mean_depth_valid < 0.25:
        recommendation["notes"].append("depth 在 marker 区域有效率偏低，3D 姿态可能不稳定")

    report = {
        "bag": str(bag_dir.resolve()),
        "modalities_present": {
            "rgb": TOPIC_RGB in topics,
            "ir": has_ir,
            "depth": TOPIC_DEPTH in topics,
        },
        "samples": len(indices),
        "rgb_resolution": list(rgb_wh),
        "depth_resolution": list(depth_wh),
        "summary": {
            "mean_depth_valid_ratio_at_markers": mean_depth_valid,
            "mean_rgb_brightness": float(np.mean(agg_rgb_bright)) if agg_rgb_bright else None,
            "mean_ir_brightness": float(np.mean(agg_ir_bright)) if agg_ir_bright else None,
            "ir_brighter_count": ir_wins if has_ir else None,
            "rgb_brighter_count": rgb_wins if has_ir else None,
        },
        "recommendation": recommendation,
        "frames": per_frame,
    }
    (out_dir / "modality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="RGB/IR/depth 模态对比报告（阶段 1）")
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.bag_dir.is_dir():
        print(f"找不到 bag: {args.bag_dir}", file=sys.stderr)
        return 1

    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "exports" / args.bag_dir.name
    )
    report = analyze_bag(args.bag_dir, out, args.samples)

    print("=== 模态对比报告 ===")
    print(json.dumps(report["modalities_present"], ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print("推荐:", json.dumps(report["recommendation"], ensure_ascii=False, indent=2))
    print(f"详细报告: {out / 'modality_report.json'}")
    print(f"对比图: {out / 'modality_frames'}")
    if not report["modalities_present"]["ir"]:
        print("\n[!]", IR_RECORDING_HINT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
