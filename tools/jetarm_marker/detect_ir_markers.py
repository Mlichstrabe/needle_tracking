#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测 JetArm IR 图像里的 4 个针上反光 marker，并导出 CSV/预览视频/检查图。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.ir_marker_detect import (  # noqa: E402
    DetectParams,
    MarkerTracker,
    draw_live_overlay,
    ir_array_to_u8,
)
from tools.jetarm_marker.rosbag_io import TOPIC_IR, load_image_frames, pick_frame_indices  # noqa: E402


def _write_contact_sheet(path: Path, overlays: List[Tuple[str, object]], sample_count: int) -> None:
    if not overlays:
        return
    indices = pick_frame_indices(len(overlays), min(sample_count, len(overlays)))
    cells: List[Image.Image] = []
    for idx in indices:
        label, bgr = overlays[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((420, 280))
        cell = Image.new("RGB", (440, 330), "white")
        cell.paste(image, (10, 35))
        draw = ImageDraw.Draw(cell)
        draw.text((10, 10), label, fill=(0, 0, 0))
        cells.append(cell)
    cols = 2
    rows = int(math.ceil(len(cells) / cols))
    sheet = Image.new("RGB", (cols * 440, rows * 330), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * 440, (i // cols) * 330))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def detect_ir_markers(
    bag_dir: Path,
    *,
    start_index: int,
    end_index: Optional[int],
    output_csv: Path,
    preview_video: Optional[Path],
    contact_sheet: Optional[Path],
    threshold_percentile: float,
    min_area: float,
    max_area: float,
    min_circularity: float,
    edge_margin: int,
    max_match_px: float,
    enforce_match_gate: bool,
) -> dict:
    frames = load_image_frames(bag_dir, TOPIC_IR, start_index=start_index, end_index=end_index)
    if not frames:
        raise RuntimeError(f"no IR frames found in {bag_dir}")

    params = DetectParams(
        threshold_percentile=threshold_percentile,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
        edge_margin=edge_margin,
        max_match_px=max_match_px,
    )
    tracker = MarkerTracker(params=params)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    video = None
    overlays: List[Tuple[str, object]] = []
    valid_count = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        header = [
            "frame_id",
            "ir_index",
            "timestamp_ns",
            "frame_valid",
            "candidate_count",
            "rom_rms_mm",
            "axis_length_ratio_2d",
        ]
        for i in range(4):
            header += [f"m{i}_u", f"m{i}_v", f"m{i}_valid"]
        writer = csv.writer(f)
        writer.writerow(header)

        for local_i, frame in enumerate(frames):
            ir_index = start_index + local_i
            gray = ir_array_to_u8(frame.array)
            result = tracker.process(gray, enforce_match_gate=enforce_match_gate)
            valid = result.frame_valid
            if valid:
                valid_count += 1

            row: List[object] = [
                local_i,
                ir_index,
                int(frame.timestamp_ns),
                int(valid),
                result.candidate_count,
                "" if result.rom_rms_mm is None else f"{result.rom_rms_mm:.3f}",
                "" if result.axis_length_ratio_2d is None else f"{result.axis_length_ratio_2d:.4f}",
            ]
            for marker_i in range(4):
                if result.selected is None:
                    row += ["", "", 0]
                else:
                    row += [
                        f"{result.selected[marker_i, 0]:.3f}",
                        f"{result.selected[marker_i, 1]:.3f}",
                        int(result.track_valid),
                    ]
            writer.writerow(row)

            label = (
                f"ir_index={ir_index} valid={int(valid)} blobs={result.candidate_count} "
                f"ratio={result.axis_length_ratio_2d}"
            )
            overlay = draw_live_overlay(result)
            overlays.append((label, overlay))
            if preview_video is not None:
                if video is None:
                    preview_video.parent.mkdir(parents=True, exist_ok=True)
                    h, w = overlay.shape[:2]
                    video = cv2.VideoWriter(str(preview_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
                video.write(overlay)

    if video is not None:
        video.release()
    if contact_sheet is not None:
        _write_contact_sheet(contact_sheet, overlays, sample_count=12)

    summary = {
        "bag": str(Path(bag_dir).resolve()),
        "topic": TOPIC_IR,
        "frames": len(frames),
        "valid_frames": valid_count,
        "valid_ratio": valid_count / max(len(frames), 1),
        "output_csv": str(output_csv.resolve()),
        "preview_video": str(preview_video.resolve()) if preview_video else None,
        "contact_sheet": str(contact_sheet.resolve()) if contact_sheet else None,
        "threshold_percentile": threshold_percentile,
        "min_area": min_area,
        "max_area": max_area,
        "min_circularity": min_circularity,
        "edge_margin": edge_margin,
        "max_match_px": max_match_px,
        "enforce_match_gate": enforce_match_gate,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _print_chinese_summary(summary: dict) -> None:
    print("IR 反光 marker 检测完成")
    print(f"输入 bag：{summary['bag']}")
    print(f"使用话题：{summary['topic']}")
    print(f"总帧数：{summary['frames']}")
    print(f"有效检测帧：{summary['valid_frames']}")
    print(f"有效率：{summary['valid_ratio'] * 100:.1f}%")
    print(f"CSV 输出：{summary['output_csv']}")
    print(f"预览视频：{summary['preview_video']}")
    print(f"检查图：{summary['contact_sheet']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="检测 JetArm IR bag 中的 4 个反光 marker。")
    parser.add_argument("bag_dir", type=Path, help="rosbag 目录，不是单个 db3 文件")
    parser.add_argument("--start-index", type=int, default=0, help="起始 IR 帧序号")
    parser.add_argument("--end-index", type=int, default=None, help="结束 IR 帧序号，包含该帧")
    parser.add_argument("--threshold-percentile", type=float, default=98.5, help="亮点阈值百分位")
    parser.add_argument("--min-area", type=float, default=12.0, help="亮点最小面积")
    parser.add_argument("--max-area", type=float, default=1800.0, help="亮点最大面积")
    parser.add_argument("--min-circularity", type=float, default=0.15, help="亮点最小圆度")
    parser.add_argument("--edge-margin", type=int, default=14, help="忽略图像边缘多少像素")
    parser.add_argument("--max-match-px", type=float, default=70.0, help="相邻帧 marker 匹配最大位移")
    parser.add_argument(
        "--enforce-match-gate",
        action="store_true",
        help="当 marker 位移超过 --max-match-px 时标为无效；默认关闭，便于先做检测质量检查。",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument("--video", type=Path, default=None, help="输出预览 MP4 路径")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="输出检查图 JPG 路径")
    args = parser.parse_args()

    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "ir_detection" / f"{args.bag_dir.name}_ir_markers.csv"
    )
    video = args.video
    if video is None:
        video = out.with_suffix(".mp4")
    contact = args.contact_sheet
    if contact is None:
        contact = out.with_name(out.stem + "_contact.jpg")

    try:
        summary = detect_ir_markers(
            args.bag_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            output_csv=out,
            preview_video=video,
            contact_sheet=contact,
            threshold_percentile=args.threshold_percentile,
            min_area=args.min_area,
            max_area=args.max_area,
            min_circularity=args.min_circularity,
            edge_margin=args.edge_margin,
            max_match_px=args.max_match_px,
            enforce_match_gate=args.enforce_match_gate,
        )
    except Exception as exc:
        print(f"IR marker 检测失败：{exc}", file=sys.stderr)
        return 1
    _print_chinese_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
