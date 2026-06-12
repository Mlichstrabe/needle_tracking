#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3+ 一键流水线（需先完成阶段 1 模态对比）。

阶段 1 请先运行:
  python tools/jetarm_marker/compare_modality_report.py data/jetarm_marker/bags/<bag>

用法:
  python tools/jetarm_marker/run_pipeline.py --bag marker_static_clean_01 --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list) -> None:
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="marker_static_clean_01")
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--end", type=int, default=380)
    parser.add_argument("--frame-index", type=int, default=238)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过阶段 1 模态报告检查（不推荐）",
    )
    args = parser.parse_args()

    bag_dir = _REPO_ROOT / "data" / "jetarm_marker" / "bags" / args.bag
    if not bag_dir.is_dir():
        print(f"缺少 bag: {bag_dir}", file=sys.stderr)
        return 1

    modality_report = (
        _REPO_ROOT / "data" / "jetarm_marker" / "exports" / args.bag / "modality_report.json"
    )
    if not args.force and not modality_report.is_file():
        print(
            "请先完成阶段 1 模态对比:\n"
            f"  python tools/jetarm_marker/compare_modality_report.py {bag_dir}\n"
            "若 bag 无 IR，请在 JetArm 上运行 launch_depth_cam_with_ir.sh + record_modality_bag.sh 后重试。",
            file=sys.stderr,
        )
        return 2

    py = sys.executable
    _run([py, "tools/jetarm_marker/bag_probe.py", str(bag_dir)])
    _run([py, "tools/jetarm_marker/export_modality_compare.py", str(bag_dir), "--samples", "12"])

    init_json = _REPO_ROOT / "data" / "jetarm_marker" / "inits" / f"{args.bag}_init.json"
    preview = _REPO_ROOT / "data" / "jetarm_marker" / "inits" / f"{args.bag}_init_preview.png"
    _run([
        py, "tools/jetarm_marker/auto_init_markers.py",
        "--bag", str(bag_dir),
        "--frame-index", str(args.frame_index),
        "--bag-name", args.bag,
        "-o", str(init_json),
        "--save-preview", str(preview),
    ])

    track_csv = _REPO_ROOT / "data" / "jetarm_marker" / "tracking" / f"{args.bag}_track2d.csv"
    _run([
        py, "tools/jetarm_marker/track_markers.py",
        "--bag", str(bag_dir),
        "--init", str(init_json),
        "--start", str(args.start),
        "--end", str(args.end),
        "-o", str(track_csv),
    ])

    geom = _REPO_ROOT / "data" / "jetarm_marker" / "geometry" / "needle_geometry.json"
    scene = _REPO_ROOT / "data" / "jetarm_marker" / "geometry" / "scene_transform.json"
    pose_csv = _REPO_ROOT / "data" / "jetarm_marker" / "tracking" / f"{args.bag}_pose.csv"
    _run([
        py, "tools/jetarm_marker/pose_from_markers.py",
        "--bag", str(bag_dir),
        "--track", str(track_csv),
        "--geometry", str(geom),
        "--scene", str(scene),
        "-o", str(pose_csv),
    ])

    print(f"\n完成。Pose CSV: {pose_csv}")
    if not args.no_replay:
        print("启动回放: python tools/jetarm_marker/replay_pose_csv.py", pose_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
