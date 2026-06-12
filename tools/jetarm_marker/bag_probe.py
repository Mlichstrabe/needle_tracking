#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1：探测 ROS2 bag（topic、帧数、时长）。

用法:
  python tools/jetarm_marker/bag_probe.py data/jetarm_marker/bags/marker_static_clean_01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.rosbag_io import (  # noqa: E402
    DEFAULT_TOPICS,
    probe_bag,
    save_probe_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="探测 JetArm marker bag")
    parser.add_argument(
        "bag_dir",
        type=Path,
        help="ros2 bag 目录（含 metadata.yaml）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="probe_summary.json 输出路径（默认 data/jetarm_marker/exports/<bag名>/）",
    )
    args = parser.parse_args()

    bag_dir = args.bag_dir
    if not bag_dir.is_dir():
        print(f"错误: 找不到 bag 目录: {bag_dir}", file=sys.stderr)
        return 1

    result = probe_bag(bag_dir)
    out = args.output
    if out is None:
        out = (
            _REPO_ROOT
            / "data"
            / "jetarm_marker"
            / "exports"
            / bag_dir.name
            / "probe_summary.json"
        )

    save_probe_summary(result, out)

    print(f"Bag: {result.bag_path}")
    if result.duration_ns:
        print(f"时长: {result.duration_ns / 1e9:.2f} s")
    print("Topics:")
    topic_set = {t.topic for t in result.topics}
    for t in result.topics:
        mark = " *" if t.topic in DEFAULT_TOPICS else ""
        print(f"  {t.topic}: {t.count} msgs ({t.msgtype}){mark}")

    missing = [tp for tp in DEFAULT_TOPICS if tp not in topic_set]
    if missing:
        print("\n警告: 缺少 V1 期望 topic:")
        for tp in missing:
            print(f"  - {tp}")

    print(f"\n已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
