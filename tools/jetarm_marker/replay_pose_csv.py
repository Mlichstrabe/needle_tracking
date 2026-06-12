#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回放 JetArm 针姿态 CSV，在 needle_tracking 的 3D 视图里显示一根运动的针。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.jetarm_marker.needle_gl_view import (  # noqa: E402
    PoseReplayWindow,
    configure_console_encoding,
    load_pose_frames,
)

configure_console_encoding()

from PyQt5.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="回放 JetArm marker pose CSV，在 3D 窗口中显示针。")
    parser.add_argument("pose_csv", type=Path, help="pose_from_ir_depth.py 输出的 CSV（旧 RGB 路径见 legacy/）")
    parser.add_argument("--fps", type=float, default=15.0, help="播放帧率")
    parser.add_argument("--needle-length-mm", type=float, default=162.0, help="显示用针长")
    args = parser.parse_args()

    if not args.pose_csv.is_file():
        print(f"找不到 CSV：{args.pose_csv}", file=sys.stderr)
        return 1

    frames = load_pose_frames(args.pose_csv)
    if not frames:
        print("CSV 中没有可回放的 valid=1 针姿态帧。", file=sys.stderr)
        return 1

    print(f"已读取可回放帧：{len(frames)}")
    print(f"输入 CSV：{args.pose_csv.resolve()}")
    print("提示：窗口内只播放 valid=1 的帧，已剔除 m1 疑似误识别帧。")

    app = QApplication(sys.argv)
    win = PoseReplayWindow(frames, fps=args.fps, needle_length_mm=args.needle_length_mm)
    win.resize(1200, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
