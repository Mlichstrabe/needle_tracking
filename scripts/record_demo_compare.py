#!/usr/bin/env python3
"""
三画面对比演示录制：IR | 深度伪彩 | 针姿示意（OpenCV 合成 MP4）。

不依赖主窗口。用于可行性演示。

用法:
  # 无相机：合成动画约 10s
  python scripts/record_demo_compare.py --demo -o demo_compare.mp4

  # 有 RealSense：录 IR + 深度 + CSV 驱动针姿示意
  python scripts/record_demo_compare.py -o live_compare.mp4 --seconds 15 \\
      --poses-csv data/jetarm_marker/sample_poses.csv

  # 仅相机两路 + 静态针姿
  python scripts/record_demo_compare.py -o live.mp4 --seconds 10 --no-poses

依赖: opencv-python, numpy；RealSense 需 pyrealsense2。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PANEL_W = 640
PANEL_H = 480
OUT_FPS = 30


def load_poses_csv(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tip = np.array(
                [float(row["tip_x"]), float(row["tip_y"]), float(row["tip_z"])],
                dtype=float,
            )
            axis = np.array(
                [float(row["axis_x"]), float(row["axis_y"]), float(row["axis_z"])],
                dtype=float,
            )
            n = np.linalg.norm(axis)
            if n > 1e-9:
                axis = axis / n
            rows.append((tip, axis))
    return rows


def _label_panel(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        out,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return out


def synth_ir(frame_idx: int, w: int, h: int) -> np.ndarray:
    """模拟 IR：渐变 + 轻微噪声 + 移动亮斑（演示用）。"""
    t = frame_idx / OUT_FPS
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 40 + 30 * np.sin(x * 0.02 + t * 2)
    blob = 80 * np.exp(-((x - w * (0.4 + 0.1 * math.sin(t))) ** 2) / 8000)
    blob += 60 * np.exp(-((y - h * 0.55) ** 2) / 6000)
    gray = np.clip(base + blob + np.random.randn(h, w) * 4, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def synth_depth(frame_idx: int, w: int, h: int) -> np.ndarray:
    """模拟深度伪彩。"""
    t = frame_idx / OUT_FPS
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 400 + 120 * np.sin(x * 0.015 + t) + 80 * np.cos(y * 0.02 - t * 0.7)
    z = np.clip(z, 300, 900)
    norm = ((z - 300) / 600 * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def render_needle_panel(
    w: int,
    h: int,
    tip: np.ndarray,
    axis: np.ndarray,
    needle_len: float = 162.0,
) -> np.ndarray:
    """简易 3D→2D 针体示意（俯仰 + 偏航投影）。"""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (8, 12, 18)

    cx, cy = w // 2, int(h * 0.55)
    scale = 0.35

    ax, ay, az = axis[0], axis[1], axis[2]
    yaw = math.atan2(ay, ax)
    pitch = math.atan2(-az, math.hypot(ax, ay) + 1e-9)

    def proj(p: np.ndarray) -> Tuple[int, int]:
        px = cx + int((p[0] * math.cos(yaw) - p[1] * math.sin(yaw)) * scale)
        py = cy + int(
            (-p[2] * math.cos(pitch) + p[0] * math.sin(yaw) * math.sin(pitch)) * scale
        )
        return px, py

    tail = tip - axis * needle_len
    p0 = proj(tail)
    p1 = proj(tip)
    cv2.line(canvas, p0, p1, (230, 230, 230), 4, cv2.LINE_AA)
    cv2.circle(canvas, p1, 10, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, p0, 8, (0, 255, 120), -1, cv2.LINE_AA)

    # 地面网格
    for i in range(-4, 5):
        x0 = cx + i * 40
        cv2.line(canvas, (x0, cy + 80), (x0 + 30, cy + 110), (40, 50, 60), 1)
    cv2.putText(
        canvas,
        "Scene (demo projection)",
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 130, 140),
        1,
        cv2.LINE_AA,
    )
    return canvas


def default_pose_for_frame(frame_idx: int, poses: Optional[List]) -> Tuple[np.ndarray, np.ndarray]:
    if poses:
        tip, axis = poses[frame_idx % len(poses)]
        return tip.copy(), axis.copy()
    t = frame_idx / OUT_FPS
    tip = np.array([20 * math.sin(t), 15 * math.cos(t * 0.8), 0.0])
    axis = np.array([0.1 * math.sin(t), 0.1 * math.cos(t), -1.0])
    axis /= np.linalg.norm(axis)
    return tip, axis


def resize_panel(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    if bgr.shape[1] == w and bgr.shape[0] == h:
        return bgr
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def compose_triptych(ir_bgr, depth_bgr, needle_bgr) -> np.ndarray:
    a = _label_panel(resize_panel(ir_bgr, PANEL_W, PANEL_H), "IR")
    b = _label_panel(resize_panel(depth_bgr, PANEL_W, PANEL_H), "Depth")
    c = _label_panel(resize_panel(needle_bgr, PANEL_W, PANEL_H), "3D Needle (demo)")
    return np.hstack([a, b, c])


class RealSenseGrabber:
    def __init__(self):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.profile = self.pipeline.start(cfg)
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        depth = frames.get_depth_frame()
        ir = frames.get_infrared_frame(1)
        if not depth or not ir:
            return None, None
        ir_bgr = cv2.cvtColor(np.asanyarray(ir.get_data()), cv2.COLOR_GRAY2BGR)
        depth_bgr = np.asanyarray(self.colorizer.colorize(depth).get_data())
        return ir_bgr, depth_bgr

    def stop(self):
        self.pipeline.stop()


def record(
    out_path: str,
    seconds: float,
    demo: bool,
    poses_csv: Optional[str],
    no_poses: bool,
    show_preview: bool,
):
    poses = None if no_poses else load_poses_csv(poses_csv) if poses_csv else None
    if poses_csv and not poses:
        print(f"警告: {poses_csv} 无有效行，使用默认动画针姿")

    grabber = None
    if not demo:
        try:
            grabber = RealSenseGrabber()
            print("RealSense 已启动 (IR + Depth)")
        except Exception as e:
            print(f"无法打开 RealSense，改用 --demo 合成画面: {e}")
            demo = True

    n_frames = max(1, int(seconds * OUT_FPS))
    out_w, out_h = PANEL_W * 3, PANEL_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, OUT_FPS, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {out_path}")

    print(f"录制 {n_frames} 帧 @ {OUT_FPS} fps -> {out_path}")
    t0 = time.perf_counter()
    for i in range(n_frames):
        if demo:
            ir = synth_ir(i, PANEL_W, PANEL_H)
            dep = synth_depth(i, PANEL_W, PANEL_H)
        else:
            ir, dep = grabber.read()
            if ir is None:
                ir = synth_ir(i, PANEL_W, PANEL_H)
                dep = synth_depth(i, PANEL_W, PANEL_H)

        tip, axis = default_pose_for_frame(i, poses)
        needle = render_needle_panel(PANEL_W, PANEL_H, tip, axis)
        frame = compose_triptych(ir, dep, needle)
        writer.write(frame)

        if show_preview:
            cv2.imshow("record_demo_compare (q=abort)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户中断")
                break

    writer.release()
    if grabber:
        grabber.stop()
    if show_preview:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t0
    print(f"完成，耗时 {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="IR | Depth | 3D 对比演示 MP4")
    parser.add_argument("-o", "--output", default="demo_compare.mp4", help="输出 MP4 路径")
    parser.add_argument("--seconds", type=float, default=10.0, help="录制时长（秒）")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="无相机：合成 IR/深度（默认真机失败时也会回退）",
    )
    parser.add_argument(
        "--poses-csv",
        default=None,
        help="针姿 CSV（同 jetarm_marker_replay）",
    )
    parser.add_argument("--no-poses", action="store_true", help="不用 CSV，内置正弦针姿")
    parser.add_argument("--preview", action="store_true", help="录制时弹窗预览")
    args = parser.parse_args()

    out = args.output
    if not os.path.isabs(out):
        out = os.path.join(_ROOT, out)

    if args.demo:
        record(out, args.seconds, True, args.poses_csv, args.no_poses, args.preview)
        return

    # 未指定 --demo：先尝试 RealSense，失败则合成
    record(out, args.seconds, False, args.poses_csv, args.no_poses, args.preview)


if __name__ == "__main__":
    main()