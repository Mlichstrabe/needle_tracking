#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2：在第一张清晰帧上手动点选 4 个 marker，保存 init JSON。

用法:
  python tools/jetarm_marker/init_markers.py \\
      data/jetarm_marker/exports/marker_static_clean_01/frames/frame_003_idx0012_rgb.png

操作:
  - 左键依次点击 4 个反光球（顺序即 m0..m3）
  - 按 u 撤销上一个点
  - 按 s 保存并退出
  - 按 q 放弃退出
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
except ImportError as exc:
    raise SystemExit("需要 opencv-python: pip install -r requirements-jetarm-marker.txt") from exc


class MarkerInitSession:
    def __init__(self, image_path: Path, bag_name: str, frame_ref: str):
        self.image_path = image_path
        self.bag_name = bag_name
        self.frame_ref = frame_ref
        self.points: List[Tuple[float, float]] = []
        self._base = cv2.imread(str(image_path))
        if self._base is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

    def _draw(self):
        vis = self._base.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"m{i}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        hint = f"已选 {len(self.points)}/4 | u=撤销 s=保存 q=退出"
        cv2.putText(vis, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((float(x), float(y)))

    def run(self) -> List[Tuple[float, float]]:
        win = "init_markers — 点击 4 个反光球"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.on_mouse)

        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                self.points = []
                break
            if key == ord("u") and self.points:
                self.points.pop()
            if key == ord("s") and len(self.points) == 4:
                break

        cv2.destroyAllWindows()
        return self.points


def main() -> int:
    parser = argparse.ArgumentParser(description="手动初始化 4 marker 像素坐标")
    parser.add_argument("image", type=Path, help="RGB 或 IR 关键帧 PNG")
    parser.add_argument(
        "--bag",
        default="unknown",
        help="关联 bag 名称（写入 JSON）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 JSON（默认 data/jetarm_marker/inits/<bag>_init.json）",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"错误: 找不到图像 {args.image}", file=sys.stderr)
        return 1

    session = MarkerInitSession(args.image, args.bag, str(args.image.name))
    points = session.run()

    if len(points) != 4:
        print("未保存：需要恰好 4 个点。")
        return 1

    out = args.output or (
        _REPO_ROOT / "data" / "jetarm_marker" / "inits" / f"{args.bag}_init.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bag": args.bag,
        "frame_image": str(args.image.resolve()),
        "frame_ref": args.image.name,
        "modality": "rgb_or_ir",
        "markers": [
            {"id": i, "u": u, "v": v} for i, (u, v) in enumerate(points)
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 4 点初始化: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
