"""
P0：离线回放 JetArm 针姿 CSV，驱动 3D 视图（不依赖相机）。

CSV 列（表头）:
  tip_x, tip_y, tip_z, axis_x, axis_y, axis_z [, confidence]

运行:
  python scripts/jetarm_marker_replay.py path/to/poses.csv
"""
from __future__ import annotations

import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from core.jetarm_geometry_config import load_geometry
from ui.widgets.gl_widget import GLVisualizationWidget


def load_poses_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tip = [float(row["tip_x"]), float(row["tip_y"]), float(row["tip_z"])]
            axis = [float(row["axis_x"]), float(row["axis_y"]), float(row["axis_z"])]
            conf = float(row.get("confidence") or 1.0)
            rows.append((tip, axis, conf))
    return rows


class ReplayWindow(QMainWindow):
    def __init__(self, poses, interval_ms: int = 33):
        super().__init__()
        self.setWindowTitle("JetArm Marker 回放 (P0)")
        self._poses = poses
        self._idx = 0

        geom = load_geometry()
        nl = float(geom.get("needle_length_mm", 162.0))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.gl = GLVisualizationWidget()
        self.gl.needle_length = nl
        self.gl.set_marker_replay_mode(True)
        layout.addWidget(self.gl)

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        if not self._poses:
            return
        tip, axis, conf = self._poses[self._idx % len(self._poses)]
        self.gl.set_marker_needle_pose(tip, axis, confidence=conf)
        self._idx += 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    poses = load_poses_csv(path)
    if not poses:
        print("CSV 无有效行")
        sys.exit(1)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    w = ReplayWindow(poses)
    w.resize(960, 720)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()