# Single Marker IMU Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent JetArm demo tool that tracks one stable IR/depth marker for translation and uses IMU quaternion data for needle orientation.

**Architecture:** Add a focused single-marker anchor module that owns m2 tracking, depth lookup, and hold/lost state. Add a lightweight geometry config and a standalone PyQt demo window that combines JetArm JARD frames, optional IMU serial data, and `GLVisualizationWidget` needle rendering. Keep this separate from `MainWindow` and the existing four-marker pose path.

**Tech Stack:** Python, PyQt5, pyqtgraph OpenGL, OpenCV, NumPy, existing `DeviceManager`, `imu_kinematics`, `ir_depth_stream_protocol`, `camera_math`, and `GLVisualizationWidget`.

---

## File Structure

- Create `data/jetarm_marker/geometry/anchor_imu_needle.json`
  - Demo geometry/config values for anchor marker, m2-to-tip distance, hold behavior, and depth window.
- Create `tools/jetarm_marker/single_marker_anchor.py`
  - Pure logic: config load, single blob tracking, depth backprojection, `tracking/coasting/lost` state.
- Create `tools/jetarm_marker/live_anchor_imu_needle.py`
  - Standalone GUI: IR panel, depth/anchor status, 3D needle, JetArm TCP worker, optional IMU serial connection.
- Modify `tools/jetarm_marker/README.md`
  - Add the new demo command and describe that it is single-marker translation + IMU orientation, not visual 6DoF.
- Optional test file `tools/jetarm_marker/_anchor_selftest.py`
  - Tiny script-style smoke checks because this repo does not currently have pytest test infrastructure.

Existing files used but not structurally refactored:

- `core/device_manager.py` for serial IMU data.
- `core/imu_kinematics.py` for `needle_axis_scene_normalized`.
- `tools/jetarm_marker/camera_math.py` for `depth_median_window` and `backproject_mm`.
- `tools/jetarm_marker/ir_marker_detect.py` for blob detection.
- `tools/jetarm_marker/ir_depth_stream_protocol.py` for JARD TCP frames.
- `ui/widgets/gl_widget.py` for `set_marker_needle_pose`.

---

### Task 1: Add Anchor Geometry Config

**Files:**
- Create: `data/jetarm_marker/geometry/anchor_imu_needle.json`

- [ ] **Step 1: Create the config file**

Write:

```json
{
  "anchor_marker": "m2",
  "m2_to_tip_mm": 140.0,
  "needle_length_mm": 162.0,
  "max_jump_px": 80.0,
  "max_hold_frames": 10,
  "depth_half_window": 13,
  "min_depth_pixels": 3,
  "z_min_mm": 50.0,
  "z_max_mm": 2000.0
}
```

- [ ] **Step 2: Validate JSON loads**

Run:

```powershell
python -c "import json; from pathlib import Path; p=Path('data/jetarm_marker/geometry/anchor_imu_needle.json'); print(json.loads(p.read_text(encoding='utf-8'))['anchor_marker'])"
```

Expected:

```text
m2
```

- [ ] **Step 3: Commit**

```powershell
git add data/jetarm_marker/geometry/anchor_imu_needle.json
git commit -m "Add single-marker anchor geometry config."
```

---

### Task 2: Implement Single-Marker Anchor Logic

**Files:**
- Create: `tools/jetarm_marker/single_marker_anchor.py`
- Optional smoke test: `tools/jetarm_marker/_anchor_selftest.py`

- [ ] **Step 1: Create the logic module skeleton**

Create `tools/jetarm_marker/single_marker_anchor.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single IR/depth marker anchor for translation-only visual tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from tools.jetarm_marker.camera_math import backproject_mm, depth_median_window
from tools.jetarm_marker.ir_marker_detect import Blob, DetectParams, detect_bright_blobs

AnchorState = Literal["tracking", "coasting", "lost"]

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "jetarm_marker"
    / "geometry"
    / "anchor_imu_needle.json"
)


@dataclass
class AnchorConfig:
    anchor_marker: str = "m2"
    m2_to_tip_mm: float = 140.0
    needle_length_mm: float = 162.0
    max_jump_px: float = 80.0
    max_hold_frames: int = 10
    depth_half_window: int = 13
    min_depth_pixels: int = 3
    z_min_mm: float = 50.0
    z_max_mm: float = 2000.0


@dataclass
class AnchorResult:
    state: AnchorState
    uv: Optional[np.ndarray]
    position_mm: Optional[np.ndarray]
    depth_mm: Optional[float]
    depth_pixels: int
    candidate_count: int
    hold_frames: int
    reason: str


def load_anchor_config(path: Optional[Path] = None) -> AnchorConfig:
    p = path or _DEFAULT_CONFIG
    data = json.loads(p.read_text(encoding="utf-8"))
    return AnchorConfig(
        anchor_marker=str(data.get("anchor_marker", "m2")),
        m2_to_tip_mm=float(data.get("m2_to_tip_mm", 140.0)),
        needle_length_mm=float(data.get("needle_length_mm", 162.0)),
        max_jump_px=float(data.get("max_jump_px", 80.0)),
        max_hold_frames=int(data.get("max_hold_frames", 10)),
        depth_half_window=int(data.get("depth_half_window", 13)),
        min_depth_pixels=int(data.get("min_depth_pixels", 3)),
        z_min_mm=float(data.get("z_min_mm", 50.0)),
        z_max_mm=float(data.get("z_max_mm", 2000.0)),
    )
```

- [ ] **Step 2: Add the tracker implementation**

Append to `single_marker_anchor.py`:

```python
class SingleMarkerAnchorTracker:
    """Tracks one bright marker and converts it to a 3D camera-space anchor."""

    def __init__(self, config: AnchorConfig, params: Optional[DetectParams] = None) -> None:
        self.config = config
        self.params = params or DetectParams(use_rom=False)
        self.previous_uv: Optional[np.ndarray] = None
        self.last_position_mm: Optional[np.ndarray] = None
        self.hold_frames = 0
        self.hold_enabled = True
        self.force_reacquire = False

    def reset(self) -> None:
        self.previous_uv = None
        self.last_position_mm = None
        self.hold_frames = 0
        self.force_reacquire = True

    def set_hold_enabled(self, enabled: bool) -> None:
        self.hold_enabled = bool(enabled)

    def choose_blob(self, blobs: list[Blob]) -> Optional[Blob]:
        if not blobs:
            return None
        if self.previous_uv is None or self.force_reacquire:
            self.force_reacquire = False
            return blobs[0]
        prev = self.previous_uv
        best: Optional[Blob] = None
        best_dist = float("inf")
        for blob in blobs:
            uv = np.array([blob.u, blob.v], dtype=np.float64)
            dist = float(np.linalg.norm(uv - prev))
            if dist < best_dist:
                best = blob
                best_dist = dist
        if best is None or best_dist > self.config.max_jump_px:
            return None
        return best

    def update(self, gray_u8: np.ndarray, depth: Optional[np.ndarray], depth_info: dict) -> AnchorResult:
        blobs = detect_bright_blobs(gray_u8, params=self.params)
        blob = self.choose_blob(blobs)
        if blob is None:
            return self._coast_or_lost(None, None, 0, len(blobs), "no_blob_or_jump")

        uv = np.array([blob.u, blob.v], dtype=np.float64)
        self.previous_uv = uv.copy()
        if depth is None:
            return self._coast_or_lost(uv, None, 0, len(blobs), "no_depth_frame")

        depth_mm, n_valid = depth_median_window(
            depth,
            float(uv[0]),
            float(uv[1]),
            half_win=self.config.depth_half_window,
            z_min_mm=self.config.z_min_mm,
            z_max_mm=self.config.z_max_mm,
            encoding=str(depth_info.get("encoding", "16UC1")),
        )
        if depth_mm is None or n_valid < self.config.min_depth_pixels:
            return self._coast_or_lost(uv, depth_mm, n_valid, len(blobs), "no_valid_depth")

        x, y, z = backproject_mm(
            float(uv[0]),
            float(uv[1]),
            float(depth_mm),
            float(depth_info["fx"]),
            float(depth_info["fy"]),
            float(depth_info["cx"]),
            float(depth_info["cy"]),
        )
        pos = np.array([x, y, z], dtype=np.float64)
        self.last_position_mm = pos.copy()
        self.hold_frames = 0
        return AnchorResult("tracking", uv, pos, float(depth_mm), int(n_valid), len(blobs), 0, "ok")

    def _coast_or_lost(
        self,
        uv: Optional[np.ndarray],
        depth_mm: Optional[float],
        depth_pixels: int,
        candidate_count: int,
        reason: str,
    ) -> AnchorResult:
        self.hold_frames += 1
        if self.hold_enabled and self.last_position_mm is not None and self.hold_frames <= self.config.max_hold_frames:
            return AnchorResult(
                "coasting",
                uv if uv is not None else self.previous_uv,
                self.last_position_mm.copy(),
                depth_mm,
                int(depth_pixels),
                int(candidate_count),
                int(self.hold_frames),
                reason,
            )
        return AnchorResult(
            "lost",
            uv if uv is not None else self.previous_uv,
            None,
            depth_mm,
            int(depth_pixels),
            int(candidate_count),
            int(self.hold_frames),
            reason,
        )
```

- [ ] **Step 3: Add a script-style smoke test**

Create `tools/jetarm_marker/_anchor_selftest.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from tools.jetarm_marker.single_marker_anchor import AnchorConfig, SingleMarkerAnchorTracker


def main() -> int:
    cfg = AnchorConfig(max_jump_px=20.0, max_hold_frames=2, depth_half_window=2, min_depth_pixels=1)
    tracker = SingleMarkerAnchorTracker(cfg)
    info = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "encoding": "16UC1"}

    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[48:53, 48:53] = 255
    depth = np.full((100, 100), 500, dtype=np.uint16)
    result = tracker.update(gray, depth, info)
    assert result.state == "tracking", result
    assert result.position_mm is not None
    assert abs(float(result.position_mm[2]) - 500.0) < 1e-6

    gray2 = np.zeros((100, 100), dtype=np.uint8)
    result2 = tracker.update(gray2, depth, info)
    assert result2.state == "coasting", result2
    assert result2.position_mm is not None

    tracker.update(gray2, depth, info)
    result4 = tracker.update(gray2, depth, info)
    assert result4.state == "lost", result4
    print("anchor selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test**

Run:

```powershell
python tools\jetarm_marker\_anchor_selftest.py
```

Expected:

```text
anchor selftest OK
```

- [ ] **Step 5: Compile the new module**

Run:

```powershell
python -m py_compile tools\jetarm_marker\single_marker_anchor.py tools\jetarm_marker\_anchor_selftest.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Commit**

```powershell
git add tools/jetarm_marker/single_marker_anchor.py tools/jetarm_marker/_anchor_selftest.py
git commit -m "Add single-marker anchor tracker."
```

---

### Task 3: Add Needle Fusion Helpers

**Files:**
- Modify: `tools/jetarm_marker/single_marker_anchor.py`

- [ ] **Step 1: Add IMU/demo axis helper and pose function**

Append to `single_marker_anchor.py`:

```python
def demo_axis() -> np.ndarray:
    """Fixed display axis for translation-only testing."""
    return np.array([0.0, 0.0, -1.0], dtype=np.float64)


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return demo_axis()
    return a / n


def anchored_needle_pose(anchor_position_mm: np.ndarray, axis_unit: np.ndarray, config: AnchorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tip, axis, tail) in display/camera millimeters.

    The anchor is m2. First version treats camera coordinates as display coordinates.
    """
    axis = normalize_axis(axis_unit)
    anchor = np.asarray(anchor_position_mm, dtype=np.float64).reshape(3)
    tip = anchor - axis * float(config.m2_to_tip_mm)
    tail = tip - axis * float(config.needle_length_mm)
    return tip, axis, tail
```

- [ ] **Step 2: Extend selftest for anchored pose**

Add to `tools/jetarm_marker/_anchor_selftest.py` after the existing assertions:

```python
    from tools.jetarm_marker.single_marker_anchor import anchored_needle_pose

    tip, axis, tail = anchored_needle_pose(np.array([0.0, 0.0, 500.0]), np.array([0.0, 0.0, -1.0]), cfg)
    assert np.allclose(axis, np.array([0.0, 0.0, -1.0]))
    assert np.allclose(tip, np.array([0.0, 0.0, 640.0]))
    assert np.allclose(tail, np.array([0.0, 0.0, 802.0]))
```

The expected `tail` uses the default `needle_length_mm=162.0` from `AnchorConfig`.

- [ ] **Step 3: Run the selftest**

```powershell
python tools\jetarm_marker\_anchor_selftest.py
```

Expected:

```text
anchor selftest OK
```

- [ ] **Step 4: Commit**

```powershell
git add tools/jetarm_marker/single_marker_anchor.py tools/jetarm_marker/_anchor_selftest.py
git commit -m "Add anchor-to-needle pose helpers."
```

---

### Task 4: Implement the Standalone Demo Window

**Files:**
- Create: `tools/jetarm_marker/live_anchor_imu_needle.py`

- [ ] **Step 1: Create imports, worker, and utility functions**

Create `tools/jetarm_marker/live_anchor_imu_needle.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-marker translation + IMU orientation demo window."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.device_manager import DeviceManager  # noqa: E402
from core.imu_kinematics import needle_axis_scene_normalized  # noqa: E402
from tools.jetarm_marker.ir_depth_stream_protocol import IrDepthFrame, connect_ir_depth_stream, iter_ir_depth_frames  # noqa: E402
from tools.jetarm_marker.ir_marker_detect import DetectParams  # noqa: E402
from tools.jetarm_marker.live_ir_depth_compare import depth_to_bgr  # noqa: E402
from tools.jetarm_marker.live_pose_estimate import load_depth_camera_info  # noqa: E402
from tools.jetarm_marker.needle_gl_view import configure_console_encoding  # noqa: E402
from tools.jetarm_marker.single_marker_anchor import (  # noqa: E402
    AnchorConfig,
    AnchorResult,
    SingleMarkerAnchorTracker,
    anchored_needle_pose,
    demo_axis,
    load_anchor_config,
)
from ui.widgets.gl_widget import GLVisualizationWidget  # noqa: E402

configure_console_encoding()

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QImage, QKeyEvent, QPixmap  # noqa: E402
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget  # noqa: E402


def bgr_to_pixmap(bgr: np.ndarray, width: int = 520) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaledToWidth(width, Qt.SmoothTransformation)


def stream_worker(host: str, port: int, out_q: "queue.Queue[IrDepthFrame]") -> None:
    while True:
        try:
            sock = connect_ir_depth_stream(host, port)
            for frame in iter_ir_depth_frames(sock):
                try:
                    while not out_q.empty():
                        out_q.get_nowait()
                except queue.Empty:
                    pass
                out_q.put(frame)
        except Exception as exc:
            print(f"[stream] {exc}; reconnect in 3s", file=sys.stderr)
            time.sleep(3.0)
```

- [ ] **Step 2: Add the Qt window class**

Append:

```python
class AnchorImuNeedleWindow(QMainWindow):
    def __init__(
        self,
        *,
        frame_queue: "queue.Queue[IrDepthFrame]",
        config: AnchorConfig,
        depth_info: dict,
        demo_axis_enabled: bool,
        serial_port: Optional[str],
    ) -> None:
        super().__init__()
        self.setWindowTitle("JetArm 单点位移 + IMU 姿态")
        self._queue = frame_queue
        self._config = config
        self._depth_info = depth_info
        self._tracker = SingleMarkerAnchorTracker(config, DetectParams(use_rom=False))
        self._demo_axis_enabled = demo_axis_enabled
        self._latest_quaternion: Optional[np.ndarray] = None
        self._last_pose_valid = False
        self._fps = 0.0
        self._device: Optional[DeviceManager] = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        panels = QHBoxLayout()

        self._ir_label = QLabel("IR")
        self._ir_label.setAlignment(Qt.AlignCenter)
        self._depth_label = QLabel("Depth")
        self._depth_label.setAlignment(Qt.AlignCenter)
        self._gl = GLVisualizationWidget()
        self._gl.set_marker_replay_mode(True)
        self._gl.needle_length = config.needle_length_mm
        self._gl.view.setCameraPosition(distance=400, elevation=25, azimuth=45)

        panels.addWidget(self._ir_label, stretch=1)
        panels.addWidget(self._depth_label, stretch=1)
        panels.addWidget(self._gl, stretch=2)
        root.addLayout(panels, stretch=1)

        self._status = QLabel("waiting")
        root.addWidget(self._status)

        if serial_port:
            self._device = DeviceManager(serial_port)
            self._device.data_received.connect(self._on_imu_data)
            self._device.error_occurred.connect(lambda msg: print(f"[imu] {msg}", file=sys.stderr))
            self._device.connect()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def closeEvent(self, event) -> None:
        if self._device is not None:
            self._device.disconnect()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
            return
        if key == Qt.Key_R:
            self._tracker.reset()
        if key == Qt.Key_Space:
            self._tracker.reset()
        if key == Qt.Key_H:
            self._tracker.set_hold_enabled(not self._tracker.hold_enabled)
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._tracker.params.threshold_percentile = min(99.9, self._tracker.params.threshold_percentile + 0.5)
        if key == Qt.Key_Minus:
            self._tracker.params.threshold_percentile = max(90.0, self._tracker.params.threshold_percentile - 0.5)
        if key == Qt.Key_BracketRight:
            self._config.depth_half_window = min(25, self._config.depth_half_window + 1)
        if key == Qt.Key_BracketLeft:
            self._config.depth_half_window = max(1, self._config.depth_half_window - 1)
        super().keyPressEvent(event)

    def _on_imu_data(self, data: dict) -> None:
        q = data.get("quaternion")
        if q is not None:
            self._latest_quaternion = np.asarray(q, dtype=np.float64).reshape(4)
```

- [ ] **Step 3: Add frame processing and rendering**

Append:

```python
    def _axis(self) -> tuple[np.ndarray, str]:
        if self._demo_axis_enabled or self._latest_quaternion is None:
            return demo_axis(), "demo"
        axis = needle_axis_scene_normalized(self._latest_quaternion)
        if axis is None:
            return demo_axis(), "invalid"
        return np.asarray(axis, dtype=np.float64), "imu"

    def _tick(self) -> None:
        t0 = time.perf_counter()
        try:
            frame = self._queue.get_nowait()
        except queue.Empty:
            return

        result = self._tracker.update(frame.gray, frame.depth, self._depth_info)
        axis, axis_source = self._axis()
        self._render_panels(frame, result)

        if result.position_mm is not None:
            tip, axis_unit, _tail = anchored_needle_pose(result.position_mm, axis, self._config)
            if result.state == "tracking" or result.state == "coasting":
                self._gl.set_marker_needle_pose(tip, axis_unit, confidence=1.0 if result.state == "tracking" else 0.4)
                self._last_pose_valid = True
        else:
            self._last_pose_valid = False

        dt = time.perf_counter() - t0
        if dt > 1e-6:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        pos = result.position_mm
        pos_text = "m2=—" if pos is None else f"m2=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})mm"
        depth_text = "depth=—" if result.depth_mm is None else f"depth={result.depth_mm:.0f}mm/{result.depth_pixels}px"
        self._status.setText(
            f"fps={self._fps:.1f}  state={result.state}  reason={result.reason}  "
            f"{pos_text}  {depth_text}  imu={axis_source}  "
            f"hold={result.hold_frames}  candidates={result.candidate_count}  "
            f"thr={self._tracker.params.threshold_percentile:.1f}  win={self._config.depth_half_window}  "
            "[space/r]reacquire [+/-]thr [/[]depth [h]hold [q]quit"
        )

    def _render_panels(self, frame: IrDepthFrame, result: AnchorResult) -> None:
        ir_bgr = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        if result.uv is not None:
            u, v = int(round(result.uv[0])), int(round(result.uv[1]))
            color = (0, 255, 0) if result.state == "tracking" else ((0, 255, 255) if result.state == "coasting" else (0, 0, 255))
            cv2.circle(ir_bgr, (u, v), 12, color, 2)
            cv2.putText(ir_bgr, "m2", (u + 12, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(ir_bgr, "IR anchor", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self._ir_label.setPixmap(bgr_to_pixmap(ir_bgr))

        if frame.depth is None:
            depth_bgr = np.zeros_like(ir_bgr)
            cv2.putText(depth_bgr, "NO DEPTH", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            depth_bgr = depth_to_bgr(frame.depth, encoding=str(self._depth_info.get("encoding", "16UC1")))
            if result.uv is not None:
                u, v = int(round(result.uv[0])), int(round(result.uv[1]))
                cv2.circle(depth_bgr, (u, v), 12, (0, 255, 255), 2)
                label = "no depth" if result.depth_mm is None else f"{result.depth_mm:.0f}mm {result.depth_pixels}px"
                cv2.putText(depth_bgr, label, (u + 12, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self._depth_label.setPixmap(bgr_to_pixmap(depth_bgr))
```

- [ ] **Step 4: Add CLI entry point**

Append:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Single marker translation + IMU orientation demo")
    parser.add_argument("--host", default="192.168.55.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--serial", default=None, help="IMU serial port, e.g. COM3")
    parser.add_argument("--demo-axis", action="store_true", help="Use fixed axis without IMU")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_anchor_config(args.config)
    depth_info = load_depth_camera_info()
    frame_q: queue.Queue[IrDepthFrame] = queue.Queue(maxsize=2)
    threading.Thread(target=stream_worker, args=(args.host, args.port, frame_q), daemon=True).start()

    app = QApplication(sys.argv)
    win = AnchorImuNeedleWindow(
        frame_queue=frame_q,
        config=config,
        depth_info=depth_info,
        demo_axis_enabled=args.demo_axis,
        serial_port=args.serial,
    )
    win.resize(1500, 760)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Compile**

Run:

```powershell
python -m py_compile tools\jetarm_marker\live_anchor_imu_needle.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Run demo-axis mode**

First start JetArm stream:

```powershell
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
```

Then:

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --demo-axis
```

Expected:

- Window opens.
- Left panel shows IR with one highlighted m2.
- Middle panel shows depth with mm text.
- Right panel shows needle translating when m2 moves.
- Status shows `imu=demo`.

- [ ] **Step 7: Commit**

```powershell
git add tools/jetarm_marker/live_anchor_imu_needle.py
git commit -m "Add single-marker IMU anchor demo window."
```

---

### Task 5: Add Documentation and IMU Run Verification

**Files:**
- Modify: `tools/jetarm_marker/README.md`

- [ ] **Step 1: Add README section**

Add after the real-time 3D section:

````markdown
## 单点位移 + IMU 姿态演示

当 4 个 marker 不稳定时，先用一个稳定反光点（默认 m2）测位移，用 IMU 给针体姿态：

```powershell
# 只验证相机单点位移
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --demo-axis

# 接入 IMU 姿态
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --serial COM3
```

该工具不做 CT 配准，也不声明视觉 6DoF；它只验证“单点相机位移 + IMU 姿态”能否稳定演示整根针移动与旋转。

快捷键：`space`/`r` 重新捕捉 m2，`+`/`-` 调 IR 阈值，`[`/`]` 调 depth 窗口，`h` 开关 hold，`q` 退出。
````

- [ ] **Step 2: Compile all touched Python files**

Run:

```powershell
python -m py_compile tools\jetarm_marker\single_marker_anchor.py tools\jetarm_marker\live_anchor_imu_needle.py tools\jetarm_marker\_anchor_selftest.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run smoke selftest**

```powershell
python tools\jetarm_marker\_anchor_selftest.py
```

Expected:

```text
anchor selftest OK
```

- [ ] **Step 4: Run demo-axis field test**

```powershell
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --demo-axis
```

Expected:

- Status reaches `state=tracking` when a bright marker is visible.
- Moving the marker forward/back changes the displayed `m2.z`.
- Short losses produce `coasting`, then `lost` if held too long.

- [ ] **Step 5: Run IMU field test**

Use the correct COM port for the JY901S-style IMU:

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --serial COM3
```

Expected:

- Status shows `imu=imu` after quaternion frames arrive.
- Translating marker moves the full needle.
- Rotating the IMU changes the needle axis.

- [ ] **Step 6: Commit docs**

```powershell
git add tools/jetarm_marker/README.md
git commit -m "Document single-marker IMU anchor demo."
```

---

## Self-Review

Spec coverage:

- Single marker m2 tracking: Task 2.
- Depth backprojection and hold/lost state: Task 2.
- IMU/demo-axis orientation: Tasks 3 and 4.
- Standalone GUI: Task 4.
- Config file: Task 1.
- Error handling and test sequence: Tasks 2, 4, and 5.

Completeness scan:

- No unresolved requirement markers remain.

Type consistency:

- `AnchorConfig`, `AnchorResult`, and `SingleMarkerAnchorTracker` are defined before being used.
- `anchored_needle_pose` returns `(tip, axis, tail)` and GUI uses `tip` and `axis`.
- `DeviceManager.data_received` supplies `quaternion`, matching `needle_axis_scene_normalized`.

Known implementation caution:

- The plan intentionally keeps the demo out of `MainWindow`.
- Existing uncommitted files `live_ir_depth_compare.py`, `live_triple_view.py`, and the modified `live_pose_estimate.py` are unrelated to this implementation plan unless the user explicitly wants them included later.
