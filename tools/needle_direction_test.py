"""
针体方向测试（跳过 CT / 选点 / 引导逻辑）

目标：
  1) 只连接 IMU，实时显示针向 direction（单位向量）
  2) 固定针尖在原点 tip=(0,0,0)，按 needle_length 反推 IMU 位置 imu = tip - direction * needle_length
  3) 记录轨迹并可视化：对比「Yaw 约束前」与「Yaw 约束后」的 XY 轨迹与分量曲线

Yaw 约束解释（用于演示“约束前后效果”）：
  - 以“启动记录时刻”的水平投影方向为参考 forward_ref
  - 每帧只保留 tilt（与重力/上方向的夹角），把 yaw（绕上方向旋转）锁到 forward_ref 平面
  - 这样：转动手柄产生的 yaw 变化会被“抹掉”，轨迹在 XY 平面更稳定

运行：
  python tools/needle_direction_test.py

依赖：
  PyQt5, numpy, pyserial, pyqtgraph
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

try:
    import serial
except Exception:  # pragma: no cover
    serial = None

from core.imu_kinematics import needle_axis_scene_normalized


@dataclass
class ImuFrame:
    t: float
    quat: np.ndarray  # [w,x,y,z]
    acc: Optional[np.ndarray]  # [ax,ay,az] m/s²


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v * 0.0
    return v / n


def _wrap180(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return float(deg)


def _tilt_only_direction(
    d: np.ndarray,
    up: np.ndarray,
    forward_ref: np.ndarray,
) -> np.ndarray:
    """
    约束 yaw：保留与 up 的夹角（tilt），把水平方向固定在 forward_ref 平面。
    """
    up = _normalize(up)
    d = _normalize(d)
    f = forward_ref - up * float(np.dot(forward_ref, up))
    f = _normalize(f) if float(np.linalg.norm(f)) > 1e-9 else np.array([1.0, 0.0, 0.0])

    # tilt angle between -up and d
    # clamp dot to avoid nan
    dot = float(np.clip(np.dot(d, -up), -1.0, 1.0))
    tilt = float(np.arccos(dot))  # 0 means pointing down

    # constrained direction in plane spanned by (-up) and f
    return _normalize(np.cos(tilt) * (-up) + np.sin(tilt) * f)


def _yaw_deg_in_frame(d: np.ndarray, up: np.ndarray, forward_ref: np.ndarray) -> float:
    """
    把方向 d 投影到水平面，计算其相对 forward_ref 的“水平朝向角”（度）。
    这不是 IMU 的原始 yaw，而是“在当前 up/forward 参考下的方位角”，专门用于可视化。
    """
    up = _normalize(up)
    f = forward_ref - up * float(np.dot(forward_ref, up))
    if float(np.linalg.norm(f)) < 1e-9:
        f = np.array([1.0, 0.0, 0.0], dtype=float)
    f = _normalize(f)
    y = _normalize(np.cross(up, f))

    h = d - up * float(np.dot(d, up))
    if float(np.linalg.norm(h)) < 1e-9:
        return 0.0
    h = _normalize(h)

    ang = float(np.degrees(np.arctan2(float(np.dot(h, y)), float(np.dot(h, f)))))
    return _wrap180(ang)


class SerialReader(QObject):
    frame = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._ser = None
        self._run = False
        self._th = None
        self._acc = None
        self._quat = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self, port: str, baud: int = 115200) -> bool:
        if serial is None:
            self.error.emit("未安装 pyserial")
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.5)
            self._run = True
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
            return True
        except Exception as e:
            self.error.emit(str(e))
            return False

    def disconnect(self):
        self._run = False
        if self._th:
            self._th.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def _loop(self):
        buf = bytearray()
        while self._run and self._ser:
            try:
                if self._ser.in_waiting:
                    buf.extend(self._ser.read(self._ser.in_waiting))
                while len(buf) >= 11:
                    if buf[0] != 0x55:
                        buf.pop(0)
                        continue
                    if len(buf) < 11:
                        break
                    frame = bytes(buf[:11])
                    if (sum(frame[:10]) & 0xFF) != frame[10]:
                        buf.pop(0)
                        continue
                    buf = buf[11:]
                    typ = frame[1]
                    if typ == 0x51:
                        ax = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 16 * 9.8
                        ay = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 16 * 9.8
                        az = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 16 * 9.8
                        self._acc = np.array([ax, ay, az], dtype=float)
                    elif typ == 0x59:
                        q0 = struct.unpack("<h", frame[2:4])[0] / 32768.0
                        q1 = struct.unpack("<h", frame[4:6])[0] / 32768.0
                        q2 = struct.unpack("<h", frame[6:8])[0] / 32768.0
                        q3 = struct.unpack("<h", frame[8:10])[0] / 32768.0
                        self._quat = np.array([q0, q1, q2, q3], dtype=float)
                        # 发一帧（有 acc 就带上，没有也没关系）
                        self.frame.emit(ImuFrame(t=time.time(), quat=self._quat.copy(), acc=None if self._acc is None else self._acc.copy()))
            except Exception as e:
                self.error.emit(str(e))
                break


class NeedleDirectionTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("针体方向测试（Yaw 约束前/后对比）")
        self.resize(980, 720)

        self.reader = SerialReader()
        self.reader.frame.connect(self._on_frame)
        self.reader.error.connect(self._on_error)

        self._recording = False
        self._t0 = 0.0
        self._needle_length = 200.0
        self._yaw_lock_enabled = True
        self._forward_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        self._up_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        self._acc_seen = False

        # data buffers
        self.ts = []
        self.raw_xyz = []        # imu position xyz (raw)
        self.lock_xyz = []       # imu position xyz (yaw locked)
        self.raw_yaw = []        # horizontal azimuth (deg) in ref frame
        self.lock_yaw = []
        self.yaw_diff = []       # raw - locked (deg)
        self.xy_err = []         # |raw_xy - lock_xy|

        self._build_ui()
        self._refresh_ports()

        self._ui = QTimer(self)
        self._ui.setInterval(50)
        self._ui.timeout.connect(self._refresh_plots)
        self._ui.start()

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)

        top = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(160)
        top.addWidget(QLabel("串口"))
        top.addWidget(self.port_combo)
        btn_ref = QPushButton("刷新")
        btn_ref.clicked.connect(self._refresh_ports)
        top.addWidget(btn_ref)
        self.btn_conn = QPushButton("连接")
        self.btn_conn.clicked.connect(self._connect)
        top.addWidget(self.btn_conn)
        self.btn_disc = QPushButton("断开")
        self.btn_disc.clicked.connect(self._disconnect)
        top.addWidget(self.btn_disc)
        top.addStretch()
        root.addLayout(top)

        cfg = QHBoxLayout()
        cfg.addWidget(QLabel("针长（IMU→针尖）"))
        self.spin_len = QDoubleSpinBox()
        self.spin_len.setRange(50, 400)
        self.spin_len.setDecimals(1)
        self.spin_len.setValue(self._needle_length)
        self.spin_len.setSuffix(" mm")
        self.spin_len.valueChanged.connect(lambda v: setattr(self, "_needle_length", float(v)))
        cfg.addWidget(self.spin_len)

        self.chk_yaw = QCheckBox("Yaw 约束（锁定水平朝向）")
        self.chk_yaw.setChecked(True)
        self.chk_yaw.toggled.connect(lambda v: setattr(self, "_yaw_lock_enabled", bool(v)))
        cfg.addWidget(self.chk_yaw)

        self.btn_start = QPushButton("开始记录（并设置参考朝向）")
        self.btn_start.clicked.connect(self._start_recording)
        cfg.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop_recording)
        cfg.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear)
        cfg.addWidget(self.btn_clear)

        cfg.addStretch()
        root.addLayout(cfg)

        self.lbl_dir = QLabel("direction raw: --    direction yaw-locked: --")
        self.lbl_dir.setStyleSheet("font-size: 14px; font-weight: bold;")
        root.addWidget(self.lbl_dir)

        self.lbl_hint = QLabel("提示：想看到约束效果，请保持倾角不变，只绕竖直轴转动（Yaw）。")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

        # plots
        plots = QHBoxLayout()

        self.plot_xy = pg.PlotWidget(title="XY 轨迹（IMU 位置）")
        self.plot_xy.setAspectLocked(True)
        self.plot_xy.showGrid(x=True, y=True, alpha=0.3)
        self.plot_xy.addLegend(offset=(10, 10))
        self.curve_xy_raw = self.plot_xy.plot([], [], pen=pg.mkPen("#4fc3f7", width=2), name="raw")
        self.curve_xy_lock = self.plot_xy.plot([], [], pen=pg.mkPen("#81c784", width=2), name="yaw-locked")
        plots.addWidget(self.plot_xy, 1)

        self.plot_xyz = pg.PlotWidget(title="X/Y/Z 分量 vs 时间（IMU 位置）")
        self.plot_xyz.showGrid(x=True, y=True, alpha=0.3)
        self.plot_xyz.addLegend(offset=(10, 10))
        self.cx_raw = self.plot_xyz.plot([], [], pen=pg.mkPen("#4fc3f7", width=1.5), name="raw X")
        self.cy_raw = self.plot_xyz.plot([], [], pen=pg.mkPen("#4fc3f7", style=Qt.DashLine), name="raw Y")
        self.cz_raw = self.plot_xyz.plot([], [], pen=pg.mkPen("#4fc3f7", style=Qt.DotLine), name="raw Z")
        self.cx_lock = self.plot_xyz.plot([], [], pen=pg.mkPen("#81c784", width=1.5), name="lock X")
        self.cy_lock = self.plot_xyz.plot([], [], pen=pg.mkPen("#81c784", style=Qt.DashLine), name="lock Y")
        self.cz_lock = self.plot_xyz.plot([], [], pen=pg.mkPen("#81c784", style=Qt.DotLine), name="lock Z")
        plots.addWidget(self.plot_xyz, 1)

        root.addLayout(plots)

        self.plot_yaw = pg.PlotWidget(title="Yaw 可视化（更直观）：水平朝向角 & 差值")
        self.plot_yaw.showGrid(x=True, y=True, alpha=0.3)
        self.plot_yaw.addLegend(offset=(10, 10))
        self.yaw_raw_curve = self.plot_yaw.plot([], [], pen=pg.mkPen("#4fc3f7", width=2), name="raw yaw")
        self.yaw_lock_curve = self.plot_yaw.plot([], [], pen=pg.mkPen("#81c784", width=2), name="locked yaw")
        self.yaw_diff_curve = self.plot_yaw.plot([], [], pen=pg.mkPen("#ffb74d", width=2), name="raw-locked")
        self.plot_yaw.setLabel("left", "角度 (°)")
        self.plot_yaw.setLabel("bottom", "时间 (s)")
        root.addWidget(self.plot_yaw)

        self.plot_err = pg.PlotWidget(title="raw vs locked 轨迹间距（XY）")
        self.plot_err.showGrid(x=True, y=True, alpha=0.3)
        self.err_curve = self.plot_err.plot([], [], pen=pg.mkPen("#ff7043", width=2))
        self.plot_err.setLabel("left", "距离 (mm)")
        self.plot_err.setLabel("bottom", "时间 (s)")
        root.addWidget(self.plot_err)

        self.lbl_status = QLabel("状态：未连接")
        root.addWidget(self.lbl_status)

    def _refresh_ports(self):
        self.port_combo.clear()
        try:
            import serial.tools.list_ports

            for p in sorted(serial.tools.list_ports.comports()):
                self.port_combo.addItem(p.device, p.device)
        except Exception:
            self.port_combo.addItem("COM3", "COM3")
        if self.port_combo.count() == 0:
            self.port_combo.addItem("(无串口)", "")

    def _connect(self):
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "提示", "请选择串口")
            return
        ok = self.reader.connect(port, 115200)
        if not ok:
            QMessageBox.warning(self, "失败", f"无法打开 {port}")
            return
        self.lbl_status.setText(f"状态：已连接 {port}")

    def _disconnect(self):
        self.reader.disconnect()
        self.lbl_status.setText("状态：未连接")

    def _on_error(self, msg: str):
        self.lbl_status.setText(f"错误：{msg}")

    def _start_recording(self):
        if not self.reader.connected:
            QMessageBox.warning(self, "提示", "请先连接 IMU")
            return
        if len(self.raw_xyz) == 0:
            # 将下一帧方向作为参考 forward
            self._forward_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        self._recording = True
        self._t0 = time.time()
        self.ts = []
        self.raw_xyz = []
        self.lock_xyz = []
        self.raw_yaw = []
        self.lock_yaw = []
        self.yaw_diff = []
        self.xy_err = []
        self.lbl_status.setText("状态：记录中…（第一帧会设置参考朝向）")

    def _stop_recording(self):
        self._recording = False
        self.lbl_status.setText("状态：已停止（可切换 Yaw 约束勾选查看对比）")

    def _clear(self):
        self.ts = []
        self.raw_xyz = []
        self.lock_xyz = []
        self.raw_yaw = []
        self.lock_yaw = []
        self.yaw_diff = []
        self.xy_err = []
        self._recording = False
        self._forward_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        # 立即清空曲线（否则下一次刷新因无数据直接 return，旧曲线会残留）
        self.curve_xy_raw.setData([], [])
        self.curve_xy_lock.setData([], [])
        self.cx_raw.setData([], [])
        self.cy_raw.setData([], [])
        self.cz_raw.setData([], [])
        self.cx_lock.setData([], [])
        self.cy_lock.setData([], [])
        self.cz_lock.setData([], [])
        self.yaw_raw_curve.setData([], [])
        self.yaw_lock_curve.setData([], [])
        self.yaw_diff_curve.setData([], [])
        self.err_curve.setData([], [])
        self.lbl_status.setText("状态：已清空")

    def _on_frame(self, fr: ImuFrame):
        # raw direction from current kinematics mapping
        d_list = needle_axis_scene_normalized(fr.quat.tolist())
        if d_list is None:
            return
        d_raw = _normalize(np.asarray(d_list, dtype=float))

        # up reference: from accel if available, else keep default up
        if fr.acc is not None:
            self._acc_seen = True
            g = _normalize(np.asarray(fr.acc, dtype=float))
            # gravity points down, so up is -g
            self._up_ref = _normalize(-g)

        # set forward reference on first sample after start_recording
        if self._recording and len(self.ts) == 0:
            horiz = d_raw - self._up_ref * float(np.dot(d_raw, self._up_ref))
            if float(np.linalg.norm(horiz)) > 1e-6:
                self._forward_ref = _normalize(horiz)
            self.lbl_status.setText("状态：记录中…（参考朝向已锁定）")

        if self._yaw_lock_enabled:
            d_lock = _tilt_only_direction(d_raw, self._up_ref, self._forward_ref)
        else:
            d_lock = d_raw.copy()

        self.lbl_dir.setText(
            f"direction raw: [{d_raw[0]:+.3f}, {d_raw[1]:+.3f}, {d_raw[2]:+.3f}]    "
            f"direction yaw-locked: [{d_lock[0]:+.3f}, {d_lock[1]:+.3f}, {d_lock[2]:+.3f}]"
        )

        if not self._recording:
            return

        t = fr.t - self._t0
        tip = np.zeros(3, dtype=float)
        imu_raw = tip - d_raw * self._needle_length
        imu_lock = tip - d_lock * self._needle_length

        self.ts.append(float(t))
        self.raw_xyz.append(imu_raw.astype(float))
        self.lock_xyz.append(imu_lock.astype(float))

        # yaw 可视化（在参考 forward/up 坐标系下）
        raw_y = _yaw_deg_in_frame(d_raw, self._up_ref, self._forward_ref)
        lock_y = _yaw_deg_in_frame(d_lock, self._up_ref, self._forward_ref)
        self.raw_yaw.append(raw_y)
        self.lock_yaw.append(lock_y)
        self.yaw_diff.append(_wrap180(raw_y - lock_y))

        # raw vs locked 的 XY 间距（用 mm 便于观察）
        self.xy_err.append(float(np.linalg.norm((imu_raw - imu_lock)[:2])))

    def _refresh_plots(self):
        if not self.ts:
            return
        ts = np.asarray(self.ts, dtype=float)
        raw = np.asarray(self.raw_xyz, dtype=float)
        lock = np.asarray(self.lock_xyz, dtype=float)

        # XY
        self.curve_xy_raw.setData(raw[:, 0], raw[:, 1])
        self.curve_xy_lock.setData(lock[:, 0], lock[:, 1])

        # XYZ vs time
        self.cx_raw.setData(ts, raw[:, 0])
        self.cy_raw.setData(ts, raw[:, 1])
        self.cz_raw.setData(ts, raw[:, 2])
        self.cx_lock.setData(ts, lock[:, 0])
        self.cy_lock.setData(ts, lock[:, 1])
        self.cz_lock.setData(ts, lock[:, 2])

        # yaw curves
        if self.raw_yaw:
            ry = np.asarray(self.raw_yaw, dtype=float)
            ly = np.asarray(self.lock_yaw, dtype=float)
            dy = np.asarray(self.yaw_diff, dtype=float)
            self.yaw_raw_curve.setData(ts, ry)
            self.yaw_lock_curve.setData(ts, ly)
            self.yaw_diff_curve.setData(ts, dy)

        # xy error curve
        if self.xy_err:
            e = np.asarray(self.xy_err, dtype=float)
            self.err_curve.setData(ts, e)

        # 动态提示：是否收到加速度（用于 up 参考）
        if self._acc_seen:
            self.lbl_hint.setText("提示：已收到加速度帧（up 参考可靠）。保持倾角不变，仅绕竖直轴转动更容易看到约束效果。")
        else:
            self.lbl_hint.setText(
                "提示：未收到加速度帧（0x51）。up 参考使用默认 Z 轴，约束效果可能不明显。"
                "可用 Wit 上位机确认是否开启加速度输出。"
            )

    def closeEvent(self, event):
        self.reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = NeedleDirectionTest()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

