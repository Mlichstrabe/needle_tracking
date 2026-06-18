"""
磁力计效果对比测试（WT901 / JY901 协议）

用法（项目根目录）:
    python tools/magnetometer_test.py

说明:
  - 本程序在 PC 端只「读取」0x54 磁场与 0x53 欧拉角，不做融合。
  - 针体/主程序用的四元数是芯片内部融合结果，磁力计是否参与由 Wit 上位机/模块配置决定。
  - 对比「有磁 / 无磁」需要你在 Wit 里改输出模式后各录一段:
      A: 9 轴融合 / 开启磁场参与（默认）
      B: 6 轴或关闭磁场融合后重连，再录一段

操作:
  1. 连接串口，看「磁力计帧」是否在增加
  2. 点「开始记录 A」静止 20 秒 → 再点「靠近金属/电机」做标记（可选）
  3. 在 Wit 里改成 6 轴/关磁融合，重连
  4. 点「开始记录 B」同样录 20 秒
  5. 点「对比 A/B 图表」查看 Yaw 稳定性与磁场变化
"""
from __future__ import annotations

import csv
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import deque

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

try:
    import serial
except ImportError:
    serial = None

LOG_DIR = _ROOT / "imu_calibration_logs"


def _configure_matplotlib_chinese() -> Optional[str]:
    """配置 matplotlib 中文字体，避免右下角说明文字乱码。"""
    try:
        from matplotlib import font_manager, rcParams
    except ImportError:
        return None
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen:
        rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
    return chosen


def _wrap180(x: float) -> float:
    while x > 180:
        x -= 360
    while x < -180:
        x += 360
    return x


def _yaw_delta_deg(prev: float, curr: float) -> float:
    return _wrap180(curr - prev)


@dataclass
class MagSample:
    t: float
    euler: np.ndarray  # roll, pitch, yaw (deg)
    mag: Optional[np.ndarray]  # mx, my, mz raw
    mag_count: int
    euler_count: int


class MagSerialReader(QObject):
    """读取 0x51/52/53/54/59，在每次 0x53 或 0x54 时发一帧快照。"""

    sample = pyqtSignal(object)
    error = pyqtSignal(str)
    stats = pyqtSignal(int, int)  # mag_frames, euler_frames

    def __init__(self):
        super().__init__()
        self._ser = None
        self._run = False
        self._th = None
        self._mag_count = 0
        self._euler_count = 0
        self.cache = {
            "acc": None,
            "gyro": None,
            "euler": None,
            "quat": None,
            "mag": None,
        }

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
            self._mag_count = 0
            self._euler_count = 0
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

    def send_mag_calib_start(self) -> bool:
        if not self.connected:
            return False
        try:
            self._ser.write(b"\xFF\xAA\x01\x07\x00")
            return True
        except Exception:
            return False

    def send_mag_calib_end(self) -> bool:
        if not self.connected:
            return False
        try:
            self._ser.write(b"\xFF\xAA\x01\x00\x00")
            return True
        except Exception:
            return False

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
                    self._parse(frame[1], frame)
            except Exception as e:
                self.error.emit(str(e))
                break

    def _parse(self, typ: int, frame: bytes):
        try:
            if typ == 0x51:
                ax = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 16 * 9.8
                ay = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 16 * 9.8
                az = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 16 * 9.8
                self.cache["acc"] = np.array([ax, ay, az])
            elif typ == 0x52:
                gx = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 2000 * np.pi / 180
                gy = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 2000 * np.pi / 180
                gz = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 2000 * np.pi / 180
                self.cache["gyro"] = np.array([gx, gy, gz])
            elif typ == 0x53:
                roll = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 180
                pitch = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 180
                yaw = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 180
                self.cache["euler"] = np.array([roll, pitch, yaw])
                self._euler_count += 1
                self._emit()
            elif typ == 0x54:
                mx = struct.unpack("<h", frame[2:4])[0]
                my = struct.unpack("<h", frame[4:6])[0]
                mz = struct.unpack("<h", frame[6:8])[0]
                self.cache["mag"] = np.array([mx, my, mz], dtype=float)
                self._mag_count += 1
                self._emit()
            elif typ == 0x59:
                q = [
                    struct.unpack("<h", frame[2:4])[0] / 32768.0,
                    struct.unpack("<h", frame[4:6])[0] / 32768.0,
                    struct.unpack("<h", frame[6:8])[0] / 32768.0,
                    struct.unpack("<h", frame[8:10])[0] / 32768.0,
                ]
                self.cache["quat"] = np.array(q)
        except Exception:
            pass

    def _emit(self):
        if self.cache["euler"] is None:
            return
        mag = None if self.cache["mag"] is None else self.cache["mag"].copy()
        self.sample.emit(
            MagSample(
                t=time.time(),
                euler=self.cache["euler"].copy(),
                mag=mag,
                mag_count=self._mag_count,
                euler_count=self._euler_count,
            )
        )
        self.stats.emit(self._mag_count, self._euler_count)


@dataclass
class RecordedSession:
    label: str
    rows: List[dict] = field(default_factory=list)
    events: List[Tuple[float, str]] = field(default_factory=list)


class MagnetometerTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("磁力计效果对比测试")
        self.resize(1000, 760)

        self.reader = MagSerialReader()
        self.reader.sample.connect(self._on_sample)
        self.reader.error.connect(self._on_error)
        self.reader.stats.connect(self._on_stats)

        self._recording: Optional[str] = None  # "A" or "B"
        self._t_rec0 = 0.0
        self.session_a = RecordedSession("A")
        self.session_b = RecordedSession("B")

        self._live_t: deque = deque(maxlen=600)
        self._live_yaw: deque = deque(maxlen=600)
        self._live_mag_b: deque = deque(maxlen=600)

        self._build_ui()
        self._refresh_ports()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(80)
        self._plot_timer.timeout.connect(self._refresh_live_plot)
        self._plot_timer.start()

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)

        hint = QLabel(
            "PC 程序不融合磁力计；对比的是芯片输出姿态在不同 Wit 配置下的差异。\n"
            "步骤: Wit 9轴模式录 A → Wit 6轴/关磁融合录 B → 点「对比 A/B」。"
            " 若「磁力计帧」一直为 0，请在上位机勾选输出磁场(0x54)。"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(140)
        row.addWidget(QLabel("串口"))
        row.addWidget(self.port_combo)
        row.addWidget(self._make_btn("刷新", self._refresh_ports))
        row.addWidget(self._make_btn("连接", self._connect))
        row.addWidget(self._make_btn("断开", self._disconnect))
        row.addStretch()
        root.addLayout(row)

        self.lbl_status = QLabel("未连接")
        root.addWidget(self.lbl_status)

        self.lbl_live = QLabel("Yaw: --  |  |B|: --  |  磁航向: --  |  磁力计帧/欧拉帧: 0/0")
        self.lbl_live.setStyleSheet("font-size: 14px; font-weight: bold;")
        root.addWidget(self.lbl_live)

        rec = QHBoxLayout()
        self.edit_label_a = QLineEdit("9轴融合(默认)")
        self.edit_label_b = QLineEdit("6轴或关磁融合")
        rec.addWidget(QLabel("A 备注"))
        rec.addWidget(self.edit_label_a)
        rec.addWidget(self._make_btn("开始记录 A", lambda: self._start_rec("A")))
        rec.addWidget(self._make_btn("停止", self._stop_rec))
        root.addLayout(rec)

        rec2 = QHBoxLayout()
        rec2.addWidget(QLabel("B 备注"))
        rec2.addWidget(self.edit_label_b)
        rec2.addWidget(self._make_btn("开始记录 B", lambda: self._start_rec("B")))
        rec2.addWidget(self._make_btn("标记:靠近金属", lambda: self._mark_event("靠近金属")))
        rec2.addWidget(self._make_btn("标记:远离金属", lambda: self._mark_event("远离金属")))
        root.addLayout(rec2)

        cal = QHBoxLayout()
        cal.addWidget(self._make_btn("磁力计校准开始(3s)", self._mag_calib_start))
        cal.addWidget(self._make_btn("磁力计校准结束", self._mag_calib_end))
        cal.addWidget(self._make_btn("对比 A/B 图表", self._plot_compare))
        cal.addWidget(self._make_btn("导出 CSV", self._export_csv))
        cal.addStretch()
        root.addLayout(cal)

        self.plot_live = pg.PlotWidget(title="实时: Yaw(橙) 与 |B|(蓝)")
        self.plot_live.showGrid(x=True, y=True, alpha=0.3)
        self.plot_live.addLegend()
        self.curve_yaw = self.plot_live.plot([], [], pen=pg.mkPen("#ffb74d", width=2), name="Yaw")
        self.curve_mag = self.plot_live.plot([], [], pen=pg.mkPen("#4fc3f7", width=2), name="|B|")
        root.addWidget(self.plot_live, stretch=1)

        self.plot_xy = pg.PlotWidget(title="实时: 磁场水平分量 (mx, my)")
        self.plot_xy.setAspectLocked(True)
        self.plot_xy.showGrid(x=True, y=True, alpha=0.3)
        self.curve_xy = self.plot_xy.plot([], [], pen=None, symbol="o", symbolSize=3, symbolBrush="#81c784")
        root.addWidget(self.plot_xy, stretch=1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        root.addWidget(self.log)

    def _make_btn(self, text: str, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        return b

    def _log(self, msg: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _refresh_ports(self):
        self.port_combo.clear()
        try:
            import serial.tools.list_ports

            for p in sorted(serial.tools.list_ports.comports()):
                self.port_combo.addItem(p.device, p.device)
        except ImportError:
            self.port_combo.addItem("COM3", "COM3")

    def _connect(self):
        port = self.port_combo.currentData()
        if not port:
            return
        if self.reader.connect(port):
            self.lbl_status.setText(f"已连接 {port}")
            self._log(f"连接 {port}")

    def _disconnect(self):
        self.reader.disconnect()
        self.lbl_status.setText("未连接")

    def _on_error(self, msg: str):
        self._log(f"错误: {msg}")

    def _on_stats(self, mag_n: int, euler_n: int):
        self.lbl_status.setText(
            f"已连接 | 累计帧: 磁力计 {mag_n} / 欧拉 {euler_n}"
            + (" | 正在录 " + self._recording if self._recording else "")
        )

    def _on_sample(self, s: MagSample):
        yaw = float(s.euler[2])
        if s.mag is not None:
            mx, my, mz = s.mag
            mag_b = float(np.linalg.norm(s.mag))
            mag_heading = float(np.degrees(np.arctan2(my, mx)))
        else:
            mag_b = float("nan")
            mag_heading = float("nan")

        self.lbl_live.setText(
            f"Yaw: {yaw:+.2f}°  |  |B|: {mag_b:.0f}  |  磁航向: {mag_heading:+.1f}°  |  "
            f"帧 {s.mag_count}/{s.euler_count}"
        )

        t = s.t
        self._live_t.append(t)
        self._live_yaw.append(yaw)
        self._live_mag_b.append(mag_b)

        if self._recording:
            sess = self.session_a if self._recording == "A" else self.session_b
            t_rel = t - self._t_rec0
            prev_yaw = sess.rows[-1]["yaw"] if sess.rows else yaw
            row = {
                "t": t_rel,
                "roll": float(s.euler[0]),
                "pitch": float(s.euler[1]),
                "yaw": yaw,
                "yaw_step": _yaw_delta_deg(prev_yaw, yaw),
                "mx": float(s.mag[0]) if s.mag is not None else "",
                "my": float(s.mag[1]) if s.mag is not None else "",
                "mz": float(s.mag[2]) if s.mag is not None else "",
                "mag_abs": mag_b if s.mag is not None else "",
            }
            c = self.reader.cache
            if c.get("acc") is not None:
                row["ax"], row["ay"], row["az"] = (float(x) for x in c["acc"])
            if c.get("gyro") is not None:
                row["gx"], row["gy"], row["gz"] = (float(x) for x in c["gyro"])
            if c.get("quat") is not None:
                row["qw"], row["qx"], row["qy"], row["qz"] = (float(x) for x in c["quat"])
            sess.rows.append(row)

    def _refresh_live_plot(self):
        if len(self._live_t) < 2:
            return
        t0 = self._live_t[0]
        ts = np.array(self._live_t) - t0
        self.curve_yaw.setData(ts, np.array(self._live_yaw))
        mb = np.array(self._live_mag_b)
        if np.any(np.isfinite(mb)):
            self.curve_mag.setData(ts, mb)

        sess = self.session_a if self._recording == "A" else (
            self.session_b if self._recording == "B" else None
        )
        if sess and sess.rows:
            mx = [r["mx"] for r in sess.rows if r["mx"] != ""]
            my = [r["my"] for r in sess.rows if r["my"] != ""]
            if mx:
                self.curve_xy.setData(mx[-300:], my[-300:])

    def _start_rec(self, which: str):
        if not self.reader.connected:
            QMessageBox.warning(self, "提示", "请先连接串口")
            return
        self._recording = which
        self._t_rec0 = time.time()
        sess = self.session_a if which == "A" else self.session_b
        sess.rows.clear()
        sess.events.clear()
        label = self.edit_label_a.text() if which == "A" else self.edit_label_b.text()
        sess.label = label or which
        self._log(f"开始记录 {which} ({sess.label})")

    def _stop_rec(self):
        if self._recording:
            self._log(f"停止记录 {self._recording}，共 {len(self._rows_for(self._recording))} 点")
        self._recording = None

    def _rows_for(self, which: str) -> List[dict]:
        return self.session_a.rows if which == "A" else self.session_b.rows

    def _mark_event(self, name: str):
        if not self._recording:
            QMessageBox.information(self, "提示", "请先开始记录 A 或 B")
            return
        sess = self.session_a if self._recording == "A" else self.session_b
        t_rel = time.time() - self._t_rec0
        sess.events.append((t_rel, name))
        self._log(f"标记 [{self._recording}] t={t_rel:.1f}s {name}")

    def _mag_calib_start(self):
        if self.reader.send_mag_calib_start():
            self._log("磁力计校准开始（请 3 秒内缓慢旋转 IMU）")
            QTimer.singleShot(3000, self._mag_calib_end)

    def _mag_calib_end(self):
        if self.reader.send_mag_calib_end():
            self._log("磁力计校准结束")

    def _session_stats(self, rows: List[dict]) -> dict:
        if len(rows) < 5:
            return {}
        yaw = np.array([r["yaw"] for r in rows], dtype=float)
        # 用 unwrap 后的相邻差计算真实跳变（避免 ±180° 折叠成 2° 的假小跳变）
        yaw_u = np.degrees(np.unwrap(np.radians(yaw)))
        steps_u = np.abs(np.diff(yaw_u))
        mag_abs = np.array(
            [r["mag_abs"] for r in rows if r["mag_abs"] != ""], dtype=float
        )
        big_jumps = int(np.sum(steps_u > 5.0)) if len(steps_u) else 0
        out = {
            "n": len(rows),
            "yaw_std": float(np.std(yaw)),
            "yaw_range": float(np.max(yaw) - np.min(yaw)),
            "yaw_max_step": float(np.max(steps_u)) if len(steps_u) else 0.0,
            "yaw_step_mean": float(np.mean(steps_u)) if len(steps_u) else 0.0,
            "yaw_big_jumps": big_jumps,
            "yaw_total_change": float(abs(yaw_u[-1] - yaw_u[0])) if len(yaw_u) else 0.0,
        }
        if len(mag_abs) > 5:
            out["mag_std"] = float(np.std(mag_abs))
            out["mag_range"] = float(np.max(mag_abs) - np.min(mag_abs))
        return out

    def _plot_compare(self):
        if not self.session_a.rows or not self.session_b.rows:
            QMessageBox.warning(
                self,
                "提示",
                "需要 A、B 两段记录。\n"
                "A: Wit 9轴默认配置\n"
                "B: Wit 改为 6轴或关闭磁融合后重连再录",
            )
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            QMessageBox.critical(self, "提示", "pip install matplotlib")
            return

        _configure_matplotlib_chinese()

        sa = self._session_stats(self.session_a.rows)
        sb = self._session_stats(self.session_b.rows)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("磁力计效果对比 (A vs B)", fontsize=13)

        def plot_yaw(ax, sess: RecordedSession, color: str):
            t = [r["t"] for r in sess.rows]
            y_raw = np.array([r["yaw"] for r in sess.rows], dtype=float)
            y = np.degrees(np.unwrap(np.radians(y_raw)))
            ax.plot(t, y, color=color, label=sess.label, linewidth=1.2)
            for te, ev in sess.events:
                ax.axvline(te, color="gray", linestyle=":", alpha=0.6)
                ax.text(te, ax.get_ylim()[1], ev, fontsize=7, rotation=90)

        ax = axes[0, 0]
        plot_yaw(ax, self.session_a, "#4fc3f7")
        plot_yaw(ax, self.session_b, "#81c784")
        ax.set_title("Yaw 时间曲线（unwrap 后，避免 ±180 折线）")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("Yaw 展开 (°)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for sess, c in ((self.session_a, "#4fc3f7"), (self.session_b, "#81c784")):
            t = [r["t"] for r in sess.rows if r["mag_abs"] != ""]
            m = [r["mag_abs"] for r in sess.rows if r["mag_abs"] != ""]
            if t:
                ax.plot(t, m, color=c, label=sess.label, linewidth=1.0)
        ax.set_title("|B| 磁场模长")
        ax.set_xlabel("时间 (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        labels = ["Yaw 最大跳变", "大跳变次数(>5°)", "|B| σ"]
        va = [
            sa.get("yaw_max_step", 0),
            sa.get("yaw_big_jumps", 0),
            sa.get("mag_std", 0),
        ]
        vb = [
            sb.get("yaw_max_step", 0),
            sb.get("yaw_big_jumps", 0),
            sb.get("mag_std", 0),
        ]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, va, w, label=self.session_a.label, color="#4fc3f7")
        ax.bar(x + w / 2, vb, w, label=self.session_b.label, color="#81c784")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("稳定性指标（越小通常越稳）")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, 1]
        ax.axis("off")
        lines = [
            f"A ({self.session_a.label}): n={sa.get('n', 0)}",
            f"  最大单帧跳变={sa.get('yaw_max_step', 0):.1f}°  "
            f"大跳变(>5°)={sa.get('yaw_big_jumps', 0)}次",
            f"  |B| σ={sa.get('mag_std', float('nan')):.0f}",
            "",
            f"B ({self.session_b.label}): n={sb.get('n', 0)}",
            f"  最大单帧跳变={sb.get('yaw_max_step', 0):.1f}°  "
            f"大跳变(>5°)={sb.get('yaw_big_jumps', 0)}次",
            f"  |B| σ={sb.get('mag_std', float('nan')):.0f}",
            "",
            "如何解读:",
            "• 最大跳变/大跳变次数 越小越稳",
            "• B 的 |B|σ 明显小于 A → 9轴对磁场更敏感",
            "• 两段 Yaw 都乱 → 测试时转动太大或磁干扰强",
            "• 训练若主要看倾角，可弱化 Yaw/用 6 轴",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10)

        plt.tight_layout()
        plt.show()

        self._log(
            f"对比完成 A: maxStep={sa.get('yaw_max_step', 0):.1f}° "
            f"大跳变={sa.get('yaw_big_jumps', 0)} |B|σ={sa.get('mag_std', 0):.0f} | "
            f"B: maxStep={sb.get('yaw_max_step', 0):.1f}° "
            f"大跳变={sb.get('yaw_big_jumps', 0)} |B|σ={sb.get('mag_std', 0):.0f}"
        )

    def _export_csv(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for which, sess in ("A", self.session_a), ("B", self.session_b):
            if not sess.rows:
                continue
            path = LOG_DIR / f"mag_test_{which}_{ts}.csv"
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(sess.rows[0].keys()))
                w.writeheader()
                w.writerows(sess.rows)
            self._log(f"已导出 {path}")

    def closeEvent(self, event):
        self.reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MagnetometerTestWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
