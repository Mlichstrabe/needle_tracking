"""
IMU 精度自检（无需量角器）

用法（项目根目录）:
    python tools/imu_accuracy_test.py

思路（都不用手输角度）:
  1. 静止噪声 — 放稳 5 秒自动采样，看抖动有多大
  2. 重力对照 — 摆 2～3 个姿势点「捕获」，用加速度计推算倾角 vs IMU 欧拉角差多少
  3. 相对转角 — 先捕获姿态 A，转 90°/180° 再捕获 B，看 IMU 转角是否对
  4. Yaw 漂移 — 静止 30 秒看 Yaw 是否自己漂

有光学定位（反光球）时，应以光学为真值；本工具只评估 IMU 自身。
"""
from __future__ import annotations

import csv
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import serial
except ImportError:
    serial = None

LOG_DIR = _ROOT / "imu_calibration_logs"
STABLE_GYRO = 0.08  # rad/s 以下认为基本静止
STABLE_ACC_STD = 0.35  # m/s² 以下认为未在晃动


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """两姿态四元数夹角（度）。q = [w,x,y,z]"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, dot)
    return float(np.degrees(2.0 * np.arccos(dot)))


def _tilt_from_accel(acc: np.ndarray) -> Tuple[float, float]:
    """静止时用加速度计估算 Roll/Pitch（度），仅作对照参考。"""
    a = np.asarray(acc, dtype=float)
    n = np.linalg.norm(a)
    if n < 3.0:
        return float("nan"), float("nan")
    ax, ay, az = a / n
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay * ay + az * az)))
    roll = np.degrees(np.arctan2(ay, az))
    return roll, pitch


def _wrap180(x: float) -> float:
    while x > 180:
        x -= 360
    while x < -180:
        x += 360
    return x


@dataclass
class ImuSnapshot:
  euler: np.ndarray
  acc: np.ndarray
  quat: np.ndarray
  gyro: np.ndarray
  t: float = field(default_factory=time.time)


class ImuSerialReader(QObject):
    """轻量串口读 IMU（0x51 加速度 + 0x52 陀螺 + 0x53 欧拉 + 0x59 四元数）。"""

    sample = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._ser = None
        self._run = False
        self._thread = None
        self.cache = {
            "acc": None,
            "gyro": None,
            "euler": None,
            "quat": None,
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
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            self.error.emit(str(e))
            return False

    def disconnect(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
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
                self._emit_sample()
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

    def _emit_sample(self):
        if self.cache["euler"] is None or self.cache["acc"] is None:
            return
        g = self.cache["gyro"]
        q = self.cache["quat"]
        snap = ImuSnapshot(
            euler=self.cache["euler"].copy(),
            acc=self.cache["acc"].copy(),
            gyro=g.copy() if g is not None else np.zeros(3),
            quat=q.copy() if q is not None else np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.sample.emit(snap)


class ImuAccuracyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU 精度自检（自动，无需量角器）")
        self.resize(760, 720)

        self.reader = ImuSerialReader()
        self.reader.sample.connect(self._on_sample)
        self.reader.error.connect(self._on_error)

        self._buf: Deque[ImuSnapshot] = deque(maxlen=120)
        self._stable = False
        self._pose_a: Optional[ImuSnapshot] = None
        self._gravity_poses: List[dict] = []
        self._noise_result: Optional[dict] = None
        self._drift_rows: List[Tuple[float, float]] = []

        self._noise_timer: Optional[QTimer] = None
        self._drift_timer: Optional[QTimer] = None
        self._noise_samples: List[ImuSnapshot] = []
        self._session = LOG_DIR / f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self._build_ui()

        self._ui = QTimer(self)
        self._ui.setInterval(150)
        self._ui.timeout.connect(self._refresh_status)
        self._ui.start()

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        layout = QVBoxLayout(c)

        top = QLabel(
            "不用量角器：把 IMU 放稳 → 等「已静止」变绿 → 点对应按钮。\n"
            "有 NDI/光学跟踪时，针尖真值请用光学；本工具只判断 IMU 自身稳不稳、偏不偏。"
        )
        top.setWordWrap(True)
        layout.addWidget(top)

        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(160)
        row.addWidget(QLabel("串口"))
        row.addWidget(self.port_combo, 1)
        btn_ref = QPushButton("刷新")
        btn_ref.clicked.connect(self._refresh_ports)
        row.addWidget(btn_ref)
        self.btn_conn = QPushButton("连接")
        self.btn_conn.clicked.connect(self._connect)
        row.addWidget(self.btn_conn)
        self.btn_disc = QPushButton("断开")
        self.btn_disc.clicked.connect(self._disconnect)
        row.addWidget(self.btn_disc)
        layout.addLayout(row)

        self.lbl_live = QLabel("Roll --  Pitch --  Yaw --  |  静止: 检测中…")
        self.lbl_live.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.lbl_live)

        self.lbl_accel = QLabel("加速度: --")
        layout.addWidget(self.lbl_accel)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Tab1 噪声 ---
        t1 = QWidget()
        v1 = QVBoxLayout(t1)
        v1.addWidget(QLabel(
            "① 噪声：IMU 放桌面不动，点「测 5 秒噪声」。\n"
            "结果看标准差：Roll/Pitch 通常 < 0.5° 较好；Yaw 可能更大（磁力计）。"
        ))
        self.btn_noise = QPushButton("测 5 秒噪声")
        self.btn_noise.clicked.connect(self._start_noise_test)
        v1.addWidget(self.btn_noise)
        self.txt_noise = QTextEdit()
        self.txt_noise.setReadOnly(True)
        self.txt_noise.setMaximumHeight(120)
        v1.addWidget(self.txt_noise)
        tabs.addTab(t1, "① 静止噪声")

        # --- Tab2 重力对照 ---
        t2 = QWidget()
        v2 = QVBoxLayout(t2)
        v2.addWidget(QLabel(
            "② 重力对照：摆 3 个差别大的姿势（如平放、侧立、竖起），每个等变绿后点「捕获姿势」。\n"
            "程序用加速度计推算倾角，与 IMU 的 Roll/Pitch 对比——差值大说明倾角不可信或轴向定义不一致。"
        ))
        self.btn_pose = QPushButton("捕获当前姿势")
        self.btn_pose.clicked.connect(self._capture_gravity_pose)
        v2.addWidget(self.btn_pose)
        self.txt_gravity = QTextEdit()
        self.txt_gravity.setReadOnly(True)
        v2.addWidget(self.txt_gravity)
        tabs.addTab(t2, "② 重力对照")

        # --- Tab3 相对转角 ---
        t3 = QWidget()
        v3 = QVBoxLayout(t3)
        v3.addWidget(QLabel(
            "③ 相对转角：用直角挡板/书角把转动限制在 90° 或 180°（只需转准，不用读数）。\n"
            "先「记录姿态 A」→ 转动 → 再「记录姿态 B」，看 IMU 算出的转角是否接近。"
        ))
        h = QHBoxLayout()
        h.addWidget(QLabel("期望转角"))
        self.combo_turn = QComboBox()
        self.combo_turn.addItems(["90", "180"])
        h.addWidget(self.combo_turn)
        h.addStretch()
        v3.addLayout(h)
        self.btn_a = QPushButton("记录姿态 A")
        self.btn_a.clicked.connect(self._capture_pose_a)
        v3.addWidget(self.btn_a)
        self.btn_b = QPushButton("记录姿态 B 并计算")
        self.btn_b.clicked.connect(self._capture_pose_b)
        v3.addWidget(self.btn_b)
        self.txt_turn = QTextEdit()
        self.txt_turn.setReadOnly(True)
        self.txt_turn.setMaximumHeight(100)
        v3.addWidget(self.txt_turn)
        tabs.addTab(t3, "③ 相对转角")

        # --- Tab4 漂移 ---
        t4 = QWidget()
        v4 = QVBoxLayout(t4)
        v4.addWidget(QLabel("④ Yaw 漂移：水平放稳 30 秒，看 Yaw 是否自己慢慢走。"))
        self.btn_drift = QPushButton("测 30 秒 Yaw 漂移")
        self.btn_drift.clicked.connect(self._start_drift_test)
        v4.addWidget(self.btn_drift)
        self.txt_drift = QTextEdit()
        self.txt_drift.setReadOnly(True)
        v4.addWidget(self.txt_drift)
        tabs.addTab(t4, "④ Yaw 漂移")

        btn_plot = QPushButton("生成图表（汇总以上测试）")
        btn_plot.clicked.connect(self._plot_all)
        layout.addWidget(btn_plot)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)

        self._refresh_ports()

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
            self._log(f"已连接 {port}")
        else:
            QMessageBox.warning(self, "失败", "无法打开串口")

    def _disconnect(self):
        self.reader.disconnect()
        self._log("已断开")

    def _on_error(self, msg: str):
        self._log(f"错误: {msg}")

    def _on_sample(self, snap: ImuSnapshot):
        self._buf.append(snap)

    def _is_stable(self) -> bool:
        if len(self._buf) < 15:
            return False
        recent = list(self._buf)[-20:]
        gyros = np.array([s.gyro for s in recent])
        accs = np.array([s.acc for s in recent])
        gnorm = np.linalg.norm(gyros, axis=1).mean()
        acc_std = np.std(accs, axis=0).max()
        return gnorm < STABLE_GYRO and acc_std < STABLE_ACC_STD

    def _mean_snap(self, n: int = 30) -> Optional[ImuSnapshot]:
        if len(self._buf) < 5:
            return None
        chunk = list(self._buf)[-n:]
        euler = np.mean([s.euler for s in chunk], axis=0)
        acc = np.mean([s.acc for s in chunk], axis=0)
        quat = np.mean([s.quat for s in chunk], axis=0)
        quat = quat / np.linalg.norm(quat)
        return ImuSnapshot(euler=euler, acc=acc, quat=quat, gyro=chunk[-1].gyro)

    def _refresh_status(self):
        snap = self._mean_snap(15)
        self._stable = self._is_stable()
        if snap is None:
            return
        st = "已静止 ✓" if self._stable else "未静止 — 请放稳"
        color = "#6f6" if self._stable else "#f96"
        self.lbl_live.setText(
            f"Roll {snap.euler[0]:+.2f}°   Pitch {snap.euler[1]:+.2f}°   "
            f"Yaw {snap.euler[2]:+.2f}°   |  {st}"
        )
        self.lbl_live.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        self.lbl_accel.setText(
            f"加速度 [{snap.acc[0]:+.2f}, {snap.acc[1]:+.2f}, {snap.acc[2]:+.2f}] m/s²"
        )
        self.btn_conn.setEnabled(not self.reader.connected)
        self.btn_disc.setEnabled(self.reader.connected)

    def _require_stable(self) -> bool:
        if not self._stable:
            QMessageBox.warning(self, "未静止", "请把 IMU 放稳，等字样变成「已静止 ✓」再操作。")
            return False
        return True

    def _start_noise_test(self):
        if not self.reader.connected:
            QMessageBox.warning(self, "提示", "请先连接串口")
            return
        self._noise_samples = []
        self.btn_noise.setEnabled(False)
        self.txt_noise.setText("采集中… 请保持静止 5 秒")
        self._log("开始 5 秒噪声测试")

        self._noise_timer = QTimer(self)
        self._noise_timer.setInterval(50)

        def tick():
            if self._buf:
                self._noise_samples.append(self._buf[-1])

        self._noise_timer.timeout.connect(tick)
        self._noise_timer.start()

        QTimer.singleShot(5000, self._finish_noise_test)

    def _finish_noise_test(self):
        if self._noise_timer:
            self._noise_timer.stop()
        self.btn_noise.setEnabled(True)
        if len(self._noise_samples) < 10:
            self.txt_noise.setText("样本太少，请重试")
            return
        e = np.array([s.euler for s in self._noise_samples])
        std = np.std(e, axis=0)
        span = e.max(axis=0) - e.min(axis=0)
        self._noise_result = {"std": std, "span": span, "n": len(e)}
        lines = [
            f"样本数: {len(e)}",
            f"Roll  抖动 σ={std[0]:.3f}°  范围={span[0]:.3f}°",
            f"Pitch 抖动 σ={std[1]:.3f}°  范围={span[1]:.3f}°",
            f"Yaw   抖动 σ={std[2]:.3f}°  范围={span[2]:.3f}°",
            "",
            "参考: Roll/Pitch σ < 0.5° 较理想；Yaw 室内常 1°～3° 或更大。",
        ]
        self.txt_noise.setText("\n".join(lines))
        self._log("噪声测试完成")
        self._save_csv_row("noise", std0=std[0], std1=std[1], std2=std[2])

    def _capture_gravity_pose(self):
        if not self._require_stable():
            return
        snap = self._mean_snap()
        if snap is None:
            return
        acc_r, acc_p = _tilt_from_accel(snap.acc)
        imu_r, imu_p = float(snap.euler[0]), float(snap.euler[1])
        d_r = _wrap180(imu_r - acc_r)
        d_p = _wrap180(imu_p - acc_p)
        name = f"姿势{len(self._gravity_poses) + 1}"
        self._gravity_poses.append({
            "name": name,
            "imu_r": imu_r, "imu_p": imu_p,
            "acc_r": acc_r, "acc_p": acc_p,
            "d_r": d_r, "d_p": d_p,
        })
        block = (
            f"【{name}】\n"
            f"  IMU      Roll {imu_r:+.2f}°  Pitch {imu_p:+.2f}°\n"
            f"  加速度计  Roll {acc_r:+.2f}°  Pitch {acc_p:+.2f}°\n"
            f"  差值      ΔRoll {d_r:+.2f}°  ΔPitch {d_p:+.2f}°\n"
        )
        self.txt_gravity.append(block)
        self._log(f"捕获 {name} ΔRoll={d_r:+.1f}° ΔPitch={d_p:+.1f}°")

    def _capture_pose_a(self):
        if not self._require_stable():
            return
        self._pose_a = self._mean_snap()
        if self._pose_a:
            self.txt_turn.setText(
                f"姿态 A 已记录: Roll {self._pose_a.euler[0]:+.1f}° "
                f"Pitch {self._pose_a.euler[1]:+.1f}° Yaw {self._pose_a.euler[2]:+.1f}°\n"
                "请转动后点「记录姿态 B」"
            )
            self._log("记录姿态 A")

    def _capture_pose_b(self):
        if self._pose_a is None:
            QMessageBox.warning(self, "提示", "请先记录姿态 A")
            return
        if not self._require_stable():
            return
        snap_b = self._mean_snap()
        if snap_b is None:
            return
        expected = float(self.combo_turn.currentText())
        measured = _quat_angle_deg(self._pose_a.quat, snap_b.quat)
        err = measured - expected
        self.txt_turn.append(
            f"\n期望转角 {expected:.0f}°  |  IMU 测得 {measured:.2f}°  |  误差 {err:+.2f}°"
        )
        self._log(f"相对转角 测得={measured:.1f}° 误差={err:+.1f}°")
        self._save_csv_row("relative_turn", expected=expected, measured=measured, err=err)
        self._pose_a = None

    def _start_drift_test(self):
        if not self.reader.connected:
            return
        self._drift_rows = []
        self.btn_drift.setEnabled(False)
        self.txt_drift.setText("采集中… 水平静止 30 秒")
        self._log("开始 Yaw 漂移测试")
        t0 = time.time()

        self._drift_timer = QTimer(self)

        def tick():
            if self._buf:
                s = self._buf[-1]
                self._drift_rows.append((time.time() - t0, float(s.euler[2])))

        self._drift_timer.timeout.connect(tick)
        self._drift_timer.start(100)
        QTimer.singleShot(30000, self._finish_drift)

    def _finish_drift(self):
        if self._drift_timer:
            self._drift_timer.stop()
        self.btn_drift.setEnabled(True)
        if len(self._drift_rows) < 5:
            self.txt_drift.setText("数据不足")
            return
        ys = np.array([r[1] for r in self._drift_rows])
        # unwrap yaw for drift rate
        y_unwrap = np.unwrap(np.radians(ys))
        y_unwrap = np.degrees(y_unwrap)
        total = y_unwrap[-1] - y_unwrap[0]
        self.txt_drift.setText(
            f"30 秒内 Yaw 变化约 {total:+.2f}°\n"
            f"起始 {ys[0]:+.2f}° → 结束 {ys[-1]:+.2f}°\n"
            "参考: 静止时 |变化| < 2° 较好；> 5° 说明磁干扰或需校准。"
        )
        self._log(f"Yaw 漂移 30s 变化 {total:+.2f}°")

    def _save_csv_row(self, kind: str, **kw):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        new = not self._session.exists()
        with open(self._session, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "kind", "data"])
            w.writerow([datetime.now().isoformat(), kind, str(kw)])

    def _plot_all(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            from matplotlib import font_manager
        except ImportError:
            QMessageBox.critical(self, "提示", "pip install matplotlib")
            return

        # 解决 Windows 下中文缺字警告（DejaVu Sans 不含大部分中文）
        try:
            preferred = [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "PingFang SC",
            ]
            available = {f.name for f in font_manager.fontManager.ttflist}
            for name in preferred:
                if name in available:
                    rcParams["font.family"] = name
                    break
            rcParams["axes.unicode_minus"] = False
        except Exception:
            # 字体设置失败不影响绘图，只是可能继续出现警告
            pass

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("IMU 自动自检汇总", fontsize=13)

        # 噪声
        ax = axes[0, 0]
        if self._noise_result:
            std = self._noise_result["std"]
            ax.bar(["Roll", "Pitch", "Yaw"], std, color=["#4fc3f7", "#81c784", "#ffb74d"])
            ax.set_ylabel("标准差 (°)")
            ax.set_title("静止噪声 (5s)")
            ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="0.5° 参考线")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "未做噪声测试", ha="center", va="center")
        ax.grid(axis="y", alpha=0.3)

        # 重力对照
        ax = axes[0, 1]
        if self._gravity_poses:
            x = np.arange(len(self._gravity_poses))
            w = 0.35
            dr = [p["d_r"] for p in self._gravity_poses]
            dp = [p["d_p"] for p in self._gravity_poses]
            ax.bar(x - w / 2, dr, w, label="ΔRoll")
            ax.bar(x + w / 2, dp, w, label="ΔPitch")
            ax.set_xticks(x)
            ax.set_xticklabels([p["name"] for p in self._gravity_poses])
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_ylabel("IMU − 加速度计 (°)")
            ax.set_title("重力对照（各姿势差值）")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "未捕获姿势", ha="center", va="center")
        ax.grid(axis="y", alpha=0.3)

        # Yaw 漂移曲线
        ax = axes[1, 0]
        if self._drift_rows:
            ts, ys = zip(*self._drift_rows)
            ax.plot(ts, ys, "-", color="#ffb74d")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("Yaw (°)")
            ax.set_title("30s Yaw 漂移")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "未做漂移测试", ha="center", va="center")

        # 说明
        ax = axes[1, 1]
        ax.axis("off")
        tips = [
            "如何理解（不用量角器）:",
            "",
            "• 噪声小 → 读数稳，适合训练显示",
            "• 重力对照 Δ 大 → 倾角不可信或轴向与 Wit 定义不一致",
            "• 相对转角误差大 → 动态/磁干扰，或转动不是直角",
            "• Yaw 漂移大 → 远离金属，做磁力计校准",
            "",
            "针尖 31° 安装误差 → 需光学跟踪或一次性的",
            "「针轴标定」，本工具测的是 IMU 盒子本身。",
        ]
        ax.text(0.05, 0.95, "\n".join(tips), va="top", fontsize=10)

        plt.tight_layout()
        plt.show()

    def closeEvent(self, event):
        self.reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = ImuAccuracyWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
