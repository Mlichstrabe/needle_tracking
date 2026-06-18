"""
磁力计 A/B 对比工具（9 轴融合 vs 6 轴 / 关磁融合）

用法（项目根目录）:
    python tools/mag_ab_compare.py

WitMotion 上位机操作要点（WT901 / JY901 类）:
  1. 连接 USB，确认串口波特率 115200，输出勾选：欧拉角(0x53)、四元数(0x59)、磁场(0x54)。
  2. **A 段（9 轴）**: 保持默认「九轴/磁场参与融合」→ 本工具录 20 s（尽量静止，可选点「靠近金属」标记）。
  3. **改配置**: Wit 中改为「六轴」或关闭磁场融合 → 断电重连或重启模块。
  4. **B 段（6 轴）**: 重新连接串口 → 再录 20 s，条件与 A 尽量相同。
  5. 点「生成对比报告」查看针轴稳定性、Yaw、|B| 与综合结论；可导出/导入 CSV 离线重绘。

说明:
  - PC 端不融合磁力计；对比的是芯片在不同 Wit 配置下输出的姿态与磁场。
  - 针轴指标与主程序一致，使用 core.imu_kinematics.needle_axis_scene_normalized(四元数)。
  - Yaw 跳变统计对 Yaw 做 unwrap 后再算相邻帧差，避免 ±180° 折叠。
"""
from __future__ import annotations

import csv
import json
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from core.imu_kinematics import needle_axis_scene_normalized

try:
    import serial
except ImportError:
    serial = None

LOG_DIR = _ROOT / "imu_calibration_logs"
CAPTURE_SEC_DEFAULT = 20

# 150s guided script for stressing magnetic gating vs always-on magnetic fusion.
GUIDED_SCRIPT_150 = [
    (0, 10, "1 clean_static", "Keep the IMU completely still in a clean magnetic area.", "clean_static_start"),
    (10, 35, "2 clean_slow_motion", "Slowly rotate the IMU in the clean area.", "clean_slow_motion_start"),
    (35, 50, "3 clean_static_recheck", "Put the IMU down and keep it still again.", "clean_static_recheck_start"),
    (50, 80, "4 disturbed_static", "Keep the IMU still; move metal/phone/magnet from 20cm to 5cm and back.", "disturbed_static_start"),
    (80, 110, "5 disturbed_slow_motion", "Keep the disturbance close; slowly rotate the IMU.", "disturbed_slow_motion_start"),
    (110, 130, "6 recovery_static", "Move the disturbance away; keep the IMU still.", "recovery_static_start"),
    (130, 150, "7 recovery_slow_motion", "In the clean area, slowly rotate the IMU again.", "recovery_slow_motion_start"),
]

CSV_FIELDS = [
    "t",
    "event",
    "roll",
    "pitch",
    "yaw",
    "yaw_step",
    "mx",
    "my",
    "mz",
    "mag_abs",
    "qw",
    "qx",
    "qy",
    "qz",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "needle_nx",
    "needle_ny",
    "needle_nz",
    "needle_step_deg",
]

WORKFLOW_LABELS = [
    "① 连接串口",
    "② 录 A（9 轴）",
    "③ Wit 改 6 轴",
    "④ 录 B（6 轴）",
    "⑤ 出报告",
]


def _configure_matplotlib_chinese() -> Optional[str]:
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


def _dir_angle_deg(d0: np.ndarray, d1: np.ndarray) -> float:
    n0 = np.linalg.norm(d0)
    n1 = np.linalg.norm(d1)
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    dot = float(np.clip(np.dot(d0 / n0, d1 / n1), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _normalize_pct_time(t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros_like(t)
    span = float(t[-1] - t[0])
    if span < 1e-6:
        return np.zeros_like(t)
    return (t - t[0]) / span * 100.0


def _pct_improve(a: float, b: float) -> float:
    """越小越好：正值表示 B 优于 A 的改善百分比。"""
    if not np.isfinite(a) or abs(a) < 1e-9:
        return 0.0
    return float((a - b) / abs(a) * 100.0)


@dataclass
class ImuSample:
    t: float
    euler: Optional[np.ndarray]
    mag: Optional[np.ndarray]
    quat: Optional[np.ndarray]
    acc: Optional[np.ndarray]
    gyro: Optional[np.ndarray]
    mag_count: int
    euler_count: int
    quat_count: int


class MagAbSerialReader(QObject):
    """0x51–0x54、0x59；以 0x59/0x53 触发快照（含缓存字段）。"""

    sample = pyqtSignal(object)
    error = pyqtSignal(str)
    stats = pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self._ser = None
        self._run = False
        self._th: Optional[threading.Thread] = None
        self._mag_count = 0
        self._euler_count = 0
        self._quat_count = 0
        self.cache: Dict[str, Optional[np.ndarray]] = {
            "acc": None,
            "gyro": None,
            "euler": None,
            "quat": None,
            "mag": None,
        }

    @property
    def connected(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    def connect(self, port: str, baud: int = 115200) -> bool:
        if serial is None:
            self.error.emit("未安装 pyserial: pip install pyserial")
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.5)
            self._run = True
            self._mag_count = 0
            self._euler_count = 0
            self._quat_count = 0
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
            elif typ == 0x59:
                q = np.array(
                    [
                        struct.unpack("<h", frame[2:4])[0] / 32768.0,
                        struct.unpack("<h", frame[4:6])[0] / 32768.0,
                        struct.unpack("<h", frame[6:8])[0] / 32768.0,
                        struct.unpack("<h", frame[8:10])[0] / 32768.0,
                    ]
                )
                self.cache["quat"] = q
                self._quat_count += 1
                self._emit()
        except Exception:
            pass

    def _emit(self):
        if self.cache["euler"] is None and self.cache["quat"] is None:
            return
        euler = (
            None if self.cache["euler"] is None else self.cache["euler"].copy()
        )
        mag = None if self.cache["mag"] is None else self.cache["mag"].copy()
        quat = None if self.cache["quat"] is None else self.cache["quat"].copy()
        acc = None if self.cache["acc"] is None else self.cache["acc"].copy()
        gyro = None if self.cache["gyro"] is None else self.cache["gyro"].copy()
        self.sample.emit(
            ImuSample(
                t=time.time(),
                euler=euler,
                mag=mag,
                quat=quat,
                acc=acc,
                gyro=gyro,
                mag_count=self._mag_count,
                euler_count=self._euler_count,
                quat_count=self._quat_count,
            )
        )
        self.stats.emit(self._mag_count, self._euler_count, self._quat_count)


@dataclass
class RecordedSession:
    label: str
    rows: List[dict] = field(default_factory=list)
    events: List[Tuple[float, str]] = field(default_factory=list)


def _row_from_sample(
    s: ImuSample, t_rel: float, prev_yaw: float, prev_needle: Optional[np.ndarray]
) -> dict:
    roll = pitch = yaw = float("nan")
    if s.euler is not None:
        roll, pitch, yaw = float(s.euler[0]), float(s.euler[1]), float(s.euler[2])
    yaw_step = _yaw_delta_deg(prev_yaw, yaw) if np.isfinite(yaw) else 0.0

    mx = my = mz = mag_abs = ""
    if s.mag is not None:
        mx, my, mz = float(s.mag[0]), float(s.mag[1]), float(s.mag[2])
        mag_abs = float(np.linalg.norm(s.mag))

    qw = qx = qy = qz = ""
    needle = None
    needle_step = ""
    if s.quat is not None:
        qw, qx, qy, qz = (float(s.quat[i]) for i in range(4))
        needle = needle_axis_scene_normalized(s.quat)
        if needle is not None and prev_needle is not None:
            needle_step = _dir_angle_deg(
                np.array(prev_needle), np.array(needle)
            )

    ax = ay = az = gx = gy = gz = ""
    if s.acc is not None:
        ax, ay, az = (float(s.acc[i]) for i in range(3))
    if s.gyro is not None:
        gx, gy, gz = (float(s.gyro[i]) for i in range(3))

    return {
        "t": t_rel,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "yaw_step": yaw_step,
        "mx": mx,
        "my": my,
        "mz": mz,
        "mag_abs": mag_abs,
        "qw": qw,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "needle_nx": needle[0] if needle else "",
        "needle_ny": needle[1] if needle else "",
        "needle_nz": needle[2] if needle else "",
        "needle_step_deg": needle_step,
        "_needle_vec": needle,
    }


def compute_session_metrics(rows: List[dict]) -> dict:
    if len(rows) < 5:
        return {"n": len(rows)}

    yaw = np.array([r["yaw"] for r in rows], dtype=float)
    yaw = yaw[np.isfinite(yaw)]
    out: dict = {"n": len(rows)}

    if len(yaw) >= 5:
        yaw_u = np.degrees(np.unwrap(np.radians(yaw)))
        steps_u = np.abs(np.diff(yaw_u))
        out["yaw_std"] = float(np.std(yaw))
        out["yaw_max_step"] = float(np.max(steps_u)) if len(steps_u) else 0.0
        out["yaw_big_jumps"] = int(np.sum(steps_u > 5.0)) if len(steps_u) else 0

    roll = np.array([r["roll"] for r in rows], dtype=float)
    pitch = np.array([r["pitch"] for r in rows], dtype=float)
    roll = roll[np.isfinite(roll)]
    pitch = pitch[np.isfinite(pitch)]
    if len(roll) >= 5 and len(pitch) >= 5:
        out["tilt_std"] = float(np.sqrt(np.std(roll) ** 2 + np.std(pitch) ** 2))

    mag_abs = np.array(
        [r["mag_abs"] for r in rows if r["mag_abs"] != ""], dtype=float
    )
    if len(mag_abs) >= 5:
        out["mag_std"] = float(np.std(mag_abs))
        out["mag_mean"] = float(np.mean(mag_abs))

    nsteps = [
        float(r["needle_step_deg"])
        for r in rows
        if r.get("needle_step_deg") not in ("", None)
    ]
    if len(nsteps) >= 5:
        ns = np.array(nsteps, dtype=float)
        out["needle_step_std"] = float(np.std(ns))
        out["needle_max_step"] = float(np.max(ns))
        out["needle_big_jumps"] = int(np.sum(ns > 5.0))

    return out


def _winner_line(name: str, key: str, sa: dict, sb: dict, lower_better: bool = True) -> str:
    va = sa.get(key)
    vb = sb.get(key)
    if va is None or vb is None:
        return f"{name}: 数据不足"
    if abs(va - vb) < 1e-6:
        return f"{name}: 相当 (A={va:.3g}, B={vb:.3g})"
    if lower_better:
        win = "B" if vb < va else "A"
    else:
        win = "B" if vb > va else "A"
    imp = _pct_improve(va, vb) if lower_better else _pct_improve(vb, va)
    return f"{name}: {win} 胜  (A={va:.3g}, B={vb:.3g}, B相对改善 {imp:+.1f}%)"


def plot_ab_report(sess_a: RecordedSession, sess_b: RecordedSession) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _configure_matplotlib_chinese()
    sa = compute_session_metrics(sess_a.rows)
    sb = compute_session_metrics(sess_b.rows)

    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 0.85], hspace=0.35)
    fig.suptitle("磁力计 A/B 对比报告", fontsize=14)

    def _rows_series(rows: List[dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
        t = np.array([r["t"] for r in rows], dtype=float)
        if key == "yaw":
            y = np.array([r["yaw"] for r in rows], dtype=float)
            y = np.degrees(np.unwrap(np.radians(y)))
        elif key == "mag_abs":
            mask = [r["mag_abs"] != "" for r in rows]
            t = t[mask]
            y = np.array([r["mag_abs"] for r in rows if r["mag_abs"] != ""], dtype=float)
        elif key == "needle_stability":
            t_list, y_list = [], []
            prev = None
            for r in rows:
                if r["needle_nx"] == "":
                    continue
                cur = np.array([r["needle_nx"], r["needle_ny"], r["needle_nz"]])
                if prev is not None:
                    t_list.append(r["t"])
                    y_list.append(_dir_angle_deg(prev, cur))
                prev = cur
            return np.array(t_list), np.array(y_list)
        else:
            y = np.array([r[key] for r in rows], dtype=float)
        return t, y

    plot_specs = [
        ("mag_abs", "|B| 模长", gs[0, 0]),
        ("yaw", "Yaw（unwrap）", gs[0, 1]),
        ("needle_stability", "针轴帧间角变化 (°)", gs[1, 0]),
    ]

    for key, title, spec in plot_specs:
        ax = fig.add_subplot(spec)
        for sess, color in ((sess_a, "#4fc3f7"), (sess_b, "#81c784")):
            t, y = _rows_series(sess.rows, key)
            if len(t) < 2:
                continue
            tp = _normalize_pct_time(t)
            ax.plot(tp, y, color=color, label=sess.label, linewidth=1.1)
            t_span = float(sess.rows[-1]["t"] - sess.rows[0]["t"]) if sess.rows else 1.0
            for te, ev in sess.events:
                if t_span > 1e-6:
                    ax.axvline(te / t_span * 100.0, color="#ffb74d", linestyle=":", alpha=0.7)
        ax.set_title(title + "（时间归一化 0–100%）")
        ax.set_xlabel("进度 (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, 1])
    metrics = [
        ("倾角 σ", "tilt_std"),
        ("Yaw 最大跳变", "yaw_max_step"),
        ("Yaw 大跳变次数", "yaw_big_jumps"),
        ("|B| σ", "mag_std"),
        ("针轴最大跳变", "needle_max_step"),
        ("针轴大跳变次数", "needle_big_jumps"),
    ]
    labels = []
    va_list = []
    vb_list = []
    colors_b = []
    for lbl, k in metrics:
        va = sa.get(k)
        vb = sb.get(k)
        if va is None and vb is None:
            continue
        va = float(va) if va is not None else 0.0
        vb = float(vb) if vb is not None else 0.0
        labels.append(lbl)
        va_list.append(va)
        vb_list.append(vb)
        imp = _pct_improve(va, vb)
        colors_b.append("#66bb6a" if imp > 0 else "#ef5350" if imp < 0 else "#9e9e9e")

    if labels:
        x = np.arange(len(labels))
        w = 0.36
        ax_bar.bar(x - w / 2, va_list, w, label=sess_a.label, color="#4fc3f7")
        bars_b = ax_bar.bar(x + w / 2, vb_list, w, label=sess_b.label, color="#81c784")
        for i, (bar, imp) in enumerate(zip(bars_b, [_pct_improve(a, b) for a, b in zip(va_list, vb_list)])):
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{imp:+.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors_b[i],
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=20, ha="right")
        ax_bar.set_title("指标对比（越小越好；绿色 % 表示 B 更优）")
        ax_bar.legend(fontsize=8)
        ax_bar.grid(axis="y", alpha=0.3)

    ax_verdict = fig.add_subplot(gs[2, :])
    ax_verdict.axis("off")
    lines = [
        f"A — {sess_a.label}  (n={sa.get('n', 0)})",
        f"B — {sess_b.label}  (n={sb.get('n', 0)})",
        "",
        "分项结论:",
        _winner_line("倾角 σ", "tilt_std", sa, sb),
        _winner_line("Yaw 最大跳变", "yaw_max_step", sa, sb),
        _winner_line("Yaw 大跳变(>5°)", "yaw_big_jumps", sa, sb),
        _winner_line("|B| σ", "mag_std", sa, sb),
        _winner_line("针轴最大跳变", "needle_max_step", sa, sb),
        _winner_line("针轴大跳变(>5°)", "needle_big_jumps", sa, sb),
        "",
        "解读提示:",
        "• 训练对准若主要看针轴/倾角，可优先看「针轴」与「倾角 σ」。",
        "• B 的 |B|σ 明显小于 A → 9 轴对磁场更敏感。",
        "• 两段 Yaw 都差 → 环境磁干扰或测试时转动过大。",
        "• 竖线：录制时标记的「靠近金属」事件（按各段时长映射到 0–100%）。",
    ]
    ax_verdict.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)

    plt.tight_layout()
    plt.show()


def export_session(sess: RecordedSession, which: str, folder: Path) -> Tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = folder / f"mag_ab_{which}_{ts}.csv"
    meta_path = folder / f"mag_ab_{which}_{ts}_meta.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sess.rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    meta = {"label": sess.label, "events": sess.events}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, meta_path


def import_session(path: Path) -> RecordedSession:
    label = path.stem
    events: List[Tuple[float, str]] = []
    meta_path = path.parent / f"{path.stem}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = meta.get("label", label)
        events = [(float(t), str(n)) for t, n in meta.get("events", [])]

    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: raw.get(k, "") for k in CSV_FIELDS}
            for k in ("t", "roll", "pitch", "yaw", "yaw_step"):
                if row[k] != "":
                    row[k] = float(row[k])
            for k in ("mx", "my", "mz", "mag_abs", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz", "needle_nx", "needle_ny", "needle_nz", "needle_step_deg"):
                if row[k] != "":
                    row[k] = float(row[k])
            rows.append(row)
    which = "A" if "_A_" in path.name or path.name.endswith("_A.csv") else "B"
    return RecordedSession(label=label or which, rows=rows, events=events)


class MagAbCompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("磁力计 A/B 对比（9 轴 vs 6 轴）")
        self.resize(1100, 820)

        self.reader = MagAbSerialReader()
        self.reader.sample.connect(self._on_sample)
        self.reader.error.connect(self._on_error)
        self.reader.stats.connect(self._on_stats)

        self.session_a = RecordedSession("9轴融合(默认)")
        self.session_b = RecordedSession("6轴/关磁融合")
        self._workflow_step = 0
        self._recording: Optional[str] = None
        self._t_rec0 = 0.0
        self._capture_left = 0
        self._prev_needle: Optional[List[float]] = None
        self._guided_last_phase: Optional[str] = None

        self._mag_baseline: Optional[float] = None
        self._baseline_samples: Deque[float] = deque(maxlen=80)

        self._live_needle_steps: Deque[float] = deque(maxlen=120)
        self._live_yaw_steps: Deque[float] = deque(maxlen=120)
        self._live_mag_b: Deque[float] = deque(maxlen=400)
        self._live_mx: Deque[float] = deque(maxlen=400)
        self._live_my: Deque[float] = deque(maxlen=400)
        self._live_t: Deque[float] = deque(maxlen=400)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()
        self._refresh_ports()
        self._update_workflow_ui()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(80)
        self._plot_timer.timeout.connect(self._refresh_live_plots)
        self._plot_timer.start()

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)

        self._step_labels: List[QLabel] = []
        step_row = QHBoxLayout()
        for i, text in enumerate(WORKFLOW_LABELS):
            lb = QLabel(text)
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(
                "padding:6px;border-radius:4px;background:#2a2f38;color:#888;"
            )
            self._step_labels.append(lb)
            step_row.addWidget(lb, 1)
        root.addLayout(step_row)

        wit = QLabel(
            "③ 请在 Wit 上位机改为六轴/关闭磁场融合后，断电重连，再点「连接」→ 录 B。\n"
            "输出务必含：四元数 0x59、欧拉 0x53、磁场 0x54（A/B 才能比针轴与 |B|）。"
        )
        wit.setWordWrap(True)
        wit.setStyleSheet("color:#90caf9;padding:4px;")
        root.addWidget(wit)

        conn = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(140)
        conn.addWidget(QLabel("串口"))
        conn.addWidget(self.port_combo)
        conn.addWidget(self._btn("刷新", self._refresh_ports))
        conn.addWidget(self._btn("连接", self._connect))
        conn.addWidget(self._btn("断开", self._disconnect))
        conn.addStretch()
        root.addLayout(conn)

        self.lbl_status = QLabel("未连接")
        root.addWidget(self.lbl_status)

        self.lbl_phase = QLabel("阶段: 等待连接")
        self.lbl_phase.setStyleSheet("font-size:15px;font-weight:bold;color:#ffb74d;")
        root.addWidget(self.lbl_phase)

        self.lbl_live = QLabel(
            "抖动分: --  |  |B| 相对基线: --  |  Yaw: --  |  帧 磁/欧/四: 0/0/0"
        )
        root.addWidget(self.lbl_live)

        self.progress = QProgressBar()
        self.progress.setRange(0, CAPTURE_SEC_DEFAULT)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m 秒")
        root.addWidget(self.progress)

        rec = QGridLayout()
        self.edit_label_a = QLineEdit("9轴融合(默认)")
        self.edit_label_b = QLineEdit("6轴/关磁融合")
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(5, 300)
        self.spin_duration.setValue(CAPTURE_SEC_DEFAULT)
        self.spin_duration.setSuffix(" 秒")

        rec.addWidget(QLabel("A 备注"), 0, 0)
        rec.addWidget(self.edit_label_a, 0, 1)
        rec.addWidget(self._btn("一键录 A", lambda: self._start_timed_capture("A")), 0, 2)
        rec.addWidget(self._btn("停止", self._stop_rec), 0, 3)

        rec.addWidget(QLabel("B 备注"), 1, 0)
        rec.addWidget(self.edit_label_b, 1, 1)
        rec.addWidget(self._btn("一键录 B", lambda: self._start_timed_capture("B")), 1, 2)
        rec.addWidget(self._btn("标记:靠近金属", lambda: self._mark_event("靠近金属")), 1, 3)

        rec.addWidget(QLabel("录制时长"), 2, 0)
        rec.addWidget(self.spin_duration, 2, 1)
        rec.addWidget(self._btn("生成对比报告", self._plot_report), 2, 2)
        rec.addWidget(self._btn("导出 CSV", self._export_csv), 2, 3)

        self.chk_guided = QCheckBox("150s 磁干扰分段引导（干净→干扰静止→干扰转动→恢复）")
        self.chk_guided.setToolTip(
            "勾选后：录制时长自动设为 150s，按秒提示当前该做的动作，"
            "并在各阶段自动写入 event，便于离线脚本重点分析 50-110s 磁干扰段。"
        )
        self.chk_guided.stateChanged.connect(self._on_guided_toggle)
        rec.addWidget(self.chk_guided, 3, 0, 1, 4)
        root.addLayout(rec)

        tools = QHBoxLayout()
        tools.addWidget(self._btn("导入 A CSV", lambda: self._import_csv("A")))
        tools.addWidget(self._btn("导入 B CSV", lambda: self._import_csv("B")))
        tools.addWidget(self._btn("磁校准开始", self._mag_calib_start))
        tools.addWidget(self._btn("磁校准结束", self._mag_calib_end))
        tools.addStretch()
        root.addLayout(tools)

        dash = QHBoxLayout()
        self.plot_jitter = pg.PlotWidget(title="实时: 针轴帧间角 (°)")
        self.plot_jitter.showGrid(x=True, y=True, alpha=0.3)
        self.curve_needle = self.plot_jitter.plot([], [], pen=pg.mkPen("#ce93d8", width=2))
        dash.addWidget(self.plot_jitter, 1)

        self.plot_magpct = pg.PlotWidget(title="实时: |B| 与相对基线 %")
        self.plot_magpct.showGrid(x=True, y=True, alpha=0.3)
        self.plot_magpct.addLegend()
        self.curve_mag = self.plot_magpct.plot([], [], pen=pg.mkPen("#4fc3f7", width=2), name="|B|")
        self.curve_magpct = self.plot_magpct.plot(
            [], [], pen=pg.mkPen("#ffb74d", width=2), name="相对基线%"
        )
        dash.addWidget(self.plot_magpct, 1)

        self.plot_xy = pg.PlotWidget(title="实时: 磁场水平 scatter (mx, my)")
        self.plot_xy.setAspectLocked(True)
        self.plot_xy.showGrid(x=True, y=True, alpha=0.3)
        self.curve_xy = self.plot_xy.plot(
            [], [], pen=None, symbol="o", symbolSize=3, symbolBrush="#81c784"
        )
        dash.addWidget(self.plot_xy, 1)
        root.addLayout(dash, stretch=1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        root.addWidget(self.log)

    def _btn(self, text: str, slot):
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
            self._mag_baseline = None
            self._baseline_samples.clear()
            if self.session_a.rows and not self.session_b.rows:
                self._set_workflow_step(3)
                self.lbl_phase.setText("阶段: 已连接 — Wit 改 6 轴后，请录 B")
            else:
                self._set_workflow_step(1)
                self.lbl_phase.setText("阶段: 已连接 — 可开始录 A（保持静止）")

    def _disconnect(self):
        self._stop_rec()
        self.reader.disconnect()
        self.lbl_status.setText("未连接")
        self._set_workflow_step(0)
        self.lbl_phase.setText("阶段: 等待连接")

    def _set_workflow_step(self, step: int):
        self._workflow_step = step
        self._update_workflow_ui()

    def _update_workflow_ui(self):
        for i, lb in enumerate(self._step_labels):
            if i < self._workflow_step:
                style = "padding:6px;border-radius:4px;background:#1b5e20;color:#c8e6c9;"
            elif i == self._workflow_step:
                style = "padding:6px;border-radius:4px;background:#1565c0;color:#fff;font-weight:bold;"
            else:
                style = "padding:6px;border-radius:4px;background:#2a2f38;color:#888;"
            lb.setStyleSheet(style)

    def _on_error(self, msg: str):
        self._log(f"错误: {msg}")

    def _on_stats(self, mag_n: int, euler_n: int, quat_n: int):
        extra = f" | 录制 {self._recording}" if self._recording else ""
        self.lbl_status.setText(
            f"已连接 | 帧 磁/欧/四: {mag_n}/{euler_n}/{quat_n}{extra}"
        )

    def _on_sample(self, s: ImuSample):
        yaw = float(s.euler[2]) if s.euler is not None else float("nan")
        mag_b = float(np.linalg.norm(s.mag)) if s.mag is not None else float("nan")

        if np.isfinite(mag_b):
            self._live_mag_b.append(mag_b)
            self._live_t.append(s.t)
            if s.mag is not None:
                self._live_mx.append(float(s.mag[0]))
                self._live_my.append(float(s.mag[1]))
            if self._mag_baseline is None:
                self._baseline_samples.append(mag_b)
                if len(self._baseline_samples) >= 40:
                    self._mag_baseline = float(np.median(self._baseline_samples))

        needle_step = 0.0
        if s.quat is not None:
            nd = needle_axis_scene_normalized(s.quat)
            if nd is not None and self._prev_needle is not None:
                needle_step = _dir_angle_deg(np.array(self._prev_needle), np.array(nd))
                self._live_needle_steps.append(needle_step)
            if nd is not None:
                self._prev_needle = nd

        if self._recording and s.euler is not None:
            prev_yaw = self.session_a.rows[-1]["yaw"] if (
                self._recording == "A" and self.session_a.rows
            ) else (
                self.session_b.rows[-1]["yaw"]
                if self._recording == "B" and self.session_b.rows
                else yaw
            )
            if isinstance(prev_yaw, str):
                prev_yaw = yaw
            self._live_yaw_steps.append(abs(_yaw_delta_deg(prev_yaw, yaw)))

        jitter = (
            float(np.std(self._live_needle_steps))
            if len(self._live_needle_steps) >= 5
            else float("nan")
        )
        pct = float("nan")
        if self._mag_baseline and np.isfinite(mag_b) and self._mag_baseline > 1e-6:
            pct = (mag_b - self._mag_baseline) / self._mag_baseline * 100.0

        self.lbl_live.setText(
            f"抖动分(针轴σ): {jitter:.2f}°  |  |B|基线偏差: {pct:+.1f}%  |  "
            f"Yaw: {yaw:+.1f}°  |  帧 {s.mag_count}/{s.euler_count}/{s.quat_count}"
        )

        if self._recording:
            sess = self.session_a if self._recording == "A" else self.session_b
            prev_yaw = sess.rows[-1]["yaw"] if sess.rows else yaw
            prev_n = None
            if sess.rows and sess.rows[-1].get("needle_nx") != "":
                prev_n = [
                    sess.rows[-1]["needle_nx"],
                    sess.rows[-1]["needle_ny"],
                    sess.rows[-1]["needle_nz"],
                ]
            row = _row_from_sample(s, s.t - self._t_rec0, prev_yaw, prev_n)
            row.pop("_needle_vec", None)
            row["event"] = ""
            if self.chk_guided.isChecked():
                ph = self._guided_phase(float(row["t"]))
                if ph is not None:
                    row["event"] = ph[2]
            sess.rows.append(row)

    def _refresh_live_plots(self):
        if self._live_needle_steps:
            n = len(self._live_needle_steps)
            self.curve_needle.setData(np.arange(n), np.array(self._live_needle_steps))

        if len(self._live_t) >= 2:
            t0 = self._live_t[0]
            ts = np.array(self._live_t) - t0
            mb = np.array(self._live_mag_b)
            self.curve_mag.setData(ts, mb)
            if self._mag_baseline and self._mag_baseline > 1e-6:
                pct = (mb - self._mag_baseline) / self._mag_baseline * 100.0
                self.curve_magpct.setData(ts, pct)
            if self._live_mx:
                self.curve_xy.setData(
                    list(self._live_mx)[-300:], list(self._live_my)[-300:]
                )

    def _on_guided_toggle(self):
        """开启分段引导时锁定 150s 时长。"""
        if self.chk_guided.isChecked():
            self.spin_duration.setValue(150)
            self.spin_duration.setEnabled(False)
            self._log("已开启 150s 磁干扰分段引导：将按干净→干扰静止→干扰转动→恢复提示并自动标记。")
        else:
            self.spin_duration.setEnabled(True)

    def _guided_phase(self, elapsed: float):
        """返回 elapsed 秒所处的脚本阶段 (start,end,name,action,event)；超范围返回 None。"""
        for ph in GUIDED_SCRIPT_150:
            if ph[0] <= elapsed < ph[1]:
                return ph
        return None

    def _start_timed_capture(self, which: str):
        if not self.reader.connected:
            QMessageBox.warning(self, "提示", "请先连接串口")
            return
        if self._recording:
            QMessageBox.information(self, "提示", "已在录制中")
            return

        dur = int(self.spin_duration.value())
        self.progress.setRange(0, dur)
        self.progress.setValue(0)

        self._recording = which
        self._t_rec0 = time.time()
        self._capture_left = dur
        sess = self.session_a if which == "A" else self.session_b
        sess.rows.clear()
        sess.events.clear()
        label = self.edit_label_a.text() if which == "A" else self.edit_label_b.text()
        sess.label = label or which

        self._prev_needle = None
        self._live_needle_steps.clear()
        self._live_yaw_steps.clear()
        self._guided_last_phase = None

        self._log(f"开始 {dur}s 录制 {which} ({sess.label})")
        if self.chk_guided.isChecked():
            self.lbl_phase.setText("录制开始：1 clean_static — IMU 放稳，保持干净静止 10s")
        else:
            self.lbl_phase.setText(
                f"录制 {which}: 静止保持… 剩余 {dur}s（可点「靠近金属」标记）"
            )
        self._countdown_timer.start()

        if which == "A":
            self._set_workflow_step(1)
        else:
            self._set_workflow_step(3)

    def _on_countdown_tick(self):
        if not self._recording:
            self._countdown_timer.stop()
            return
        dur = int(self.spin_duration.value())
        elapsed = dur - self._capture_left
        self.progress.setValue(elapsed)

        if self.chk_guided.isChecked():
            ph = self._guided_phase(elapsed)
            if ph is not None:
                start, end, name, action, ev = ph
                left_in_phase = int(end - elapsed)
                self.lbl_phase.setText(
                    f"{name}（{action}） — 本阶段剩 {left_in_phase}s ｜ 总剩 {self._capture_left}s"
                )
                # 进入新阶段时自动打标记
                if ev and self._guided_last_phase != name:
                    self._mark_event(ev)
                self._guided_last_phase = name
        else:
            self.lbl_phase.setText(
                f"录制 {self._recording}: 静止保持… 剩余 {self._capture_left}s"
            )
        self._capture_left -= 1
        if self._capture_left < 0:
            self._countdown_timer.stop()
            which = self._recording
            self._stop_rec()
            if which == "A":
                self._set_workflow_step(2)
                self.lbl_phase.setText(
                    "A 完成 — 请在 Wit 改为六轴/关磁后重连，再录 B"
                )
                QMessageBox.information(
                    self,
                    "A 段完成",
                    "请在 WitMotion 中改为六轴或关闭磁场融合，\n"
                    "断电重连后点击「连接」，再点「一键录 B」。",
                )
            else:
                self._set_workflow_step(4)
                self.lbl_phase.setText("B 完成 — 可点「生成对比报告」")
                QMessageBox.information(
                    self,
                    "B 段完成",
                    "两段数据已就绪，请点击「生成对比报告」。",
                )

    def _stop_rec(self):
        if self._recording:
            sess = self.session_a if self._recording == "A" else self.session_b
            self._log(f"停止 {self._recording}，共 {len(sess.rows)} 点")
        self._recording = None
        self._countdown_timer.stop()
        self.progress.setValue(0)

    def _mark_event(self, name: str):
        if not self._recording:
            QMessageBox.information(self, "提示", "请先开始录制 A 或 B")
            return
        sess = self.session_a if self._recording == "A" else self.session_b
        t_rel = time.time() - self._t_rec0
        sess.events.append((t_rel, name))
        self._log(f"标记 [{self._recording}] t={t_rel:.1f}s {name}")

    def _plot_report(self):
        if not self.session_a.rows or not self.session_b.rows:
            QMessageBox.warning(
                self,
                "提示",
                "需要 A、B 两段数据（录制或导入 CSV）。",
            )
            return
        try:
            plot_ab_report(self.session_a, self.session_b)
        except ImportError:
            QMessageBox.critical(self, "提示", "请安装 matplotlib: pip install matplotlib")
            return
        sa = compute_session_metrics(self.session_a.rows)
        sb = compute_session_metrics(self.session_b.rows)
        self._log(
            f"报告 A 针轴大跳变={sa.get('needle_big_jumps', '-')} "
            f"B={sb.get('needle_big_jumps', '-')}"
        )

    def _export_csv(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for which, sess in (("A", self.session_a), ("B", self.session_b)):
            if not sess.rows:
                continue
            p, m = export_session(sess, which, LOG_DIR)
            self._log(f"导出 {p.name} + {m.name}")

    def _import_csv(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"导入 {which} 段 CSV",
            str(LOG_DIR),
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            sess = import_session(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return
        if which == "A":
            self.session_a = sess
            self.edit_label_a.setText(sess.label)
        else:
            self.session_b = sess
            self.edit_label_b.setText(sess.label)
        self._log(f"已导入 {which}: {len(sess.rows)} 点")
        if self.session_a.rows and self.session_b.rows:
            self._set_workflow_step(4)

    def _mag_calib_start(self):
        if self.reader.send_mag_calib_start():
            self._log("磁力计校准开始（约 3 s 内缓慢旋转）")
            QTimer.singleShot(3000, self._mag_calib_end)

    def _mag_calib_end(self):
        if self.reader.send_mag_calib_end():
            self._log("磁力计校准结束")

    def closeEvent(self, event):
        self._stop_rec()
        self.reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MagAbCompareWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
