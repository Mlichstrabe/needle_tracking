"""
磁力计融合对比 GUI —— 一段九轴数据，四条轨迹任选两个直观对比。

直接运行（需 numpy/matplotlib/PyQt5 的环境，如项目 imu-env）:
    python tools/mag_fusion/compare_gui.py

轨迹（同一段九轴 CSV）:
  • 九轴       = 芯片四元数
  • 六轴       = PC 纯陀螺+加速度
  • 芯片自适应融合 = 连续自适应磁融合（q_adaptive/q_pc）
  • 30s脉冲融合    = 平时纯六轴，每 30s 磁质量合格时 0.5s 脉冲校正

默认对比：芯片自适应融合 vs 磁力计常驻。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QWidget,
)

from tools.mag_fusion.fusion import (  # noqa: E402
    AdaptiveMagConfig,
    PeriodicMagConfig,
    estimate_b0_adaptive,
    run_fusion_compare,
    run_fusion_series,
)
from tools.mag_fusion.metrics import compute_track  # noqa: E402
from tools.mag_fusion.plot_compare import plot_pair_comparison, plot_three_comparison  # noqa: E402
from tools.mag_fusion.replay import ReplayTable, load_csv  # noqa: E402

LOG_DIR = _ROOT / "imu_calibration_logs"

#TRACK_NATIVE9 = "原生九轴"
#TRACK_SIX = "纯六轴"
#TRACK_FUSED9 = "融合九轴"

TRACK_NATIVE9 = "新九轴"
TRACK_SIX = "六轴"
TRACK_FUSED9 = "芯片自适应融合"
TRACK_PERIODIC = "30s脉冲融合"
TRACK_FULL_MAG = "磁力计常驻"


def _find_latest_csv() -> Optional[Path]:
    if not LOG_DIR.exists():
        return None
    cands: List[Path] = []
    for pat in ("mag_ab_*.csv", "mag_test_*.csv", "*.csv"):
        cands.extend(LOG_DIR.glob(pat))
    # 排除融合输出的 sidecar
    cands = [p for p in cands if not p.stem.endswith("_fused")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


class CompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("磁力计融合对比 — 三选二")
        self.resize(720, 360)

        self.table: Optional[ReplayTable] = None
        self.t = self.mag = self.gyro = None
        self.q_chip = self.q_pc = self.q_6 = self.q_periodic = self.q_full_mag = self.w_mag = None
        self.w_periodic = self.w_full_mag = None
        self.pulse_fired = self.pulse_skipped = 0

        self._build_ui()

        latest = _find_latest_csv()
        if latest:
            self._load_csv(latest)
            self._generate(auto=True)
        else:
            self._log("未找到 CSV，请点「选择 CSV」。先用 mag_ab_compare 录一段九轴数据。")

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        g = QGridLayout(c)

        # 行0：CSV
        self.lbl_csv = QLabel("（未加载）")
        self.lbl_csv.setStyleSheet("font-weight:bold;")
        g.addWidget(QLabel("数据文件:"), 0, 0)
        g.addWidget(self.lbl_csv, 0, 1, 1, 3)
        btns = QHBoxLayout()
        b1 = QPushButton("选择 CSV")
        b1.clicked.connect(self._pick_csv)
        b2 = QPushButton("加载最新")
        b2.clicked.connect(self._load_latest)
        btns.addWidget(b1)
        btns.addWidget(b2)
        wb = QWidget()
        wb.setLayout(btns)
        g.addWidget(wb, 0, 4)

        # 行1：融合开关 + 参数
        self.chk_fusion = QCheckBox("启用融合算法（融合九轴）")
        self.chk_fusion.setChecked(True)
        self.chk_fusion.stateChanged.connect(self._on_fusion_toggle)
        g.addWidget(self.chk_fusion, 1, 0, 1, 2)

        g.addWidget(QLabel("eps |B|容差:"), 1, 2)
        self.sp_eps = QDoubleSpinBox()
        self.sp_eps.setRange(0.02, 0.5)
        self.sp_eps.setSingleStep(0.01)
        self.sp_eps.setValue(0.15)
        self.sp_eps.valueChanged.connect(self._on_param_changed)
        g.addWidget(self.sp_eps, 1, 3)

        g.addWidget(QLabel("kp_mag 慢校正:"), 2, 2)
        self.sp_kp = QDoubleSpinBox()
        self.sp_kp.setRange(0.0, 1.0)
        self.sp_kp.setSingleStep(0.05)
        self.sp_kp.setValue(0.30)
        self.sp_kp.valueChanged.connect(self._on_param_changed)
        g.addWidget(self.sp_kp, 2, 3)

        g.addWidget(QLabel("脉冲间隔(s):"), 1, 4)
        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(5.0, 120.0)
        self.sp_interval.setSingleStep(5.0)
        self.sp_interval.setValue(30.0)
        self.sp_interval.valueChanged.connect(self._on_param_changed)
        g.addWidget(self.sp_interval, 2, 4)

        # 行2：三选二
        g.addWidget(QLabel("对比 A:"), 3, 0)
        self.cmb_a = QComboBox()
        g.addWidget(self.cmb_a, 3, 1)
        g.addWidget(QLabel("对比 B:"), 4, 0)
        self.cmb_b = QComboBox()
        g.addWidget(self.cmb_b, 4, 1)
        self._refresh_track_options(default=True)

        btn_row = QHBoxLayout()
        self.btn_go = QPushButton("生成对比图")
        self.btn_go.setStyleSheet("font-weight:bold;padding:6px;")
        self.btn_go.clicked.connect(lambda: self._generate(auto=False))
        btn_row.addWidget(self.btn_go)
        self.btn_fusion_ab = QPushButton("三者对比")
        self.btn_fusion_ab.setToolTip("六轴 vs 芯片自适应融合 vs 磁力计常驻")
        self.btn_fusion_ab.clicked.connect(self._compare_three_tracks)
        btn_row.addWidget(self.btn_fusion_ab)
        self.btn_fullmag_ab = QPushButton("自适应 vs 常驻磁")
        self.btn_fullmag_ab.setToolTip("芯片自适应融合 vs 磁力计常驻（w_mag=1，无阻带）")
        self.btn_fullmag_ab.clicked.connect(self._compare_fullmag_ab)
        btn_row.addWidget(self.btn_fullmag_ab)
        bw = QWidget()
        bw.setLayout(btn_row)
        g.addWidget(bw, 4, 2, 1, 3)

        # 行5：日志
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        g.addWidget(self.log, 5, 0, 1, 5)

    def _log(self, msg: str):
        self.log.append(msg)

    def _on_fusion_toggle(self):
        self._refresh_track_options(default=False)
        if self.table is not None:
            self._recompute()

    def _on_param_changed(self, *_args):
        if self.table is not None:
            self._recompute()

    def _refresh_track_options(self, default: bool):
        opts = [TRACK_NATIVE9, TRACK_SIX]
        if self.chk_fusion.isChecked():
            opts = [TRACK_PERIODIC, TRACK_FULL_MAG, TRACK_FUSED9, TRACK_NATIVE9, TRACK_SIX]
        prev_a = self.cmb_a.currentText()
        prev_b = self.cmb_b.currentText()
        for cmb in (self.cmb_a, self.cmb_b):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItems(opts)
            cmb.blockSignals(False)
        if default or prev_a not in opts:
            self.cmb_a.setCurrentText(TRACK_FUSED9 if self.chk_fusion.isChecked() else TRACK_NATIVE9)
        else:
            self.cmb_a.setCurrentText(prev_a)
        if default or prev_b not in opts:
            self.cmb_b.setCurrentText(TRACK_FULL_MAG if self.chk_fusion.isChecked() else TRACK_SIX)
        else:
            self.cmb_b.setCurrentText(prev_b)

    def _compare_fusion_ab(self):
        if not self.chk_fusion.isChecked():
            self.chk_fusion.setChecked(True)
        self.cmb_a.setCurrentText(TRACK_PERIODIC)
        self.cmb_b.setCurrentText(TRACK_FUSED9)
        self._generate(auto=False)

    def _compare_three_tracks(self):
        if self.table is None:
            QMessageBox.warning(self, "提示", "请先加载 CSV。")
            return
        if not self.chk_fusion.isChecked():
            self.chk_fusion.setChecked(True)
        if self.q_pc is None or self.q_full_mag is None or self.q_6 is None:
            self._recompute()
        tracks = [
            (TRACK_SIX, self.q_6, compute_track(TRACK_SIX, self.q_6, gyro=self.gyro)),
            (TRACK_FUSED9, self.q_pc, compute_track(TRACK_FUSED9, self.q_pc, gyro=self.gyro)),
            (TRACK_FULL_MAG, self.q_full_mag, compute_track(TRACK_FULL_MAG, self.q_full_mag, gyro=self.gyro)),
        ]
        try:
            nwin = plot_three_comparison(
                self.table.t,
                tracks,
                mag=self.table.mag,
                title=f"{TRACK_SIX} vs {TRACK_FUSED9} vs {TRACK_FULL_MAG}",
                show=True,
            )
            self._log(f"已打开 {nwin} 个三者对比窗口：六轴 / 芯片自适应融合 / 磁力计常驻")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "绘图失败", str(e))
            self._log(f"三者对比绘图失败: {e}")

    def _compare_fullmag_ab(self):
        if not self.chk_fusion.isChecked():
            self.chk_fusion.setChecked(True)
        self.cmb_a.setCurrentText(TRACK_FUSED9)
        self.cmb_b.setCurrentText(TRACK_FULL_MAG)
        self._generate(auto=False)

    def _pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 IMU CSV", str(LOG_DIR if LOG_DIR.exists() else _ROOT), "CSV (*.csv)"
        )
        if path:
            self._load_csv(Path(path))

    def _load_latest(self):
        latest = _find_latest_csv()
        if latest:
            self._load_csv(latest)
        else:
            QMessageBox.warning(self, "提示", "imu_calibration_logs 下没有 CSV。")

    def _load_csv(self, path: Path):
        try:
            table = load_csv(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", str(e))
            self._log(f"读取失败: {e}")
            return
        self.table = table
        self.t = table.t
        self.mag = table.mag
        self.gyro = table.gyro
        self.q_chip = table.q_chip
        self.lbl_csv.setText(f"{path.name}  ({len(table.t)} 帧, schema={table.schema})")
        self._log(f"已加载 {path.name}，{len(table.t)} 帧。")
        if table.schema != "fusion":
            self._log("注意: 该 CSV 缺四元数列，原生九轴将用欧拉角近似。")
        self._diagnose()
        self._recompute()

    def _diagnose(self):
        """数据质量体检：判断这段数据是否适合做九轴/六轴对比。"""
        t, mag, gyro = self.table.t, self.table.mag, self.table.gyro
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        hz = (1.0 / np.median(dt)) if len(dt) else float("nan")
        mag_abs = np.linalg.norm(mag, axis=1)
        b0, _, _ = estimate_b0_adaptive(t, mag, gyro, AdaptiveMagConfig())
        peak = float(np.nanmax(mag_abs)) if np.any(np.isfinite(mag_abs)) else float("nan")
        ratio = (peak / b0) if (b0 and b0 > 0 and np.isfinite(peak)) else float("inf")
        still_frac = float(np.mean(np.linalg.norm(gyro, axis=1) < 0.12))
        self._log(
            f"体检: ≈{hz:.0f}Hz | |B|基线≈{b0:.0f} 峰值≈{peak:.0f} (×{ratio:.1f}) | "
            f"静止帧 {still_frac*100:.0f}%"
        )
        if ratio > 1.3:
            self._log("⚠ 磁干扰强(|B|峰值>1.3×基线)：磁不可信，九轴优势难体现，建议远离金属重录。")
        elif still_frac < 0.10:
            self._log("⚠ 几乎无静止段：静止漂移指标不可靠，建议开头静止 5 秒再动作。")
        else:
            self._log("✓ 磁环境较干净，适合对比。")

    def _recompute(self):
        if self.table is None:
            return
        adaptive_cfg = AdaptiveMagConfig(eps=self.sp_eps.value(), kp_mag_max=self.sp_kp.value())
        periodic_cfg = PeriodicMagConfig(
            eps=adaptive_cfg.eps,
            b0_window_s=adaptive_cfg.b0_window_s,
            kp_mag_max=adaptive_cfg.kp_mag_max,
            interval_s=self.sp_interval.value(),
        )
        force6 = not self.chk_fusion.isChecked()
        if force6:
            self.q_pc, self.q_6, self.w_mag = run_fusion_series(
                self.table.t,
                self.table.gyro,
                self.table.acc,
                self.table.mag,
                self.table.q_chip,
                cfg=adaptive_cfg,
                force_6dof=True,
            )
            self.q_periodic = self.q_6.copy()
            self.q_full_mag = self.q_6.copy()
            self.w_periodic = np.zeros(len(self.table.t))
            self.w_full_mag = np.zeros(len(self.table.t))
            self.pulse_fired = self.pulse_skipped = 0
            self._log(
                f"已重算: 融合关（仅六轴） eps={adaptive_cfg.eps:.2f} kp_mag={adaptive_cfg.kp_mag_max:.2f}"
            )
            return

        out = run_fusion_compare(
            self.table.t,
            self.table.gyro,
            self.table.acc,
            self.table.mag,
            self.table.q_chip,
            adaptive_cfg=adaptive_cfg,
            periodic_cfg=periodic_cfg,
        )
        self.q_pc = out["q_adaptive"]
        self.q_periodic = out["q_periodic"]
        self.q_6 = out["q_6dof"]
        self.w_mag = out["w_adaptive"]
        self.w_periodic = out["w_periodic"]
        self.q_full_mag, _, self.w_full_mag = run_fusion_series(
            self.table.t,
            self.table.gyro,
            self.table.acc,
            self.table.mag,
            self.table.q_chip,
            cfg=adaptive_cfg,
            force_full_mag=True,
        )
        self.pulse_fired = out["pulse_fired"]
        self.pulse_skipped = out["pulse_skipped"]
        self._log(
            f"已重算: eps={adaptive_cfg.eps:.2f} kp_mag={adaptive_cfg.kp_mag_max:.2f} "
            f"间隔={periodic_cfg.interval_s:.0f}s | "
            f"原融合 w均值={float(np.mean(self.w_mag)):.3f} | "
            f"30s脉冲 触发={self.pulse_fired} 跳过={self.pulse_skipped} "
            f"w均值={float(np.mean(self.w_periodic)):.3f} | "
            f"常驻磁 w均值={float(np.mean(self.w_full_mag)):.3f}"
        )

    def _quats_for(self, name: str):
        if name == TRACK_NATIVE9:
            return self.q_chip
        if name == TRACK_SIX:
            return self.q_6
        if name == TRACK_FUSED9:
            return self.q_pc
        if name == TRACK_PERIODIC:
            return self.q_periodic
        if name == TRACK_FULL_MAG:
            return self.q_full_mag
        return None

    def _generate(self, auto: bool):
        if self.table is None:
            QMessageBox.warning(self, "提示", "请先加载 CSV。")
            return
        if self.q_pc is None:
            self._recompute()
        name_a = self.cmb_a.currentText()
        name_b = self.cmb_b.currentText()
        if name_a == name_b:
            QMessageBox.warning(self, "提示", "请选择两条不同的轨迹。")
            return
        qa = self._quats_for(name_a)
        qb = self._quats_for(name_b)
        if qa is None or qb is None:
            QMessageBox.warning(self, "提示", "所选轨迹不可用（融合未开启？）。")
            return
        ma = compute_track(name_a, qa, gyro=self.gyro)
        mb = compute_track(name_b, qb, gyro=self.gyro)
        self._log(
            f"对比 {name_a} vs {name_b}: "
            f"抖动RMS {ma.needle_jitter_deg_rms:.3f} vs {mb.needle_jitter_deg_rms:.3f}°; "
            f"静止漂移 {ma.yaw_drift_still_deg:.2f} vs {mb.yaw_drift_still_deg:.2f}°"
        )
        try:
            nwin = plot_pair_comparison(
                self.table.t,
                qa,
                qb,
                name_a,
                name_b,
                ma,
                mb,
                mag=self.table.mag,
                events=None,
                title=f"{name_a}  vs  {name_b}",
                show=True,
                multi_window=True,
            )
            self._log(
                f"已打开 {nwin} 个对比窗口：①针轴抖动 ②帧间跳变(突变) "
                f"③Yaw漂移 ④静止放大(长测) ⑤磁场 ⑥柱状图 ⑦结论"
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "绘图失败", str(e))
            self._log(f"绘图失败: {e}")


def main():
    app = QApplication(sys.argv)
    w = CompareWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
