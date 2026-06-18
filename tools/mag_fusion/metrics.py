"""
离线融合结果指标：针轴抖动、Yaw 跳变、磁场统计；可按 event 分段。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.imu_kinematics import needle_axis_scene_normalized  # noqa: E402

from tools.mag_fusion.fusion import quat_angle_deg  # noqa: E402

STILL_GYRO_RAD_S = 0.12  # 判定静止（用于航向漂移指标）


def _yaw_from_quat(q: Sequence[float]) -> float:
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return float(np.degrees(np.arctan2(siny, cosy)))


def _unwrap_yaw_series(yaw_deg: np.ndarray) -> np.ndarray:
    if len(yaw_deg) == 0:
        return yaw_deg
    return np.degrees(np.unwrap(np.radians(yaw_deg)))


@dataclass
class QuaternionTrackMetrics:
    label: str
    n: int = 0
    needle_axis_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    needle_jitter_deg_rms: float = 0.0
    yaw_max_step_deg: float = 0.0
    yaw_big_jumps: int = 0
    yaw_total_change_deg: float = 0.0
    yaw_drift_still_deg: float = 0.0  # 静止段航向游走范围（越小越好；六轴会漂、九轴应被磁拉住）
    yaw_end_drift_deg: float = 0.0  # 终端累计偏移：末段均值 − 起始段均值（长测时六轴会越漂越大）
    n_still: int = 0
    vs_chip_mean_deg: float = 0.0
    vs_chip_max_deg: float = 0.0


@dataclass
class MagFieldMetrics:
    mag_abs_mean: float = 0.0
    mag_abs_std: float = 0.0
    mag_abs_range: float = 0.0
    w_mag_mean: float = 0.0
    w_mag_fraction: float = 0.0


@dataclass
class SegmentMetrics:
    name: str
    t_start: float
    t_end: float
    chip: QuaternionTrackMetrics = field(default_factory=lambda: QuaternionTrackMetrics("chip"))
    pc: QuaternionTrackMetrics = field(default_factory=lambda: QuaternionTrackMetrics("pc"))
    six_dof: QuaternionTrackMetrics = field(default_factory=lambda: QuaternionTrackMetrics("6dof"))
    mag: MagFieldMetrics = field(default_factory=MagFieldMetrics)


def _needle_axes(quats: np.ndarray) -> np.ndarray:
    axes = []
    for q in quats:
        d = needle_axis_scene_normalized(q)
        if d is None:
            axes.append([np.nan, np.nan, np.nan])
        else:
            axes.append(d)
    return np.array(axes, dtype=float)


def _track_metrics(
    label: str,
    quats: np.ndarray,
    ref: Optional[np.ndarray] = None,
    still_mask: Optional[np.ndarray] = None,
) -> QuaternionTrackMetrics:
    m = QuaternionTrackMetrics(label=label, n=len(quats))
    if len(quats) < 2:
        return m

    axes = _needle_axes(quats)
    valid = np.all(np.isfinite(axes), axis=1)
    if int(np.sum(valid)) < 2:
        return m

    a = axes[valid]
    m.needle_axis_std = tuple(float(np.std(a[:, i])) for i in range(3))
    # 相对均值方向的角抖动 RMS
    mean_axis = a.mean(axis=0)
    mn = np.linalg.norm(mean_axis)
    if mn > 1e-6:
        mean_axis = mean_axis / mn
        dots = np.clip(np.sum(a * mean_axis, axis=1), -1.0, 1.0)
        ang = np.degrees(np.arccos(dots))
        m.needle_jitter_deg_rms = float(np.sqrt(np.mean(ang ** 2)))

    yaw = np.array([_yaw_from_quat(q) for q in quats[valid]])
    yaw_u = _unwrap_yaw_series(yaw)
    steps = np.abs(np.diff(yaw_u))
    m.yaw_max_step_deg = float(np.max(steps)) if len(steps) else 0.0
    m.yaw_big_jumps = int(np.sum(steps > 5.0))
    m.yaw_total_change_deg = float(abs(yaw_u[-1] - yaw_u[0])) if len(yaw_u) else 0.0

    # 终端累计偏移：末段均值 − 起始段均值（不依赖静止判定，长测时最能体现六轴单向漂移）
    if len(yaw_u) >= 10:
        k = max(3, len(yaw_u) // 20)  # 取首/尾各 ~5% 帧求均值，抗噪
        m.yaw_end_drift_deg = float(abs(np.mean(yaw_u[-k:]) - np.mean(yaw_u[:k])))

    # 静止段航向漂移：静止时姿态本不应变化，纯六轴会单向漂走，九轴应被磁约束住
    if still_mask is not None and len(still_mask) == len(quats):
        sm = np.asarray(still_mask, dtype=bool)[valid]
        if int(np.sum(sm)) >= 5:
            yw_still = yaw_u[sm]
            m.yaw_drift_still_deg = float(np.max(yw_still) - np.min(yw_still))
            m.n_still = int(np.sum(sm))

    if ref is not None and len(ref) == len(quats):
        angles = []
        for i in range(len(quats)):
            if np.all(np.isfinite(quats[i])) and np.all(np.isfinite(ref[i])):
                angles.append(quat_angle_deg(quats[i], ref[i]))
        if angles:
            m.vs_chip_mean_deg = float(np.mean(angles))
            m.vs_chip_max_deg = float(np.max(angles))

    return m


def compute_track(
    label: str,
    quats: np.ndarray,
    gyro: Optional[np.ndarray] = None,
    ref: Optional[np.ndarray] = None,
) -> QuaternionTrackMetrics:
    """计算单条轨迹指标；传入 gyro 时额外算静止段航向漂移。供 GUI/外部调用。"""
    quats = np.asarray(quats, dtype=float)
    still_mask = None
    if gyro is not None and len(gyro) == len(quats):
        still_mask = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1) < STILL_GYRO_RAD_S
    return _track_metrics(label, quats, ref=ref, still_mask=still_mask)


def compute_mag_metrics(mag: np.ndarray, w_mag: np.ndarray) -> MagFieldMetrics:
    mag_abs = np.linalg.norm(mag, axis=1)
    fin = mag_abs[np.isfinite(mag_abs)]
    mm = MagFieldMetrics()
    if len(fin) < 1:
        return mm
    mm.mag_abs_mean = float(np.mean(fin))
    mm.mag_abs_std = float(np.std(fin)) if len(fin) > 1 else 0.0
    mm.mag_abs_range = float(np.max(fin) - np.min(fin))
    if len(w_mag) == len(fin):
        mm.w_mag_mean = float(np.mean(w_mag))
        mm.w_mag_fraction = float(np.mean(w_mag > 0.5))
    return mm


def compute_all_metrics(
    t: np.ndarray,
    q_chip: np.ndarray,
    q_pc: np.ndarray,
    q_6dof: np.ndarray,
    mag: np.ndarray,
    w_mag: np.ndarray,
    events: Optional[List[Tuple[float, str]]] = None,
    gyro: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """全段 + 可选 event 分段指标。传入 gyro 时额外计算静止段航向漂移。"""
    still_mask: Optional[np.ndarray] = None
    if gyro is not None and len(gyro) == len(q_chip):
        still_mask = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1) < STILL_GYRO_RAD_S

    chip = _track_metrics("q_chip", q_chip, still_mask=still_mask)
    pc = _track_metrics("q_pc", q_pc, ref=q_chip, still_mask=still_mask)
    six = _track_metrics("q_6dof", q_6dof, ref=q_chip, still_mask=still_mask)
    mag_m = compute_mag_metrics(mag, w_mag)

    segments: List[SegmentMetrics] = []
    if events:
        bounds = [float(t[0])] + [te for te, _ in events] + [float(t[-1])]
        names = ["段0"] + [ev for _, ev in events]
        for i in range(len(bounds) - 1):
            t0, t1 = bounds[i], bounds[i + 1]
            mask = (t >= t0) & (t <= t1)
            if int(np.sum(mask)) < 5:
                continue
            sm_seg = still_mask[mask] if still_mask is not None else None
            seg = SegmentMetrics(
                name=names[i] if i < len(names) else f"段{i}",
                t_start=t0,
                t_end=t1,
                chip=_track_metrics("chip", q_chip[mask], still_mask=sm_seg),
                pc=_track_metrics("pc", q_pc[mask], ref=q_chip[mask], still_mask=sm_seg),
                six_dof=_track_metrics("6dof", q_6dof[mask], ref=q_chip[mask], still_mask=sm_seg),
                mag=compute_mag_metrics(mag[mask], w_mag[mask]),
            )
            segments.append(seg)

    return {
        "chip": chip,
        "pc": pc,
        "six_dof": six,
        "mag": mag_m,
        "segments": segments,
    }


def verdict_text(metrics: Dict[str, object]) -> str:
    """根据指标生成简短中文结论，重点对比「优化九轴 q_pc」与「纯六轴 q_6dof」。"""
    chip: QuaternionTrackMetrics = metrics["chip"]
    pc: QuaternionTrackMetrics = metrics["pc"]
    six: QuaternionTrackMetrics = metrics["six_dof"]
    mag: MagFieldMetrics = metrics["mag"]

    lines = [
        "三方对比（q_pc=优化九轴 / q_6dof=纯六轴 / q_chip=芯片九轴）:",
        f"  芯片九轴 : 抖动RMS={chip.needle_jitter_deg_rms:.3f}°  "
        f"Yaw最大步={chip.yaw_max_step_deg:.2f}°  静止漂移={chip.yaw_drift_still_deg:.2f}°",
        f"  纯六轴   : 抖动RMS={six.needle_jitter_deg_rms:.3f}°  "
        f"Yaw最大步={six.yaw_max_step_deg:.2f}°  静止漂移={six.yaw_drift_still_deg:.2f}°",
        f"  优化九轴 : 抖动RMS={pc.needle_jitter_deg_rms:.3f}°  "
        f"Yaw最大步={pc.yaw_max_step_deg:.2f}°  静止漂移={pc.yaw_drift_still_deg:.2f}°",
        f"  终端累计偏移(越小越好): 芯片={chip.yaw_end_drift_deg:.1f}°  "
        f"六轴={six.yaw_end_drift_deg:.1f}°  优化九轴={pc.yaw_end_drift_deg:.1f}°",
        f"  磁修正平均权重≈{mag.w_mag_mean:.2f}（启用比例 {mag.w_mag_fraction * 100:.0f}%）。",
        "",
    ]

    # 短期稳定性：优化九轴是否追平六轴（容许 +10%）
    short_ok = pc.needle_jitter_deg_rms <= six.needle_jitter_deg_rms * 1.10
    # 长期不漂：静止航向漂移是否更小
    has_drift = (pc.n_still >= 5 and six.n_still >= 5)
    drift_win = has_drift and pc.yaw_drift_still_deg < six.yaw_drift_still_deg

    if short_ok and drift_win:
        lines.append("结论: 优化九轴短期抖动追平六轴，且静止航向漂移更小 → 各方面不劣于、长期优于六轴。")
    elif short_ok and not has_drift:
        lines.append("结论: 优化九轴短期已追平六轴；本段缺足够静止帧，无法评估长期漂移，建议补录开头静止 5s。")
    elif short_ok and not drift_win:
        lines.append("结论: 短期已追平，但静止漂移未占优；可适当增大 kp_mag（慢校正更强）或检查磁校准。")
    else:
        lines.append("结论: 优化九轴短期抖动仍大于六轴；建议减小 kp_mag、收紧 eps/dip_tol，或确认磁力计已校准。")

    if has_drift and six.yaw_drift_still_deg > 1.0:
        lines.append(f"提示: 纯六轴静止漂移达 {six.yaw_drift_still_deg:.2f}°，正是九轴用磁的价值所在。")

    return "\n".join(lines)
