"""
Mahony 6DOF 姿态积分 + 自适应磁力计航向修正。

四元数约定: [w, x, y, z]（与 JY901 / 主程序一致）。
单位: 陀螺 rad/s，加速度 m/s²，磁场为模块原始计数（仅用于相对变化）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

G_NORM = 9.80665


@dataclass
class AdaptiveMagConfig:
    """
    自适应磁融合参数。

    默认 use_chip_guided=True：运动/磁脏时贴近芯片九轴，仅静止且磁干净时
    才用低增益 Mahony 磁修正，从而与「常开磁 Mahony」拉开差距并接近 Wit 九轴。
    """

    eps: float = 0.12  # |B| 容差（略收紧）
    b0_window_s: float = 3.0
    still_gyro_rad_s: float = 0.12
    dB_std_factor: float = 3.0
    dip_tol_deg: float = 15.0
    kp_mag_max: float = 0.12  # 仅静止磁净时参与，增益宜小
    kp_acc: float = 2.0
    ki: float = 0.0
    # --- v2：与常开磁区分、贴近芯片 ---
    use_chip_guided: bool = True
    mag_only_when_still: bool = True  # 运动时 w=0，避免动态磁拉扯
    weight_exponent: float = 3.0  # w^exp，强压低中等质量段的磁参与
    min_mag_quality: float = 0.18  # 低于该质量的磁修正直接关闭
    mag_dir_tol_deg: float = 18.0  # 芯片姿态下的磁场方向偏离阈值
    chip_blend_base: float = 0.97  # 常态贴近芯片九轴
    chip_blend_mag_scale: float = 0.03  # w→0 时≈100% 芯片；w→1 时≈97% 芯片 + 3% Mahony磁
    max_deviation_from_chip_deg: float = 2.0  # 限制单帧磁修正相对芯片的偏离
    dirty_six_axis_fallback: float = 0.50  # 脏磁时混入六轴，避免芯片/PC磁航向被干扰拖走
    fallback_weight_exponent: float = 2.0  # 磁质量越高，越快回到芯片自适应输出


@dataclass
class PeriodicMagConfig(AdaptiveMagConfig):
    """30s 脉冲磁融合：平时纯六轴，每隔 interval_s 在磁可信时短脉冲校正。"""

    interval_s: float = 30.0  # 两次校正尝试的最小间隔
    pulse_s: float = 0.5  # 单次脉冲持续时长（秒）
    quality_min: float = 0.5  # 触发时刻自适应质量分阈值（0~1）
    kp_mag_pulse: float = 0.45  # 脉冲内磁增益（可略高于连续融合）


def _initial_quat(q_chip: np.ndarray) -> np.ndarray:
    for i in range(len(q_chip)):
        qi = q_chip[i]
        if np.all(np.isfinite(qi)) and np.linalg.norm(qi) > 0.5:
            return qi.copy()
    return np.array([1.0, 0.0, 0.0, 0.0])


def _integrate_fusion(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q0: np.ndarray,
    w_mag: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> np.ndarray:
    """Mahony 积分，w_mag 逐帧乘 kp_mag_max。"""
    n = len(t)
    q_out = np.zeros((n, 4), dtype=float)
    fus = MahonyFusion(q0, kp_acc=cfg.kp_acc, ki=cfg.ki, kp_mag=0.0)
    use_pulse_scale = isinstance(cfg, PeriodicMagConfig)
    for i in range(n):
        dt = float(t[i] - t[i - 1]) if i > 0 else 0.01
        if i == 0:
            dt = 0.01
        km = float(w_mag[i])
        if use_pulse_scale and km > 0:
            fus.kp_mag = cfg.kp_mag_pulse * km
        else:
            fus.kp_mag = cfg.kp_mag_max * km
        m = mag[i] if np.all(np.isfinite(mag[i])) else None
        q_out[i] = fus.step(gyro[i], acc[i], m, dt)
    return q_out


def _integrate_chip_guided_fusion(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q_chip: np.ndarray,
    q0: np.ndarray,
    w_mag: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> np.ndarray:
    """
    芯片引导自适应：Mahony(加权磁) 后按磁可信度与芯片 slerp，并限制相对芯片偏差。
    运动/磁脏 → 接近 q_chip；静止磁净 → 允许小幅 PC 磁慢校正。
    """
    n = len(t)
    q_out = np.zeros((n, 4), dtype=float)
    fus = MahonyFusion(q0, kp_acc=cfg.kp_acc, ki=cfg.ki, kp_mag=0.0)
    for i in range(n):
        dt = float(t[i] - t[i - 1]) if i > 0 else 0.01
        if i == 0:
            dt = 0.01

        if np.all(np.isfinite(q_chip[i])) and np.linalg.norm(q_chip[i]) > 0.5:
            q_ref = quat_normalize(q_chip[i])
        else:
            q_ref = fus.q.copy()

        w = float(w_mag[i])
        blend = cfg.chip_blend_base + cfg.chip_blend_mag_scale * (1.0 - w)
        blend = float(np.clip(blend, 0.0, 1.0))

        # 运动或磁不可信：直接跟芯片（与常开磁拉开差距）
        if blend >= 0.999:
            q_out[i] = q_ref
            fus.q = q_ref.copy()
            continue

        # 每帧以芯片为锚点积分，避免 Mahony 独自漂到 180°
        fus.q = q_ref.copy()
        fus.kp_mag = cfg.kp_mag_max * w
        m = mag[i] if np.all(np.isfinite(mag[i])) else None
        q_m = fus.step(gyro[i], acc[i], m, dt)
        if float(np.dot(q_m, q_ref)) < 0.0:
            q_m = -q_m

        q_blend = quat_slerp(q_m, q_ref, blend)

        cap = cfg.max_deviation_from_chip_deg
        if cap > 1e-6:
            dev = quat_angle_deg(q_blend, q_ref)
            if dev > cap:
                q_blend = quat_slerp(q_ref, q_blend, cap / dev)

        q_out[i] = q_blend
        fus.q = q_blend.copy()
    return q_out


def _blend_dirty_mag_with_six_axis(
    q_adaptive: np.ndarray,
    q_6dof: np.ndarray,
    w_mag: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> np.ndarray:
    """磁质量差时向六轴回退；磁质量好时回到芯片自适应输出。"""
    if len(q_adaptive) != len(q_6dof) or len(w_mag) != len(q_adaptive):
        return q_adaptive

    floor = float(np.clip(cfg.dirty_six_axis_fallback, 0.0, 1.0))
    exp = max(float(cfg.fallback_weight_exponent), 0.01)
    if floor >= 1.0 - 1e-9:
        return q_adaptive

    out = np.zeros_like(q_adaptive)
    for i in range(len(q_adaptive)):
        if not (
            np.all(np.isfinite(q_adaptive[i]))
            and np.all(np.isfinite(q_6dof[i]))
            and np.linalg.norm(q_adaptive[i]) > 0.5
            and np.linalg.norm(q_6dof[i]) > 0.5
        ):
            out[i] = q_adaptive[i]
            continue
        quality = float(np.clip(w_mag[i], 0.0, 1.0))
        chip_share = floor + (1.0 - floor) * (quality ** exp)
        out[i] = quat_slerp(q_6dof[i], q_adaptive[i], chip_share)
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton 积 q⊗r，均为 [w,x,y,z]。"""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=float,
    )


def rotate_vector_by_quat(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate body-frame vector v by quaternion q=[w,x,y,z]."""
    q0, q1, q2, q3 = quat_normalize(np.asarray(q, dtype=float))
    bx, by, bz = np.asarray(v, dtype=float)
    return np.array(
        [
            (1 - 2 * (q2 * q2 + q3 * q3)) * bx
            + 2 * (q1 * q2 - q0 * q3) * by
            + 2 * (q1 * q3 + q0 * q2) * bz,
            2 * (q1 * q2 + q0 * q3) * bx
            + (1 - 2 * (q1 * q1 + q3 * q3)) * by
            + 2 * (q2 * q3 - q0 * q1) * bz,
            2 * (q1 * q3 - q0 * q2) * bx
            + 2 * (q2 * q3 + q0 * q1) * by
            + (1 - 2 * (q1 * q1 + q2 * q2)) * bz,
        ],
        dtype=float,
    )


def quat_angle_deg(q0: Sequence[float], q1: Sequence[float]) -> float:
    """两姿态四元数夹角（度）。"""
    a = quat_normalize(np.asarray(q0, dtype=float))
    b = quat_normalize(np.asarray(q1, dtype=float))
    dot = min(1.0, abs(float(np.dot(a, b))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    """球面插值，t=0→q0，t=1→q1。"""
    a = quat_normalize(np.asarray(q0, dtype=float))
    b = quat_normalize(np.asarray(q1, dtype=float))
    t = float(np.clip(t, 0.0, 1.0))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    s = float(np.sin(theta))
    if s < 1e-9:
        return a.copy()
    s0 = float(np.sin((1.0 - t) * theta) / s)
    s1 = float(np.sin(t * theta) / s)
    return quat_normalize(s0 * a + s1 * b)


class MahonyFusion:
    """Mahony AHRS：陀螺积分 + 加速度倾角 + 可选磁航向。"""

    def __init__(
        self,
        q0: Optional[Sequence[float]] = None,
        *,
        kp_acc: float = 2.0,
        ki: float = 0.0,
        kp_mag: float = 0.0,
    ):
        self.q = quat_normalize(
            np.asarray(q0 if q0 is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
        )
        self.kp_acc = float(kp_acc)
        self.ki = float(ki)
        self.kp_mag = float(kp_mag)
        self._e_int = np.zeros(3, dtype=float)

    def step(
        self,
        gyro: Sequence[float],
        acc: Sequence[float],
        mag: Optional[Sequence[float]],
        dt: float,
    ) -> np.ndarray:
        if dt <= 0.0 or not np.isfinite(dt):
            return self.q.copy()

        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = acc

        e = np.zeros(3, dtype=float)
        an = float(np.linalg.norm([ax, ay, az]))
        if an > 0.5 * G_NORM:
            axn, ayn, azn = ax / an, ay / an, az / an
            # 估计重力在体坐标的方向（由 q 推出）
            qw, qx, qy, qz = q
            vx = 2.0 * (qx * qz - qw * qy)
            vy = 2.0 * (qw * qx + qy * qz)
            vz = qw * qw - qx * qx - qy * qy + qz * qz
            ex = ayn * vz - azn * vy
            ey = azn * vx - axn * vz
            ez = axn * vy - ayn * vx
            e += np.array([ex, ey, ez])

        if mag is not None and self.kp_mag > 1e-9:
            mx, my, mz = mag
            mn = float(np.linalg.norm([mx, my, mz]))
            if mn > 1e-6:
                mx, my, mz = mx / mn, my / mn, mz / mn
                qw, qx, qy, qz = q
                # 磁场旋转到水平面（Mahony 经典形式）
                hx = 2.0 * mx * (0.5 - qy * qy - qz * qz) + 2.0 * my * (qx * qy - qw * qz) + 2.0 * mz * (
                    qx * qz + qw * qy
                )
                hy = 2.0 * mx * (qx * qy + qw * qz) + 2.0 * my * (0.5 - qx * qx - qz * qz) + 2.0 * mz * (
                    qy * qz - qw * qx
                )
                bx = float(np.sqrt(hx * hx + hy * hy))
                bz = 2.0 * mx * (qx * qz - qw * qy) + 2.0 * my * (qy * qz + qw * qx) + 2.0 * mz * (
                    0.5 - qx * qx - qy * qy
                )
                wx = 2.0 * bx * (0.5 - qy * qy - qz * qz) + 2.0 * bz * (qx * qz - qw * qy)
                wy = 2.0 * bx * (qx * qy - qw * qz) + 2.0 * bz * (qw * qx + qy * qz)
                wz = 2.0 * bx * (qw * qy + qx * qz) + 2.0 * bz * (0.5 - qx * qx - qy * qy)
                ex = my * wz - mz * wy
                ey = mz * wx - mx * wz
                ez = mx * wy - my * wx
                e += np.array([ex, ey, ez])

        if self.ki > 0:
            self._e_int += e * dt
            self._e_int = np.clip(self._e_int, -0.5, 0.5)

        g_corr = np.array([gx, gy, gz]) + self.kp_acc * e + self.ki * self._e_int
        qa = np.array([0.0, g_corr[0], g_corr[1], g_corr[2]], dtype=float)
        q_dot = 0.5 * quat_multiply(q, qa)
        self.q = quat_normalize(q + q_dot * dt)
        return self.q.copy()


def _still_mask(gyro: np.ndarray, thresh: float) -> np.ndarray:
    gnorm = np.linalg.norm(gyro, axis=1)
    return gnorm < thresh


def _dip_angle_deg(acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """逐帧加速度（重力方向）与磁场向量夹角（度）。静止时即磁倾角的余角，应近似恒定。"""
    a = np.asarray(acc, dtype=float)
    m = np.asarray(mag, dtype=float)
    an = np.linalg.norm(a, axis=1)
    mn = np.linalg.norm(m, axis=1)
    out = np.full(len(a), np.nan)
    valid = (an > 1e-6) & (mn > 1e-6)
    dots = np.sum(a[valid] * m[valid], axis=1) / (an[valid] * mn[valid])
    dots = np.clip(dots, -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(dots))
    return out


def _mag_world_dir_weights(
    t: np.ndarray,
    mag: np.ndarray,
    q_ref: Optional[np.ndarray],
    gyro: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> np.ndarray:
    """Gate magnetic correction by field direction in the chip quaternion frame."""
    n = len(t)
    if q_ref is None or len(q_ref) != n:
        return np.ones(n, dtype=float)

    dirs = np.full((n, 3), np.nan, dtype=float)
    for i in range(n):
        if not (
            np.all(np.isfinite(q_ref[i]))
            and np.linalg.norm(q_ref[i]) > 0.5
            and np.all(np.isfinite(mag[i]))
        ):
            continue
        v = rotate_vector_by_quat(q_ref[i], mag[i])
        vn = float(np.linalg.norm(v))
        if vn > 1e-6:
            dirs[i] = v / vn

    t0 = float(t[0]) if n else 0.0
    in_win = (t - t0) <= cfg.b0_window_s
    still = _still_mask(gyro, cfg.still_gyro_rad_s)
    sel = in_win & still & np.all(np.isfinite(dirs), axis=1)
    if int(np.sum(sel)) < 5:
        sel = in_win & np.all(np.isfinite(dirs), axis=1)
    if int(np.sum(sel)) < 5:
        return np.ones(n, dtype=float)

    d0 = np.nanmedian(dirs[sel], axis=0)
    d0n = float(np.linalg.norm(d0))
    if d0n <= 1e-6 or not np.all(np.isfinite(d0)):
        return np.ones(n, dtype=float)
    d0 = d0 / d0n

    w = np.zeros(n, dtype=float)
    valid = np.all(np.isfinite(dirs), axis=1)
    dots = np.clip(np.sum(dirs[valid] * d0, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    w[valid] = _soft(ang / max(cfg.mag_dir_tol_deg, 1e-6))
    return w


def estimate_b0_adaptive(
    t: np.ndarray,
    mag: np.ndarray,
    gyro: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> Tuple[float, float, int]:
    """
    在前 b0_window_s 内的静止段估计 B0 与 dB/dt 阈值。
    返回 (B0, dB_thresh, 用于估计的样本数)。
    """
    mag_abs = np.linalg.norm(mag, axis=1)
    still = _still_mask(gyro, cfg.still_gyro_rad_s)
    t0 = float(t[0]) if len(t) else 0.0
    in_win = (t - t0) <= cfg.b0_window_s
    sel = still & in_win & np.isfinite(mag_abs)
    if int(np.sum(sel)) < 5:
        sel = in_win & np.isfinite(mag_abs)
    if int(np.sum(sel)) < 3:
        sel = np.isfinite(mag_abs)
    vals = mag_abs[sel]
    if len(vals) < 1:
        return float(np.nanmedian(mag_abs)), float("inf"), 0
    b0 = float(np.median(vals))
    b_std = float(np.std(vals)) if len(vals) > 1 else 0.0
    dB_thresh = cfg.dB_std_factor * max(b_std, 1.0)
    return b0, dB_thresh, int(np.sum(sel))


def _estimate_dip0(
    t: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    gyro: np.ndarray,
    cfg: AdaptiveMagConfig,
) -> float:
    """静止窗口内的磁倾角基线（中位数）。无法估计时返回 nan（届时跳过 dip 门控）。"""
    dip = _dip_angle_deg(acc, mag)
    still = _still_mask(gyro, cfg.still_gyro_rad_s)
    t0 = float(t[0]) if len(t) else 0.0
    in_win = (t - t0) <= cfg.b0_window_s
    sel = still & in_win & np.isfinite(dip)
    if int(np.sum(sel)) < 5:
        sel = np.isfinite(dip)
    if int(np.sum(sel)) < 3:
        return float("nan")
    return float(np.median(dip[sel]))


def _soft(x: np.ndarray) -> np.ndarray:
    """把已归一化的「偏离度」x≥0 映射到权重 ∈[0,1]：x=0→1，x≥1→0，中间线性。"""
    return np.clip(1.0 - x, 0.0, 1.0)


def adaptive_mag_weights(
    t: np.ndarray,
    mag: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    cfg: AdaptiveMagConfig,
    q_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    逐帧连续软权重 w_mag∈[0,1]（优化九轴的核心）。

    由三项相乘：
      1) 模长一致性  —— |B| 越接近 B0 越可信；
      2) 变化率平稳  —— d|B|/dt 越小越可信；
      3) 磁倾角一致性 —— 当前 dip 越接近静止基线 dip0 越可信（识别铁磁干扰）。
    连续加权避免了硬 0/1 切换在阈值边界的抖动。
    """
    n = len(t)
    w = np.zeros(n, dtype=float)
    mag_abs = np.linalg.norm(mag, axis=1)
    b0, dB_thresh, _ = estimate_b0_adaptive(t, mag, gyro, cfg)
    if not np.isfinite(b0) or b0 <= 0:
        return w, b0, dB_thresh

    # 1) 模长偏离度：|B| 偏离 B0 超过 eps·B0 即衰减到 0
    band_dev = np.abs(mag_abs - b0) / max(cfg.eps * b0, 1e-6)
    w_band = _soft(band_dev)

    # 2) 变化率偏离度
    dB = np.zeros(n, dtype=float)
    dB[1:] = np.abs(np.diff(mag_abs))
    dt = np.diff(t)
    dt = np.where(dt > 1e-6, dt, np.nan)
    rate = np.zeros(n, dtype=float)
    rate[1:] = dB[1:] / dt
    if np.isfinite(dB_thresh) and dB_thresh > 0:
        w_rate = _soft(rate / dB_thresh)
    else:
        w_rate = np.ones(n, dtype=float)

    # 3) 磁倾角一致性
    dip0 = _estimate_dip0(t, acc, mag, gyro, cfg)
    if np.isfinite(dip0):
        dip = _dip_angle_deg(acc, mag)
        dip_dev = np.abs(dip - dip0) / max(cfg.dip_tol_deg, 1e-6)
        w_dip = _soft(dip_dev)
        w_dip[~np.isfinite(dip)] = 0.0
    else:
        w_dip = np.ones(n, dtype=float)

    w_dir = _mag_world_dir_weights(t, mag, q_ref, gyro, cfg)

    w = w_band * w_rate * w_dip * w_dir
    w[~np.isfinite(mag_abs)] = 0.0
    if cfg.mag_only_when_still:
        w = w * _still_mask(gyro, cfg.still_gyro_rad_s).astype(float)
    exp = float(cfg.weight_exponent)
    if abs(exp - 1.0) > 1e-6:
        w = np.power(np.clip(w, 0.0, 1.0), exp)
    w[w < cfg.min_mag_quality] = 0.0
    # 兜底：重复/异常时间戳(dt≈0)会让变化率项变 nan/inf，污染整段融合 → 一律视为不可信(权重0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    return w, b0, dB_thresh


def full_mag_weights(mag: np.ndarray) -> np.ndarray:
    """
    直通磁融合权重：有有效磁场则 w_mag=1，不做 |B|/变化率/倾角 门控（无阻带）。
  用于对比「自适应软加权」与「普通常开磁力计修正」。
    """
    mag_abs = np.linalg.norm(mag, axis=1)
    w = np.ones(len(mag), dtype=float)
    w[~np.isfinite(mag_abs) | (mag_abs <= 1e-6)] = 0.0
    return w


def periodic_mag_weights(
    t: np.ndarray,
    mag: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    cfg: PeriodicMagConfig,
) -> Tuple[np.ndarray, float, float, int, int]:
    """
    30s 脉冲磁权重：平时 0；每到 interval 且磁质量合格时，开启 pulse_s 窗口。
    返回 (w, b0, dB_thresh, 触发次数, 跳过次数[磁脏])。
    """
    n = len(t)
    w = np.zeros(n, dtype=float)
    w_qual, b0, dB_thresh = adaptive_mag_weights(t, mag, gyro, acc, cfg)
    if n == 0:
        return w, b0, dB_thresh, 0, 0

    t0 = float(t[0])
    t_end_all = float(t[-1])
    fired = skipped = 0
    k = 0
    while True:
        t_trig = t0 + k * cfg.interval_s
        if t_trig > t_end_all + 1e-6:
            break
        i = int(np.searchsorted(t, t_trig, side="left"))
        if i >= n:
            break
        # 触发后 1s 内取质量峰值，避免恰落在 w=0 的帧上
        win = (t >= t_trig) & (t < t_trig + 1.0)
        if not np.any(win):
            k += 1
            continue
        q = float(np.max(w_qual[win]))
        if q >= cfg.quality_min:
            mask = (t >= t_trig) & (t <= t_trig + cfg.pulse_s)
            w[mask] = np.maximum(w[mask], q)
            fired += 1
        else:
            skipped += 1
        k += 1

    return w, b0, dB_thresh, fired, skipped


def run_fusion_series(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q_chip: np.ndarray,
    *,
    cfg: Optional[AdaptiveMagConfig] = None,
    force_6dof: bool = False,
    force_full_mag: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对整段序列融合，返回 (q_pc, q_6dof, w_mag)。

    - 默认: q_pc 为连续自适应磁融合（软权重门控）
    - force_full_mag: q_pc 为常开磁 Mahony（w_mag=1，无阻带）
    - force_6dof: q_pc 同 q_6dof，均不用磁
    """
    cfg = cfg or AdaptiveMagConfig()
    q0 = _initial_quat(q_chip)

    if force_6dof:
        w_mag = np.zeros(len(t), dtype=float)
    elif force_full_mag:
        w_mag = full_mag_weights(mag)
    else:
        q_gate = q_chip if cfg.use_chip_guided else None
        w_mag, _, _ = adaptive_mag_weights(t, mag, gyro, acc, cfg, q_ref=q_gate)

    w_6 = np.zeros(len(t), dtype=float)
    q_6 = _integrate_fusion(t, gyro, acc, mag, q0, w_6, cfg)

    if force_6dof:
        q_pc = q_6.copy()
    elif force_full_mag or not cfg.use_chip_guided:
        q_pc = _integrate_fusion(t, gyro, acc, mag, q0, w_mag, cfg)
    else:
        q_pc = _integrate_chip_guided_fusion(
            t, gyro, acc, mag, q_chip, q0, w_mag, cfg
        )
        q_pc = _blend_dirty_mag_with_six_axis(q_pc, q_6, w_mag, cfg)
    return q_pc, q_6, w_mag


def legacy_adaptive_config(cfg: Optional[AdaptiveMagConfig] = None) -> AdaptiveMagConfig:
    """旧版纯 Mahony 软加权（无芯片回拉、运动时也允许磁）。"""
    base = cfg or AdaptiveMagConfig()
    return AdaptiveMagConfig(
        eps=base.eps,
        b0_window_s=base.b0_window_s,
        still_gyro_rad_s=base.still_gyro_rad_s,
        dB_std_factor=base.dB_std_factor,
        dip_tol_deg=base.dip_tol_deg,
        kp_mag_max=0.30,
        kp_acc=base.kp_acc,
        ki=base.ki,
        use_chip_guided=False,
        mag_only_when_still=False,
        weight_exponent=1.0,
        chip_blend_base=0.0,
        chip_blend_mag_scale=0.0,
        max_deviation_from_chip_deg=180.0,
    )


def run_adaptive_vs_fullmag(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q_chip: np.ndarray,
    *,
    cfg: Optional[AdaptiveMagConfig] = None,
    include_legacy: bool = False,
) -> dict:
    """自适应(v2 芯片引导) vs 常开磁 Mahony；可选附带旧版自适应。"""
    cfg = cfg or AdaptiveMagConfig()
    q_adapt, q_6, w_adapt = run_fusion_series(
        t, gyro, acc, mag, q_chip, cfg=cfg
    )
    q_full, _, w_full = run_fusion_series(
        t, gyro, acc, mag, q_chip, cfg=cfg, force_full_mag=True
    )
    out = {
        "q_chip": q_chip,
        "q_adaptive": q_adapt,
        "q_full_mag": q_full,
        "q_6dof": q_6,
        "w_adaptive": w_adapt,
        "w_full_mag": w_full,
        "cfg": cfg,
    }
    if include_legacy:
        leg = legacy_adaptive_config(cfg)
        q_leg, _, w_leg = run_fusion_series(
            t, gyro, acc, mag, q_chip, cfg=leg
        )
        out["q_adaptive_legacy"] = q_leg
        out["w_adaptive_legacy"] = w_leg
    return out


def run_periodic_fusion_series(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q_chip: np.ndarray,
    *,
    cfg: Optional[PeriodicMagConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """30s 脉冲磁融合。返回 (q_periodic, q_6dof, w_mag, 触发次数, 跳过次数)。"""
    cfg = cfg or PeriodicMagConfig()
    q0 = _initial_quat(q_chip)
    w_mag, _, _, fired, skipped = periodic_mag_weights(t, mag, gyro, acc, cfg)
    q_pc = _integrate_fusion(t, gyro, acc, mag, q0, w_mag, cfg)
    w_6 = np.zeros(len(t), dtype=float)
    q_6 = _integrate_fusion(t, gyro, acc, mag, q0, w_6, cfg)
    return q_pc, q_6, w_mag, fired, skipped


def run_fusion_compare(
    t: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    q_chip: np.ndarray,
    *,
    adaptive_cfg: Optional[AdaptiveMagConfig] = None,
    periodic_cfg: Optional[PeriodicMagConfig] = None,
) -> dict:
    """一次算出：连续融合、30s脉冲融合、纯六轴、芯片四元数。"""
    adaptive_cfg = adaptive_cfg or AdaptiveMagConfig()
    periodic_cfg = periodic_cfg or PeriodicMagConfig(
        eps=adaptive_cfg.eps,
        b0_window_s=adaptive_cfg.b0_window_s,
        kp_mag_max=adaptive_cfg.kp_mag_max,
    )
    q_adapt, q_6a, w_adapt = run_fusion_series(
        t, gyro, acc, mag, q_chip, cfg=adaptive_cfg
    )
    q_period, q_6b, w_period, fired, skipped = run_periodic_fusion_series(
        t, gyro, acc, mag, q_chip, cfg=periodic_cfg
    )
    return {
        "q_chip": q_chip,
        "q_adaptive": q_adapt,
        "q_periodic": q_period,
        "q_6dof": q_6a,
        "w_adaptive": w_adapt,
        "w_periodic": w_period,
        "pulse_fired": fired,
        "pulse_skipped": skipped,
        "periodic_cfg": periodic_cfg,
    }


def synthesize_demo(duration_s: float = 6.0, hz: float = 100.0) -> dict:
    """生成短合成序列，供无真实 CSV 时自检。"""
    n = int(duration_s * hz)
    t = np.linspace(0, duration_s, n)
    dt = 1.0 / hz
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = np.zeros((n, 3))
    acc = np.tile([0.0, 0.0, G_NORM], (n, 1))
    mag = np.tile([200.0, 0.0, 400.0], (n, 1))
    q_chip = np.zeros((n, 4))

    for i in range(1, n):
        # 2s 后绕 Z 缓慢转动
        wz = 0.3 if t[i] > 2.0 else 0.0
        gyro[i] = [0.0, 0.0, wz]
        qa = np.array([0.0, 0.0, 0.0, wz])
        q_dot = 0.5 * quat_multiply(q_true, qa)
        q_true = quat_normalize(q_true + q_dot * dt)
        q_chip[i] = q_true
        # 4s 后模拟磁干扰：|B| 突变
        if t[i] > 4.0:
            mag[i] = [500.0, 100.0, 200.0]
        else:
            mag[i] = [200.0, 0.0, 400.0]

    q_chip[0] = [1.0, 0.0, 0.0, 0.0]
    return {"t": t, "gyro": gyro, "acc": acc, "mag": mag, "q_chip": q_chip}


if __name__ == "__main__":
    d = synthesize_demo()
    cfg = AdaptiveMagConfig()
    q_pc, q_6, w = run_fusion_series(
        d["t"], d["gyro"], d["acc"], d["mag"], d["q_chip"], cfg=cfg
    )
    print(
        f"合成自检 n={len(d['t'])}  B0段 w_mag 均值={w[:300].mean():.2f}  "
        f"干扰段 w_mag 均值={w[400:].mean():.2f}  "
        f"末帧 q_pc vs chip 夹角={quat_angle_deg(q_pc[-1], d['q_chip'][-1]):.2f}°"
    )
