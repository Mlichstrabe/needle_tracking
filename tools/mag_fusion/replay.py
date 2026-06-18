"""
从 CSV 加载 IMU 流并规范为融合所需列。

支持两种导出格式:
  1. fusion（推荐）: t, ax..az, gx..gz, mx..mz, qw..qz [, event]
  2. mag_test（tools/magnetometer_test.py，需含扩展列）:
     旧版仅 roll/pitch/yaw/mx/my/mz 无法离线融合，会抛出明确错误。

列名别名见 COLUMN_ALIASES。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COLUMN_ALIASES: Dict[str, List[str]] = {
    "t": ["t", "time", "timestamp", "sec"],
    "ax": ["ax", "acc_x", "acc0"],
    "ay": ["ay", "acc_y", "acc1"],
    "az": ["az", "acc_z", "acc2"],
    "gx": ["gx", "gyro_x", "gyro0"],
    "gy": ["gy", "gyro_y", "gyro1"],
    "gz": ["gz", "gyro_z", "gyro2"],
    "mx": ["mx", "mag_x"],
    "my": ["my", "mag_y"],
    "mz": ["mz", "mag_z"],
    "qw": ["qw", "q0", "quat_w"],
    "qx": ["qx", "q1", "quat_x"],
    "qy": ["qy", "q2", "quat_y"],
    "qz": ["qz", "q3", "quat_z"],
    "event": ["event", "note", "mark"],
    "mag_abs": ["mag_abs", "mag_b", "|B|"],
    "roll": ["roll", "imu_roll"],
    "pitch": ["pitch", "imu_pitch"],
    "yaw": ["yaw", "imu_yaw"],
}

FUSION_REQUIRED = ("t", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")
QUAT_OPTIONAL = ("qw", "qx", "qy", "qz")


@dataclass
class ReplayTable:
    """规范后的回放表。"""

    t: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    mag: np.ndarray
    q_chip: np.ndarray
    events: List[Tuple[float, str]]
    source_path: Path
    schema: str  # "fusion" | "mag_test_extended"


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _resolve_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    """csv 列名 -> 规范名。"""
    norm = {_normalize_header(f): f for f in fieldnames}
    resolved: Dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = _normalize_header(a)
            if key in norm:
                resolved[canon] = norm[key]
                break
    return resolved


def _read_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV 无表头: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _col_float(rows: List[dict], col: str, default: float = np.nan) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(col, "")
        if v == "" or v is None:
            out.append(default)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(default)
    return np.array(out, dtype=float)


def euler_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX 欧拉角（度）→ 四元数 [w,x,y,z]。"""
    r = np.radians(roll_deg) * 0.5
    p = np.radians(pitch_deg) * 0.5
    y = np.radians(yaw_deg) * 0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def load_csv(path: str | Path) -> ReplayTable:
    path = Path(path)
    fieldnames, rows = _read_rows(path)
    if not rows:
        raise ValueError(f"CSV 无数据行: {path}")

    col = _resolve_columns(fieldnames)
    missing = [k for k in FUSION_REQUIRED if k not in col]
    if missing:
        has_euler = all(k in col for k in ("roll", "pitch", "yaw"))
        hint = (
            "缺少融合所需列: "
            + ", ".join(missing)
            + "。\n"
            "请用 magnetometer_test 重新导出（需含 ax..gz、qw..qz），"
            "或使用含完整 IMU 流的 fusion CSV。"
        )
        if has_euler and "mx" in col:
            hint += "\n检测到仅有欧拉角无陀螺/加速度——旧 mag_test 格式不支持 PC 融合。"
        raise ValueError(hint)

    t = _col_float(rows, col["t"])
    if not np.all(np.isfinite(t)):
        # 若时间为空，用行号代替
        t = np.arange(len(rows), dtype=float) * 0.01
    elif np.max(t) > 1e9:
        t = t - t[0]

    acc = np.column_stack([_col_float(rows, col[k]) for k in ("ax", "ay", "az")])
    gyro = np.column_stack([_col_float(rows, col[k]) for k in ("gx", "gy", "gz")])
    mag = np.column_stack([_col_float(rows, col[k]) for k in ("mx", "my", "mz")])

    n = len(rows)
    q_chip = np.full((n, 4), np.nan)
    has_quat = all(k in col for k in QUAT_OPTIONAL)
    if has_quat:
        q_chip = np.column_stack([_col_float(rows, col[k]) for k in QUAT_OPTIONAL])
        for i in range(n):
            if not np.all(np.isfinite(q_chip[i])) or np.linalg.norm(q_chip[i]) < 0.5:
                q_chip[i] = np.nan
    elif all(k in col for k in ("roll", "pitch", "yaw")):
        roll = _col_float(rows, col["roll"])
        pitch = _col_float(rows, col["pitch"])
        yaw = _col_float(rows, col["yaw"])
        for i in range(n):
            if np.all(np.isfinite([roll[i], pitch[i], yaw[i]])):
                q_chip[i] = euler_to_quat_wxyz(roll[i], pitch[i], yaw[i])

    # 无效四元数填 identity
    for i in range(n):
        if not np.all(np.isfinite(q_chip[i])) or np.linalg.norm(q_chip[i]) < 0.5:
            q_chip[i] = [1.0, 0.0, 0.0, 0.0]

    events: List[Tuple[float, str]] = []
    if "event" in col:
        ev_col = col["event"]
        last_ev = ""
        for r in rows:
            ev = (r.get(ev_col) or "").strip()
            if ev and ev != last_ev:
                try:
                    events.append((float(r[col["t"]]), ev))
                except (KeyError, ValueError):
                    pass
            if ev:
                last_ev = ev

    schema = "fusion" if has_quat else "mag_test_extended"
    return ReplayTable(
        t=t,
        gyro=gyro,
        acc=acc,
        mag=mag,
        q_chip=q_chip,
        events=events,
        source_path=path,
        schema=schema,
    )


def write_fusion_output(
    table: ReplayTable,
    q_pc: np.ndarray,
    q_6dof: np.ndarray,
    w_mag: np.ndarray,
    out_path: Path,
) -> Path:
    """写出 sidecar CSV：原时间列 + q_chip + q_pc + q_6dof + w_mag + |B|。"""
    out_path = Path(out_path)
    mag_abs = np.linalg.norm(table.mag, axis=1)
    header = [
        "t",
        "qw_chip",
        "qx_chip",
        "qy_chip",
        "qz_chip",
        "qw_pc",
        "qx_pc",
        "qy_pc",
        "qz_pc",
        "qw_6dof",
        "qx_6dof",
        "qy_6dof",
        "qz_6dof",
        "w_mag",
        "mag_abs",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(table.t)):
            w.writerow(
                [
                    table.t[i],
                    *table.q_chip[i],
                    *q_pc[i],
                    *q_6dof[i],
                    w_mag[i],
                    mag_abs[i],
                ]
            )
    return out_path
