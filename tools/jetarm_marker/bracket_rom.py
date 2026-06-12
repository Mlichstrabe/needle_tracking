#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""针架 4 marker ROM：边长约束 + 2D 相似匹配（类 Aimooe .aimtool 简化版）。"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_ROM = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "jetarm_marker"
    / "geometry"
    / "bracket_rom.json"
)


@dataclass(frozen=True)
class RomPair:
    i: int
    j: int
    d_mm: float
    label: str = ""


@dataclass
class BracketRom:
    marker_ids: Tuple[str, ...]
    pairs: Tuple[RomPair, ...]
    axis_ratio: float

    @classmethod
    def from_json(cls, path: Path) -> "BracketRom":
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs = tuple(
            RomPair(
                int(item["a"]),
                int(item["b"]),
                float(item["d"]),
                str(item.get("label", "")),
            )
            for item in data["pairwise_mm"]
        )
        axis_ratio = float(data.get("axis_ratio_m2m1_over_m0m3", 59.0 / 95.0))
        ids = tuple(str(x) for x in data["marker_ids"])
        return cls(marker_ids=ids, pairs=pairs, axis_ratio=axis_ratio)

    def distance_matrix_mm(self) -> np.ndarray:
        mat = np.zeros((4, 4), dtype=np.float64)
        for p in self.pairs:
            mat[p.i, p.j] = p.d_mm
            mat[p.j, p.i] = p.d_mm
        return mat

    def template_2d_mm(self) -> np.ndarray:
        """由边长恢复平面四边形顶点（m0 原点，m1 沿 +x）。"""
        d01 = self._dist(0, 1)
        d03 = self._dist(0, 3)
        d13 = self._dist(1, 3)
        p0 = np.array([0.0, 0.0])
        p1 = np.array([d01, 0.0])
        x3 = (d01 * d01 + d03 * d03 - d13 * d13) / max(2.0 * d01, 1e-6)
        y3_sq = max(d03 * d03 - x3 * x3, 0.0)
        p3 = np.array([x3, math.sqrt(y3_sq)])
        d02 = self._dist(0, 2)
        d12 = self._dist(1, 2)
        d32 = self._dist(3, 2)
        best_p2: Optional[np.ndarray] = None
        best_err = float("inf")
        for sign in (-1.0, 1.0):
            for t in np.linspace(0.05, 0.95, 19):
                cand = (1 - t) * p0 + t * p1 + sign * np.array([0.0, 1.0])
                err = abs(np.linalg.norm(cand - p0) - d02)
                err += abs(np.linalg.norm(cand - p1) - d12)
                err += abs(np.linalg.norm(cand - p3) - d32)
                if err < best_err:
                    best_err = err
                    best_p2 = cand
        if best_p2 is None:
            best_p2 = np.array([d02 * 0.5, d12 * 0.5])
        return np.vstack([p0, p1, best_p2, p3])

    def _dist(self, i: int, j: int) -> float:
        for p in self.pairs:
            if (p.i, p.j) == (i, j) or (p.i, p.j) == (j, i):
                return p.d_mm
        raise KeyError((i, j))


def load_default_rom() -> BracketRom:
    return BracketRom.from_json(_DEFAULT_ROM)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            mat[i, j] = d
            mat[j, i] = d
    return mat


def rom_scale_rms_mm(points_ordered: np.ndarray, rom: BracketRom) -> Tuple[float, float]:
    """已知顺序 m0..m3，最优尺度下 ROM 边长 RMS 误差（mm）。"""
    measured = []
    target = []
    for p in rom.pairs:
        measured.append(float(np.linalg.norm(points_ordered[p.i] - points_ordered[p.j])))
        target.append(p.d_mm)
    m = np.asarray(measured, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    scale = float(m @ t / max(t @ t, 1e-9))
    if scale <= 1e-9:
        return float("inf"), 0.0
    err = m / scale - t
    rms = float(np.sqrt(np.mean(err * err)))
    return rms, scale


def _similarity_align(template: np.ndarray, observed: np.ndarray) -> Tuple[float, float]:
    """2D Procrustes（相似变换）对齐后 RMS，返回 (rms_mm, scale_px_per_mm)。"""
    t = template.astype(np.float64)
    o = observed.astype(np.float64)
    t_c = t - t.mean(axis=0)
    o_c = o - o.mean(axis=0)
    t_norm = float(np.linalg.norm(t_c))
    o_norm = float(np.linalg.norm(o_c))
    if t_norm < 1e-9 or o_norm < 1e-9:
        return float("inf"), 0.0
    # 最优旋转（2D Kabsch）
    h = t_c.T @ o_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    aligned = (o_c @ r) * (t_norm / o_norm)
    diff = aligned - t_c
    rms_template = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    scale = o_norm / t_norm
    return rms_template, scale


@dataclass
class RomMatchResult:
    points: np.ndarray
    rms_mm: float
    scale_px_per_mm: float
    blob_indices: Tuple[int, ...]


def select_four_by_rom(
    blob_points: Sequence[np.ndarray],
    rom: BracketRom,
    *,
    max_rms_mm: float,
    min_pair_px: float = 18.0,
) -> Optional[RomMatchResult]:
    """从候选 2D 点中选最符合 ROM 的 4 点及 m0-m3 标号。"""
    if len(blob_points) < 4:
        return None
    template = rom.template_2d_mm()
    best: Optional[RomMatchResult] = None
    best_rms = float("inf")

    indices = list(range(len(blob_points)))
    for combo in itertools.combinations(indices, 4):
        pts = np.asarray([blob_points[i] for i in combo], dtype=np.float64)
        pairwise = _pairwise_distances(pts)
        if float(pairwise[np.triu_indices(4, k=1)].min()) < min_pair_px:
            continue
        for perm in itertools.permutations(range(4)):
            ordered = pts[list(perm)]
            rms_dist, scale_d = rom_scale_rms_mm(ordered, rom)
            rms_shape, scale_s = _similarity_align(template, ordered)
            rms = 0.5 * rms_dist + 0.5 * rms_shape
            scale = 0.5 * (scale_d + scale_s)
            if rms < best_rms:
                best_rms = rms
                best = RomMatchResult(
                    points=ordered,
                    rms_mm=rms,
                    scale_px_per_mm=scale,
                    blob_indices=tuple(combo[perm[k]] for k in range(4)),
                )

    if best is None:
        return None
    return best


def rom_match_acceptable(match: RomMatchResult, max_rms_mm: float) -> bool:
    return match.rms_mm <= max_rms_mm


def match_previous_labeled(
    points: np.ndarray, previous: np.ndarray, max_match_px: float
) -> bool:
    distances = np.linalg.norm(points - previous, axis=1)
    return float(distances.max()) <= max_match_px
