#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IR 反光 marker 2D 检测（离线与实时预览共用）。"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tools.jetarm_marker.bracket_rom import BracketRom, load_default_rom, match_previous_labeled, select_four_by_rom

DEFAULT_MIN_AXIS_LENGTH_RATIO_2D = 0.55
DEFAULT_MAX_ROM_RMS_MM = 22.0

MARKER_COLORS_BGR = (
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
)


@dataclass
class Blob:
    u: float
    v: float
    area: float
    circularity: float
    aspect: float
    mean_intensity: float
    score: float


@dataclass
class DetectParams:
    threshold_percentile: float = 98.5
    min_area: float = 12.0
    max_area: float = 1800.0
    min_circularity: float = 0.15
    edge_margin: int = 14
    max_match_px: float = 70.0
    min_axis_length_ratio_2d: float = DEFAULT_MIN_AXIS_LENGTH_RATIO_2D
    max_rom_rms_mm: float = DEFAULT_MAX_ROM_RMS_MM
    use_rom: bool = True
    rom_candidate_limit: int = 16


@dataclass
class FrameDetectResult:
    gray: np.ndarray
    blobs: List[Blob]
    selected: Optional[np.ndarray] = None
    track_valid: bool = False
    axis_length_ratio_2d: Optional[float] = None
    geometry_valid: bool = False
    rom_rms_mm: Optional[float] = None
    candidate_count: int = 0

    @property
    def frame_valid(self) -> bool:
        return self.selected is not None and self.track_valid and self.geometry_valid


@dataclass
class MarkerTracker:
    params: DetectParams = field(default_factory=DetectParams)
    rom: Optional[BracketRom] = None
    previous: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.params.use_rom and self.rom is None:
            try:
                self.rom = load_default_rom()
            except OSError:
                self.rom = None

    def process(self, gray_u8: np.ndarray, *, enforce_match_gate: bool = True) -> FrameDetectResult:
        blobs = detect_bright_blobs(gray_u8, params=self.params)
        selected: Optional[np.ndarray] = None
        track_valid = False
        rom_rms: Optional[float] = None

        if self.params.use_rom and self.rom is not None:
            blob_pts = [np.array([b.u, b.v], dtype=np.float64) for b in blobs[: self.params.rom_candidate_limit]]
            match = select_four_by_rom(blob_pts, self.rom, max_rms_mm=self.params.max_rom_rms_mm)
            if match is not None:
                selected = match.points
                rom_rms = match.rms_mm
        else:
            raw = _select_spread_four(blobs, gray_u8.shape)
            if raw is not None:
                selected, _ = _match_previous_unlabeled(raw, self.previous, self.params.max_match_px)

        if selected is not None:
            if self.previous is None:
                track_valid = True
            else:
                track_valid = match_previous_labeled(selected, self.previous, self.params.max_match_px)
            if not enforce_match_gate:
                track_valid = True
            if track_valid:
                self.previous = selected.copy()

        ratio = axis_length_ratio_2d(selected) if selected is not None else None
        ratio_ok = ratio is not None and ratio >= self.params.min_axis_length_ratio_2d
        rom_ok = (
            rom_rms is not None
            and rom_rms <= self.params.max_rom_rms_mm
        )
        if self.params.use_rom and self.rom is not None:
            geometry_valid = bool(selected is not None and rom_ok and ratio_ok)
        else:
            geometry_valid = ratio_ok

        return FrameDetectResult(
            gray=gray_u8,
            blobs=blobs,
            selected=selected,
            track_valid=track_valid,
            axis_length_ratio_2d=ratio,
            geometry_valid=geometry_valid,
            rom_rms_mm=rom_rms,
            candidate_count=len(blobs),
        )

    def reset(self) -> None:
        self.previous = None


def ir_array_to_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype == np.uint8:
        return arr.copy()

    data = arr.astype(np.float32)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.7))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def detect_bright_blobs(gray: np.ndarray, *, params: DetectParams) -> List[Blob]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = int(np.percentile(blur, params.threshold_percentile))
    threshold = max(threshold, 80)
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Blob] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < params.min_area or area > params.max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < params.min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(min(w, h) / max(w, h, 1))
        if aspect < 0.45:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        u = float(moments["m10"] / moments["m00"])
        v = float(moments["m01"] / moments["m00"])
        h_img, w_img = gray.shape[:2]
        if (
            u < params.edge_margin
            or v < params.edge_margin
            or u > w_img - params.edge_margin
            or v > h_img - params.edge_margin
        ):
            continue
        blob_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, -1)
        mean_intensity = float(cv2.mean(gray, mask=blob_mask)[0])
        score = mean_intensity * math.sqrt(area) * circularity * aspect
        blobs.append(
            Blob(
                u=u,
                v=v,
                area=area,
                circularity=circularity,
                aspect=aspect,
                mean_intensity=mean_intensity,
                score=score,
            )
        )
    return sorted(blobs, key=lambda b: b.score, reverse=True)


def _assign_angle_order(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered_indices = list(np.argsort(angles))
    right = int(np.argmax(points[:, 0]))
    start = ordered_indices.index(right)
    ordered_indices = ordered_indices[start:] + ordered_indices[:start]
    return points[ordered_indices]


def _select_spread_four(blobs: List[Blob], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if len(blobs) < 4:
        return None
    candidates = list(blobs[:16])
    max_score = max((b.score for b in candidates), default=1.0)
    diag = float(np.hypot(image_shape[1], image_shape[0]))
    best_points: Optional[np.ndarray] = None
    best_value = -float("inf")

    for combo in itertools.combinations(candidates, 4):
        points = np.asarray([[b.u, b.v] for b in combo], dtype=np.float64)
        pairwise = []
        for i in range(4):
            for j in range(i + 1, 4):
                pairwise.append(float(np.linalg.norm(points[i] - points[j])))
        if min(pairwise) < 18.0:
            continue
        mean_dist = float(np.mean(pairwise)) / max(diag, 1.0)
        hull = cv2.convexHull(points.astype(np.float32))
        hull_area = float(cv2.contourArea(hull)) / float(max(image_shape[0] * image_shape[1], 1))
        score_term = float(sum(b.score / max_score for b in combo)) / 4.0
        value = 0.35 * score_term + 1.25 * mean_dist + 2.0 * hull_area
        if value > best_value:
            best_value = value
            best_points = points

    return best_points


def _match_previous_unlabeled(
    points: np.ndarray, previous: Optional[np.ndarray], max_match_px: float
) -> Tuple[np.ndarray, bool]:
    if previous is None:
        return _assign_angle_order(points), True
    best_perm: Optional[Tuple[int, ...]] = None
    best_cost = float("inf")
    best_max = float("inf")
    for perm in itertools.permutations(range(4)):
        ordered = points[list(perm)]
        distances = np.linalg.norm(ordered - previous, axis=1)
        cost = float(distances.sum())
        if cost < best_cost:
            best_cost = cost
            best_max = float(distances.max())
            best_perm = perm
    assert best_perm is not None
    ordered = points[list(best_perm)]
    return ordered, best_max <= max_match_px


def axis_length_ratio_2d(points: Optional[np.ndarray]) -> Optional[float]:
    if points is None or points.shape != (4, 2):
        return None
    cross_len = float(np.linalg.norm(points[0] - points[3]))
    axis_len = float(np.linalg.norm(points[2] - points[1]))
    if cross_len <= 1e-9:
        return None
    return axis_len / cross_len


def draw_live_overlay(result: FrameDetectResult, *, fps: Optional[float] = None) -> np.ndarray:
    vis = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
    for blob in result.blobs[:16]:
        cv2.circle(vis, (int(round(blob.u)), int(round(blob.v))), 4, (255, 120, 0), 1)

    if result.selected is not None:
        for i, (u, v) in enumerate(result.selected):
            if not result.geometry_valid:
                color = (0, 0, 255)
            elif not result.track_valid:
                color = (0, 128, 255)
            else:
                color = MARKER_COLORS_BGR[i]
            cv2.circle(vis, (int(round(u)), int(round(v))), 9, color, 2)
            cv2.putText(
                vis,
                f"m{i}",
                (int(u) + 8, int(v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        p0, p1, p2, p3 = result.selected
        cv2.line(vis, tuple(p0.astype(int)), tuple(p3.astype(int)), (180, 180, 180), 1)
        cv2.line(vis, tuple(p2.astype(int)), tuple(p1.astype(int)), (180, 180, 180), 1)

    ratio_text = "n/a" if result.axis_length_ratio_2d is None else f"{result.axis_length_ratio_2d:.2f}"
    rom_text = "n/a" if result.rom_rms_mm is None else f"{result.rom_rms_mm:.1f}mm"
    gate = "PASS" if result.geometry_valid else "FAIL"
    track = "ok" if result.track_valid else "jump"
    lines = [
        f"candidates={result.candidate_count}  track={track}  rom_rms={rom_text}  gate={gate}  ratio={ratio_text}",
        f"frame_valid={int(result.frame_valid)}  [q]uit [r]eset [+/-] threshold",
    ]
    if fps is not None:
        lines[0] = f"fps={fps:.1f}  " + lines[0]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return vis
