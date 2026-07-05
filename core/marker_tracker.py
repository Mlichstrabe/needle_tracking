"""2D 锚点跟踪时序门限（跳变抑制 + 短时 hold）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MarkerTrackState:
    px: Optional[np.ndarray] = None
    depth_mm: Optional[float] = None
    hold_remaining: int = 0


class MarkerTemporalFilter:
    def __init__(
        self,
        max_jump_px: float = 80.0,
        max_hold_frames: int = 10,
    ):
        self.max_jump_px = float(max_jump_px)
        self.max_hold_frames = int(max_hold_frames)
        self._state = MarkerTrackState()

    def reset(self) -> None:
        self._state = MarkerTrackState()

    def update(
        self,
        px: Optional[Tuple[float, float]],
        depth_mm: Optional[float],
        visible: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
        """
        返回 (filtered_px, filtered_depth, using_hold)。
        visible=False 时递减 hold，在 hold 内仍输出上一帧。
        """
        st = self._state
        if visible and px is not None:
            p = np.array(px, dtype=float)
            if st.px is not None:
                dist = float(np.linalg.norm(p - st.px))
                if dist > self.max_jump_px:
                    if st.hold_remaining > 0:
                        st.hold_remaining -= 1
                        return st.px, st.depth_mm, True
            st.px = p
            if depth_mm is not None:
                st.depth_mm = float(depth_mm)
            st.hold_remaining = self.max_hold_frames
            return st.px, st.depth_mm, False

        if st.px is not None and st.hold_remaining > 0:
            st.hold_remaining -= 1
            return st.px, st.depth_mm, True

        st.px = None
        st.depth_mm = None
        st.hold_remaining = 0
        return None, None, False