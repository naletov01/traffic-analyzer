"""Frame filtering.

MotionFilter is a cheap activity gate run before the detector: a frame with no
motion can't hold a newly-arrived vehicle, so the pipeline can skip inference.
The predicates below report whether a processed frame is worth keeping.
"""

from typing import List, Optional

import cv2
import numpy as np

from config import frame_filter as cfg
from domain.track import Track


class MotionFilter:
    def __init__(self):
        self._prev: Optional[np.ndarray] = None

    def is_active(self, frame: np.ndarray) -> bool:
        """True if the frame differs enough from the previous one (motion)."""
        if not cfg.motion_enabled:
            return True

        small = cv2.cvtColor(
            frame[:: cfg.downscale, :: cfg.downscale], cv2.COLOR_BGR2GRAY
        )
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return True                      # first frame / size change: process it

        diff = cv2.absdiff(small, prev)
        moved_fraction = (diff > cfg.pixel_delta).mean()
        return moved_fraction >= cfg.min_fraction


def has_vehicles(tracks: List[Track]) -> bool:
    return len(tracks) > 0


def has_readable_plate(tracks: List[Track]) -> bool:
    return any(t.plate_text for t in tracks)
