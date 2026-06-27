"""Draw tracked vehicles: a box in the car's colour, labelled with id/colour/plate."""

from typing import List

import cv2
import numpy as np

from domain.track import Track


def _readable_text_color(bgr) -> tuple:
    """Black or white text depending on box-color brightness, for contrast."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        box_color = t.color_bgr
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        label = t.label()
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # background bar above the box
        ly2 = y1
        ly1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, ly2), box_color, -1)
        cv2.putText(
            frame, label, (x1 + 3, ly2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _readable_text_color(box_color), 1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame: np.ndarray, fps: float, count: int) -> np.ndarray:
    text = f"FPS: {fps:4.1f}   vehicles: {count}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame
