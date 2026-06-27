"""HSV-based vehicle colour estimation, tolerant of evening light and glare.

The heuristic white-balances the scene, samples the lower body band (the centre
is glass on head-on views), masks specular glare, and falls back to median
brightness for black/gray/white. Returns a colour name and the BGR to draw the
box with.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from config import color as cfg

# (name, hue ranges in OpenCV's 0..179 scale, draw colour in BGR)
_HUE_COLORS = [
    ("red",    [(0, 10), (170, 179)], (0, 0, 255)),
    ("orange", [(11, 20)],            (0, 128, 255)),
    ("yellow", [(21, 33)],            (0, 255, 255)),
    ("green",  [(34, 85)],            (0, 200, 0)),
    ("cyan",   [(86, 99)],            (255, 255, 0)),
    ("blue",   [(100, 130)],          (255, 0, 0)),
    ("purple", [(131, 169)],          (255, 0, 200)),
]

_ACHROMATIC_BGR = {
    "black": (40, 40, 40),
    "gray": (128, 128, 128),
    "white": (230, 230, 230),
}

# Box colour for every class name either backend can return.
_DRAW_BGR = {
    **_ACHROMATIC_BGR,
    "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 200, 0),
    "yellow": (0, 255, 255), "orange": (0, 128, 255), "purple": (255, 0, 200),
}

Gains = Tuple[float, float, float]

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from detection.color_classifier import ColorClassifier
        _classifier = ColorClassifier(cfg.model_path, cfg.classes_path)
    return _classifier


def compute_wb_gains(frame: np.ndarray) -> Gains:
    """Gray-world white-balance gains (B, G, R), from a subsampled frame."""
    means = frame[::4, ::4].reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    g = means.mean() / means
    return float(g[0]), float(g[1]), float(g[2])


def _body_patch(frame: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return frame[0:0, 0:0]
    py1 = y1 + int(h * cfg.band_top)
    py2 = y1 + int(h * cfg.band_bottom)
    px = int(w * cfg.band_side)
    return frame[max(py1, 0):py2, max(x1 + px, 0):x2 - px]


def _achromatic(median_v: float) -> Tuple[str, Tuple[int, int, int]]:
    if median_v < cfg.black_value:
        name = "black"
    elif median_v > cfg.white_value:
        name = "white"
    else:
        name = "gray"
    return name, _ACHROMATIC_BGR[name]


def estimate(frame: np.ndarray, bbox,
             wb_gains: Optional[Gains] = None) -> Tuple[str, Tuple[int, int, int]]:
    """Return (color_name, color_bgr) for the vehicle at bbox."""
    if cfg.backend == "cnn":
        x1, y1, x2, y2 = bbox
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            return "unknown", (0, 255, 0)
        name = _get_classifier().predict(crop)
        return name, _DRAW_BGR.get(name, (0, 255, 0))

    patch = _body_patch(frame, bbox)
    if patch.size == 0:
        return "unknown", (0, 255, 0)

    if cfg.white_balance:
        gb, gg, gr = wb_gains if wb_gains is not None else compute_wb_gains(frame)
        patch = np.clip(patch.astype(np.float32) * (gb, gg, gr), 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    valid = ~((v >= cfg.glare_value) & (s <= cfg.glare_saturation))
    if valid.sum() < 10:                      # nothing but glare
        return "white", _ACHROMATIC_BGR["white"]

    colored = valid & (s >= cfg.sat_threshold) & (v >= cfg.value_min)
    if colored.sum() / valid.sum() < cfg.color_min_fraction:
        return _achromatic(float(np.median(v[valid])))

    # Dominant hue among the coloured pixels.
    hist = cv2.calcHist([h[colored].astype(np.uint8)], [0], None, [180], [0, 180]).flatten()
    best_name, best_bgr, best_score = "gray", _ACHROMATIC_BGR["gray"], -1.0
    for name, ranges, bgr in _HUE_COLORS:
        score = sum(hist[lo:hi + 1].sum() for lo, hi in ranges)
        if score > best_score:
            best_name, best_bgr, best_score = name, bgr, score
    return best_name, best_bgr
