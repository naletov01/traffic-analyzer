"""Locate a licence plate within a vehicle crop (ONNX YOLOv9-t)."""

import logging
from typing import NamedTuple, Optional

import numpy as np

from config import plate as cfg

# Quieten per-frame CoreML EP warnings on empty frames (macOS-only, harmless).
logging.getLogger("open_image_models").setLevel(logging.ERROR)


class PlateBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1


class PlateDetector:
    def __init__(self):
        from open_image_models.detection.pipeline.license_plate import (
            LicensePlateDetector,
        )

        self.model = LicensePlateDetector(detection_model=cfg.detector_model)

    def detect_best(self, vehicle_crop: np.ndarray) -> Optional[PlateBox]:
        """Return the most confident plate box in the crop, or None."""
        if vehicle_crop.size == 0:
            return None

        best: Optional[PlateBox] = None
        for d in self.model.predict(vehicle_crop):
            if d.confidence < cfg.detector_conf:
                continue
            bb = d.bounding_box
            box = PlateBox(int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2),
                           float(d.confidence))
            if best is None or box.conf > best.conf:
                best = box
        return best
