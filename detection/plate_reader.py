"""Read a plate from a vehicle crop: detect -> OCR -> format.

Returns a single-frame PlateResult; combining readings across frames is the
Pipeline's job.
"""

from typing import NamedTuple, Optional

import numpy as np

from config import plate as cfg
from detection.plate_detector import PlateDetector
from detection.plate_ocr import PlateOCR
from utils.ua_plate import format_plate


class PlateResult(NamedTuple):
    text: str
    conf: float
    valid: bool        # passes the format check (always True when format_mode="none")


class PlateReader:
    def __init__(self):
        self.detector = PlateDetector()
        self.ocr = PlateOCR()

    def read(self, vehicle_crop: np.ndarray) -> Optional[PlateResult]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        box = self.detector.detect_best(vehicle_crop)
        if box is None or box.width < cfg.min_plate_width:
            return None      # no plate, or too small to read reliably

        plate_crop = vehicle_crop[box.y1:box.y2, box.x1:box.x2]
        ocr = self.ocr.read(plate_crop)
        if ocr is None:
            return None

        raw, conf = ocr
        if conf < cfg.min_ocr_conf:
            return None

        text, valid = format_plate(raw, mode=cfg.format_mode)
        if text is None or len(text) < cfg.min_text_len:
            return None      # empty or implausibly short (partial read)
        return PlateResult(text=text, conf=conf, valid=valid)
