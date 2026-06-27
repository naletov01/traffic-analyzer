"""Read text from a plate crop (fast-plate-ocr, ONNX)."""

from typing import Optional, Tuple

import cv2
import numpy as np

from config import plate as cfg


class PlateOCR:
    def __init__(self):
        from fast_plate_ocr import LicensePlateRecognizer

        self.model = LicensePlateRecognizer(cfg.ocr_model, device=cfg.device)

    def read(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Return (raw_text, confidence) or None if nothing was read."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)  # model wants 1 channel
        preds = self.model.run(gray, return_confidence=True)
        if not preds:
            return None

        pred = preds[0]
        text = (pred.plate or "").strip()
        if not text:
            return None

        conf = float(np.mean(pred.char_probs)) if pred.char_probs is not None else 1.0
        return text, conf
