"""MobileNet colour classifier (ONNX) — the distilled student model.

Takes a vehicle crop and returns a colour name. Trained by distilling a CLIP +
brightness teacher (see tools/gen_color_dataset.py, tools/train_color.py), so it
runs at edge speed where CLIP could not.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_SIZE = 128


class ColorClassifier:
    def __init__(self, model_path: str, classes_path: str):
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.classes = Path(classes_path).read_text().split()

    def predict(self, crop_bgr: np.ndarray) -> str:
        img = cv2.resize(crop_bgr, (_SIZE, _SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((img - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input: x})[0][0]
        return self.classes[int(logits.argmax())]
