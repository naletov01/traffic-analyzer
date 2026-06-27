"""YOLO vehicle detector running directly on onnxruntime (no torch / ultralytics).

Replicates Ultralytics' letterbox preprocessing and detection post-processing
(decode + per-class NMS) so the boxes match the original `.track()` output.
"""

from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime

from config import vehicle as cfg

# Detection: (x1, y1, x2, y2, confidence, class_id)
Detection = Tuple[int, int, int, int, float, int]


class YoloOnnxDetector:
    def __init__(self):
        self.session = onnxruntime.InferenceSession(
            cfg.model_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.imgsz = cfg.imgsz
        self._classes = set(cfg.classes)

    def _letterbox(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        dy, dx = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
        return canvas, r, dx, dy

    def detect(self, frame: np.ndarray) -> List[Detection]:
        canvas, r, dx, dy = self._letterbox(frame)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]

        preds = self.session.run(None, {self.input: blob})[0][0].T  # (8400, 84)
        scores = preds[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)

        keep = (conf >= cfg.track_low_thresh) & np.isin(cls, list(self._classes))
        preds, conf, cls = preds[keep], conf[keep], cls[keep]
        if len(preds) == 0:
            return []

        cxcy, wh = preds[:, :2], preds[:, 2:4]
        boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)  # xyxy, letterbox

        # Per-class NMS via a per-class coordinate offset, then one NMS pass.
        offset = cls[:, None] * (self.imgsz + 1)
        nms_in = (boxes + offset).astype(np.float32)
        wh_boxes = np.concatenate([nms_in[:, :2], nms_in[:, 2:] - nms_in[:, :2]], axis=1)
        idxs = cv2.dnn.NMSBoxes(wh_boxes.tolist(), conf.tolist(),
                                cfg.track_low_thresh, cfg.nms_iou)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        h, w = frame.shape[:2]
        out: List[Detection] = []
        for i in idxs:
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip((x1 - dx) / r, 0, w))
            y1 = int(np.clip((y1 - dy) / r, 0, h))
            x2 = int(np.clip((x2 - dx) / r, 0, w))
            y2 = int(np.clip((y2 - dy) / r, 0, h))
            out.append((x1, y1, x2, y2, float(conf[i]), int(cls[i])))
        return out
