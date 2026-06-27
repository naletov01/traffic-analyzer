"""Vehicle detection and tracking, torch-free.

Runs the YOLO detector on onnxruntime and tracks with a built-in ByteTrack, so
each vehicle gets a stable unique id. Same interface as before: update(frame)
returns the current tracks.
"""

from typing import List

import numpy as np

from config import vehicle as cfg
from detection.byte_track import ByteTrack
from detection.yolo_onnx import YoloOnnxDetector
from domain.track import Track


class VehicleTracker:
    def __init__(self):
        self.detector = YoloOnnxDetector()
        self.tracker = ByteTrack(
            high_thresh=cfg.track_high_thresh,
            new_thresh=cfg.new_track_thresh,
            match_iou=cfg.match_iou,
            buffer=cfg.track_buffer,
            min_hits=cfg.min_hits,
        )

    def update(self, frame: np.ndarray) -> List[Track]:
        detections = self.detector.detect(frame)
        tracks = []
        for t in self.tracker.update(detections):
            x1, y1, x2, y2 = (int(v) for v in t.box)
            tracks.append(Track(track_id=t.id, bbox=(x1, y1, x2, y2),
                                cls_id=t.cls, conf=t.conf))
        return tracks
