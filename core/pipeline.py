"""Per-frame orchestration: detect/track vehicles, then enrich with colour and
plate. Results are cached per track so the heavy work isn't repeated each frame.
"""

from typing import Dict, List

import numpy as np

from config import color as color_cfg
from config import plate as plate_cfg
from config import runtime as rt
from detection.color_estimator import compute_wb_gains
from detection.color_estimator import estimate as estimate_color
from detection.plate_reader import PlateReader
from detection.vehicle_tracker import VehicleTracker
from domain.track import Track
from utils.frame_filter import MotionFilter


class Pipeline:
    def __init__(self):
        self.tracker = VehicleTracker()
        self.plate_reader = PlateReader()
        self.motion = MotionFilter()
        self._cache: Dict[int, Track] = {}
        self._last_tracks: List[Track] = []
        self._frame_idx = 0
        self.frames_total = 0
        self.frames_skipped = 0

    def process(self, frame: np.ndarray) -> List[Track]:
        self._frame_idx += 1
        self.frames_total += 1

        # A static frame brings no new vehicles, so reuse the previous result.
        if not self.motion.is_active(frame):
            self.frames_skipped += 1
            return self._last_tracks

        tracks = self.tracker.update(frame)

        wb_gains = None  # computed lazily, once per frame, only if needed

        live_ids = set()
        for t in tracks:
            live_ids.add(t.track_id)
            cached = self._cache.get(t.track_id)

            # Colour: vote across frames, weighting by crop area, so the result
            # comes from the clearest (closest) view, not the distant entry frame.
            if cached:
                t.color_name = cached.color_name
                t.color_bgr = cached.color_bgr
                t.color_votes = cached.color_votes
                t.color_bgr_by_name = cached.color_bgr_by_name
                t.last_color_frame = cached.last_color_frame
            if (t.last_color_frame < 0
                    or self._frame_idx - t.last_color_frame >= rt.color_every_n_frames):
                if wb_gains is None and color_cfg.backend != "cnn":
                    wb_gains = compute_wb_gains(frame)
                name, bgr = estimate_color(frame, t.bbox, wb_gains)
                x1, y1, x2, y2 = t.bbox
                t.vote_color(name, bgr, (x2 - x1) * (y2 - y1))
                t.last_color_frame = self._frame_idx

            if cached:
                t.plate_text = cached.plate_text
                t.plate_conf = cached.plate_conf
                t.plate_votes = cached.plate_votes
                t.last_ocr_frame = cached.last_ocr_frame

            # Poll unconfirmed cars often, confirmed ones rarely (cheap refine).
            cadence = (rt.ocr_every_n_confirmed if t.plate_text
                       else rt.ocr_every_n_frames)
            if self._frame_idx - t.last_ocr_frame >= cadence:
                x1, y1, x2, y2 = t.bbox
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size:
                    result = self.plate_reader.read(crop)
                    if result:
                        t.vote_plate(result.text, result.conf,
                                     plate_cfg.min_votes_to_confirm,
                                     plate_cfg.single_read_conf)
                t.last_ocr_frame = self._frame_idx

            self._cache[t.track_id] = t

        for dead in set(self._cache) - live_ids:
            del self._cache[dead]

        self._last_tracks = tracks
        return tracks
