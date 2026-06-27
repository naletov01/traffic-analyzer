"""Compact ByteTrack-style multi-object tracker (numpy only, no torch).

Two-stage association like ByteTrack: confident detections are matched to tracks
first, then leftover low-confidence ones rescue still-unmatched tracks (which is
what keeps ids stable through brief occlusions). Motion is a light constant-
velocity predictor on the box centre; greedy IoU matching keeps it dependency-free.
"""

from typing import List, Tuple

import numpy as np

Detection = Tuple[int, int, int, int, float, int]  # x1,y1,x2,y2,conf,cls


def _iou_matrix(tb, db):
    """IoU of every track box (T,4) against every detection box (D,4) -> (T,D)."""
    tb, db = tb[:, None, :], db[None, :, :]
    xa = np.maximum(tb[..., 0], db[..., 0])
    ya = np.maximum(tb[..., 1], db[..., 1])
    xb = np.minimum(tb[..., 2], db[..., 2])
    yb = np.minimum(tb[..., 3], db[..., 3])
    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    at = (tb[..., 2] - tb[..., 0]) * (tb[..., 3] - tb[..., 1])
    ad = (db[..., 2] - db[..., 0]) * (db[..., 3] - db[..., 1])
    return inter / (at + ad - inter + 1e-9)


class _Track:
    def __init__(self, track_id, box, conf, cls):
        self.id = track_id
        self.box = np.array(box, dtype=float)        # x1,y1,x2,y2
        self.vel = np.zeros(2)                        # centre velocity
        self.conf, self.cls = conf, cls
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False

    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    def predict(self):
        w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
        cx, cy = self.center() + self.vel
        self.box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        self.time_since_update += 1

    def update(self, box, conf, cls):
        new_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        self.vel = 0.5 * self.vel + 0.5 * (new_c - self.center())
        self.box = np.array(box, dtype=float)
        self.conf, self.cls = conf, cls
        self.hits += 1
        self.time_since_update = 0


class ByteTrack:
    def __init__(self, high_thresh=0.5, new_thresh=0.6, match_iou=0.2,
                 buffer=30, min_hits=2):
        self.high_thresh = high_thresh
        self.new_thresh = new_thresh
        self.match_iou = match_iou       # minimum IoU to associate
        self.buffer = buffer
        self.min_hits = min_hits
        self._tracks: List[_Track] = []
        self._next_id = 1

    def _match(self, tracks, dets):
        """Greedy IoU matching. Returns (pairs, unmatched_tracks, unmatched_dets)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb = np.array([t.box for t in tracks])
        db = np.array([d[:4] for d in dets], dtype=float)
        iou = _iou_matrix(tb, db)
        pairs, ut, ud = [], set(range(len(tracks))), set(range(len(dets)))
        while True:
            i, j = np.unravel_index(np.argmax(iou), iou.shape)
            if iou[i, j] < self.match_iou:
                break
            pairs.append((i, j))
            ut.discard(i); ud.discard(j)
            iou[i, :] = -1
            iou[:, j] = -1
        return pairs, list(ut), list(ud)

    def update(self, detections: List[Detection]):
        for t in self._tracks:
            t.predict()

        high = [d for d in detections if d[4] >= self.high_thresh]
        low = [d for d in detections if d[4] < self.high_thresh]

        # Stage 1: existing tracks vs confident detections.
        pairs, ut, ud = self._match(self._tracks, high)
        for ti, di in pairs:
            self._tracks[ti].update(high[di][:4], high[di][4], high[di][5])

        # Stage 2: still-unmatched tracks vs low-confidence detections.
        rem = [self._tracks[i] for i in ut]
        pairs2, _, _ = self._match(rem, low)
        for ti, di in pairs2:
            rem[ti].update(low[di][:4], low[di][4], low[di][5])

        # New tracks from confident, unmatched detections.
        for di in ud:
            if high[di][4] >= self.new_thresh:
                self._tracks.append(_Track(self._next_id, high[di][:4], high[di][4], high[di][5]))
                self._next_id += 1

        # Lifecycle: confirm, then drop tracks lost longer than the buffer.
        kept = []
        for t in self._tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.time_since_update <= self.buffer:
                kept.append(t)
        self._tracks = kept

        return [t for t in self._tracks if t.confirmed and t.time_since_update == 0]
