"""Video source with two modes, chosen from the source string.

Live sources (RTSP / webcam) are read on a background thread; `read()` returns
the latest frame and drops stale ones to keep latency low. File sources are read
sequentially in order (and looped), since dropping frames would just race to EOF.
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

_LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class VideoCapture:
    def __init__(self, source: str, drop_stale: bool = True, loop_file: bool = True):
        self.source = int(source) if str(source).isdigit() else source
        self.is_live = self._is_live(source)
        self.loop_file = loop_file and not self.is_live
        self.drop_stale = drop_stale and self.is_live
        self.finished = False

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._frame_id = 0
        self._last_read_id = -1

        if self.is_live:
            # Keep OpenCV's own buffer tiny; we do our own freshest-frame logic.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._lock = threading.Lock()
            self._frame: Optional[np.ndarray] = None
            self._running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    @staticmethod
    def _is_live(source: str) -> bool:
        s = str(source)
        if s.isdigit():            # webcam index
            return True
        return s.lower().startswith(_LIVE_PREFIXES)

    # --- live (threaded, freshest-frame) ---------------------------------
    def _reader(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)   # stream hiccup — back off and retry
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def _read_live(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_id
            if self.drop_stale and self._frame_id == self._last_read_id:
                return False, None, self._frame_id   # no new frame yet
            self._last_read_id = self._frame_id
            return True, self._frame.copy(), self._frame_id

    # --- file (sequential, in order) -------------------------------------
    def _read_file(self) -> Tuple[bool, Optional[np.ndarray], int]:
        ok, frame = self.cap.read()
        if not ok:
            if self.loop_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                self.finished = True
                return False, None, self._frame_id
        self._frame_id += 1
        return True, frame, self._frame_id

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Return (ok, frame, frame_id)."""
        return self._read_live() if self.is_live else self._read_file()

    def release(self) -> None:
        if self.is_live:
            self._running = False
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
