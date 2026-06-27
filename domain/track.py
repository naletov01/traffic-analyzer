"""Data model for a single tracked vehicle."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class Track:
    """One vehicle followed across frames, identified by a stable track_id."""

    track_id: int
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2 (pixels)
    cls_id: int                              # COCO class id (car/truck/...)
    conf: float = 0.0

    # Colour, decided by voting across frames (see vote_color).
    color_name: Optional[str] = None
    color_bgr: Tuple[int, int, int] = (0, 255, 0)   # box color (defaults to green)
    color_votes: Dict[str, float] = field(default_factory=dict)
    color_bgr_by_name: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    last_color_frame: int = -1

    # Plate, decided by temporal voting across frames (see vote_plate).
    plate_text: Optional[str] = None
    plate_conf: float = 0.0
    # text -> (votes, cumulative_confidence)
    plate_votes: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    last_ocr_frame: int = -1                 # frame index of last OCR attempt

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.bbox

    def vote_color(self, name: str, bgr: Tuple[int, int, int], weight: float = 1.0) -> None:
        """Add one colour reading, weighted (by crop area) so a close, clear view
        outweighs the small distant frame when the car first appeared."""
        self.color_votes[name] = self.color_votes.get(name, 0.0) + weight
        self.color_bgr_by_name[name] = bgr
        best = max(self.color_votes, key=self.color_votes.get)
        self.color_name = best
        self.color_bgr = self.color_bgr_by_name[best]

    def vote_plate(self, text: str, conf: float, min_votes: int = 2,
                   single_read_conf: float = 0.75) -> None:
        """Add one OCR reading and promote the best candidate to plate_text.

        The candidate with the most accumulated confidence wins once it has
        `min_votes` votes, or immediately if a single reading is confident enough.
        """
        votes, total = self.plate_votes.get(text, (0, 0.0))
        self.plate_votes[text] = (votes + 1, total + conf)

        best_text, (best_votes, best_total) = max(
            self.plate_votes.items(), key=lambda kv: kv[1][1]
        )
        best_avg = best_total / best_votes
        if best_votes >= min_votes or best_avg >= single_read_conf:
            self.plate_text = best_text
            self.plate_conf = best_avg

    def label(self) -> str:
        parts = [f"ID {self.track_id}"]
        if self.color_name:
            parts.append(self.color_name)
        if self.plate_text:
            parts.append(self.plate_text)
        return " | ".join(parts)
