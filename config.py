"""Tunable parameters for the traffic analyzer, grouped by component."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"


@dataclass
class VehicleConfig:
    # YOLO11n exported to ONNX, run directly on onnxruntime (no torch).
    model_path: str = str(WEIGHTS_DIR / "yolo11n.onnx")
    classes: tuple = (2, 3, 5, 7)   # COCO ids: car, motorcycle, bus, truck
    imgsz: int = 640
    nms_iou: float = 0.5            # NMS threshold in the detector
    # ByteTrack thresholds (see detection/byte_track.py).
    track_low_thresh: float = 0.1   # detector score floor; low dets rescue tracks
    track_high_thresh: float = 0.4  # confident detections, matched first
    new_track_thresh: float = 0.5   # start a track only on a confident detection
    match_iou: float = 0.2          # min IoU to associate a detection with a track
    track_buffer: int = 30          # frames a lost track survives
    min_hits: int = 2               # matches before a track is shown


@dataclass
class ColorConfig:
    # "cnn" uses the distilled MobileNet (color_mobilenet.onnx); "heuristic" uses
    # the HSV rules below.
    backend: str = "cnn"
    model_path: str = str(WEIGHTS_DIR / "color_mobilenet.onnx")
    classes_path: str = str(WEIGHTS_DIR / "color_classes.txt")

    # --- heuristic backend parameters ---
    # Sample the lower body band, not the center: head-on the center is glass.
    band_top: float = 0.5
    band_bottom: float = 0.7
    band_side: float = 0.25
    # Gray-world white balance cancels the scene's color cast (e.g. evening blue).
    white_balance: bool = True
    # Specular glare is excluded from the statistics.
    glare_value: int = 235
    glare_saturation: int = 35
    # A car reads as colored only if this fraction of pixels clears sat_threshold.
    sat_threshold: int = 65
    color_min_fraction: float = 0.45
    value_min: int = 40
    # Achromatic split by median brightness.
    black_value: int = 65
    white_value: int = 175


@dataclass
class PlateConfig:
    detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    detector_conf: float = 0.25
    # "european-..." suits UK/EU plates; "global-plates-mobile-vit-v2-model" is broader.
    ocr_model: str = "european-plates-mobile-vit-v2-model"
    device: str = "auto"
    min_plate_width: int = 40        # px; smaller plates are skipped
    # Confirm a reading after this many votes, or one this confident.
    min_votes_to_confirm: int = 2
    single_read_conf: float = 0.80
    min_text_len: int = 6
    min_ocr_conf: float = 0.5
    format_mode: str = "none"        # "none" or "ua"


@dataclass
class FilterConfig:
    # Motion gate: skip the detector on static frames (no motion -> no new cars).
    motion_enabled: bool = True
    downscale: int = 8
    pixel_delta: int = 25
    min_fraction: float = 0.002


@dataclass
class RuntimeConfig:
    window_name: str = "Traffic Analyzer"
    # Re-estimate colour every N frames per track and vote (weighted by crop size).
    color_every_n_frames: int = 8
    # OCR cadence per track: often while unconfirmed, then slow refinement.
    ocr_every_n_frames: int = 6
    ocr_every_n_confirmed: int = 24
    drop_stale_frames: bool = True
    show_fps: bool = True


vehicle = VehicleConfig()
color = ColorConfig()
plate = PlateConfig()
frame_filter = FilterConfig()
runtime = RuntimeConfig()
