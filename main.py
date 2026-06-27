"""Entry point: open a video source, run the analyzer, show the annotated video.

    python main.py rtsp://user:pass@host:554/stream
    python main.py path/to/video.mp4
    python main.py 0                     # local webcam

Press 'q' or Esc to quit.
"""

import argparse
import sys
import time

import cv2

from config import frame_filter as ff
from config import runtime as rt
from core.capture import VideoCapture
from core.pipeline import Pipeline
from render.visualizer import draw_hud, draw_tracks
from utils.frame_filter import has_vehicles


def parse_args():
    p = argparse.ArgumentParser(description="Real-time vehicle traffic analyzer")
    p.add_argument("source", help="RTSP URL, video file path, or webcam index")
    p.add_argument("--no-display", action="store_true",
                   help="run headless (no window), e.g. for benchmarking")
    p.add_argument("--skip-empty", action="store_true",
                   help="skip rendering frames with no vehicles")
    p.add_argument("--no-motion-filter", action="store_true",
                   help="disable the cheap motion gate that skips static frames")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_motion_filter:
        ff.motion_enabled = False

    print(f"[init] opening source: {args.source}")
    try:
        cap = VideoCapture(args.source, drop_stale=rt.drop_stale_frames)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print("[init] loading models...")
    pipeline = Pipeline()
    print("[ready] press 'q' or Esc to quit")

    fps = 0.0
    prev = time.time()

    try:
        while True:
            ok, frame, _ = cap.read()

            if ok:
                tracks = pipeline.process(frame)
                show = not (args.skip_empty and not has_vehicles(tracks))

                if show:
                    now = time.time()
                    dt = now - prev
                    prev = now
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / dt)   # smoothed

                    if not args.no_display:
                        draw_tracks(frame, tracks)
                        if rt.show_fps:
                            draw_hud(frame, fps, len(tracks))
                        cv2.imshow(rt.window_name, frame)

            # waitKey must run every iteration or the window freezes on macOS.
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if cap.finished:
                break
            if not ok:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[exit] interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = pipeline.frames_total
    if total:
        skipped = pipeline.frames_skipped
        print(f"[filter] motion gate skipped {skipped}/{total} frames "
              f"({100 * skipped / total:.0f}%) of heavy inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
