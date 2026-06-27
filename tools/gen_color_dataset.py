"""Build a colour-classification dataset by auto-labelling vehicle crops.

Hybrid teacher, playing to each method's strength:
  * saturated colours (red/blue/green/yellow/orange) come from CLIP, but only
    when CLIP is confident AND the crop is actually saturated;
  * neutrals (black/gray/white) are split by body brightness using tertiles, so
    they stay balanced instead of collapsing into "gray".

The small student model is later trained on the resulting dataset/<colour>/.

    python tools/gen_color_dataset.py weights/test_plates.mp4 --out dataset
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from detection.color_estimator import compute_wb_gains
from detection.vehicle_tracker import VehicleTracker

SATURATED = ["red", "blue", "green", "yellow", "orange"]
NEUTRAL = ["black", "gray", "white"]
CLIP_CLASSES = SATURATED + ["white", "black", "gray", "silver", "brown"]
TEMPLATES = ["a photo of a {} car", "a {} car", "a {} colored vehicle"]

# CLIP neutral predictions collapsed to our three neutral classes.
_NEUTRAL_MAP = {"white": "white", "black": "black", "gray": "gray",
                "silver": "gray", "brown": "gray"}
_NEUTRAL_IDX = [CLIP_CLASSES.index(c) for c in _NEUTRAL_MAP]


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    model.eval().to(device)
    tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        feats = []
        for c in CLIP_CLASSES:
            tf = model.encode_text(tok([t.format(c) for t in TEMPLATES]).to(device))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append(tf.mean(0))
        text_feats = torch.stack(feats)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return model, preprocess, text_feats, model.logit_scale.exp().item()


def band_stats(crop, gains):
    """Median brightness and median saturation of the WB'd hood band.

    Median saturation (not a pixel fraction) separates a uniformly coloured car
    from a white van with a coloured logo: the logo barely moves the median.
    """
    h, w = crop.shape[:2]
    b = crop[int(h * 0.5):int(h * 0.7), int(w * 0.25):int(w * 0.75)]
    if b.size == 0:
        b = crop
    b = np.clip(b.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])), float(np.median(hsv[:, :, 1]))


def collect(videos, per_track, frame_step, min_size):
    items = []
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"[skip] {vid}")
            continue
        tracker = VehicleTracker()
        count = defaultdict(int)
        tag = Path(vid).stem
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % frame_step:
                continue
            gains = compute_wb_gains(frame)
            for t in tracker.update(frame):
                if count[t.track_id] >= per_track:
                    continue
                x1, y1, x2, y2 = t.bbox
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0 or min(crop.shape[:2]) < min_size:
                    continue
                medv, meds = band_stats(crop, gains)
                items.append({"tag": tag, "crop": crop.copy(), "medv": medv, "meds": meds})
                count[t.track_id] += 1
        cap.release()
        print(f"[{tag}] running total {len(items)}")
    return items


def clip_label(items, model, preprocess, text_feats, logit_scale, device, batch=32):
    def tight(c):
        h, w = c.shape[:2]
        return c[int(h * 0.12):int(h * 0.92), int(w * 0.12):int(w * 0.88)]
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([preprocess(Image.fromarray(
            cv2.cvtColor(tight(it["crop"]), cv2.COLOR_BGR2RGB))) for it in chunk]).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (logit_scale * (f @ text_feats.T)).softmax(-1)
        conf, idx = probs.max(-1)
        neutral_names = list(_NEUTRAL_MAP)
        neutral_pick = probs[:, _NEUTRAL_IDX].argmax(-1)
        for it, c, j, nj in zip(chunk, conf.tolist(), idx.tolist(), neutral_pick.tolist()):
            it["clip"], it["conf"] = CLIP_CLASSES[j], c
            it["neutral"] = _NEUTRAL_MAP[neutral_names[nj]]
        print(f"  labelled {min(i + batch, len(items))}/{len(items)}")


def assign(items, clip_conf, sat_med, black_v, white_v):
    """Hybrid label.

    Saturated colour: CLIP confident AND the band is uniformly saturated (median
    saturation high) — this rejects white vans with a coloured logo.
    Neutral: CLIP's whole-car opinion plus a brightness floor for black, so a
    glossy black hood reflecting the sky isn't mislabelled white.
    """
    for it in items:
        if (it["clip"] in SATURATED and it["conf"] >= clip_conf
                and it["meds"] >= sat_med):
            it["label"] = it["clip"]
            continue
        # CLIP rescues black (its whole-car view beats a glossy hood's brightness);
        # white vs gray is left to brightness so silver cars don't drift to white.
        v, nb = it["medv"], it["neutral"]
        if v < black_v or nb == "black":
            it["label"] = "black"
        elif v > white_v:
            it["label"] = "white"
        else:
            it["label"] = "gray"


def montage(items, path, per_class=4):
    cells = []
    by = defaultdict(list)
    for it in items:
        by[it["label"]].append(it)
    for cls in SATURATED + NEUTRAL:
        row = by.get(cls, [])[:per_class]
        for it in row:
            c = cv2.resize(it["crop"], (110, 110))
            cv2.rectangle(c, (0, 92), (109, 109), (0, 0, 0), -1)
            cv2.putText(c, cls, (3, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(c)
        for _ in range(per_class - len(row)):
            cells.append(np.zeros((110, 110, 3), np.uint8))
    grid = np.zeros((len(SATURATED + NEUTRAL) * 110, per_class * 110, 3), np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, per_class)
        grid[r * 110:r * 110 + 110, col * 110:col * 110 + 110] = c
    cv2.imwrite(path, grid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default="dataset")
    p.add_argument("--per-track", type=int, default=3)
    p.add_argument("--frame-step", type=int, default=5)
    p.add_argument("--min-size", type=int, default=40)
    p.add_argument("--clip-conf", type=float, default=0.55)
    p.add_argument("--sat-med", type=float, default=55)    # median band saturation
    p.add_argument("--black-v", type=float, default=72)    # brightness floor for black
    p.add_argument("--white-v", type=float, default=175)   # brightness ceiling for white
    p.add_argument("--montage", default=None)
    args = p.parse_args()

    device = pick_device()
    print(f"[device] {device}")
    model, preprocess, text_feats, ls = build_clip(device)

    items = collect(args.videos, args.per_track, args.frame_step, args.min_size)
    print(f"[collect] {len(items)} crops")
    clip_label(items, model, preprocess, text_feats, ls, device)
    assign(items, args.clip_conf, args.sat_med, args.black_v, args.white_v)

    out = Path(args.out)
    dist = Counter()
    for k, it in enumerate(items):
        d = out / it["label"]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{it['tag']}_{k}.jpg"), it["crop"])
        dist[it["label"]] += 1

    print(f"\n[done] {sum(dist.values())} labelled crops")
    for c in SATURATED + NEUTRAL:
        print(f"  {c:8s} {dist[c]}")
    if args.montage:
        montage(items, args.montage)
        print(f"[montage] {args.montage}")


if __name__ == "__main__":
    main()
