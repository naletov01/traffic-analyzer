"""Train the small colour classifier (student) on the auto-labelled dataset.

MobileNetV3-small, transfer-learned from ImageNet, exported to ONNX for edge use.
Only geometric augmentation is used — colour jitter would corrupt the labels.

    python tools/train_color.py --data dataset --epochs 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

IMG = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_loaders(data_dir, batch):
    train_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    full = datasets.ImageFolder(data_dir)
    n_val = int(len(full) * 0.15)
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_set.dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    val_set.dataset = datasets.ImageFolder(data_dir, transform=val_tf)
    return (DataLoader(train_set, batch, shuffle=True),
            DataLoader(val_set, batch),
            full.classes, full.targets)


def class_weights(targets, n_classes, device):
    counts = np.bincount(targets, minlength=n_classes)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="weights/color_mobilenet.onnx")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

    train_dl, val_dl, classes, targets = make_loaders(args.data, args.batch)
    print(f"[data] {len(targets)} imgs, classes={classes}")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights(targets, len(classes), device))
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        model.eval()
        correct = total = 0
        per_cls = np.zeros((len(classes), 2))  # correct, total
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
                for p, t in zip(pred, y):
                    per_cls[t, 1] += 1
                    per_cls[t, 0] += int(p == t)
        acc = correct / total
        print(f"ep{ep:2d} val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.onnx.export(
                model.cpu().eval(),
                torch.randn(1, 3, IMG, IMG),
                args.out, input_names=["input"], output_names=["logits"],
                opset_version=12, dynamic_axes={"input": {0: "n"}})
            model.to(device)

    # The exporter writes weights to a sidecar .data file; fold them back into a
    # single self-contained .onnx for easy shipping.
    import onnx
    onnx.save_model(onnx.load(args.out), args.out, save_as_external_data=False)
    Path(args.out + ".data").unlink(missing_ok=True)

    print(f"\n[best] val_acc={best:.3f}")
    print("per-class val accuracy:")
    for i, c in enumerate(classes):
        t = per_cls[i, 1]
        print(f"  {c:8s} {per_cls[i,0]/t:.2f}" if t else f"  {c:8s} (none)")
    print(f"[saved] {args.out}")
    Path("weights/color_classes.txt").write_text("\n".join(classes))
    print("[saved] weights/color_classes.txt")


if __name__ == "__main__":
    main()
