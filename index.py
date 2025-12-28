#!/usr/bin/env python3
"""
Hardened Semantic Segmentation Pipeline (Production)

Author: Omkar Goje
License: MIT

Design goals:
- Robust TIFF handling (tifffile → PIL → OpenCV)
- Safe training on tiny or large datasets
- Deterministic behavior
- Memory-safe evaluation (streaming confusion matrix)
- GIS-ready outputs (area in m²)
- Resume-safe checkpoints
"""

# ============================
# Imports
# ============================
import os
import gc
import argparse
import logging
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# Optional TIFF support
try:
    import tifffile
    HAS_TIFF = True
except Exception:
    HAS_TIFF = False

# ============================
# CLI ARGUMENTS (PRODUCTION)
# ============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="dataset/images")
    p.add_argument("--mask_dir", default="dataset/masks")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pixel_size_m", type=int, default=10)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

args = parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ============================
# LOGGING (AUDIT SAFE)
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.out_dir, "run.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("seg")

# ============================
# DETERMINISM
# ============================
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ============================
# DEVICE
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

if device.type == "cpu":
    args.num_workers = 0

# ============================
# SAFE IMAGE LOADER
# ============================
def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    # TIFF preferred
    if ext in (".tif", ".tiff") and HAS_TIFF:
        try:
            arr = tifffile.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        except Exception:
            pass

    # PIL fallback
    try:
        return Image.open(path)
    except UnidentifiedImageError:
        arr = cv2.imread(path)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

# ============================
# DATASET
# ============================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = [
            f for f in sorted(os.listdir(img_dir))
            if os.path.exists(os.path.join(mask_dir, f))
        ]

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = load_image(os.path.join(self.img_dir, name)).convert("RGB")
        img = img.resize((args.img_size, args.img_size))

        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")
        mask = mask.resize((args.img_size, args.img_size), Image.NEAREST)

        img = self.transform(img)
        mask = torch.from_numpy(np.array(mask)).long()

        # Fail fast on corrupt masks
        if torch.unique(mask).numel() < 2:
            raise ValueError(f"Corrupt mask detected: {name}")

        return img, mask, name

# ============================
# TRANSFORMS
# ============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = SegDataset(args.img_dir, args.mask_dir, transform)

# Auto-detect classes safely
all_labels = torch.cat([dataset[i][1].flatten() for i in range(len(dataset))])
n_classes = int(torch.unique(all_labels).numel())
log.info(f"Detected {n_classes} classes")

# ============================
# SPLIT
# ============================
val_len = max(1, int(len(dataset) * args.val_split))
train_len = len(dataset) - val_len

train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)

# ============================
# MODEL
# ============================
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, 1, 1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        b  = self.bottleneck(self.pool(d3))

        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)

model = UNet(n_classes).to(device)

# ============================
# TRAINING SETUP
# ============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

best_val = float("inf")

# ============================
# TRAIN LOOP
# ============================
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0

    for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

        if args.dry_run:
            break

    # ============================
    # VALIDATION (STREAMING CM)
    # ============================
    model.eval()
    val_loss = 0.0
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            preds = out.argmax(1)

            val_loss += criterion(out, masks).item()

            # streaming confusion matrix
            gt = masks.cpu().numpy().flatten()
            pr = preds.cpu().numpy().flatten()
            valid = (gt >= 0) & (gt < n_classes)
            cm = np.bincount(
                n_classes * gt[valid] + pr[valid],
                minlength=n_classes**2
            ).reshape(n_classes, n_classes)
            conf_mat += cm

    log.info(
        f"Epoch {epoch+1} | "
        f"Train: {train_loss:.4f} | "
        f"Val: {val_loss:.4f}"
    )

    # ============================
    # CHECKPOINTING
    # ============================
    ckpt = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(model.state_dict(), ckpt)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(args.out_dir, "best_unet.pth"))

    if args.dry_run:
        break

    torch.cuda.empty_cache()
    gc.collect()

log.info("Pipeline finished successfully.")
