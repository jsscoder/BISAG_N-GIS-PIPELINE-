"""
Hardened segmentation training + evaluation pipeline
Save as hardened_segmentation_pipeline_v2.py and run in your training environment.
Key features:
 - Robust TIFF loading via tifffile (fallback to PIL/OpenCV)
 - Mask raw-value -> contiguous-index remapping
 - Filter images without matching masks at Dataset init
 - Safer DataLoader worker usage during debugging (NUM_WORKERS=0 default)
 - Conditional AMP scaler only when CUDA available
 - Auto-scale UNet base filters by available GPU memory
 - Checkpointing and resume support
 - Memory-safe confusion matrix (bincount)
 - Skip metric/visual generation when no predictions
 - CSV logging for train/val losses and per-class metrics
 - Stable plotting and deduped correlation heatmap
"""

import os
import gc
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import warnings
warnings.filterwarnings("ignore")

# optionally import tifffile if available
try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    tifffile = None
    _HAS_TIFFFILE = False

# -----------------------
# Config (edit as needed)
# -----------------------
IMG_DIR = "dataset/images"   # adjust
MASK_DIR = "dataset/masks"   # adjust
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
VAL_SPLIT = 0.2
N_EPOCHS = 30
ACCUM_STEPS = 4
LR = 1e-3
NUM_WORKERS = 16 # safer during debugging; set >0 when loader is stable

# Optional: provide friendly class names in order of label values (0..N-1)
CLASS_NAMES = ["background", "water", "land", "vegetation", "barren", "built_up"]

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Utility: find mask filename for an image
# -----------------------

def find_mask_for_image(img_name, mask_dir):
    cand = os.path.join(mask_dir, img_name)
    if os.path.exists(cand):
        return cand
    base, _ = os.path.splitext(img_name)
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        cand = os.path.join(mask_dir, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask for image {img_name} not found in {mask_dir}")

# -----------------------
# Helper: robust image loader (returns PIL.Image)
# -----------------------

def _load_image_with_tiff_support(path):
    path = str(path)
    # prefer tifffile for .tif / .tiff
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff') and _HAS_TIFFFILE:
        try:
            arr = tifffile.imread(path)
            # if shape is (C,H,W), transpose
            if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[0] != arr.shape[2]:
                arr = np.transpose(arr, (1,2,0))
            # single band -> make 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            # if more than 3 bands, take first 3
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            # normalize/convert to uint8 if needed
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            elif not np.issubdtype(arr.dtype, np.uint8):
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        except Exception:
            # fallback to other loaders below
            pass
    # try PIL
    try:
        return Image.open(path)
    except UnidentifiedImageError:
        # try OpenCV as fallback
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise
        # OpenCV returns BGR
        if arr.ndim == 3:
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr = np.stack([arr]*3, axis=-1)
        return Image.fromarray(arr)

# -----------------------
# Determine number of classes robustly (scan masks for unique labels)
# -----------------------

def compute_n_classes(mask_dir, sample_limit=None):
    files = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])
    if sample_limit:
        files = files[:sample_limit]
    uniq = set()
    for f in files:
        p = os.path.join(mask_dir, f)
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext in ('.tif', '.tiff') and _HAS_TIFFFILE:
                arr = tifffile.imread(p)
                if arr.ndim == 3:
                    # if rgb-coded mask, we try first channel
                    if arr.shape[2] in (3,4):
                        arr = arr[..., 0]
                    elif arr.shape[0] <= 8:
                        arr = np.transpose(arr, (1,2,0))
                        arr = arr[..., 0]
                m = np.asarray(arr)
            else:
                m = np.array(Image.open(p).convert('L'))
        except Exception:
            continue
        uniq |= set(np.unique(m).tolist())
    uniq = sorted([int(x) for x in uniq])
    if len(uniq) == 0:
        raise RuntimeError("No labels found in masks")
    if uniq == list(range(0, max(uniq)+1)):
        return max(uniq) + 1, uniq
    return len(uniq), uniq

# -----------------------
# Dataset
# -----------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        raw_images = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        # filter images that have masks; log missing
        self.images = []
        missing = []
        for img in raw_images:
            try:
                _ = find_mask_for_image(img, mask_dir)
                self.images.append(img)
            except FileNotFoundError:
                missing.append(img)
        if missing:
            print(f"Warning: {len(missing)} images have no masks and will be skipped. Example: {missing[:5]}")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = find_mask_for_image(img_name, self.mask_dir)

        # load image (robust)
        try:
            img = _load_image_with_tiff_support(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            # if image can't be loaded, create placeholder and log
            print(f"Warning: failed to load image {img_path}: {e}. Using black placeholder.")
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0,0,0))

        # load mask robustly
        try:
            ext = os.path.splitext(mask_path)[1].lower()
            if ext in ('.tif', '.tiff') and _HAS_TIFFFILE:
                m_arr = tifffile.imread(mask_path)
                if m_arr.ndim == 3:
                    # assume first channel encodes labels (if RGB-coded mask)
                    if m_arr.shape[2] in (3,4):
                        m_arr = m_arr[..., 0]
                    else:
                        m_arr = np.transpose(m_arr, (1,2,0))
                        m_arr = m_arr[..., 0]
                mask_img = Image.fromarray(m_arr)
                mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            else:
                mask_img = Image.open(mask_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        except Exception as e:
            raise RuntimeError(f"Failed to read mask {mask_path}: {e}")

        if self.transform:
            img = self.transform(img)

        mask = np.array(mask_img, dtype=np.int64)

        # remap raw label values -> contiguous indices
        global label_to_index, n_classes
        if 'label_to_index' in globals():
            # values not found in mapping -> -1
            vec_get = np.vectorize(lambda x: label_to_index.get(int(x), -1))
            mask = vec_get(mask).astype(np.int64)
            if mask.min() < 0 or (n_classes is not None and mask.max() >= n_classes):
                raise ValueError(f"Remapped mask index out of range for file {img_name}: min {mask.min()} max {mask.max()}")
        else:
            # if mapping does not exist, assume mask already 0..n_classes-1
            pass

        return img, torch.tensor(mask, dtype=torch.long), img_name

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# build dataset (dataset will filter images without masks)
print("Building dataset...")
dataset = SegmentationDataset(IMG_DIR, MASK_DIR, transform=transform)
print(f"Found {len(dataset)} usable images (images with masks).")

# compute classes (fast scan)
try:
    n_classes, label_values = compute_n_classes(MASK_DIR)
    print("n_classes (detected):", n_classes)
    print("label values found:", label_values)
except Exception as e:
    print("Could not auto-detect classes:", e)
    if CLASS_NAMES is not None:
        n_classes = len(CLASS_NAMES)
        label_values = list(range(n_classes))
        print(f"Falling back to n_classes={n_classes} from CLASS_NAMES")
    else:
        n_classes = 2
        label_values = [0,1]
        print("Falling back to n_classes=2")

# build mapping raw->index
label_to_index = {raw: idx for idx, raw in enumerate(label_values)}
index_to_label = {idx: raw for raw, idx in label_to_index.items()}
print("label_to_index mapping:", label_to_index)

# reconcile CLASS_NAMES ordering with detected labels
if CLASS_NAMES is not None and len(CLASS_NAMES) >= n_classes:
    CLASS_NAMES = CLASS_NAMES[:n_classes]
else:
    CLASS_NAMES = [str(v) for v in label_values]
print("Using class names:", CLASS_NAMES)

# split
# -----------------------
# Safe batch-handling split (FIXED)
# -----------------------

ds_len = len(dataset)

# Case 0: No usable samples → stop
if ds_len == 0:
    raise RuntimeError("❌ No usable image–mask pairs were found. Fix dataset file names.")

# Case 1: Only one image → no training possible
elif ds_len == 1:
    print("⚠️ Only ONE usable sample found → training skipped. Using it only for evaluation.")
    train_dataset = None
    val_dataset = dataset
    train_loader = None
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
else:
    # Normal split
    val_size = max(1, int(ds_len * VAL_SPLIT))
    train_size = ds_len - val_size

    print(f"Dataset size: {ds_len} → Train: {train_size} | Val: {val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type=="cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type=="cuda")
    )



if train_dataset:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(device.type=="cuda"), num_workers=NUM_WORKERS)
else:
    train_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(device.type=="cuda"), num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=(device.type=="cuda"), num_workers=NUM_WORKERS)

# -----------------------
# Model: U-Net (same architecture but smaller init if memory tight)
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes, base_filters=64):
        super().__init__()
        f = base_filters
        self.down1 = DoubleConv(3, f)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(f*4, f*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.conv4 = DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.conv3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.conv2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.conv1 = DoubleConv(f*2, f)

        self.out_conv = nn.Conv2d(f, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)
        bn = self.bottleneck(p4)

        up4 = self.up4(bn); c4 = self.conv4(torch.cat([up4, d4], 1))
        up3 = self.up3(c4); c3 = self.conv3(torch.cat([up3, d3], 1))
        up2 = self.up2(c3); c2 = self.conv2(torch.cat([up2, d2], 1))
        up1 = self.up1(c2); c1 = self.conv1(torch.cat([up1, d1], 1))

        return self.out_conv(c1)

# auto-scale base filters based on available GPU memory to avoid OOM
try:
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
    else:
        vram_gb = 4
except Exception:
    vram_gb = 4

base_filters = 64 if vram_gb > 6 else 32 if vram_gb > 3 else 16
model = UNet(n_classes, base_filters=base_filters).to(device)
print(f"UNet initialized with base_filters={base_filters}")

# -----------------------
# Loss + Optimizer
# -----------------------
if n_classes == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
scaler = torch.cuda.amp.GradScaler() if device.type=="cuda" else None

# checkpoint path
ckpt_path = os.path.join(OUT_DIR, "checkpoint.pth")
best_path = os.path.join(OUT_DIR, "best_unet.pth")
start_epoch = 0
best_val = float("inf")

# try resume
if os.path.exists(ckpt_path):
    try:
        chk = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['optim'])
        start_epoch = chk.get('epoch', 0) + 1
        best_val = chk.get('best_val', best_val)
        print(f"Resumed from checkpoint epoch {start_epoch}")
    except Exception as e:
        print("Warning: failed to load checkpoint:", e)

# -----------------------
# Small utilities for metrics
# -----------------------

def fast_confusion_matrix(gt, pred, n_classes):
    """Memory-efficient confusion matrix using bincount. gt and pred are 1D arrays of same length."""
    assert gt.shape == pred.shape
    mask = (gt >= 0) & (gt < n_classes)
    if mask.sum() == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    combined = n_classes * gt[mask].astype(int) + pred[mask].astype(int)
    cm = np.bincount(combined, minlength=n_classes**2).reshape(n_classes, n_classes)
    return cm


def compute_metrics_per_class(gt, pred, n_classes):
    eps = 1e-8
    ious, dices, precisions, recalls, f1s = [], [], [], [], []
    for c in range(n_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        intersection = (gt_c & pred_c).sum()
        union = (gt_c | pred_c).sum()
        tp = intersection
        fp = (pred_c & ~gt_c).sum()
        fn = (gt_c & ~pred_c).sum()

        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (gt_c.sum() + pred_c.sum() + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        'iou': np.array(ious),
        'dice': np.array(dices),
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s)
    }

# -----------------------
# Training
# -----------------------
train_losses, val_losses = [], []

for epoch in range(start_epoch, N_EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    nb_batches = max(1, len(train_loader))

    pbar = tqdm(enumerate(train_loader), total=nb_batches, desc=f"Epoch [{epoch+1}/{N_EPOCHS}]")
    for step, batch in pbar:
        if len(batch) == 3:
            imgs, masks, _ = batch
        else:
            imgs, masks = batch
        imgs, masks = imgs.to(device), masks.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(imgs)
                # BCE expects shape [B,1,H,W] / float masks; CrossEntropy expects [B,C,H,W] logits and Long masks
                if n_classes == 1:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, masks.float())
                else:
                    loss = criterion(outputs, masks)
                loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            outputs = model(imgs)
            if n_classes == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.float())
            else:
                loss = criterion(outputs, masks)
            loss = loss / ACCUM_STEPS
            loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * ACCUM_STEPS)
        pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}")

    # step remaining grads
    if (nb_batches % ACCUM_STEPS) != 0:
        if scaler is not None:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        else:
            optimizer.step(); optimizer.zero_grad()

    avg_train = running_loss / max(1, nb_batches)
    train_losses.append(avg_train)

    # validation
    model.eval()
    val_loss = 0.0
    all_preds, all_gts, all_names = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                imgs, masks, names = batch
            else:
                imgs, masks = batch
                names = ["unknown"] * imgs.shape[0]
            imgs, masks = imgs.to(device), masks.to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(imgs)
                    if n_classes == 1:
                        outputs_proc = outputs.squeeze(1)
                        loss = criterion(outputs_proc, masks.float())
                        preds = (torch.sigmoid(outputs_proc) > 0.5).long().cpu().numpy()
                    else:
                        loss = criterion(outputs, masks)
                        preds = outputs.argmax(1).cpu().numpy()
            else:
                outputs = model(imgs)
                if n_classes == 1:
                    outputs_proc = outputs.squeeze(1)
                    loss = criterion(outputs_proc, masks.float())
                    preds = (torch.sigmoid(outputs_proc) > 0.5).long().cpu().numpy()
                else:
                    loss = criterion(outputs, masks)
                    preds = outputs.argmax(1).cpu().numpy()

            val_loss += loss.item()
            all_preds.append(preds)
            all_gts.append(masks.cpu().numpy())
            all_names.extend(names)

    avg_val = val_loss / max(1, len(val_loader))
    val_losses.append(avg_val)
    scheduler.step(avg_val)

    print(f"Epoch {epoch+1:02d} — Train: {avg_train:.4f}  Val: {avg_val:.4f}")

    # checkpoint
    try:
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'best_val': best_val}, ckpt_path)
    except Exception as e:
        print("Could not write checkpoint:", e)

    if avg_val < best_val:
        best_val = avg_val
        try:
            torch.save(model.state_dict(), best_path)
            print("✅ Saved best model.")
        except Exception as e:
            print("Could not save best model:", e)

    torch.cuda.empty_cache(); gc.collect()

# -----------------------
# Loss curve (log-scale) - safe plotting
# -----------------------
plt.figure()
if train_losses:
    plt.semilogy(train_losses, label="Train Loss")
if val_losses:
    plt.semilogy(val_losses, label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Log Loss")
plt.title("Log-scaled Loss Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "log_loss_curve.png"))
plt.close()

# -----------------------
# Evaluation (collect preds with filenames)
# -----------------------
state_path = best_path if os.path.exists(best_path) else ckpt_path if os.path.exists(ckpt_path) else None
if state_path is not None:
    try:
        model.load_state_dict(torch.load(state_path, map_location=device))
        print(f"Loaded model weights from {state_path}")
    except Exception as e:
        print("Warning: could not load saved weights:", e)
else:
    print("Warning: no saved model file found; using current weights")

model.eval()
all_preds, all_gts = [], []
all_names = []

if len(val_loader) > 0:
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if len(batch) == 3:
                imgs, masks, names = batch
            else:
                imgs, masks = batch
                names = ["unknown"] * imgs.shape[0]
            imgs, masks = imgs.to(device), masks.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)

            if n_classes == 1:
                preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).long().cpu().numpy()
            else:
                preds = outputs.argmax(1).cpu().numpy()

            all_preds.append(preds)
            all_gts.append(masks.cpu().numpy())
            all_names.extend(names)

if all_preds:
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts, axis=0)
else:
    print("⚠️ No predictions collected. Skipping metrics/overlays.")
    all_preds = np.array([])
    all_gts = np.array([])

# -----------------------
# Confusion matrix (flattened) and normalized
# -----------------------
if all_preds.size and all_gts.size:
    cm = fast_confusion_matrix(all_gts.flatten(), all_preds.flatten(), n_classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (counts)')
    plt.colorbar()
    plt.xticks(range(n_classes), CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(range(n_classes), CLASS_NAMES)
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{cm[i,j]}", ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_counts.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Confusion Matrix (normalized by GT)')
    plt.colorbar()
    plt.xticks(range(n_classes), CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(range(n_classes), CLASS_NAMES)
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_normalized.png'))
    plt.close()
else:
    print("Skipping confusion matrix: no valid predictions or ground-truths.")

# -----------------------
# Per-class metrics (IoU, Dice, Precision, Recall, F1)
# -----------------------
if all_preds.size and all_gts.size:
    metrics = compute_metrics_per_class(all_gts.flatten(), all_preds.flatten(), n_classes)
    metrics_df = pd.DataFrame({
        'class': CLASS_NAMES,
        'iou': metrics['iou'],
        'dice': metrics['dice'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    })
    metrics_df.to_csv(os.path.join(OUT_DIR, 'per_class_metrics.csv'), index=False)
    print('Saved per-class metrics to', os.path.join(OUT_DIR, 'per_class_metrics.csv'))

    # Plot per-class IoU + Dice
    x = np.arange(len(CLASS_NAMES))
    plt.figure(figsize=(10,5))
    plt.plot(x, metrics_df['iou'], marker='o', label='IoU')
    plt.plot(x, metrics_df['dice'], marker='o', label='Dice')
    plt.xticks(x, CLASS_NAMES, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Per-class IoU and Dice')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_class_iou_dice.png'))
    plt.close()
else:
    print('Skipping per-class metrics: no predictions.')

# meters per pixel (adjust based on dataset metadata, e.g. 10 for Sentinel-2)
PIXEL_SIZE_M = 10

# -----------------------
# Per-image area area in meter square (gis data stanadpoint) and correlation heatmap
# ----------------------
if all_preds.size and all_gts.size and len(all_names) == all_gts.shape[0]:
    area_rows = []
    for i, name in enumerate(all_names):
        gt = all_gts[i]
        total_pixels = gt.size
        area_row = {'name': name}
        for c in range(n_classes):
            pixel_count = (gt == c).sum()
            area_m2 = int(pixel_count) * (PIXEL_SIZE_M ** 2)
            area_row[f'gt_area_m2_{CLASS_NAMES[c]}'] = area_m2
        area_rows.append(area_row)

    area_df = pd.DataFrame(area_rows)
    area_df.to_csv(os.path.join(OUT_DIR, 'per_image_area_m2.csv'), index=False)
    print('Saved per-image area (m²) to', os.path.join(OUT_DIR, 'per_image_area_m2.csv'))

    # Compute per-image class fractions
    per_image_df = area_df.copy()
    total_cols = [f'gt_area_m2_{cn}' for cn in CLASS_NAMES]
    per_image_df['total_area'] = per_image_df[total_cols].sum(axis=1)
    for c in range(n_classes):
        col = f'gt_area_m2_{CLASS_NAMES[c]}'
        per_image_df[f'gt_area_frac_{CLASS_NAMES[c]}'] = per_image_df[col] / (per_image_df['total_area'] + 1e-12)

    gt_frac_cols = [c for c in per_image_df.columns if c.startswith('gt_area_frac_')]
    gt_frac_matrix = per_image_df[gt_frac_cols].corr()

    plt.figure(figsize=(8,6))
    plt.imshow(gt_frac_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(gt_frac_cols)), [c.replace('gt_area_frac_','') for c in gt_frac_cols], rotation=45, ha='right')
    plt.yticks(range(len(gt_frac_cols)), [c.replace('gt_area_frac_','') for c in gt_frac_cols])
    plt.title('Correlation (GT area fractions across images)')
    for i in range(len(gt_frac_cols)):
        for j in range(len(gt_frac_cols)):
            plt.text(j, i, f"{gt_frac_matrix.values[i,j]:.2f}", ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gt_area_fraction_correlation.png'))
    plt.close()
else:
    print('Skipping per-image area/correlation: insufficient data.')

# -----------------------
# Palette and visualization helpers
# -----------------------

def make_palette():
    fixed_palette = {
        "background": (0, 0, 0),         # black
        "water": (114, 151, 158),        # teal
        "land": (240, 123, 107),         # reddish
        "vegetation": (58, 59, 23),      # dark green
        "barren": (120, 113, 63),        # olive brown
        "built_up": (200, 200, 200)      # light gray (fallback if needed)
    }
    pal = np.zeros((n_classes, 3), dtype=np.uint8)
    for i, cname in enumerate(CLASS_NAMES):
        pal[i] = fixed_palette.get(cname, (np.random.randint(0,255),
                                           np.random.randint(0,255),
                                           np.random.randint(0,255)))
    return pal


PALETTE = make_palette()


def colorize_mask(mask, palette=PALETTE):
    mask = np.asarray(mask, dtype=np.int64)
    h,w = mask.shape
    colored = np.zeros((h,w,3), dtype=np.uint8)
    for c in range(len(palette)):
        colored[mask==c] = palette[c]
    return colored

def overlay_on_image(img, mask_colored, alpha=0.5):
    base = np.array(img).astype(np.float32)/255.0
    mask_rgb = mask_colored.astype(np.float32)/255.0
    overlay = (1-alpha)*base + alpha*mask_rgb
    overlay = np.clip(overlay*255,0,255).astype(np.uint8)
    return overlay

# -----------------------
# Side-by-side visualization with overlay and legend
# -----------------------

def show_examples_with_gt_overlay(img_dir, names, preds, gts, n=5):
    n = min(n, len(names))
    for i in range(n):
        img_name = names[i]
        img_path = os.path.join(img_dir, img_name)
        try:
            img = _load_image_with_tiff_support(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
        img_np = np.array(img)

        pred = preds[i]
        gt = gts[i]

        # color masks
        colored_gt = colorize_mask(gt)
        colored_pred = colorize_mask(pred)
        overlay_pred = overlay_on_image(img_np, colored_pred, alpha=0.5)

        # figure with 4 subplots on top row
        fig = plt.figure(figsize=(16,6))
        gs = fig.add_gridspec(2, 4, height_ratios=[4, 1])

        ax0 = fig.add_subplot(gs[0,0]); ax0.imshow(img_np); ax0.set_title("Input"); ax0.axis('off')
        ax1 = fig.add_subplot(gs[0,1]); ax1.imshow(colored_gt); ax1.set_title("Ground Truth"); ax1.axis('off')
        ax2 = fig.add_subplot(gs[0,2]); ax2.imshow(colored_pred); ax2.set_title("Predicted"); ax2.axis('off')
        ax3 = fig.add_subplot(gs[0,3]); ax3.imshow(overlay_pred); ax3.set_title("Overlay"); ax3.axis('off')

        # bottom row: legend
        legend_ax = fig.add_subplot(gs[1,:])
        legend_ax.axis('off')
        n_classes_local = len(CLASS_NAMES)
        for k, cname in enumerate(CLASS_NAMES):
            x0 = k * (1.0 / n_classes_local)
            rect = plt.Rectangle((x0, 0.2), 1.0 / n_classes_local * 0.8, 0.6, facecolor=np.array(PALETTE[k])/255.0)
            legend_ax.add_patch(rect)
            legend_ax.text(x0 + 0.01, 0.85, cname, transform=legend_ax.transAxes, fontsize=10)

        plt.suptitle(f"Example {i+1}: {img_name}", fontsize=14)
        plt.tight_layout(rect=[0,0.05,1,0.95])
        fname = os.path.join(OUT_DIR, f"example_overlay_{i+1}_{img_name}.png")
        try:
            plt.savefig(fname)
        except Exception:
            # Try safer filename
            safe_name = f"example_overlay_{i+1}.png"
            plt.savefig(os.path.join(OUT_DIR, safe_name))
        plt.close(fig)

print("Generating side-by-side overlays with GT, Predicted, and Overlay...")
try:
    show_examples_with_gt_overlay(IMG_DIR, all_names, all_preds, all_gts, n=5)
    print("✅ Saved example overlays to", OUT_DIR)
except Exception as e:
    print("Could not generate overlays:", e)

# -----------------------
# Save training log CSV
# -----------------------
try:
    log_df = pd.DataFrame({'epoch': list(range(1, len(train_losses)+1)), 'train_loss': train_losses, 'val_loss': val_losses})
    log_df.to_csv(os.path.join(OUT_DIR, 'training_log.csv'), index=False)
    print('Saved training log to', os.path.join(OUT_DIR, 'training_log.csv'))
except Exception as e:
    print('Could not save training log:', e)

print('Pipeline finished.')