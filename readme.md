# Hardened Semantic Segmentation Pipeline (U-Net)

A **production-safe, fault-tolerant semantic segmentation training + evaluation pipeline**
built with **PyTorch**. Designed for **remote sensing / GIS / TIFF-heavy datasets**.

---

## 🚀 Features

- Robust TIFF loading (`tifffile → PIL → OpenCV` fallback)
- Automatic **raw-mask → contiguous label remapping**
- Skips images **without matching masks**
- Safe DataLoader defaults (debug-friendly)
- Automatic UNet scaling based on **available GPU VRAM**
- AMP (mixed precision) **only when CUDA is available**
- Resume training from checkpoints
- Memory-safe confusion matrix (no sklearn blowups)
- Per-class IoU, Dice, Precision, Recall, F1
- Per-image **area computation (m²)** for GIS use
- Correlation heatmap of land-cover fractions
- Clean CSV + PNG outputs (ready for reports)

---

## 📁 Dataset Structure

```text
dataset/
├── images/
│   ├── img_001.tif
│   ├── img_002.png
│   └── ...
└── masks/
    ├── img_001.tif
    ├── img_002.png
    └── ...


CLASS_NAMES = [
  "background",
  "water",
  "land",
  "vegetation",
  "barren",
  "built_up"
]
📦 Outputs (outputs/)
File	Description
best_unet.pth	Best model (lowest val loss)
checkpoint.pth	Resume-training checkpoint
training_log.csv	Epoch-wise train/val loss
per_class_metrics.csv	IoU, Dice, F1 per class
confusion_matrix_*.png	Counts + normalized
per_image_area_m2.csv	GIS-ready area stats
gt_area_fraction_correlation.png	Class correlation
example_overlay_*.png	Visual sanity checks
🧠 Model

U-Net

Auto-scaled base filters:

64 (VRAM > 6GB)

32 (VRAM 3–6GB)

16 (low-memory GPUs)

🛡 Stability Guarantees

Won’t crash if:

Some images lack masks

TIFFs are multi-band

Dataset is tiny (even 1 image)

CUDA is unavailable

Skips metrics safely if no predictions

📍 GIS Notes

Pixel area conversion supported:

PIXEL_SIZE_M = 10  # Sentinel-2 example


Area output is in square meters (m²).

📜 License

MIT — use freely, modify aggressively.

👨‍💻 Author

Built for real-world remote-sensing pipelines, not toy notebooks.


---

If you want:
- **Dockerfile**
- **W&B / TensorBoard logging**
- **Multi-GPU (DDP)**
- **DeepLabV3 / Swin-UNet**
- **Inference-only script**

Say the word.