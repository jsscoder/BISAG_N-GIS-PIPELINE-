````md
# Hardened Semantic Segmentation Pipeline (U-Net)

A **production-safe, fault-tolerant semantic segmentation training and evaluation pipeline**
built with **PyTorch**, designed for **remote sensing, GIS, and TIFF-heavy datasets**.

This repository prioritizes **robustness, reproducibility, and GIS-aligned outputs** over toy or notebook-only experiments.

---

## 🇮🇳 Government of India – BISAG-N Alignment

### Institutional Context
This project is designed for **geospatial and remote sensing workflows** aligned with technical practices commonly followed by:

**Bhaskaracharya National Institute for Space Applications and Geo-informatics (BISAG-N)**  
**Government of India**

BISAG-N is an autonomous scientific society under the  
**Ministry of Electronics & Information Technology (MeitY), Government of India**,  
working in satellite-based geospatial technologies, GIS systems, and remote sensing applications.

### Intended Application Domains
- **Land Use / Land Cover (LULC) classification**
- **Satellite & aerial image semantic segmentation**
- **GIS-ready area estimation (m²)**
- **Government, academic, and research-grade geospatial datasets**

### Disclaimer
This repository is an **independent technical implementation** inspired by domain practices.  
It is **not an official BISAG-N or Government of India product** unless explicitly stated.

---

## 🚀 Key Features

### Data Handling
- Robust TIFF loading (`tifffile → PIL → OpenCV` fallback)
- Automatic **raw mask value → contiguous class index remapping**
- Images without matching masks are **automatically skipped**
- Safe DataLoader defaults (debug-friendly, worker-safe)

### Training & Performance
- U-Net with **automatic scaling based on available GPU VRAM**
- Mixed Precision (AMP) **enabled only when CUDA is available**
- Gradient accumulation support
- Resume training from checkpoints
- Stable learning-rate scheduling

### Evaluation & Analytics
- Memory-safe confusion matrix (no sklearn OOM issues)
- Per-class **IoU, Dice, Precision, Recall, F1**
- GIS-ready **per-image area computation (m²)**
- Class-wise correlation heatmaps
- Clean CSV + PNG outputs for reporting

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
````

**Rules**

* Image and mask filenames must match
* Mask values may be arbitrary integers (auto-remapped internally)

---

## 🏷 Class Configuration

```python
CLASS_NAMES = [
  "background",
  "water",
  "land",
  "vegetation",
  "barren",
  "built_up"
]
```

* Classes are **auto-detected from masks**
* Manual override is supported via `CLASS_NAMES`

---

## 📦 Outputs (`outputs/`)

| File                               | Description                                    |
| ---------------------------------- | ---------------------------------------------- |
| `best_unet.pth`                    | Best-performing model (lowest validation loss) |
| `checkpoint.pth`                   | Training resume checkpoint                     |
| `training_log.csv`                 | Epoch-wise train/validation loss               |
| `per_class_metrics.csv`            | IoU, Dice, Precision, Recall, F1               |
| `confusion_matrix_*.png`           | Raw & normalized confusion matrices            |
| `per_image_area_m2.csv`            | GIS-ready area statistics                      |
| `gt_area_fraction_correlation.png` | Class fraction correlation                     |
| `example_overlay_*.png`            | Input / GT / Prediction overlays               |

---

## 🧠 Model Architecture

### Base Model

* **U-Net**

### Automatic Capacity Scaling

| GPU VRAM | Base Filters |
| -------- | ------------ |
| > 6 GB   | 64           |
| 3–6 GB   | 32           |
| < 3 GB   | 16           |

Designed to **avoid OOM crashes** on low-memory GPUs.

---

## 🛡 Stability Guarantees

This pipeline will **not crash** if:

* Some images have no corresponding masks
* TIFFs are multi-band or non-standard
* Dataset size is extremely small (even 1 image)
* CUDA is unavailable (CPU fallback supported)

Metrics and plots are **safely skipped** if predictions are unavailable.

---

## 📍 GIS Notes

Pixel-to-area conversion is supported:

```python
PIXEL_SIZE_M = 10  # Example: Sentinel-2
```

* All area outputs are reported in **square meters (m²)**
* Suitable for LULC and geospatial reporting pipelines

---

## ⚙️ Installation

```bash
git clone https://github.com/jsscoder/BISAG_N-GIS-PIPELINE-.git
cd hardened-segmentation-pipeline
pip install -r requirements.txt
```

> For CUDA support, install PyTorch from [https://pytorch.org](https://pytorch.org)

---

## ▶️ Run Training

```bash
python hardened_segmentation_pipeline_v2.py
```

---

## 📜 License

**MIT License**
Use freely. Modify aggressively. Deploy responsibly.

---

## 👨‍💻 Author & Intent

Built for **real-world remote-sensing and GIS pipelines**,
not academic demos or notebook-only experiments.

---

## 🔮 Optional Extensions

This pipeline can be extended with:

* Docker support
* TensorBoard / Weights & Biases logging
* Multi-GPU training (DDP)
* Advanced architectures (DeepLabV3, Swin-UNet)
* Inference-only / deployment scripts

```
```
