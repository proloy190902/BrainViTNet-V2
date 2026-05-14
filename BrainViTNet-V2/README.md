# BrainViTNet-V2 🧠

**A Hybrid ResNet-ViT Architecture for Brain Tumor Classification with Uncertainty Estimation**


## 🏗️ Novel Components

| # | Component | Description |
|---|-----------|-------------|
| ① | **2.5D Slice Fusion** | Multi-scale depthwise conv (3×3, 5×5, 7×7) with gated fusion |
| ② | **Domain-Adaptive BN** | Per-domain running stats with soft gating at inference |
| ③ | **Deformable Cross-Scale Attention (DCSA)** | SE-gated multi-scale feature fusion |
| ④ | **Dual Uncertainty** | MC Dropout (epistemic) + Aleatoric head with log-variance |
| ⑤ | **Cross-Domain Contrastive Loss** | Stable supervised contrastive with logsumexp fix |
| ⑥ | **Grad-CAM++** | Explainability visualizations |
| ⑦ | **Temperature Calibration** | LBFGS-optimized post-hoc calibration |
| ⑧ | **LR Warmup + Cosine Decay** | Stable early training, smooth decay |

---

## 📁 Project Structure

```
BrainViTNet-V2/
├── src/
│   ├── model.py       # All model components & architecture
│   ├── train.py       # Training loop
│   └── evaluate.py    # Evaluation + visualization
├── results/           # Output plots saved here
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your dataset paths
Edit `TRAIN_DIR` and `TEST_DIR` in both `src/train.py` and `src/evaluate.py`.

```python
TRAIN_DIR = "/path/to/your/Training"
TEST_DIR  = "/path/to/your/Testing"
```

### 3. Train
```bash
cd src
python train.py
```

### 4. Evaluate & Visualize
```bash
python evaluate.py
```

---

## ⚙️ Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Optimizer | AdamW (lr = 1e-4, wd = 1e-4) |
| Embed Dim | 512 |
| ViT Heads | 8 |
| MC Samples (T) | 10 |
| Contrastive Temp | 0.07 |

---




---

