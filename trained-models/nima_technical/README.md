# Model Summary

The **LemGendary NIMA Technical Scorer** is a professional-grade AI model optimized for the `quality` lifecycle within the LemGendary Training Suite. 

- **Architecture**: NIMA_Model (EfficientNetV2-S (Spatial Integrity))
- **Input Resolution**: 384x384
- **Use Case**: Technical quality scorer trained on custom standardized LemGendizedQualityDataset, optimized for detecting micro-defects, noise, and artifacts.
- **Training Data**: LemGendizedQualityDataset
- **Evaluation**: Validated against SOTA quality baselines.

> [!IMPORTANT]
> **Quality Vector**: This model is specialized for **Technical Integrity**. 
> - **Primary Targets**: Noise, Blur, Compression, Sharpness.


## Usage

```python
import torch
from PIL import Image
from models.nima import NIMA_Model

# 1. Initialize
model = NIMA_Model()
model.load_state_dict(torch.load("nima_technical_latest.pth"))
model.eval()

# 2. Forward Pass
img = Image.open("photo.jpg").resize((384, 384))
probs = model(img)

# 3. Scale Calculation
scores = torch.arange(1, 11).float()
mean_score = torch.sum(probs * scores).item()
print(f"Quality Score: {mean_score:.2f}")
```

- **Input Requirements**: RGB Image Tensors normalized to ImageNet stats.
- **Output Characteristics**: Quality predictive arrays.
- **Failures**: Large aspect ratio distortions during the standard resize phases.

## System

This model is a core module within the **LemGendary AI Training Suite**. 
- **Upstream**: Compressed/Raw RGB Buffers.
- **Downstream**: Dynamic restoration feedback loops and automated sorting scripts.

## Implementation requirements

- **Hardware**: NVIDIA GeForce GTX 1650 (4G VRAM)
- **Software**: PyTorch 2.11+, CUDA 12.1.
- **Training Lifecycle**: Successfully processed over 5 total epochs securely.

# Model Characteristics

## Model initialization

The model uses a backbone pre-trained on ImageNet-1K with custom adaptation layers for the 2026 specialization phase.

## Model stats

- **Precision**: ONNX FP16 (Edge) / PyTorch FP32 (Training).
- **Latency**: Sub-50ms inference bound on target local GPU hardware.
- **Ejection**: Weight tensors are decoupled into sidecar `.data` files for WebGPU stability.

## Other details

The matrix is optimized for browser-based execution via **ONNX Runtime Web**, bypassing standard browser memory constraints.

## Stability Constraints

Trained using **Earth Mover's Distance (EMD)** with strict 0.1 Temperature Anchoring to prevent probability collapse. The batch-level PLCC penalty is explicitly disabled to preserve global True Rank Correlation (SRCC).

# Data Overview

## Training data

Collected and curated from the following high-fidelity arrays:
- **LemGendizedQualityDataset**: ~440k binary image samples.

## Demographic groups

N/A. This matrix assesses photographic composition and signal restoration integrity.

## Evaluation data

Managed via an **80/20 train/validate split** with zero sample-leakage across the validation matrix.

# Evaluation Results

## Summary

The model has been structurally converged to achieve the following SOTA baselines:
- **Baseline Achievement**: **PLCC**: 0.9832310080528259 | **SRCC**: 0.8785159108168068

## Fairness 

Stability is optimized across low-dynamic-range and high-dynamic-range scenarios equally.

## Usage limitations

The model is a statistical estimator; it should not be used as an absolute arbiter of artistic value without human oversight.

## Ethics

Developed with an emphasis on **Earth Mover's Distance** (where applicable) and **Perceptual Loss** (LPIPS) to ensure result alignment with human subjective quality judgments.
