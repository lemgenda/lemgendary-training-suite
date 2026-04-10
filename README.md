# LemGendary AI Training Suite (v7.0.0-LEMGENDARY)

> **The Professional Standard for Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision models natively on Windows with 2026-era Resilience Architecture.

---

## ⚡ 2026 Resilience Architecture

The v7.0 release introduces the **Manifold Anchor Protocol** and **Automated Singularity Rollbacks**, ensuring that even 50-hour training runs remain stable against numerical explosions (NaN) and dataset corruption.

### 🧵 The LemGendary Hub
The master orchestration console for all training activities. Automatically manages `.venv` isolation, hardware-specific library selection (NVIDIA CUDA 12.1), and sequential project execution.

### 🧬 Unified Multitask DNA
Fully synchronized with the [LemGendary Dataset Engine (v3.0)](../lemgendary-datasets/README.md).
- **Native Support**: Detection, Instance Segmentation (Polygons), and Pose Estimation (Keypoints).
- **SOTA Alignment**: Automated metric scoring against PLCC, SRCC, PSNR, and mAP@0.5:0.95.
- **Multitask Dataset Loader**: High-performance pipeline for streaming massive Kaggle datasets with zero-latency prefetching.

---

## 🛠️ Getting Started

### 1. The Models Hub
The central TUI for managing models, environment, and cloud deployments.
```powershell
./lemgendary_models_hub.ps1
```

### 2. Manual Orchestration
For advanced users who require direct pipeline control.
```bash
# Individual Model Run
python training/train.py --model nima_technical

# Global Unit Test (Dry Run)
python train_all.py --epochs 1 --yes

# Smart Cloud Streaming (Kaggle API)
python smart_orchestrator.py
```

---

## 📂 Project Anatomy

- `training/` — Core backpropagation logic and stability hooks (`train.py`, `losses.py`).
- `models/` — Specialized model architecture definitions and weight loaders.
- `trained-models/` — Production-ready artifacts (ONNX FP16/FP32).
- `technical_papers/` — High-fidelity architecture whitepapers and research notes.
- `unified_models.yaml` — The Master Registry for all 21+ supported neural networks.
- `unified_data.yaml` — Global dataset mapping and pathing overrides.

---

## 🚀 Technical Whitepapers
Deep dives into the mathematical foundations and hardware-aware optimizations of the suite:

- **[NIMA: Quality Assessment (Markdown)](technical_papers/PAPER_LEMGENDARY_NIMA.md)**
- **[NAFNet: Resurrection & Restoration (Markdown)](technical_papers/PAPER_LEMGENDARY_NAFNET.md)**
- **[Premium HTML Versions](https://lemgenda.github.io/lemgendary-training-suite/)**

---

## 🛡️ Resilience Metrics (v2026 Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Singularity Rollback** | Detection of numerical explosions (NaN) triggers 50% LR cooling and weight restoration. | ✅ Active |
| **Serial Extraction Mutex** | ZIP extractions are serialized to prevent Windows file-system contention. | ✅ Active |
| **Zero-Latency Prefetch** | Subsequent datasets are streamed/unpacked in the background during active training. | ✅ Active |
| **SOTA Continuity** | Automated epoch extension if benchmarks aren't hit within initial limits. | ✅ Active |

---
**LemGendary AI Suite | Advanced Agentic Coding 2026**
