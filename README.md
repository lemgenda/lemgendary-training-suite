# LemGendary AI Training Suite (v7.0.0-LEMGENDARY)

> **The Professional Standard for Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision models natively on Windows with 2026-era Resilience Architecture.

---

## ⚡ 2026 Resilience Architecture

The v7.2 release introduces the **Hyper-Convergence Patch (v2.6)** and **Stochastic Weight Averaging (SWA)**, ensuring that models never stall on plateaus and achieve superior generalization through manifold smoothing.

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
The master orchestration console. It manages the full lifecycle from system bootstrapping to multi-model cloud orchestration.
```powershell
./lemgendary_models_hub.ps1
```

#### 📋 Detailed Menu Structure
| Option | Action | Sub-Prompts & Details |
| :--- | :--- | :--- |
| **1. Initialize Systems** | **Environment Sync** | Installs Python 3.12, creates `.venv`, and Auto-Detects GPU. Installs PyTorch 2.4.1+cu121 (NVIDIA) or DirectML. |
| **2. Train Model** | **Interactive Selection** | Launches the **Category Submenu**: <br>• **1. Quality**: NIMA (Aesthetic/Technical) <br>• **2. Face/Det**: YOLOv8n, RetinaFace, CodeFormer <br>• **3. SuperRes**: UltraZoom Array (x2-x8) <br>• **4. Restoration**: NAFNet, MIRNet, FFANet, MPRNet <br>• **5. Hybrid**: UPN v2, Multi-Restorer, Film |
| **3. Global Orchestration** | **Continuous Train** | Executes sequential training for all 21+ models defined in `unified_models.yaml` uninterrupted. |
| **4. Deploy to Cloud** | **Kaggle Deployment** | Generates tailored instructions and topologies for remote Jupyter execution. |
| **5. Smart Cloud Ops** | **Hybrid Streaming** | Trains locally using the **Smart Prefetch Engine** to stream Kaggle data without local storage bloat. |
| **6. Unit Test (All)** | **Diagnostic Pass** | Runs precisely **1 functional epoch** across the entire model inventory to validate memory/VRAM. |
| **9. Environment Janitor** | **Orphan Purge** | Force-terminates orphaned Python/PowerShell processes and releases file-system mutexes. |
| **7. Exit** | **Full Shutdown** | Gracefully closes the orchestration hub. |

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
| **Plateau Breaker** | Detects static Loss and injects **3.0x LR Jolt** to shatter local minimums. | ✅ Active |
| **SWA Smoothing** | Shadow weight averaging across epochs for superior SOTA generalization. | ✅ Active |
| **Singularity Rollback** | Detection of numerical explosions (NaN) triggers weight restoration & cooling. | ✅ Active |
| **Regression Guard** | Hard-reset to peak weights if quality metrics drop > 5% for 3 epochs. | ✅ Active |
| **Zero-Latency Prefetch** | Subsequent datasets are streamed/unpacked in the background during active training. | ✅ Active |

---
**LemGendary AI Suite | Advanced Agentic Coding 2026**
