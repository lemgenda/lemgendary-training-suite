# LemGendary AI Training Suite (v7.5.0-LEMGENDARY)

> **The Professional Standard for Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision models natively on Windows with 2026-era Resilience Architecture.

---

## ⚡ 2026 Resilience Architecture (v4.5 Breakthrough)

The v7.5 release introduces the **Registry-First Dynamic Orchestration (v4.5)** and **Standardized Epoch Resumption**, ensuring binary parity across all 21+ neural models with zero maintenance debt. By unifying all metadata into a single source of truth, the suite now self-heals its dependency mappings across local and cloud clusters natively.

### 🧵 The LemGendary Hub
The master orchestration console for all training activities. Automatically manages `.venv` isolation, hardware-specific library selection (NVIDIA CUDA 12.1), and sequential project execution.

### 🧬 Unified Multitask DNA
Fully synchronized with the [LemGendary Dataset Engine (v3.0)](../lemgendary-datasets/README.md).
- **SOTA Alignment**: Automated metric scoring against PLCC, SRCC, PSNR, and mAP@0.5:0.95.
- **Registry-First Acquisition**: Dynamic dependency discovery removes all hardcoded dataset lists; the suite auto-fetches required Kaggle streams based on the `unified_models.yaml` registry.
- **Unified Multitask Loader**: High-performance pipeline for streaming massive datasets with zero-latency prefetching and Windows-hardened file-locking protection.

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
- `unified_models.yaml` — The Master Registry (Single Source of Truth) for all supported neural networks and data dependencies.

---

## 🚀 Technical Whitepapers
Deep dives into the mathematical foundations and hardware-aware optimizations of the suite:

- **[NIMA: Quality Assessment (Markdown)](technical_papers/PAPER_LEMGENDARY_NIMA.md)**
- **[NAFNet: Resurrection & Restoration (Markdown)](technical_papers/PAPER_LEMGENDARY_NAFNET.md)**
- **[Premium HTML Versions](https://lemgenda.github.io/lemgendary-training-suite/)**

---

## 🚀 Autonomous Smart Pipeline (v5.0 Breakthrough)

The v7.5.0-LEMGENDARY release transforms the training suite into a fully autonomous, data-driven engine. The **Smart Training Governor** now manages the entire training trajectory without manual intervention.

### 🧠 Smart Training Governor
A centralized intelligence layer that dynamically recalibrates training complexity and velocity based on real-time manifold performance:
- **Resolution Scaling**: Automatically shifts from 128px to 768px as the model converges.
- **Thermal Management**: Dynamically adjusts `softmax_temp` (0.1 → 0.05) to sharpen logits as resolution increases.
- **Smart Clamping**: Autonomous logit guardrails [15.0, 50.0] that tighten during instability and relax for high-fidelity discovery.
- **Plateau Priority**: Metric-aligned optimization levers; Fidelity-focused models (NAFNet) prioritize **Resolution**, while Perceptual-focused models (CodeFormer) prioritize **Data Variety**.

### 🛡️ Memory-Sentinel (VRAM Guard)
Hardware-aware orchestration that prevents OOM crashes and OS paging:
- **Proactive Scaling**: Predicts VRAM consumption before resolution shifts and automatically trades physical **Batch Size** for **Gradient Accumulation**.
- **Reactive Recovery**: Intercepts physical OOM errors mid-epoch, clears the CUDA cache, and performs emergency batch reductions to keep the mission alive.

### 📊 Universal SOTA Telemetry
A standardized, 17-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Metrics**: PLCC, SRCC, PSNR, SSIM, LPIPS, FID, mAP50, mAP50-95.
- **Governor State**: Data Fraction, Softmax Temp, Logit Clamp, LR, Batch Size, Accumulation.

---

## 🛡️ Resilience Architecture (v2026 Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Smart Governor** | Autonomous scaling of Res, Temp, Clamp, and Dataset Fraction. | ✅ Active |
| **Memory-Sentinel** | Hardware-aware VRAM monitoring and Batch-Accumulation trades. | ✅ Active |
| **Plateau Breaker** | Detects metric stalls and injects **2.0x LR Jolt** or **3.0x Resolution Scaling**. | ✅ Active |
| **Singularity Shield** | Detection of NaNs triggers immediate weight restoration & thermal cooling. | ✅ Active |
| **Regression Guard** | Physical checkpoint rollback if metrics drop > 5% for 3 consecutive epochs. | ✅ Active |
| **SWA Smoothing** | Shadow weight averaging across epochs for superior SOTA generalization. | ✅ Active |

---
**LemGendary AI Suite | Advanced Agentic Coding 2026**

