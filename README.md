# LemGendary AI Training Suite (v8.1.0-ULTRA-STABILIZED)

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
- **Unified Multitask Loader**: High-performance pipeline for streaming massive datasets with zero-latency prefetching, Windows-hardened file-locking protection, and a **Distributed I/O Cache** for instant multi-worker initialization.

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
| **Q. Exit** | **Full Shutdown** | Gracefully closes the orchestration hub. |

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
- **Jolt Recoil Protection**: Detects counter-productive energy injections; if regression follows a plateau break, the Governor triggers a **0.5x Recoil Damping** to force manifold stabilization.
- **Plateau Priority**: Metric-aligned optimization levers; Fidelity-focused models (NIMA) prioritize **Resolution**, while Perceptual-focused models (CodeFormer) prioritize **Data Variety**.

### 🛡️ Memory-Sentinel (VRAM Guard)
Hardware-aware orchestration that prevents OOM crashes and OS paging:
- **Proactive Scaling**: Predicts VRAM consumption before resolution shifts and automatically trades physical **Batch Size** for **Gradient Accumulation**.
- **Reactive Recovery**: Intercepts physical OOM errors mid-epoch, clears the CUDA cache, and performs emergency batch reductions to keep the mission alive.

### 📊 Universal SOTA Telemetry
A standardized, 17-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Telemetry Schema Guard**: Automatically detects, archives, and migrates legacy 8-column or 10-column logs into the 17-column hardware-aware standard.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts and bypasses incompatible metrics (e.g., LPIPS for quality) to prevent numerical poison from infiltrating the Governor's logic.
- **Auditable State**: Tracks PLCC, SRCC, PSNR, SSIM, LPIPS, FID, mAP, Data Fraction, Softmax Temp, Logit Clamp, LR, Batch Size, and Accumulation.

---

## 🏗️ Architectural Hardening: Dethroning the Mocks (v8.0.0 Breakthrough)

The v8.0.0 release marks the complete transition from placeholder "Mock" classes to **Real High-Fidelity Architectures**. This ensures that every epoch contributes to empirical model convergence across the entire 21+ model inventory.

### 🎭 Real Face & Detection Backbones
- **CodeFormer (Real)**: Replaced UNet proxies with deep **Multi-Scale Residual UNets** featuring spatial skip connections.
- **RetinaFace (Real)**: Integrated **MobileNetV2 features** with multi-task regression heads for Bboxes, Confidence, and Landmarks.
- **ParseNet (Real)**: Real **Fully Convolutional Network (FCN)** implementation for 19-class semantic face parsing.

### 💎 Professional Restoration Backbones
- **Sub-Pixel Mastery**: `UltraZoom` upgraded to a real **ESPCN** sub-pixel convolution network.
- **Feature Synergy**: Universal film and restoration models upgraded to **Residual Dense Blocks (RDB)** for maximum feature reuse and high-frequency preservation.

---

## 🛡️ Resilience Architecture (v2026 Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Smart Governor** | Autonomous scaling of Res, Temp, Clamp, and Dataset Fraction. | ✅ Active |
| **Jolt Recoil** | Detects and dampens manifold regression following plateau breaks. | ✅ Active |
| **Memory-Sentinel** | Hardware-aware VRAM monitoring and Batch-Accumulation trades. | ✅ Active |
| **Plateau-Buster** | **v5.2 Upgrade**: Strict 0.1% delta gating with a stagnation jolt counter to break training stalls. | ✅ Active |
| **SOTA Guardrail** | **v5.2 Upgrade**: Quality-Regression Mutex prevents false exports if PSNR regresses during loss drops. | ✅ Active |
| **Singularity Shield** | Detection of NaNs triggers immediate weight restoration & thermal cooling. | ✅ Active |
| **Regression Guard** | Physical checkpoint rollback if metrics drop > 5% for 3 consecutive epochs. | ✅ Active |
| **SWA Smoothing** | Shadow weight averaging across epochs for superior SOTA generalization. | ✅ Active |
| **No-Mock Protocol** | 100% Real High-Fidelity Architectures (Removed all Proxies/Mocks). | ✅ Active |

---

