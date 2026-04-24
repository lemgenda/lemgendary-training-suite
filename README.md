# LemGendary AI Training Suite (v8.1.0-ULTRA-STABILIZED)

> **The Professional Standard for Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision models natively on Windows with 2026-era Resilience Architecture.

---

### 📡 Mission Status: v2.5.0 (Resiliency v6.1.25)
🚀 **Status**: Production Deployment / Active Training Loop  
🧪 **Current Goal**: Break 35dB PSNR / Escaping 24dB Local Minima

---

## ⚡ 2026 Resilience Architecture (v4.5 Breakthrough)

The v7.5 release introduces the **Registry-First Dynamic Orchestration (v4.5)** and **Standardized Epoch Resumption**, ensuring binary parity across all 21+ neural models with zero maintenance debt. By unifying all metadata into a single source of truth, the suite now self-heals its dependency mappings across local and cloud clusters natively.

### 🧵 The LemGendary Hub
The master orchestration console for all training activities. Automatically manages `.venv` isolation, hardware-specific library selection (NVIDIA CUDA 12.1), and sequential project execution.

### 🧬 Unified Multitask DNA
Fully synchronized with the [LemGendary Dataset Engine (v3.0)](../lemgendary-datasets/README.md).
- **SOTA Alignment**: Automated metric scoring against PLCC, SRCC, PSNR, and mAP@0.5:0.95.
- **Unified Multitask Loader**: High-performance pipeline for streaming massive datasets with zero-latency prefetching and Windows-hardened file-locking protection.
- **Persistent I/O Sync (v5.8)**: Shatters cold-start disk hangs on Windows. Uses a JSON mission manifest to cache the physical file structure, reducing multi-worker initialization from **8 minutes to <1 second**.

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
| **3. Global Orchestration** | **Continuous Train** | Executes intelligent phased training with zero-latency pre-fetching and aggressive dataset SSD purging. |
| **4. Unit Test (All)** | **Diagnostic Pass** | Runs precisely **1 functional epoch** across the entire model inventory to validate memory/VRAM. |
| **5. Environment Janitor** | **Orphan Purge** | Force-terminates orphaned Python/PowerShell processes and releases file-system mutexes. |
| **Q. Exit** | **Full Shutdown** | Gracefully closes the orchestration hub. |

### 2. Manual Orchestration
For advanced users who require direct pipeline control.
```bash
# Individual Model Run
python training/train.py --model nima_technical

# Global Orchestration (Smart Caching + Phased Matrix Mode)
python train_all.py

# Global Unit Test (Dry Run)
python train_all.py --epochs 1 --yes
```

---

## 📂 Project Anatomy

- `training/` — Core backpropagation logic and stability hooks (`train.py`, `losses.py`).
- `models/` — Specialized model architecture definitions and weight loaders.
- `trained-models/` — Production-ready artifacts (ONNX FP16/FP32).
- `technical_papers/` — High-fidelity architecture whitepapers and research notes.
- `unified_models.yaml` — The Master Registry (Single Source of Truth) for all supported neural networks and data dependencies.

---

## ⚙️ Orchestrator Pipeline & Smart Governor Internals

The v8.1.0-ULTRA-STABILIZED training suite leverages a highly sophisticated, multi-phase orchestration sequence and an autonomous parameter governor. These systems ensure non-stop continuous training across the entire model inventory while aggressively optimizing both physical hardware and the mathematical training manifold.

### 1. The Smart Orchestrator Sequence
Defined in `smart_orchestrator.py`, this script manages the physical execution of the models using a mathematically structured pipeline.
- **Phased Execution Matrix**: Models are grouped by structural topology into 8 distinct phases (Phase 1: Deep Quality Assessment, Phase 2A/B: Facial Analytics & Detection, Phase 3A-E: Super-Resolution & Restoration). This controls the global memory footprint.
- **Zero-Latency Background Pre-Fetch**: While a model trains natively, the Orchestrator arms a background thread to look ahead to the *next* phase and stream Kaggle datasets into the SSD cache. This ensures the GPU never idles waiting for physical I/O.
- **Aggressive SSD Purging**: Upon completing a phase, a surgical memory purger mathematically shreds any cached datasets that are no longer required by future phases, freeing massive disk space on constrained hardware.

### 2. Smart Governor Initialization & Starting Values
When a model boots via `train.py`, the `SmartTrainingGovernor` initializes with the following default logic limits (configurable via `unified_models.yaml`):
- **Data Fraction (`initial_fraction`)**: Defaults to `10%` (0.1). This forces the model to heavily overfit to a small stochastic subset during its "Discovery Phase."
- **Fraction Increment**: Defaults to `15%` (0.15). The chunk size by which the dataset expands when a plateau is hit.
- **Resolution Ladder**: Follows a predefined progression (e.g., `[128, 256, 384, 512, 640]`) to force spatial feature extraction.
- **Thermal Values**: Softmax Temperature starts at `0.1`; Logit Clamp initialized at `20.0` (with a dynamic range of `[15.0, 45.0]`).
- **Hardware Sentinels**: Physical batch size (`16` or auto-detected); VRAM Safety Margin (`0.85` or 85%).

### 3. Autonomous Triggers & Logic
The Governor executes `audit_epoch()` at the end of every epoch, utilizing multiple sentinels to analyze the model's structural health:
- **Stagnation Plateau Guard**: Triggered if the validation metric (e.g., PSNR) fails to improve by a strict `0.1%` delta (`min_delta: 1e-3`) for `plateau_patience` (default 6) epochs.
- **Drift Sentinel**: Triggered if the validation quality regresses compared to the previous epoch. If consecutive drift occurs (>= 2), the Governor mathematically dampens the Learning Rate by `0.8x/0.7x` and tightens the logit clamp (`-5.0`) to force stability.
- **Numerical Sentinel**: If the `sentinel_trigger_rate` (NaNs or infs) exceeds 5%, the LR is cooled to seat the manifold. If stress exceeds 20%, scaling milestones are intentionally delayed.
- **Hardware/Memory Sentinel**: Predicts the next VRAM footprint before shifting resolutions using `(next_res / current_res)^1.8`. If the prediction exceeds 85%, it natively trades physical Batch Size for logical Gradient Accumulation to prevent OS paging.

### 4. Dynamic Optimizations & Shields
- **Velocity Life-Support & Defibrillation (Jolt)**: If all resolution and data scaling options are exhausted during a plateau, the Governor injects a High-Energy Manifold Jolt (multiplying the LR by `2.0x` up to `4.0x`). If the LR has "cooled to death" (`< base_lr * 0.01`), it forces a surgical multiplier to rewind the OneCycleLR curve and break the stagnation.
- **Stabilization Shield**: Immediately following a structural shift (Resolution up-scale or Batch size drop), the manifold is naturally volatile. The Governor engages a **3-epoch Stabilization Shield** that temporarily locks out the Regression Guard and Plateau detection to prevent "Self-Gaslighting" recoils.
- **Recoil Protocol**: If the model suffers a catastrophic manifold collapse and must roll back to a `_best.pth` checkpoint, the Governor triggers `recoil()`. It tacticaly steps down one rung on the resolution ladder and shrinks the dataset fraction, allowing the model to gently re-seat itself safely.

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

### 🛡️ Memory-Sentinel (VRAM Guard) & Survival Profile (v5.9)
Hardware-aware orchestration that prevents OOM crashes and OS paging:
- **Proactive Scaling**: Predicts VRAM consumption before resolution shifts and automatically trades physical **Batch Size** for **Gradient Accumulation**.
- **Survival Profile (v5.9)**: Specifically for heavy architectures (NAFNet/MIRNet) on 4GB hardware. Forces **Physical Batch Size 1** with **4x Gradient Accumulation**.
- **Reactive Recovery**: Intercepts physical OOM errors mid-epoch, clears the CUDA cache, and performs emergency batch reductions to keep the mission alive.
- **Mission Continuity Guard (v6.1)**: Manifold leak prevention. Ensures the mission continues seamlessly after a memory recovery event.
- **Plateau Breaker (v6.1.25)**: Engineered high-energy Manifold Jolt + Velocity Life-Support. Breaks 200-epoch stagnancy using kinetic LR resetting.

### 📊 Universal SOTA Telemetry
A standardized, 17-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Telemetry Schema Guard**: Automatically detects, archives, and migrates legacy 8-column or 10-column logs into the 17-column hardware-aware standard.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts and bypasses incompatible metrics to prevent numerical poison from infiltrating the Governor's logic.
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
| **Plateau-Buster** | **v5.2 Upgrade**: Strict 0.1% delta gating with a stagnation jolt counter. | ✅ Active |
| **SOTA Guardrail** | **v5.2 Upgrade**: Quality-Regression Mutex prevents false exports. | ✅ Active |
| **Singularity Shield** | Detection of NaNs triggers immediate weight restoration & thermal cooling. | ✅ Active |
| **I/O Sync** | **v5.8 Upgrade**: Persistent JSON manifold manifest shatters disk hangs. | ✅ Active |
| **Survival Profile** | **v5.9 Upgrade**: Physical/Logical batch trading for heavy models on 4GB hardware. | ✅ Active |
| **Continuity Guard** | **v6.1 Upgrade**: OOM-recovery loop leak prevention and manifold liveness. | ✅ Active |
| **Plateau Breaker** | **v6.1.25 Upgrade**: High-energy Manifold Jolt + Velocity Life-Support. | ✅ Active |
| **Pulse Persistence**| **v6.1.25 Upgrade**: Epsilon-hardened 20% intra-epoch synchronization. | ✅ Active |
| **Stabilization Shield**| **v6.1.26 Upgrade**: Strict plateau-lockout protecting the manifold after structural shifts. | ✅ Active |
| **Spatial Augmentation**| **v6.1.26 Upgrade**: Synchronous geometric flips preventing catastrophic feature memorization. | ✅ Active |

---
