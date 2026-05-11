# LemGendary AI Training Suite (v16.2.8-NUCLEAR-HARDENED)

> **The 2026 Global Standard for High-Fidelity Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models. Optimized for high-frequency artifact detection and structural restoration with **Nuclear-Hardened v16.2.8 Architecture**.

---

### 📡 Mission Status: v16.2.8 (High-Fidelity Era)
🚀 **Status**: High-Fidelity Calibration Active / Global Registry Hardened  
🧪 **Current Goal**: Finalize the **Resolution Ladder (256px-640px)** with **Ladder-Aware SOTA Guards** and **Manifold Hardening**.

---

## ⚡ High-Fidelity "Nuclear" Hardening

The v16.2.8 release marks the end of "low-resolution warm-up." The suite now enforces a strict high-fidelity baseline to ensure models learn complex textures and micro-artifacts from the very first epoch.

### 🛰️ Dynamic Memory-Sentinel (Batch Decoupling)
- **100% Autonomous Batching**: Manual `batch_size` settings have been removed from the registry. The suite now uses active VRAM probing to calculate the absolute peak physical batch size and gradient accumulation for your specific hardware.
- **Atomic Re-Audit (v17.0)**: The suite now performs a fresh hardware probe every time the resolution ladder jumps (e.g., 256px → 384px) or validation begins. This ensures peak physical throughput at low resolutions while automatically throttling for high-res stability.
- **VRAM Defibrillation**: Proactive memory purging between training and high-res validation to ensure zero-paging on restricted hardware (4GB-8GB).
- **Sub-Nuclear 4GB Lockdown (v22.0)**: On GTX 1650 cards, the suite enforces a strict **Serial-Only Mode** (0 workers) and hard-clamps pixel volume (Targets ~6-8 Batch) to prevent Windows System RAM paging. This results in a **2x performance gain** over legacy Batch 1 profiles by staying 100% inside physical VRAM.
- **Serial Recovery Shield (v17.2)**: Automatically force-disables parallel workers after a VRAM overflow to prevent Windows worker deadlocks and system freezes.
- **Terminal Progress Guard (v17.2)**: Intelligent checkpoint auditing that advances the epoch when progress hits 99.9%, preventing "Groundhog Day" training loops.

### 🧬 Hardened Resolution Ladders
- **Quality Scorer Floor**: NIMA Aesthetic and Technical scorers now start at **512px**, forcing the detection of high-frequency noise and artistic composition immediately.
- **Restoration Baseline**: All restoration models (NAFNet, MIRNet, etc.) now start at a minimum of **256px**, climbing to **640px** via the autonomous ladder.

### 🚀 Ladder-Aware SOTA Guard (v18.0)
- **Quality-Driven Progression**: Reaching SOTA targets at sub-maximal resolutions no longer terminates training. Instead, the Governor triggers a **Forced Spatial Jump** to the next rung (e.g., 256px → 384px).
- **SOTA Hardening Guard (v19.0)**: Enforces a mandatory **2-epoch "Hardening Period"** for every resolution rung. Even if SOTA is hit instantly, the model must stay for 2 epochs to solidify weights before the next jump.
- **Mission Completion**: Full SOTA export and hub synchronization only occur after the model has conquered the **Final Resolution** (e.g., 640px).

---

## 🛠️ Getting Started

### 1. The Models Hub
The master orchestration console for system bootstrapping and cloud sync.
```powershell
./lemgendary_models_hub.ps1
```

#### 📋 Detailed Menu Structure
| Option | Action | Sub-Prompts & Details |
| :--- | :--- | :--- |
| **1. Initialize Systems** | **Environment Sync** | Installs Python 3.12, creates `.venv`, and Auto-Detects GPU. Installs PyTorch 2.7.0+ and **Master SOTA Stack**. |
| **2. Train Model** | **High-Fidelity Selection**| Launches the **Hardened Category Submenu** with 24+ SOTA architectures. |
| **3. Global Orchestration**| **Continuous Train** | Executes intelligent phased training with **Stateless Resumption** and **Sawtooth Governance**. |
| **5. Environment Janitor** | **Orphan Purge** | Force-terminates orphaned processes and releases Windows file-system mutexes. |

---

## 📂 Project Anatomy (Stateless Multi-Tenant)

- `unified_models_v2.yaml` — **The Master Registry**: High-Fidelity floor, refined SOTA targets (FID/PLCC), and standardized learning rates.
- `training/optimization_engine.py` — **The Governor**: Sawtooth scaling, Turbulence Dampening, and NPP Recoil.
- `training/train.py` — **The Master Pipeline**: Features **Active Memory-Sentinel** and **Atomic SOTA Export**.
- `LemGendaryModels/` — **The Artifact Vault**: Decoupled repository for SOTA weights, metrics, and ONNX binaries.

---

## 🛡️ Feature Matrix (v16.2.8 Master Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Memory-Sentinel** | 100% autonomous physical batch and accumulation calculation. | ✅ v16.2.8 Active |
| **Serial Shield** | Force-disables workers on low-VRAM hardware after an OOM. | ✅ v17.2 Active |
| **Progress Guard** | Advances epochs at 99.9% progress to prevent training loops. | ✅ v17.2 Active |
| **4GB Iron-Clamp** | Hardware-aware pixel ceilings to prevent System RAM paging. | ✅ v22.0 Active |
| **High-Fidelity Floor**| Minimum starting resolutions of 224px-512px across all manifolds. | ✅ v16.2.8 Active |
| **Numerical Priority** | Proportional iteration scaling based on manifold stress. | ✅ v15.0 Active |
| **SOTA-Force Jump** | Quality-driven progression that propellant models to higher resolutions. | ✅ v18.0 Active |
| **Manifold Hardening**| 2-epoch mandatory stabilization period before any resolution jump. | ✅ v19.0 Active |
| **Manifold Anchor** | Loop-Proof Optimization with Failure Path Memory. | ✅ v10.0 Active |

---

## 📊 Universal SOTA Telemetry (20-Column Audit)
Standardized historical audit (`metrics.csv`) captures the complete state:
- **Standardized Schema**: Epoch, Loss, LR, Accuracy, Res, Data, Temp, Clamp, Batch, Accumulation, and Stress.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison.
- **Cloud Persistence**: Metrics are synchronized across local and cloud via the CloudSyncManager.

---

## ☁️ Dual-Repo SOTA Hub Sync & Kaggle Deployment

The Governor automatically synchronizes with your `LemGendaryModels` repository, saving `_latest.pth` and `_best.pth` directly to the Hub. It uses **Dual-Token PATs** (`SUITE_PAT` and `GITHUB_PAT`) for secure, headless authentication on Kaggle.

### Universal Hardware Inference
All inference notebooks and training engines natively fall back to **DirectML** on local machines, providing zero-config GPU acceleration for **AMD** and **Intel** graphics cards on Windows.
