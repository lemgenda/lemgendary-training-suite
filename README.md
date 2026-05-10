# LemGendary AI Training Suite (v16.2.8-NUCLEAR-HARDENED)

> **The 2026 Global Standard for High-Fidelity Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models. Optimized for high-frequency artifact detection and structural restoration with **Nuclear-Hardened v16.2.8 Architecture**.

---

### 📡 Mission Status: v16.2.8 (High-Fidelity Era)
🚀 **Status**: High-Fidelity Calibration Active / Global Registry Hardened  
🧪 **Current Goal**: Transition the entire fleet to a **224px-512px Resolution Floor** and **Dynamic Memory-Sentinel** batching.

---

## ⚡ High-Fidelity "Nuclear" Hardening

The v16.2.8 release marks the end of "low-resolution warm-up." The suite now enforces a strict high-fidelity baseline to ensure models learn complex textures and micro-artifacts from the very first epoch.

### 🛰️ Dynamic Memory-Sentinel (Batch Decoupling)
- **100% Autonomous Batching**: Manual `batch_size` settings have been removed from the registry. The suite now uses active VRAM probing to calculate the absolute peak physical batch size and gradient accumulation for your specific hardware.
- **Atomic Re-Audit (v17.0)**: The suite now performs a fresh hardware probe every time the resolution ladder jumps (e.g., 256px → 384px) or validation begins. This ensures peak physical throughput at low resolutions while automatically throttling for high-res stability.
- **VRAM Defibrillation**: Proactive memory purging between training and high-res validation to ensure zero-paging on restricted hardware (4GB-8GB).

### 🧬 Hardened Resolution Ladders
- **Quality Scorer Floor**: NIMA Aesthetic and Technical scorers now start at **512px**, forcing the detection of high-frequency noise and artistic composition immediately.
- **Restoration Baseline**: All restoration models (NAFNet, MIRNet, etc.) now start at a minimum of **256px**, bypassing the structural "blur" caused by ultra-low resolution warm-up.

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
| **High-Fidelity Floor**| Minimum starting resolutions of 224px-512px across all manifolds. | ✅ v16.2.8 Active |
| **Numerical Priority** | Proportional iteration scaling based on manifold stress. | ✅ v15.0 Active |
| **Smart Governor** | Sawtooth curriculum resets to prevent local minima traps. | ✅ v15.2 Active |
| **CloudSyncManager** | Native `kagglehub` integration for atomic model versioning. | ✅ v16.0 Active |
| **Nuclear Stealth** | Base64-keyed imports to silence IDE/Linter noise. | ✅ v15.0 Active |
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
