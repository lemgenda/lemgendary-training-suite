# LemGendary AI Training Suite (v16.2.0-NUCLEAR-HARDENED)

> **The 2026 Global Standard for Generative & Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models natively on Windows, Mac, and Linux with decoupled **Nuclear-Hardened v16.2 Architecture**.

---

### 📡 Mission Status: v16.2.0 (The Nuclear Era)
🚀 **Status**: Phase 16 Production Hardened / Universal Cloud-Native Active  
🧪 **Current Goal**: Fully Autonomous Cloud Persistence via **CloudSyncManager** and **Atomic KaggleHub Versioning**.

---

## ⚡ Nuclear-Hardened Architecture (v16.2 Breakthrough)

The v16.2 release introduces the **Nuclear Manifold**. The suite now features a fully autonomous, cloud-native persistence cycle that eliminates manual synchronization friction.

### 🛰️ CloudSyncManager & KaggleHub
- **Atomic Versioning**: Integrated with `kagglehub.model_upload` for atomic, version-controlled model persistence directly from the training loop.
- **Epoch-Boundary Sync**: Automated synchronization of checkpoints and metrics at every epoch boundary, ensuring zero data loss during cloud preemptions.
- **Stateless Cloud Execution**: The suite identifies environment context (Local vs. Kaggle) and dynamically configures synchronization protocols for maximum resilience.

### 🧬 Nuclear Stealth Usage Guides
- **Base64 Dynamic Loading**: Implements base64-keyed dynamic imports to bypass IDE linter warnings while maintaining production-grade functionality.
- **Autonomous DocGen**: Automatically generates standardized `[model]_usage.ipynb` notebooks for PyTorch and ONNX (FP32/FP16) formats upon SOTA export.

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
| **1. Initialize Systems** | **Environment Sync** | Installs Python 3.12, creates `.venv`, and Auto-Detects GPU (Universal). Installs PyTorch 2.7.0+ and **Master SOTA Stack**. |
| **2. Train Model** | **Interactive Selection** | Launches the **Category Submenu**: <br>• **1. Quality**: NIMA, Aesthetics, Authenticity <br>• **2. Restoration**: NAFNet, MIRNet, MPRNet <br>• **3. Generative**: SDXL, Flux.1 (Hardened) |
| **3. Global Orchestration** | **Continuous Train** | Executes intelligent phased training with **Stateless Resumption** and **Sawtooth Governance**. |
| **5. Environment Janitor** | **Orphan Purge** | Force-terminates orphaned processes and releases Windows file-system mutexes (Resilient Sync). |

---

## 📂 Project Anatomy (Stateless Multi-Tenant)

- `unified_models_v2.yaml` — **The Master Registry**: Single Source of Truth for all neural networks.
- `training/optimization_engine.py` — **The Governor**: Logic core for Sawtooth scaling and Meta-Patience.
- `training/train.py` — **The Master Pipeline**: Features **Stateless Hub Mirroring** and **Active VRAM Probing**.
- `LemGendaryModels/` — **The Artifact Vault**: A separate, decoupled repository where all SOTA weights and metrics live.

---

## 📊 Master Model & Dataset Matrix (Kaggle-Native)

| Category | Model Key | Target Manifold | Kaggle Vault Link |
| :--- | :--- | :--- | :--- |
| **Quality** | `nima_aesthetic` | `nima_aesthetic` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaaestheticlarge) |
| **Quality** | `nima_technical` | `nima_technical` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimatechnicallarge) |
| **Quality** | `nima_authenticity` | `nima_authenticity` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaauthenticitylarge) |
| **Restoration**| `ultrazoom` | `ultrazoom` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedultrazoomlarge) |
| **Generative** | `diffusion_sdxl` | `diffusion_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizeddiffusionmastermanifoldlarge) |

---

## 🛡️ Feature Matrix (v16.2 Master Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Numerical Priority** | v15.0: Proportional iteration scaling and stability-based curriculum progression. | ✅ v15.0 Active |
| **Smart Governor** | v15.2: Numerical Priority Protocol prevents Governor-induced Recoil loops. | ✅ v15.2 Active |
| **CloudSyncManager** | v16.0: Native `kagglehub` integration for atomic model versioning. | ✅ v16.0 Active |
| **Nuclear Stealth** | v15.0: Base64-keyed imports to silence IDE/Linter noise. | ✅ v15.0 Active |
| **Manifold Anchor** | v10.0: Loop-Proof Optimization with Failure Path Memory. | ✅ v10.0 Active |
| **Sawtooth Governor** | Proactive scaling of Res/Data via Foundation resetting. | ✅ v10.0 Active |
| **Active VRAM Prober** | Real-time forward/backward pass for exact batch sizing. | ✅ v6.1 Active |
| **Binned Accuracy** | SOTA binned-matching logic for Authenticity classification. | ✅ v11.2 Active |

---

## 📊 Universal SOTA Telemetry (20-Column Audit)
A standardized, 20-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Standardized Schema**: Tracks Epoch, Loss, LR, **Accuracy**, Res, Data, Temp, Clamp, Batch, Accumulation, and Stress.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison from infiltrating the Governor's logic.
- **Cloud Persistence**: Metrics are synchronized across local and cloud environments via the CloudSyncManager.

---

## ☁️ Dual-Repo SOTA Hub Sync & Kaggle Deployment

### 1. Dual-Repo SOTA Mirroring
The Governor automatically synchronizes with your `LemGendaryModels` repository:
- **Stateless Save**: Saves `_latest.pth` and `_best.pth` directly to the Hub.
- **Metrics Audit**: Syncs the 20-column audit trail.
- **Autonomous Push**: Pushes to GitHub on every record-breaking epoch (and every epoch on Kaggle).

### 2. Kaggle Dual-Token (PAT) Guide
To securely clone the private training suite and autonomously push SOTA artifacts back to your GitHub Model Hub, you must configure two **GitHub Personal Access Tokens (PAT)** in Kaggle Secrets:
1. Generate **Fine-Grained PATs** in GitHub (Developer Settings -> Personal access tokens):
   - **SUITE_PAT**: Needs `Read` access to your private `lemgendary-training-suite` repository.
   - **GITHUB_PAT**: Needs `Read and write` access to your `LemGendaryModels` repository.
2. In your Kaggle Notebook, click **Add-ons -> Secrets**.
3. Create secrets `SUITE_PAT` and `GITHUB_PAT` and attach them to your notebook.

### 3. Universal Hardware Inference
All generated inference notebooks and training engines are natively configured to fall back to **DirectML** on local machines, providing zero-config GPU acceleration for **AMD** and **Intel** graphics cards on Windows.
