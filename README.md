# LemGendary AI Training Suite (v12.0.0-STATELESS-RESILIENCE)

> **The 2026 Global Standard for Generative & Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models natively on Windows, Mac, and Linux with decoupled **2026-era Stateless Resilience Architecture**.

---

### 📡 Mission Status: v12.0.0 (The Stateless Era)
🚀 **Status**: Phase 12 Production Hardened / Universal Backend Active  
🧪 **Current Goal**: Autonomous Continuous Training via **Stateless Mirroring** and **Cross-Environment Checkpoint Parity**.

---

## ⚡ 2026 Stateless Resilience Architecture (v12.0 Breakthrough)

The v12.0 release introduces the **Stateless Training Manifold**. The suite no longer stores heavy binary weights locally within the code repository. Instead, it utilizes a decoupled **Models Hub** for all persistent artifacts, ensuring the core codebase remains lightweight and lightning-fast to clone.

### 🧬 Decoupled Checkpoint Management
- **Stateless Codebase**: The `lemgendary-training-suite` now ignores all binary weights (`.pth`, `.pt`). Your code repository is strictly for logic and configuration.
- **Hub-Direct Persistence**: All `_latest.pth` and `_best.pth` checkpoints are saved and loaded directly from your `LemGendaryModels` repository.
- **Intra-Epoch Resilience**: Transient progress (`_progress.pth`) is stored in a local, git-ignored `checkpoints/` folder for immediate recovery from system freezes, while epoch-level state is archived to the cloud.

### 🛰️ Autonomous Cloud Synchronization
- **Kaggle Stateless Sync**: Every single epoch on Kaggle is autonomously pushed to GitHub. This ensures that if a cloud session times out or crashes, you can resume on a fresh instance with zero data loss.
- **Proactive Sync Protocol**: The suite executes a proactive `git pull --rebase` before every push, ensuring your local and cloud environments remain in perfect, conflict-free alignment.

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

## 🛡️ Feature Matrix (v12.0 Master Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Stateless Resume** | Direct loading of checkpoints from Hub repo to bypass suite bloat. | ✅ v13.0 Active |
| **Autonomous Sync** | Every-epoch pushes on Kaggle for zero-loss cloud training. | ✅ v13.0 Active |
| **Proactive Rebase** | Conflict-free synchronization via pull-before-push protocols. | ✅ v13.0 Active |
| **Sawtooth Governor** | Proactive scaling of Res/Data via Foundation resetting. | ✅ v6.1 Active |
| **Active VRAM Prober** | Real-time forward/backward pass for exact batch sizing. | ✅ v6.1 Active |
| **Instant Damping** | Zero-patience micro-adjustments on manifold regression. | ✅ v6.1 Active |
| **Meta-Patience** | Dynamic stagnation gating based on manifold complexity. | ✅ v6.1 Active |
| **Binned Accuracy** | SOTA binned-matching logic for Authenticity classification. | ✅ v11.2 Active |

---

## 📊 Universal SOTA Telemetry (20-Column Audit)
A standardized, 20-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Standardized Schema**: Tracks Epoch, Loss, LR, Accuracy, Res, Data, Temp, Clamp, Batch, **Accumulation**, and **Stress**.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison from infiltrating the Governor's logic.
- **Stateless Persistence**: Metrics are saved directly to the Models Hub, providing a continuous audit trail across environments.

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
