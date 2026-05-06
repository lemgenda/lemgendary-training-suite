# LemGendary AI Training Suite (v11.0.0-AUTONOMOUS-CURRICULUM)

> **The 2026 Global Standard for Generative & Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models natively on Windows, Mac, and Linux with decoupled **2026-era Master Resilience Architecture**.

---

### 📡 Mission Status: v11.0.0 (The Autonomous Era)
🚀 **Status**: Phase 11 Production Hardened / Universal Backend Active  
🧪 **Current Goal**: Autonomous Curriculum Mastery via **Sawtooth Resolution Scaling** and **Real-Time VRAM Probing**.

---

## ⚡ 2026 Resilience Architecture (v11.0 Breakthrough)

The v11.0 release introduces the **Universal Curriculum Governor (v6.1)**. The suite no longer simply reacts to plateaus; it proactively manages the training manifold using a "Breadth-First" data strategy and hardware-aware structural shifts.

### 🧵 The Active Hardware Prober
Replacing legacy static caps, the suite now executes a **Single-Sample VRAM Probe** (Forward + Backward pass) before every training phase. It measures the exact byte-footprint of your specific backbone and resolution, calculating the mathematically perfect batch size for your specific GPU (from 4GB to 80GB).

### 🧬 Universal Curriculum Governor (v6.1)
- **Data-First Foundation**: Enforces 100% dataset mastery at lower resolutions before allowing any spatial expansion.
- **The Sawtooth Reset**: Upon resolution up-scaling (e.g., 384px -> 512px), the Governor resets the data fraction to 50% to allow the model to re-seat itself on the new spatial manifold without noise-overload.
- **Instant Velocity Damping**: A zero-patience guidance system. If an epoch regresses by even 0.001%, the Governor immediately applies a **2% LR Cool** and a **-0.5 unit Clamp shift** to nudge the trajectory back.
- **Meta-Patience Scaling**: Stagnation patience is no longer fixed. It scales dynamically with resolution complexity (e.g., waiting longer at 1024px than at 224px).

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
| **3. Global Orchestration** | **Continuous Train** | Executes intelligent phased training with **Zero-Latency Data Fetching** and **Sawtooth Governance**. |
| **5. Environment Janitor** | **Orphan Purge** | Force-terminates orphaned processes and releases Windows file-system mutexes (Resilient Sync). |

---

## 📊 Master Model & Dataset Matrix (Kaggle-Native)

| Category | Model Key | Target Manifold | Kaggle Vault Link |
| :--- | :--- | :--- | :--- |
| **Quality** | `nima_aesthetic` | `nima_aesthetic` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaaestheticlarge) |
| **Quality** | `nima_technical` | `nima_technical` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimatechnicallarge) |
| **Quality** | `nima_authenticity` | `nima_authenticity` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaauthenticitylarge) |
| **Restoration**| `ultrazoom` | `ultrazoom` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedultrazoomlarge) |
| **Restoration**| `nafnet_debluring` | `nafnet_debluring` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednafnetdebluringlarge) |
| **Generative** | `diffusion_sdxl` | `diffusion_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizeddiffusionmastermanifoldlarge) |

---

## 📂 Project Anatomy (Decoupled Multi-Tenant)

- `unified_models_v2.yaml` — **The Master Registry**: Single Source of Truth for all neural networks. Includes dynamic `target_effective_batch` and `min_delta` parameters.
- `training/optimization_engine.py` — **The Governor**: Logic core for Sawtooth scaling, Meta-Patience, and Instant Damping.
- `training/train.py` — **The Master Pipeline**: Features **Active VRAM Probing** and **Resilient IO Sync** (Windows file-lock protection).
- `data/dataset.py` — **High-Speed Data Core**: Features hardware-aware CV2 augmentations and synchronous geometric flips.

---

## ⚙️ Smart Governor v6.1 Internals (The "Sawtooth" Protocol)

The v11.0.0 suite leverages a "Structure-First" optimization sequence:

### 1. The Foundation Phase
- **Mastery Gating**: The model starts at the lowest resolution in its ladder (e.g. 224px).
- **Breadth Expansion**: The Governor expands the `sample_fraction` (e.g. 10% -> 100%) as plateaus are reached.
- **Goal**: Achieve maximum possible SRCC/Accuracy on the full dataset before touching resolution.

### 2. The Sawtooth Shift
- **Structural Expansion**: Once the foundation is mastered, the Governor jumps to the next resolution rung.
- **The Reset**: To prevent manifold shock, the **Data Fraction is reset to 50%**. This forces the model to focus on the new spatial features before being hit with the full dataset again.
- **Hardware Re-Probe**: The engine re-measures VRAM footprint and recalibrates batch sizes for the new resolution.

### 3. Resilience & Guidance
- **Instant Velocity Damping**: Zero-patience micro-adjustments (2% LR Cool) on every regression.
- **Resilient IO Sync**: All CSV and Checkpoint file operations use a retry-lock mechanism to prevent Windows `PermissionError` crashes during Hub synchronization.
- **Meta-Patience**: Patience scales with complexity: `effective_patience = base_patience * (Resolution / Base_Res)`.

---

## 🛡️ Feature Matrix (2026 Master Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Sawtooth Governor** | Proactive scaling of Res/Data via Foundation resetting. | ✅ v6.1 Active |
| **Active VRAM Prober** | Real-time forward/backward pass for exact batch sizing. | ✅ v6.1 Active |
| **Instant Damping** | Zero-patience micro-adjustments on manifold regression. | ✅ v6.1 Active |
| **Meta-Patience** | Dynamic stagnation gating based on manifold complexity. | ✅ v6.1 Active |
| **Resilient IO Sync** | Retry-lock file protection for Windows Hub stability. | ✅ v6.1 Active |
| **Universal Backend** | Native support for CUDA, MPS, XPU, and DirectML. | ✅ v10.1 Active |
| **SOTA Guardrail** | Quality-Regression Mutex prevents false exports. | ✅ v5.2 Active |
| **Continuity Guard** | OOM-recovery loop leak prevention and manifold liveness. | ✅ v6.1 Active |

---

## 📊 Universal SOTA Telemetry (Multi-Metric 2026)
A standardized, 17-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Multi-Metric Tracking**: Hard-enforced tracking of PSNR, SSIM, LPIPS, FID, and **SRCC/PLCC Accuracy**.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison from infiltrating the Governor's logic.
- **Auditable State**: Tracks Data Fraction, Softmax Temp, Logit Clamp, LR, Batch Size, and Accumulation.

---

## ☁️ Dual-Repo SOTA Hub Sync & Kaggle Deployment
The 2026 training suite features a completely decoupled model deployment architecture, allowing you to train seamlessly locally or on Kaggle, and push models directly to a secondary GitHub Model Hub.

### 1. Dual-Repo SOTA Mirroring
When training completes (or SOTA is breached), the Governor automatically:
- Synthesizes `_best.pth` checkpoints.
- Compiles standalone ONNX matrices (`LemGendary[Model].onnx`).
- Generates fully-configured Kaggle Inference Notebooks.
- Instantly pushes these artifacts to your designated `LemGendaryModels` repository via automated Git commands.

### 2. Kaggle Auto-Push (PAT Guide)
To allow Kaggle to autonomously push completed SOTA artifacts back to your GitHub Model Hub (without downloading massive zip files), you must configure a **GitHub Personal Access Token (PAT)** in Kaggle:
1. Generate a **Fine-Grained PAT** in GitHub (Developer Settings -> Personal access tokens) with `Read and write` access to your Model Hub repository.
2. In your Kaggle Notebook, click **Add-ons -> Secrets**.
3. Create a new secret named `GITHUB_PAT` and paste your token.
4. Attach it to your notebook. The suite will detect this and automatically execute cloud-syncs on every SOTA breach.

### 3. Universal Hardware Inference
All generated inference notebooks and training engines are natively configured to fall back to **DirectML** on local machines. This provides immediate, zero-config GPU acceleration for **AMD** and **Intel** graphics cards on Windows while preserving maximum **NVIDIA CUDA** performance on Kaggle.
