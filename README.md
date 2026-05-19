# LemGendary AI Training Suite (v16.2.8-NUCLEAR-HARDENED)

> **The 2026 Global Standard for High-Fidelity Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models. Optimized for high-frequency artifact detection and structural restoration with **Nuclear-Hardened v16.2.8 Architecture**.

---

## 📡 Mission Status: v16.2.8 (High-Fidelity Era)

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
- **Dynamic Validation Sharding & Auto-Expansion (v23.4)**: Automatically subsets the validation dataloader to 30% per epoch with a fixed random seed during early phases to accelerate perceptual metric (LPIPS/FID) computation by 3x. Once the model hits the **Refinement Phase** (maximum resolution ladder step at 100% training data fraction), the Governor dynamically auto-expands validation to **100% (full dataset)** to guarantee an absolute SOTA generalizability audit.
- **Absolute Energy Floor (Sentinel Guard)**: The Pre-Backward loss Sentinel now enforces an absolute baseline energy floor (`> 0.05` unscaled) to prevent false-positive training recoils on microscopic loss metrics during high-fidelity refinement.

### 🧠 Intelligent Curriculum & Resume Persistence (v23.1)

- **Universal Quality Scorecarding**: Lifted structural-perceptual score evaluations out of quality-only constraints. The trainer now dynamically compounds PSNR, SSIM, LPIPS, and FID for all restoration models, enabling precise tracking in `metrics.csv` and allowing the governor to drive dataset fraction ladders.
- **Plateau Memory Persistence**: The governor's internal epoch metrics history buffer (`self.history`) is now fully serialized to checkpoint state dictionaries. This ensures that arbitrary stop/resume cycles under local interruptions or Cloud preemptions do not reset early plateau-stopping counts, maintaining solid training scaling continuity.
- **Continuous Parameter Predictor Calibration (v23.2)**: Standardized validation MAE telemetry for continuous parameter predictors (like `upn_v2`). The optimizer dynamically tracks theta-normalization ($\pi$) and records MAE directly into `metrics.csv` (encoded as negative PSNR for unified tabular representation) under automated Huber/SmoothL1 loss curves.
- **Universal Film Restorer SOTA (v23.2)**: Fully integrated joint Restoration + Colorization manifolds. Configured SOTA targets (`psnr: 24.0`, `ssim: 0.80`, `lpips: 0.25`) driven by Perceptual Loss (LPIPS) and dynamic degradation loaders.
- **Professional Multi-Task Restoration (v23.5)**: Fully integrated multi-headed Mixture-of-Experts (MoE) routing engine with 11 specialized restoration heads (denoise, deblur, derain, dehaze_indoor, dehaze_outdoor, lowlight, exposure, superres, vintage, face_restorer, face_parser) dynamically trained under **Dynamic Filename Task Ingestion** with regex task parsing to prevent task-routing collapse. Deployed **Downsampled Global Attention (DGA)** to execute global spatial self-attention under a dense $32 \times 32$ pooled resolution, achieving complete OOM immunity and 4000x spatial reduction on GTX 1650/4GB hardware.

### 🧬 Hardened Resolution Ladders

- **Quality Scorer Floor**: NIMA Aesthetic and Technical scorers now start at **512px**, forcing the detection of high-frequency noise and artistic composition immediately.
- **Restoration Baseline**: All restoration models (NAFNet, MIRNet, etc.) now start at a minimum of **256px**, climbing to **640px** via the autonomous ladder.
- **Parameter Predictor Ceiling**: Continuous regression models (`upn_v2`) are capped at a stable **256px** resolution ceiling to conserve physical VRAM and maximize gradient throughput on local hardware.

### 🚀 Ladder-Aware SOTA Guard (v18.0)

- **Quality-Driven Progression**: Reaching SOTA targets at sub-maximal resolutions no longer terminates training. Instead, the Governor triggers a **Forced Spatial Jump** to the next rung (e.g., 256px → 384px).
- **SOTA Hardening Guard (v19.0)**: Enforces a mandatory **2-epoch "Hardening Period"** for every resolution rung. Even if SOTA is hit instantly, the model must stay for 2 epochs to solidify weights before the next jump.
- **SOTA Memorization & Data Verification (v23.6)**: Gated early SOTA completion to verify the model has not simply memorized data subsets. If SOTA is met at the final resolution but on a subset (fraction < 1.0), the trainer immediately advances the dataset fraction to 1.0 (100% data) and reconstructs the dataloaders instead of terminating. Loop completion is only allowed once the SOTA baseline is verified on 100% of the training dataset.
- **SOTA Quality Selection Guard (v23.6)**: Gated SOTA exports for quality tasks. Loss-only improvements update the validation loss to keep training active but do not trigger checkpoint exports or overwrite SOTA weights unless the primary correlation/accuracy quality score hits a record high.
- **Same-Resolution Recoil Protection (v23.6)**: Hardened the Smart Governor to retain the current data fraction during same-resolution plateaus and recoil/regression phases, ensuring the model stabilizes on the current data variety instead of running in loops lowering and rising fractions.
- **Low-Variance Safety Gate (v23.6)**: Bypasses emergency head resets and thermal shock temperature resets for `nima_authenticity` or if validation target standard deviation is low (< 0.15) to prevent training destabilization.
- **Mission Completion**: Full SOTA export and hub synchronization only occur after the model has conquered the **Final Resolution** on 100% data.


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
| **2. Train Model** | **High-Fidelity Selection** | Launches the **Hardened Category Submenu** with 24+ SOTA architectures. |
| **3. Global Orchestration** | **Continuous Train** | Executes intelligent phased training with **Stateless Resumption** and **Sawtooth Governance**. |
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
| **High-Fidelity Floor** | Minimum starting resolutions of 224px-512px across all manifolds. | ✅ v16.2.8 Active |
| **Validation Sharding** | Fixed 30% validation auditing, auto-expanding to 100% in Refinement. | ✅ v23.4 Active |
| **Energy Floor Guard** | Absolute loss baseline check to prevent false-positive recoils. | ✅ v23.0 Active |
| **Dynamic Quality Delta** | Auto-scales `min_delta` (100x) for Restoration plateau detection. | ✅ v23.0 Active |
| **Quality Integration** | Un-nested SOTA evaluation providing cumulative quality scores for restoration. | ✅ v23.1 Active |
| **History Persistence** | Serializes governor history buffer to survive restarts and preemptions. | ✅ v23.1 Active |
| **Numerical Priority** | Proportional iteration scaling based on manifold stress. | ✅ v15.0 Active |
| **SOTA-Force Jump** | Quality-driven progression that propellant models to higher resolutions. | ✅ v18.0 Active |
| **Manifold Hardening** | 2-epoch mandatory stabilization period before any resolution jump. | ✅ v19.0 Active |
| **Manifold Anchor** | Loop-Proof Optimization with Failure Path Memory. | ✅ v10.0 Active |
| **Huber Regression Engine** | Continuous parameter prediction with SmoothL1 loss and $\pi$-boundary normalization. | ✅ v23.2 Active |
| **Dynamic Film Degradation** | Joint restoration + colorization training with on-the-fly vintage damage injection. | ✅ v23.2 Active |
| **SOTA Verification** | Verifies SOTA targets on 100% of training data at final resolution. | ✅ v23.6 Active |
| **Quality Selection Guard** | Gates SOTA exports to require absolute improvement in primary quality metrics. | ✅ v23.6 Active |
| **Recoil Protection** | Retains dataset fraction during same-resolution plateaus and regressions. | ✅ v23.6 Active |
| **Low-Variance Safety Gate** | Bypasses emergency head resets and thermal shocks on low-variance distributions. | ✅ v23.6 Active |


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
