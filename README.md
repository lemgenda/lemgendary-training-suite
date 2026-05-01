# LemGendary AI Training Suite (v10.1.0-MASTER-DYNAMIC-VRAM)

> **The 2026 Global Standard for Generative & Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models natively on Windows, Mac, and Linux with decoupled **2026-era Master Resilience Architecture**.

---

### 📡 Mission Status: v10.1.0 (Dynamic VRAM Sync)
🚀 **Status**: Phase 10 Production Hardened / Universal Backend Active  
🧪 **Current Goal**: Orchestrate high-fidelity 1024px fine-tuning with **Anticipatory VRAM Governance** to prevent OS paging on consumer hardware.

---

## ⚡ 2026 Resilience Architecture (v10.1 Breakthrough)

The v10.1 release introduces **Real-Time Headroom Probing** and a **Universal Backend Selector**. The suite no longer assumes hardware limits; it actively senses free VRAM (accounting for browsers/OS overhead) and dynamically re-seats the training manifold to prevent system-wide paging and I/O stalls.

### 🧵 The LemGendary Hub
The master orchestration console for all training activities. Automatically manages `.venv` isolation, hardware-specific library selection (NVIDIA CUDA, Apple MPS, Intel XPU, or DirectML), and sequential project execution.

### 🧬 Unified Multitask DNA
Fully synchronized with the [LemGendary Dataset Engine (v6.0)](../lemgendary-datasets/README.md).
- **Universal Backend Selector**: Prioritizes NVIDIA CUDA > Apple MPS > Intel XPU > DirectML > CPU.
- **Headroom-Aware Memory-Sentinel**: Probes `mem_get_info` in real-time to calculate batch sizes based on **Actual Free VRAM**, not theoretical capacity.
- **Time-Aware Checkpoint Governor**: Targets a strict **15-minute Resiliency Window**, scaling save frequency based on epoch velocity to protect both SSD endurance and training progress.

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
| **1. Initialize Systems** | **Environment Sync** | Installs Python 3.12, creates `.venv`, and Auto-Detects GPU (Universal). Installs PyTorch 2.7.0+ and **Master SOTA Stack** (PEFT/Diffusers/BNB). |
| **2. Train Model** | **Interactive Selection** | Launches the **Category Submenu**: <br>• **1. Quality**: NIMA, Aesthetics, Authenticity, **Anime NSFW Classification** <br>• **2. Face/Det**: YOLOv8n, RetinaFace, CodeFormer <br>• **3. SuperRes**: UltraZoom Array (x2-x8) <br>• **4. Restoration**: NAFNet, MIRNet, FFANet, MPRNet <br>• **5. Hybrid**: UPN v2, Multi-Restorer, Film <br>• **6. Generative**: SDXL, Flux.1 (Hardened) <br>• **7. Multimodal**: LLaVA v1.6, BLIP-2 (QLoRA) |
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

## 📊 Master Model & Dataset Matrix (Kaggle-Native)

The following table defines the structural mapping between the neural backbones and their high-fidelity LemGendized manifolds stored in the `lemtreursi` Kaggle vault.

| Category | Model Key | Target Manifold | Kaggle Vault Link |
| :--- | :--- | :--- | :--- |
| **Quality** | `nima_aesthetic` | `nima_aesthetic` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaaestheticlarge) |
| **Quality** | `nima_technical` | `nima_technical` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimatechnicallarge) |
| **Quality** | `nima_authenticity` | `nima_authenticity` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaauthenticitylarge) |
| **Safety** | `anime_nsfw_classification` | `classification_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedclassificationmastermanifoldlarge) |
| **Face** | `codeformer` | `codeformer` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedcodeformerlarge) |
| **Face** | `parsenet` | `parsenet` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedparsenetlarge) |
| **Detection** | `retinaface_mobilenet` | `retinaface_mobilenet` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedretinafacemobilenetlarge) |
| **Detection** | `yolov8n` | `yolov8n` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedyolov8nlarge) |
| **Restoration**| `ultrazoom` | `ultrazoom` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedultrazoomlarge) |
| **Restoration**| `nafnet_debluring` | `nafnet_debluring` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednafnetdebluringlarge) |
| **Restoration**| `nafnet_denoising` | `nafnet_denoising` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizednafnetdenoisinglarge) |
| **Hybrid** | `upn_v2` | `upn_v2` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedupnv2large) |
| **Hybrid** | `professional_multitask_restoration` | `professional_multitask_restoration` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedprofessionalmultitaskrestorationlarge) |
| **Generative** | `diffusion_sdxl` | `diffusion_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizeddiffusionmastermanifoldlarge) |
| **Generative** | `diffusion_flux` | `diffusion_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizeddiffusionmastermanifoldlarge) |
| **Multimodal** | `vlm_llava` | `vision_language_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedvisionlanguagemastermanifoldlarge) |
| **Multimodal** | `vlm_blip2` | `vision_language_master_manifold` | [Access Manifold](https://www.kaggle.com/datasets/lemtreursi/lemgendizedvisionlanguagemastermanifoldlarge) |

---

## 📂 Project Anatomy (Decoupled Multi-Tenant)

- `unified_models_v2.yaml` — **The Master Registry**: Single Source of Truth for all 21+ neural networks and data dependencies.
- `lemgendary_models_hub.ps1` — **The Master Hub**: Interactive PowerShell console for environment management and training.
- `train_all.py` — **The Global Orchestrator**: High-velocity phased training script with automated variety scaling.
- `training/` — Core backpropagation logic, loss engines (`losses.py`), and the autonomous `SmartTrainingGovernor`.
- `models/` — Specialized model architectures (Restoration, Generative, Multimodal) and task-aware heads.
- `data/` — Data logic core (`dataset.py`) and dynamic configuration generators (YOLO/VLM).
- `export/` — SOTA production export engines for ONNX (WebGPU) and standalone PyTorch artifacts.
- `checkpoints/` — Strictly holds transient, active epoch binaries (`_best.pth`, `_last.pth`).
- `../LemGendaryModels/` — Decoupled root target for final SOTA production deployment.
- `../LemGendaryDatasets/` — Decoupled root for manifold storage and autonomous Kaggle/HF acquisition.
- `technical_papers/` — High-fidelity architecture whitepapers and research notes.

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

### 📊 Universal SOTA Telemetry (Multi-Metric 2026)
A standardized, 17-column historical audit (`metrics.csv`) that captures the complete state of the training manifold:
- **Multi-Metric Tracking**: Hard-enforced tracking of PSNR, SSIM, LPIPS, FID, and newly added **Classification Accuracy** for the Authenticity Scorer.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts and bypasses incompatible metrics to prevent numerical poison from infiltrating the Governor's logic.
- **Auditable State**: Tracks Data Fraction, Softmax Temp, Logit Clamp, LR, Batch Size, and Accumulation.

---

## 🏗️ Architectural Hardening: Dethroning the Mocks (v9.0.0 Breakthrough)

The v9.0.0 release marks the complete integration of **True Generative Operations** and structural multi-tier fallbacks.

### 🎭 Master Generative & VLM Hardening (v2026 Engine)
- **Flux.1 & SDXL (LoRA Enabled)**: 12B parameter transformer backbones are now trainable on consumer hardware via **PEFT (Parameter-Efficient Fine-Tuning)**.
- **Multimodal QLoRA**: LLaVA v1.6 and BLIP-2 are hardened with **4-bit NormalFloat (NF4)** quantization, allowing high-fidelity vision-language reasoning to seat in standard VRAM.
- **Task-Aware Loss Engines**: Integrated EMD (Earth Mover's Distance) and SRCC Rank-Boost for aesthetic discovery, alongside Flow Matching for Flux.

### 💎 Universal 2026 Data Protocol
- **Protocol Routing**: `MultiTaskDataset` now autonomously routes between **Kaggle Mirrors** (`kaggle://lemtreursi/`) and **HuggingFace Origins** (`hf://`).
- **Disk Space Guard**: 2.5x volume check prevents massive Master Manifolds (1.2TB+) from causing volume overflows during acquisition.
- **Post-SOTA Guardian**: Interactive console hooks intercept mission completion to actively wipe local dataset caches, permanently defending disk space on constrained hardware.

### 💎 Professional Restoration Backbones
- **Sub-Pixel Mastery**: `UltraZoom` upgraded to a real **ESPCN** sub-pixel convolution network.
- **Feature Synergy**: Universal film and restoration models upgraded to **Residual Dense Blocks (RDB)** for maximum feature reuse and high-frequency preservation.

---

## 🛡️ Resilience Architecture (v2026 Engine)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Smart Governor** | Autonomous scaling of Res, Temp, Clamp, and Dataset Fraction. | ✅ Active |
| **Jolt Recoil** | Detects and dampens manifold regression following plateau breaks. | ✅ Active |
| **Memory-Sentinel** | **v10.1 Upgrade**: Real-time VRAM probing (`mem_get_info`) for dynamic batch sizing. | ✅ Active |
| **Intra-Epoch Sentinel**| **v10.1 Upgrade**: Anticipatory batch downscaling if system VRAM pressure spikes. | ✅ Active |
| **Time-Aware Save** | **v10.1 Upgrade**: Target 15-min save window based on real-time iteration velocity. | ✅ Active |
| **Universal Backend** | **v10.1 Upgrade**: Native support for CUDA, MPS, XPU, and DirectML. | ✅ Active |
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

## 🛡️ Industry Standards, Competitive Positioning & SOTA Audit (2026)

The LemGendary Training Suite has been benchmarked against the 2026 MLOps and Generative/Restoration landscape. Below is a strategic analysis of its architectural superiority.

### 1. Industry Standards Audit
| Standard | LemGendary Implementation | Status |
| :--- | :--- | :--- |
| **Versioned Lineage** | Single-Source-of-Truth via `unified_models_v2.yaml` and `unified_data.yaml` registry anchors. | **Pass (Elite)** |
| **Autonomous MLOps** | Phased `train_all.py` Orchestrator with surgical disk purging and zero-latency pre-fetching. | **Pass (Innovative)** |
| **AI Observability** | The `SmartTrainingGovernor` provides real-time drift detection, regression guards, and hardware sentinels. | **Pass (Advanced)** |
| **Policy-as-Code** | Integrated `anime_nsfw_classification` acts as a gatekeeper for generative manifold sanitation. | **Pass (Strong)** |

### 2. Competitive Landscape Analysis
| Framework | Focus | LemGendary Advantage |
| :--- | :--- | :--- |
| **BasicSR / MMEditing** | Restoration | **Decoupled Architecture**: LemGendary separates manifolds from logic, enabling global Kaggle/HF sync without code changes. |
| **HF Diffusers / PEFT** | Generative | **Governance Layer**: LemGendary uses these as engines but adds a structural "Governor" to prevent catastrophic concept drift. |
| **SageMaker / TFX** | Enterprise Ops | **Hardware Efficiency**: Achieving same-tier "Policy-as-Code" and CI/CD at 1/10th the cost via 4-bit/SWA consumer-grade optimization. |

### 3. Unique Selling Points (USPs)
*   **The Smart Training Governor**: Unlike standard schedulers, the Governor "feels" the manifold, dynamically scaling resolution and data fractions to maintain stability.
*   **Manifold-Agnostic Resolution**: The Tiered Dataloader resolves data across local volumes, Kaggle Vaults, and HuggingFace Hubs with protocol-aware recovery.
*   **SOTA Guard (Regression Shield)**: Mathematically detects quality regression and physically purges "poisoned" checkpoints to force a return to the stable manifold.

---
