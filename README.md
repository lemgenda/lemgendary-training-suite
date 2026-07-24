# LemGendary AI Training Suite (v16.3.0-NUCLEAR-HARDENED)

> **The 2026 Global Standard for High-Fidelity Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models. Optimized for high-frequency artifact detection and structural restoration with **Nuclear-Hardened v16.3.0 Architecture** (now featuring Live Polarity Shields and Absolute Anti-Loop Guards).

---

## 📡 Mission Status: v16.2.9 (High-Fidelity Era)

🚀 **Status**: High-Fidelity Calibration Active / Global Registry Hardened  
🧪 **Current Goal**: Finalize the **Resolution Ladder (256px-640px)** with **Ladder-Aware SOTA Guards** and **Manifold Hardening**.

---

## ⚡ High-Fidelity "Nuclear" Hardening & Technical Backlog

The deep architectural backlog, including the **Memory-Sentinel**, **Sawtooth Governance**, **Resolution Ladders**, **SOTA Validation Guards**, and **Judicial Audit** logic, has been fully consolidated into our official whitepapers.

- **Dynamic Segmentation Loading (v16.3.1)**: MultiTaskDataset now natively ingests standard 1-channel `.png` segmentation maps directly from `masks/` allowing training of architectures like ParseNet.
- **Dynamic Restoration Degradations (v16.3.1)**: MultiTaskDataset dynamically synthesizes blur, haze, noise, rain, and combinations using Albumentations for models like UpnV2 where target maps are solely pristine ground truth.
- **Adaptive Loop-Breaker & Loop Guards (v16.3.2)**: SmartTrainingGovernor dynamically tracks checkpoint rollback history to detect resolution-regression locks. It automatically triggers spatial resolution promotion (Strategy A) or dynamic drift-gate relaxation (Strategy B), protected by an anti-self-fighting `breakout_lock` retreat shield, to resolve infinite training loops autonomously.
- **Autonomous Resolution Escalation & Dynamic Limits (v16.3.3)**: `SmartTrainingGovernor` dynamically detects resolution-regression locks when rollback thresholds are met and automatically promotes the training resolution to the next rung in `res_ladder` (`256px -> 384px`), resetting sample fraction to 15% for a fresh warmup. `get_active_regression_limit()` dynamically relaxes `regression_limit` during loop recovery so static YAML limits never block resolution escalation.

For an exhaustive breakdown of the Training Suite architecture, please consult the [Master Training Suite Guide](file:///c:/Development/python/model-training/lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md) in the `lemgendary-docs` repository.

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

## 🛡️ Feature Matrix (v16.2.9 Master Engine)

All features have been exhaustively documented in the [Master Training Suite Guide](file:///c:/Development/python/model-training/lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md).

---

## 📊 Universal SOTA Telemetry (20-Column Audit)

Standardized historical audit (`metrics.csv`) captures the complete state:

- **Standardized Schema**: Epoch, Loss, LR, Accuracy, Res, Data, Temp, Clamp, Batch, Accumulation, and Stress.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison.
- **Cloud Persistence**: Metrics are synchronized across local and cloud via the CloudSyncManager.

---

## ⚖️ Standalone Judicial Audit CLI (`judicial_audit_api.py`)

A fully decoupled, zero-dependency validation wrapper designed for CI/CD integration and isolated model auditing.

**Key Capabilities:**

- **Framework Agnostic Loader:** Loads raw PyTorch `.pth` dictionaries, full PyTorch exported objects, and ONNX compiled graphs dynamically.
- **Auto-Casting & Guard Rails:** Dynamically detects ONNX input types (`float16`/`float32`) and automatically guards against double-softmax distributions by scanning raw logit distributions.
- **Fast-Path Correlator:** Bypasses complex multi-process training augmenters for single-threaded PIL loads (eliminating Windows DataLoader worker-hangs), outputting standard `PLCC` (Pearson) and `SRCC` (Spearman) scores natively.
- **JSON Pipeline Exporter:** Automatically pipes results into a standard JSON schema for automated regression auditing.

**Usage:**

```powershell
python judicial_audit_api.py --model_path .\export\model.onnx --dataset_dir .\test_data --labels_csv .\test_data\labels.csv --output_json report.json
```

---

## ☁️ Dual-Repo SOTA Hub Sync & Kaggle Deployment

The Governor automatically synchronizes with your `LemGendaryModels` repository, saving `_latest.pth` and `_best.pth` directly to the Hub. It uses **Dual-Token PATs** (`SUITE_PAT` and `GITHUB_PAT`) for secure, headless authentication on Kaggle.

### Multi-GPU DataParallel Capabilities (Kaggle Scale)

The training suite natively intercepts execution environments with multiple GPUs (e.g., Kaggle Tesla T4 x2) and automatically wraps compatible models in PyTorch's `nn.DataParallel` API.

- **Dynamic Batch Distribution**: Seamlessly splits large high-fidelity pixel matrices (e.g., `768px`) across available GPUs, doubling effective throughput.
- **Seamless CPU Checkpointing**: Intelligently intercepts the `.pth` save hooks, stripping the `module.` prefix injected by `DataParallel` before saving to disk. This guarantees that Kaggle Multi-GPU checkpoints can be effortlessly downloaded and evaluated natively on standalone Windows environments or CPU deployments without manual layer re-mapping.
- **ONNX Trace Resilience (FakeTensor Guards)**: Dynamically wraps unmapped `FakeTensor` memory pointer access (`data_ptr()`) during FX/ONNX graph tracing within the DataParallel multi-GPU engine to prevent false-positive segmentation faults during structural graph export.

### Universal Hardware Inference

All inference notebooks and training engines natively fall back to **DirectML** on local machines, providing zero-config GPU acceleration for **AMD** and **Intel** graphics cards on Windows.

- **Hardware-Aware Resolution Capping**: Dynamically limits maximum training and validation resolution (e.g. `max_allowed_local_resolution: 640`) on local environments to prevent 4GB VRAM exhaustion, while permitting 1024px+ scaling on robust cloud infrastructures.
