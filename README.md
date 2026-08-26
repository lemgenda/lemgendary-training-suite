# LemGendary AI Training Suite (v16.3.0-NUCLEAR-HARDENED)

> **The 2026 Global Standard for High-Fidelity Vision Model Training.**
>
> A unified, industrial-grade orchestration layer for training, optimizing, and deploying SOTA vision and multimodal models. Optimized for high-frequency artifact detection and structural restoration with **Nuclear-Hardened v16.3.0 Architecture** (now featuring Live Polarity Shields and Absolute Anti-Loop Guards).

---

## [SYNC] Mission Status: v16.2.9 (High-Fidelity Era)

[LAUNCH] **Status**: High-Fidelity Calibration Active / Global Registry Hardened  
[GOAL] **Current Goal**: Finalize the **Resolution Ladder (256px-640px)** with **Ladder-Aware SOTA Guards** and **Manifold Hardening**.

---

## [FAST] High-Fidelity "Nuclear" Hardening & Technical Backlog

The deep architectural backlog, including the **Memory-Sentinel**, **Sawtooth Governance**, **Resolution Ladders**, **SOTA Validation Guards**, and **Judicial Audit** logic, has been fully consolidated into our official whitepapers.

- **Dynamic Segmentation Loading (v16.3.1)**: MultiTaskDataset now natively ingests standard 1-channel `.png` segmentation maps directly from `masks/` allowing training of architectures like ParseNet.
- **Dynamic Restoration Degradations (v16.3.1)**: MultiTaskDataset dynamically synthesizes blur, haze, noise, rain, and combinations using Albumentations for models like UpnV2 where target maps are solely pristine ground truth.
- **Adaptive Loop-Breaker & Loop Guards (v16.3.2)**: SmartTrainingGovernor dynamically tracks checkpoint rollback history to detect resolution-regression locks. It automatically triggers spatial resolution promotion (Strategy A) or dynamic drift-gate relaxation (Strategy B), protected by an anti-self-fighting `breakout_lock` retreat shield, to resolve infinite training loops autonomously.
- **Autonomous Resolution Escalation & Dynamic Limits (v16.3.3)**: `SmartTrainingGovernor` dynamically detects resolution-regression locks when rollback thresholds are met and automatically promotes the training resolution to the next rung in `res_ladder` (`256px -> 384px`), resetting sample fraction to 15% for a fresh warmup. `get_active_regression_limit()` dynamically relaxes `regression_limit` during loop recovery so static YAML limits never block resolution escalation.
- **Proven-Manifold Protection & Intra-Resolution Data Recoil (v16.0.0)**: If a model regresses during dataset fraction expansion on a resolution where it already achieved a high peak score (`best_quality >= 0.75 * target_quality_score` or `> 85.0`), the `SmartTrainingGovernor` blocks premature Spatial Retreats (resolution drops) and instead executes Intra-Resolution Data Recoil. It steps dataset fraction back to the last safe fraction on the high-resolution manifold (e.g. `75% -> 55%` at `512px`) while cooling the learning rate by 50% and locking stabilization for 5 epochs.
- **Autonomous SOTA Hyperparameter Adaptation (v17.5)**: `SmartTrainingGovernor` dynamically adjusts loss function hyperparameters on the fly without manual mid-training YAML edits. When PLCC/SRCC or EMD plateau below target benchmarks (`SRCC > 0.9100`, `EMD < 0.0700`), the Governor automatically scales pairwise `rank_weight` (up to `1.5`), tightens `rank_margin` (down to `0.05`), and sharpens `softmax_temp` during the `REFINEMENT` phase.
- **Differentiable Soft-Spearman Loss & Rank Memory Bank (v19.0)**: Eliminates micro-batch ranking starvation under low VRAM ($b=2$) by evaluating sigmoid-ranked correlation over a historical FIFO queue ($N=32$), scaling pairwise comparisons to $\binom{32}{2} = 496$ pairs per backward pass.
- **Spatial Statistical Pooling ($\text{Mean} \oplus \text{Std}$)**: Doubles feature sensitivity to localized micro-defects and safety triggers by retaining spatial variance alongside global averages.
- **Headless Kaggle Cloud Engine**: Full CLI and PowerShell API orchestration for launching, monitoring, and pulling high-VRAM GPU training runs (Tesla T4 x2 / P100) headlessly.
- **Universal Post-Training Target Audit & Interactive Guidance**: Diagnostic gap analysis against `sota_targets` upon epoch ceiling completion with interactive cloud escalation and export options.
- **Walk-Forward Curriculum Orchestrator (v17.1)**: Automated 6-fold spatial-temporal expansion matrix (Phase 1: 4 Pairs $\to$ Phase 4: 16 Pairs) for robust financial regime generalization without manual intervention.
- **Forex High-Entropy Resilience (v17.2)**: `SmartTrainingGovernor` bypasses standard Turbulence Shields for financial manifolds, doubling the Intense Cyclical Learning Rate (Jolt Protocol) intensity ($2.0\times$ multiplier over 5-epoch windows) while extending absolute plateau patience to prevent false-positive retreats.

For an exhaustive breakdown of the Training Suite architecture, please consult the [Master Training Suite Guide](file:///c:/Development/python/model-training/lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md) in the `lemgendary-docs` repository.

---

## Getting Started

### 1. The Models Hub

The master orchestration console for system bootstrapping and cloud sync.

```powershell
./lemgendary_models_hub.ps1
```

#### Detailed Menu Structure

| Option | Action | Sub-Prompts & Details |
| :--- | :--- | :--- |
| **1. Initialize Systems** | **Environment Sync** | Installs Python 3.12, creates `.venv`, and Auto-Detects GPU. Installs PyTorch 2.7.0+ and **Master SOTA Stack**. |
| **2. Train Model Locally** | **Two-Level Domain Selection** | Select from parent domains (**Image Manipulation & Restoration**, **Image Generation & Multimodal**, **Financial & Time-Series**) with 24+ SOTA architectures. |
| **3. Single-Epoch Unit Test** | **Fleet Smoke Test** | Diagnostic 1-epoch execution across all registered models. |
| **4. Kaggle Cloud Engine** | **Headless GPU Orchestration** | Launch, stream telemetry, and pull trained checkpoints from Kaggle Cloud GPU headlessly. |

---

## Project Anatomy (Stateless Multi-Tenant)

- `unified_models_v2.yaml` — **The Master Registry**: High-Fidelity floor, refined SOTA targets (FID/PLCC), and standardized learning rates.
- `training/optimization_engine.py` — **The Governor**: Sawtooth scaling, Turbulence Dampening, and NPP Recoil.
- `training/train.py` — **The Master Pipeline**: Features **Active Memory-Sentinel** and **Atomic SOTA Export**.
- `LemGendaryModels/` — **The Artifact Vault**: Decoupled repository for SOTA weights, metrics, and ONNX binaries.

---

## [GUARD] Feature Matrix (v16.2.9 Master Engine)

All features have been exhaustively documented in the [Master Training Suite Guide](file:///c:/Development/python/model-training/lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md).

---

## [METRICS] Universal SOTA Telemetry (Dynamic Hardware-Aware Schema)

Standardized historical audit (`metrics.csv`) automatically scales based on the active domain:

- **28-Column Image Telemetry**: Epoch, Loss, LR, PLCC, SRCC, PSNR, SSIM, LPIPS, FID, mAP50, mIoU, Accuracy, Res, Data, Temp, Clamp, Batch, Accumulation, Stress.
- **21-Column Financial Telemetry**: Epoch, Loss, LR, DirAcc, WinRate, ProfitFactor, Sharpe, Sortino, MaxDD, TP_MAE, SL_MAE, Quality_Score, Pairs, Data, Temp, Clamp, Batch, Accumulation, Stress.
- **Auto-Recovery**: Instantly detects domain/column mismatch upon resume, archiving corrupted/legacy logs to `_legacy.csv` and initializing a fresh schema.
- **Metrics Sanitizer**: Explicitly sanitizes `inf`/`NaN` artifacts to prevent numerical poison.
- **Cloud Persistence**: Metrics are synchronized across local and cloud via the CloudSyncManager.

---

## Standalone Judicial Audit CLI (`judicial_audit_api.py`)

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

## Dual-Repo SOTA Hub Sync & Kaggle Deployment

The Governor automatically synchronizes with your `LemGendaryModels` repository, saving `_latest.pth` and `_best.pth` directly to the Hub. It uses **Dual-Token PATs** (`SUITE_PAT` and `GITHUB_PAT`) for secure, headless authentication on Kaggle.

### Multi-GPU DataParallel Capabilities (Kaggle Scale)

The training suite natively intercepts execution environments with multiple GPUs (e.g., Kaggle Tesla T4 x2) and automatically wraps compatible models in PyTorch's `nn.DataParallel` API.

- **Dynamic Batch Distribution**: Seamlessly splits large high-fidelity pixel matrices (e.g., `768px`) across available GPUs, doubling effective throughput.
- **Seamless CPU Checkpointing**: Intelligently intercepts the `.pth` save hooks, stripping the `module.` prefix injected by `DataParallel` before saving to disk. This guarantees that Kaggle Multi-GPU checkpoints can be effortlessly downloaded and evaluated natively on standalone Windows environments or CPU deployments without manual layer re-mapping.
- **ONNX Trace Resilience (FakeTensor Guards)**: Dynamically wraps unmapped `FakeTensor` memory pointer access (`data_ptr()`) during FX/ONNX graph tracing within the DataParallel multi-GPU engine to prevent false-positive segmentation faults during structural graph export.
- **Real-Time SOTA Export Device Re-Anchoring**: Guarantees that multi-GPU DataParallel parameters and buffers are atomically restored to the primary accelerator (`cuda:0`) across all SOTA export cycles via `finally` execution blocks, with proactive start-of-epoch device alignment verification.
- **Read-Only Dataset Manifold Resilience**: Safely skips dataset directory writes when running in read-only environments (such as Kaggle `/kaggle/input` mounts/symlinks) without interrupting training or model export workflows.

### Universal Hardware Inference

All inference notebooks and training engines natively fall back to **DirectML** on local machines, providing zero-config GPU acceleration for **AMD** and **Intel** graphics cards on Windows.

- **Hardware-Aware Resolution Capping**: Dynamically limits maximum training and validation resolution (e.g. `max_allowed_local_resolution: 640`) on local environments to prevent 4GB VRAM exhaustion, while permitting 1024px+ scaling on robust cloud infrastructures.

---

## [PROGRESS] Forex Trading Model (`forex_predictor`)

The LemGendary Training Suite includes a **production-grade Forex prediction model** trained on MetaTrader 5 OHLCV data across all major currency pairs and all timeframes.

### Architecture: Multi-Scale CNN-Transformer Hybrid

```text
[M1 branch] [M5 branch] [M15 branch] [H1 branch] [H4 branch] [D1 branch]
     │            │            │            │           │            │
Causal TCN   Causal TCN   Causal TCN   Causal TCN  Causal TCN  Causal TCN
(4 layers)   (4 layers)   (4 layers)   (4 layers)  (4 layers)  (4 layers)
     └────────────┴────────────┴────────────┴───────────┴────────────┘
                       Cross-Timeframe Attention (4 heads)
                       + Currency Pair Embedding (8 pairs)
                                     │
                        Fused Feature Manifold [d=256]
                       ┌─────────────┴─────────────┐
                 Direction Head               Magnitude Head
               3-class Softmax            Regression (Softplus)
            [Down / Sideways / Up]        [TP pips, SL pips]
              + Confidence Score
```

**Design Principles:**

- **Stateless** — no hidden state between calls → safe for ONNX + MT5 EA
- **Causal Conv1D only** — zero future lookahead, zero data leakage
- **Confidence-gated magnitude** — low-confidence bars don't corrupt regression
- **ONNX-compatible** — exports cleanly for MT5 EA deployment

### Lookback Windows

| Timeframe | Window | Span |
| :--- | :--- | :--- |
| M1 | 512 bars | ~8.5 hours |
| M5 | 288 bars | ~1 day |
| M15 | 192 bars | ~2 days |
| H1 | 168 bars | ~1 week |
| H4 | 90 bars | ~2.5 weeks |
| D1 | 252 bars | ~1 year |

### Governor Curriculum

The Governor's `res_ladder` is repurposed as a **timeframe expansion ladder**. Training starts on M1 only (FOUNDATION phase) and expands to all 6 timeframes as metrics stabilize.

### Key Files

| File | Purpose |
| :--- | :--- |
| `models/forex_predictor.py` | Model architecture (ForexPredictor) |
| `data/mt5_pipeline.py` | MT5 data download, indicator computation, label generation |
| `data/forex_dataset.py` | ForexDataset (Governor-aligned fractional sampling) |
| `training/losses.py` | ForexDualLoss (CE direction + Huber magnitude) |
| `export/mt5_signal.py` | ONNX export + live signal generator + MQL5 EA stub |

### MT5 Deployment

Once a demo account is available:

```powershell
# 1. Download data (using credentials or MT5_LOGIN/MT5_PASSWORD/MT5_SERVER environment variables)
python data/mt5_pipeline.py --mode download --pairs EURUSD GBPUSD --timeframes 60 240 --login <ACCOUNT_NUMBER> --password <PASSWORD> --server <SERVER_NAME>

# 2. Train
python training/train.py --model forex_predictor

# 3. Export to ONNX
python export/mt5_signal.py --mode export --checkpoint LemGendaryModels/forex_predictor/forex_predictor_best.pth

# 4. Generate live signal
python export/mt5_signal.py --mode signal --onnx LemGendaryModels/forex_predictor/forex_predictor.onnx --pair EURUSD --login <ACCOUNT_NUMBER> --password <PASSWORD> --server <SERVER_NAME>

# 5. Generate MQL5 EA stub
python export/mt5_signal.py --mode mql5_stub --out export/LemGendaryForexEA.mq5
```

> [WARNING] **Live Trading Safety**: The model emits signals only. SL/TP enforcement and max drawdown kill-switch must be implemented in the MQL5 EA layer independently of the model.
