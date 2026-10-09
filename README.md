# LemGendary Model Training Suite

> Industrial-standard training, evaluation, export, and telemetry orchestration suite for Vision, Restoration, and Financial Time-Series deep learning models.
>
> **Architecture details and technical whitepapers:** [Master Training Suite Guide (PAPER_TRAINING_SUITE.md)](../lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md) · [Master CLI Manual (MANUAL_CLI.md)](../lemgendary-docs/MD-Papers/MANUAL_CLI.md) · [Master API Manual (MANUAL_API.md)](../lemgendary-docs/MD-Papers/MANUAL_API.md) · [Documentation Hub](https://lemgenda.github.io/ai-training-whitepapers/index.html)

---

## Current Status

| Metric | Value |
| :--- | :--- |
| **Version** | `v16.8.0-STABLE` |
| **Phase** | Post-Modernization S.O.L.I.D. Architecture Complete (Phases 0-14 Complete) |
| **Next** | Production Modernization & S.O.L.I.D. Architecture Complete — Ecosystem Ready |
| **Verified Models** | 24+ production architectures, multi-platform execution (Local, Kaggle T4x2/P100, Google Colab) |
| **Roadmap** | [training_suite_refactoring_roadmap.md](../lemgendary-docs/roadmaps/training_suite_refactoring_roadmap.md) |

---

### v16.9.17 — YOLO FP32 ONNX Separate Weights Sidecar Export & Kaggle Cloud Execution Engine

- **YOLO FP32 ONNX Separate Weights Sidecar (`training/governance/yolo_governor.py`)** — Upgraded ONNX export for YOLOv8n to package external weight tensors into a separate binary sidecar via `onnx.external_data_helper.convert_model_to_external_data`. Generates both `LemGendaryModels/yolov8n/yolov8n.onnx` (topology definition) and `LemGendaryModels/yolov8n/yolov8n.onnx.data` (raw weight parameters), conforming to enterprise ONNX size and runtime loading specifications.
- **Kaggle Cloud Training Launch & Live Monitoring Engine (`training/server/jobs.py`, `training/server/routes/training.py`)** — Added backend endpoints and background execution workers to push training jobs directly to Kaggle GPU clusters (`POST /api/training/kaggle/train`), attach live telemetry WebSocket streams to active Kaggle kernel logs (`POST /api/training/kaggle/monitor`), probe kernel execution status (`GET /api/training/kaggle/status`), and automatically pull remote checkpoints upon epoch completion (`POST /api/training/kaggle/pull`).
- **P0 Model Registry & Status Normalization (`unified_models_v2.yaml`, `training/server/routes/gui.py`, `training/server/routes/models.py`)** — Standardized all model statuses across the ecosystem to the authoritative 8-token vocabulary (`PLANNED`, `SPECIFICATION`, `DATASET_READY`, `TRAINING`, `TRAINED`, `VALIDATED`, `PRODUCTION`, `DEPRECATED`). Resolved NIMA Mobile backbone contradiction uniformly to `MobileNetV3-Small` and documented dual backbones (`MobileNetV1-0.25` and `ResNet-50`) for RetinaFace.

### v16.9.16 — Real-Time Batch/Epoch Progress Telemetry, Code-Only Suite Isolation & First-Principles VRAM Formula

- **Live Minibatch & Epoch Telemetry (`training/training/epoch.py`, `training/training/engine.py`)** — Added real-time visual progress logging to terminal during epoch iteration. Reports loss, running average loss, learning rate, and elapsed execution seconds every 10 minibatches (and upon final minibatch completion). Epoch initiation and completion banners announce current quality scores, PSNR/SSIM reconstruction metrics, and SOTA records.
- **Code-Only Training Suite Isolation (`training/core_loop.py`, `training/governance/yolo_governor.py`)** — Enforced complete separation between training code and generated model artifacts. All checkpoints, production exports, training history, and metrics CSV files are routed strictly to `LemGendaryModels/<model_key>/`. Eliminated local `checkpoints/` and model subdirectories inside `export/`.
- **First-Principles Continuous VRAM Batch Formula (`training/governance/yolo_governor.py`, `unified_models_v2.yaml`)** — Replaced hardcoded $4 \times 4$ VRAM tier lookup table with continuous first-principles memory arithmetic ($\text{VRAM}_{\text{GB}} \cdot 1024 \cdot \text{safety} - \text{static\_overhead}$) divided by resolution-scaled activation memory. Externalized all governor tuning parameters to YAML under `yolov8n > optimization`.
- **KaggleHub Checkpoint Probe Timeout & Non-Blocking Discovery (`training/checkpoint/recovery.py`)** — Wrapped remote checkpoint probing in a 5-second non-blocking timeout with clear console diagnostics to prevent silent execution hangs during cloud startup.

### v16.9.16 — Universal Export & Checkpoint Governance Engine Across All Architectures & Methods

- **Universal Tri-Format Export (`training/export/universal_exporter.py` & `training/export/onnx.py`)** — Standardized export pipeline across all model architectures and all training modes (Local, Kaggle, Colab). On every new `best.pth` checkpoint, exports: (1) `[ModelName].onnx` in FP16 precision with integrated self-contained weights, (2) `[ModelName]_FP32.onnx` + `[ModelName]_FP32.onnx.data` in FP32 precision with separate external binary weights sidecar (`onnx.external_data_helper.convert_model_to_external_data`), and (3) `[ModelName].pt` standard PyTorch FP32 model checkpoint. Target destination is strictly `LemGendaryModels/[ModelName]/`.
- **Strict 4-Tier Checkpoint Hierarchy (`training/checkpoint/lifecycle_manager.py`)** — Implemented `CheckpointLifecycleManager` enforcing uniform checkpoint governance:
  1. `progress.pth`: Intra-epoch checkpoint evaluated every 15 minutes during training and validation passes, strictly bounded to `[5%, 50%]` progress (never saved below 5% or above 50% regardless of time elapsed).
  2. `latest.pth`: Saved at epoch completion after both train and val passes complete; immediately purges `progress.pth` upon creation and resets the 15-minute timer.
  3. `best.pth`: Saved whenever the model achieves a new best `Quality_Score`; immediately triggers tri-format export and refreshes `LemGendaryModels/[ModelName]/README.md`.
  4. `vault_[target].pth`: Milestone checkpoints persisting historical best scores for individual metrics defined in `sota_targets` (e.g., `vault_psnr.pth`, `vault_ssim.pth`, `vault_plcc.pth`, `vault_map50.pth`).
- **Zero Pollution Isolation Guarantee** — All checkpoints persist to `LemGendaryModels/[ModelName]/checkpoints/`, runtime auxiliary configs and intermediate artifacts route to `LemGendaryModels/[ModelName]/training/`, and epoch statistics append exclusively to `LemGendaryModels/[ModelName]/metrics.csv`. Completely eliminated file generation in `lemgendary-training-suite/`.
- **Registry Schema Resilience & Kaggle Pipeline Parity (`training/core_loop.py` & `data/yolo_config_gen.py`)** — Resolved model info resolution for both top-level and nested `models` dictionaries in `unified_models_v2.yaml`, preventing empty metadata and task-type misclassification in cloud environments. Dynamic YOLO configuration files now materialize directly into `LemGendaryModels/[model_key]/training/`.

### v16.9.15 — Dynamic Sawtooth VRAM Batch Governor & Generalized Kaggle Dataset Auto-Binding

- **Dynamic Sawtooth VRAM Batch Scaling (`training/governance/yolo_governor.py`)** — Upgraded `compute_safe_batch_size` across all VRAM tiers. On 4GB hardware (`<= 4.5 GB`), updated safe batch limits to `{320: 32, 480: 16, 640: 8, 1024: 4}`, unlocking batch size 16 at resolution 480 for YOLOv8n (which previously was throttled to 8 despite consuming only 1.52 GB VRAM). Implemented dynamic runtime Sawtooth Governor feedback: dynamically probes peak VRAM pressure via `torch.cuda.max_memory_allocated()`; if pressure exceeds 90%, it dynamically halves the batch size to protect against OOM spikes; if VRAM headroom is abundant (< 60% capacity), it automatically promotes the stage batch size up to the safe capacity limit.
- **Generalized Kaggle Attached Dataset Auto-Binding (`training/data/manifold.py`, `data/dataset.py`, `data/yolo_config_gen.py`)** — Decoupled Kaggle cloud training from specific dataset name requirements (such as `LemGendizedMirNetExposureKaggleReady`). Added `is_valid_manifold_dir` and `scan_kaggle_input_manifolds` in `ManifoldResolver`, detecting manifolds with `shards/`, `images/`, `targets/`, `dataset_info.yaml`, `dataset-metadata.json`, and `.parquet` containers. In Kaggle environments, training automatically scans `/kaggle/input` and auto-binds to whichever dataset or datasets are attached to the notebook as input, eliminating `ValueError: num_samples=0` on container and custom-named manifolds.

### v16.9.14 — Plateau-Driven Ladder Advancement, Global Epoch Synchronization & Validation Display

- **Plateau- and Overfitting-Driven Rung Advancement (`training/governance/yolo_governor.py`)** — Removed fixed per-rung epoch budgets. Every fraction step and resolution rung now advances only when the quality score (`0.7 * mAP50-95 + 0.3 * mAP50`) fails to improve by more than `rung_plateau_min_delta` (0.001) for `rung_plateau_patience` (5) epochs after at least `rung_min_epochs` (3), or when overfitting signatures appear at any fraction. SOTA extension cycles at the top rung are uncapped, and the global epoch ceiling auto-extends by 100 when approached.
- **13-Stage Gradual Ladder** — 320px: 30% to 50% to 70% to 85% to 100%; 480px and 640px: 50% to 65% to 80% to 100% (15%-20% increments, restart at 50% on each resolution jump).
- **Global Epoch Synchronization** — `trainer.start_epoch`, `trainer.epochs` and the LR scheduler are aligned at `on_pretrain_routine_end`; optimizer, scaler and EMA state are restored with `_load_checkpoint_state` instead of `resume=True`. Progress bar, Governor banners, `metrics.csv` and WebSocket telemetry now show the same monotonic epoch. Resumption no longer marks a stage complete from epoch counts.
- **Validation Display** — Validator description overridden to `Validation`, replacing the raw metric header in the progress bar, with a per-epoch summary of train loss, validation loss (box, cls, dfl), mAP50 and mAP50-95.
- **Checkpoint and Export Isolation** — Checkpoints live only in `LemGendaryModels/yolov8n/checkpoints/`; `yolov8n.pt` and `yolov8n.onnx` are exported to `LemGendaryModels/yolov8n/` whenever SOTA targets (`mAP50 >= 0.54`, `mAP50-95 >= 0.39`) are reached.

### v16.9.13 — Intra-Resolution Fraction Progression & Governor Overfitting Rescue Protocol

- **Intra-Resolution Fraction Ladder Progression (`training/governance/yolo_governor.py`)** — Refactored `build_ladder_curriculum` to ensure training traverses dataset manifold fractions from initial (30%) through mid-fidelity (70%) to full data (100%) on the lowest resolution rung (320px) before escalating to higher spatial resolutions (`480px`, `640px`). Prevents spatial jumping on partial data slices and anchors basic feature extraction on the complete dataset distribution.
- **Governor Overfitting Rescue Protocol (`on_fit_epoch_end`)** — Introduced active overfitting detection during partial fraction training (`fraction < 1.0`). If train loss decreases while validation loss increases over a 3-epoch window, or validation mAP plateaus while train loss drops, the governor immediately triggers `trainer.stop = True`, halts the partial stage early, and escalates to the next expanded dataset fraction to introduce sample variety and break the overfitting attractor.
- **Multi-Fraction Checkpoint Discovery & Resumption Sync** — Enhanced stage indexing and checkpoint serialization to encode both spatial resolution and dataset fraction (`stage{idx}_{res}px_f{pct}`). Checkpoint inspection and metrics CSV parsing determine exact prior stage completion and enable seamless mid-stage resumption without losing epoch counts or resetting optimizer states.

### v16.9.12 — Checkpoint Resumption Protocol, Authoritative Persistence & Literature Citations

- **Mid-Rung Checkpoint Resumption & Stage Skip Protocol (`training/governance/yolo_governor.py`)** — Solved the issue where YOLO training restarted from epoch 1 instead of continuing from checkpoint. Interrogates checkpoint metadata (`torch.load`) to extract `ckpt_epoch` and `ckpt_imgsz`. If a resolution ladder rung was already satisfied, the governor skips it immediately. If interrupted mid-stage, it sets `resume=True` and resumes from `ckpt_epoch + 2` without resetting optimizer buffers.
- **Authoritative Single Source of Truth Persistence (`LemGendaryModels/`)** — Fixed relative path traversal in `training/checkpoint/recovery.py` and aligned `yolo_governor.py`, `core_loop.py`, and `engine.py` so that `LemGendaryModels/<model_key>/checkpoints/` and `LemGendaryModels/<model_key>/metrics.csv` are the authoritative persistence locations across all models.
- **Scientific Literature & Reference Citations (`training/doc_generator.py`)** — Embedded canonical research paper titles, authors, conference/journal publications, arXiv links, and BibTeX citations across all 22 model READMEs in `LemGendaryModels/<model_key>/README.md`.

### v16.9.11 — Real-Time Checkpoint Parity & Sub-Second Minibatch Cancellation Protocol

- **Dual-Path Real-Time Checkpoint Mirroring (`training/governance/yolo_governor.py`)** — Added `_synchronize_checkpoints` invoked on every completed epoch (`on_fit_epoch_end`) and spatial ladder transition. Intermediate weights (`best.pt`, `best.pth`, `last.pt`, `progress.pth`) are mirrored immediately into both `checkpoints/yolov8n/` and `LemGendaryModels/yolov8n/checkpoints/` alongside `metrics.csv`, eliminating delay in weight availability.
- **Sub-Second Minibatch Cancellation Hook (`on_train_batch_end`)** — Registered an inner-loop callback with the Ultralytics trainer that checks `cancel_check()` after every single minibatch and immediately sets `trainer.stop = True`. Propagated cancellation listeners through `TrainingService.train()`, `_run_yolo_native()`, and `run_training()`, enabling instant termination upon user stop request without waiting for epoch completion.
- **Ultralytics Checkpoint Inspector Support (`training/services/checkpoint_service.py`)** — Enhanced `CheckpointService.inspect_checkpoint()` to inspect Ultralytics `.pt` checkpoints natively, extracting parameter counts (3.15M), layer counts (129 layers), and `best_fitness` metrics for the desktop GUI checkpoint inspector.

### v16.9.10 — Governed YOLO Curriculum Runner & Sawtooth VRAM Pacing

- **Governed YOLO Curriculum Runner (`training/governance/yolo_governor.py`)** — Implemented `YOLOCurriculumGovernor` and `run_governed_yolo_training`. Coordinates multi-stage spatial resolution ladders (`320px -> 480px -> 640px`), hardware-aware Sawtooth VRAM batch allocation, dataset fraction scaling (`0.3 -> 0.6 -> 1.0`), numerical AMP stabilization on GTX 16xx hardware, cross-stage checkpoint handoff, and real-time telemetry synchronization with `TelemetryEngine` and GUI WebSocket callbacks.
- **In-Process Dynamic Stage Transition & SOTA Early Stopping** — Connected YOLO training loop to autonomous early stopping upon reaching target SOTA thresholds (`map50 >= 0.54, map50_95 >= 0.39`) at the 640px full-dataset rung, with automated stage escalation on validation plateau.
- **`core_loop.py` Delegation Architecture** — Refactored `_run_yolo_native` to delegate execution directly to `run_governed_yolo_training`, replacing monolithic fixed-resolution runs with autonomous curriculum progression while preserving native Ultralytics inner-loop optimizations.
- **Telemetry Parity & Full Training Certification** — Integrated per-epoch metric synchronization to `checkpoints/yolov8n/metrics.csv` and `LemGendaryModels/yolov8n/metrics.csv`, ensuring the desktop GUI receives resolution ladder progress, data fraction advancement, and 3-pillar SOTA convergence badges.

### v16.9.9 — In-Process YOLOv8n Execution, Ladder Stage Propagation & Hyperparameter Hydration

- **In-Process YOLOv8n Native Trainer Routing** — Integrated `_run_yolo_native` directly into `TrainingService.train()` when `model_key == "yolov8n"`. Prevents `ValueError` factory crashes by delegating execution to the Ultralytics trainer in-process and returning a valid `TrainingSummary`.
- **Resolution & Sawtooth Governor Parameter Propagation** — Added `ladder_stage` and `enable_sawtooth` to `QuickTrainRequest` in `training/server/routes/gui.py`, forwarded them through `jobs.py` to `TrainingService.train()`, and passed them into `args.resolution` and `args.enable_sawtooth`. Updated `_run_yolo_native` to honor `args.resolution` (as `imgsz`) and `args.lr` (as `lr0`). Added `--resolution` and `--enable-sawtooth` CLI options to `build_cli_parser()`.
- **Model Hyperparameter Defaults Hydration** — Enriched `GET /api/gui/models/with-stats` to include `learning_rate`, `batch_size`, and `default_epochs` in the model return payload, enabling desktop clients to synchronize form defaults immediately upon model selection.

### v16.9.8 — Authoritative 3-Pillar SOTA Convergence & Forex Multi-Timeframe Confluence Telemetry

- **Authoritative 3-Pillar Fully-Trained Verification Invariant** — Enforced non-negotiable triple-gate criterion for model status certification: `fully_trained` strictly requires (1) 100% of defined SOTA metrics satisfied across all specified targets in `sota_targets`, (2) full traversal through highest resolution ladder rung (`max_res_completed >= target_res`), and (3) completion on 100% training data fraction (`data_fraction_completed >= 0.99`). Sub-unitary or partial runs are designated strictly as `partially_trained`.
- **Multi-Metric SOTA Registry & Directional Gate** — Built `METRIC_REGISTRY` mapping 23 metrics across vision restoration, quality scoring, object detection, and financial domains. Automatically resolves directional criteria (lower-is-better for LPIPS, FID, MAE, MaxDD, TP/SL errors; higher-is-better for PSNR, SSIM, SRCC, PLCC, Accuracy, DirAcc, WinRate, Sharpe).
- **Forex Multi-Timeframe Confluence Ladder (`res_ladder: [1, 5, 15, 60, 240, 1440]`)** — Mapped resolution ladder to MetaTrader 5 candle intervals (`M1`, `M5`, `M15`, `H1`, `H4`, `D1`), isolated 8 financial scorecard metrics from vision datasets, and updated `core_loop.py` to extract pairs and active timeframes directly from YAML `kwargs`.
- **Desktop GUI Telemetry Hydration** — Enriched `GET /api/gui/models/with-stats` with `sota_targets_total`, `sota_targets_met`, `sota_all_met`, `sota_details`, `ladder_type`, `is_forex`, `ladder_passed`, and `data_fraction_passed`.

### v16.9.0 — Cross-Suite E2E Testing, WebDataset Parity & Tripartite Sidecar Mesh

- **Cross-Suite End-to-End Integration Testing** — Implemented and verified `tests/test_integration_e2e.py` validating that streaming WebDataset `.tar` shards produced by `lemgendary-datasets` are auto-discovered, decoded into tensors, and ingested by the training engine without intermediate filesystem extraction.
- **Tripartite Health Mesh Compatibility** — Validated FastAPI sidecar daemon route schemas and health telemetry on port 8200 (`/api/training/models`, `/health`), guaranteeing reliable status reporting across the unified desktop client.
- **Full Compliance Testing** — Verified 100% test pass rate and compliance under `lem-env validate --project lemgendary-training-suite`.

### v16.8.0 — Container Dependencies, Model Registry Modernization & Zero-Duplication Storage Integration

- **Unified Container Dependencies Manifest Synchronization (Phase 1)** — Synchronized `requirements.txt` from `lemgendary-env-manager` SSOT manifest (`requirements-training.txt`) establishing verified runtime support for all 5 container formats: `webdataset==1.0.2`, `mosaicml-streaming>=0.9.0,<1.0.0`, `litdata>=0.2.0,<1.0.0`, `pyarrow==25.0.1`, and `zstandard>=0.23.0`.
- **Sidecar Web Service Dependencies Alignment (Phase 1)** — Embedded missing web server dependencies (`fastapi==0.141.1`, `uvicorn==0.53.0`, `pydantic==2.13.5`, `pydantic-core==2.46.5`, `httpx==0.28.1`) in the SSOT manifest.
- **Cloud Notebook Auto-Installers Hardening (Phase 1)** — Upgraded `training/notebooks/cells/deps.py` with automated fallback installation of core streaming container dependencies in both Kaggle and Google Colab execution targets.
- **Virtual Environment Runtime Validation (Phase 1)** — Installed and verified zero-conflict execution across all 5 container format loaders, PyTorch 2.5.1, Diffusers 0.40.0, and Transformers 5.17.0 in `lemgendary-training-suite\.venv`.
- **Unified Models Registry Modernization (Phase 2)** — Upgraded `unified_models_v2.yaml` to Version 2.3.0, registering `canonical_format` (`parquet`, `litdata`, `mds`, `webdataset`, `directory`) across all 22 active neural architectures.
- **Zero-Duplication Container Integration & Multi-Modal Streaming (Phase 4)** — Aligned training data pipelines with modern single-format container manifolds (`.tar`, `.mds`, `chunk*.bin`), supporting self-contained multi-modal restoration samples (`target.webp`, `mask.webp`) with zero-overhead memory mapping.
- **WebDataset Reader Multi-Modal & 10-Bin Distribution Hardening (Phase 5)** — Upgraded `training/data/containers/webdataset.py` with multi-modal paired sample parsing, extracting ground-truth targets (`target.webp`), segmentation masks (`mask.webp`), text captions, and full JSON payloads containing 10-bin human perceptual quality distributions for NIMA models. Added recursive and nested directory search (`shards/*.tar`, `shards/<split>/*.tar`).
- **LitData Streaming Container Reader Hardening (Phase 5)** — Upgraded `training/data/containers/litdata.py` to auto-discover nested litdata directories (`litdata/<split>`, `<split>/litdata`, `litdata`), guard `StreamingDataset` initialization behind `index.json` presence to eliminate un-indexed directory stub hangs, handle variable-shape tensor sample dicts, and gracefully recover from initialization faults with zero silent failures.
- **MosaicML Streaming (MDS) Reader Resilience (Phase 5)** — Hardened `training/data/containers/mds.py` with multi-tier directory resolution (`mds/<split>`, `mds`), dual `LocalDataset` and `StreamingDataset` initialization with fallback to zero-latency raw index parsing, and multi-modal sample extraction.
- **Universal Container Resolver Ecosystem (Phase 5)** — Upgraded `training/data/containers/__init__.py` (`resolve_container_reader`) to prioritize top-level `format` and `canonical_format` declarations in `dataset_info.yaml`, with heuristic fallback detecting nested container subdirectories (`shards/`, `mds/`, `litdata/`, `parquet/`).
- **Workspace-Level Notebook Directory Resolution (Phase 6)** — Fixed workspace directory resolution in `training/notebooks/builders/base.py` (`_get_workspace_root`), eliminating incorrect path traversal outside the project and synchronizing notebooks strictly to `kaggle_training` and `colab_training`.
- **Notebook Generation & Baseline Parity Verification (Phase 6)** — Synchronized cell header titles in `training/notebooks/builders/colab.py` with `lemgendary-datasets` and refreshed test fixtures in `tests/fixtures/baseline_notebooks/`, passing all 8 notebook subsystem and all 6 container reader unit tests with 100% success.

### v16.2.9 — Parallel Strategy Audit, Modernized Manifold Sync & DDP Tooling

- **Parallel Strategies Audit & DDP Engine** — Completed full audit of Single, DataParallel (DP), and DistributedDataParallel (DDP) execution pathways. Upgraded `training/parallel/ddp.py` with multi-GPU initialization using PyTorch distributed process groups (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`), gradient accumulation via `no_sync()`, broadcast sync, and safe distributed cleanup. Added `preferred_parallel` field per model in `unified_models_v2.yaml` (`ddp` for large vision backbones, `dp` for multitask restoration, `single` for lightweight / sequential models).
- **Modernized Dataset Format Synchronization** — Added full lossless `.webp` format and container fallback decoding to `data/dataset.py`, ensuring native loader compatibility with all modernized manifold container standards (Directory, Parquet, MDS, LitData, WebDataset). Fixed bare exception handlers to eliminate silent failures.
- **YOLO Ultralytics Trainer Delegation** — Enforced delegation guard in `models/factory.py` directing YOLOv8 architectures to Ultralytics native trainer with dedicated multi-GPU DDP support.
- **Distributed Training Launcher Tool** — Introduced `tools/launch_ddp.py` wrapping `torchrun` for multi-GPU execution in local cluster, cloud, and Kaggle environments with dynamic worker configuration and automated port assignment.
- **Diagnostic & Verification Hardening** — Resolved all IDE type and parameter mismatch diagnostics in `training/core_loop.py` for Forex dataset loading, checkpoint recovery discovery, resume state handling, export parameters, and notebook generation paths. Validated with 100% pass rate under `lem-env validate`.

### v16.2.8 — Full S.O.L.I.D. Modernization, Sidecar Daemon & Canonical Presets (Phases 0-14)

Completed the comprehensive 2026 architectural refactoring and modernization roadmap across all 15 milestone phases:

- **Phase 14: Root Decluttering & Zero-Suppression Hardening** — Evicted residual binary blobs (`yolov8n.pt`, `face_landmarker.task`) to `.gitignore`-protected storage (`checkpoints/`). Pruned temporary directories and enforced clean project root containing exclusively canonical entrypoints (`cli.py`, `lemgendary_models_hub.ps1`), configuration manifests (`config.yaml`, `unified_models_v2.yaml`, `presets.yaml`, `openapi.json`), dependencies manifests, and subpackages. Maintained 100% zero-suppression policy with zero compiler errors and zero linter warnings.
- **Phase 13: Canonical Presets & Desktop GUI Integration** — Defined canonical training presets in `presets.yaml` (`quick-sota`, `debug-tiny`, `walk-forward`, `restoration-ultra`, `detection-yolo`). Implemented Desktop GUI dashboard aggregation endpoints in `training/server/routes/gui.py` (`GET /api/gui/state`, `GET /api/gui/models/with-stats`, `POST /api/gui/quick-train`). Exported frozen OpenAPI 3.1 contract (`openapi.json`) for TypeScript client generation in `lemgendary-ai-studio-gui`.
- **Phase 12: FastAPI Sidecar Daemon & Persistent Job Queue** — Established background sidecar service on `127.0.0.1:8200` with ACID SQLite persistent job queue in `.lemtrain_server/jobs.db` using WAL mode. Added local token security (`.lemtrain_server/token`), process PID management, restart recovery for interrupted jobs, thread pool background workers, and real-time streaming WebSockets at `/api/ws/jobs/{job_id}/logs` and `/api/ws/logs`. Added `lemtrain server start/stop/status/openapi` CLI commands.
- **Phase 11: In-Process S.O.L.I.D. Services Layer & Canonical Typer CLI** — Decomposed monolithic script workflows into dedicated single-responsibility services (`TrainingService`, `EvaluationService`, `CheckpointService`, `ExportService`, `CloudSyncService`, `NotebookService`, `AuditService`). Replaced fragmented operational scripts with canonical Typer CLI (`training/cli/lemtrain.py` and root `cli.py`) exposing `train`, `eval`, `export`, `notebooks`, `checkpoints`, `presets`, `audit`, `sync`, and `server`. Relocated operational scripts to `tools/` with backward-compatible shims.
- **Phase 10: Modular Training Engine Coordinator** — Modularized the multi-epoch training engine into `training/training/` (`context.py`, `amp.py`, `optimizer.py`, `epoch.py`, `validation.py`, `engine.py`). Deconstructed monolithic `core_loop.py` from 4,513 lines down to a clean facade delegating execution to `run_training(ctx)`. Verified exact 3-epoch metric trajectory match against baseline.
- **Phase 9: Modular Notebook Cell Architecture & Cross-Repo Sync** — Built granular cell generator framework (`training/notebooks/cells/`) and platform builders for Kaggle and Google Colab. Reduced `notebook_generator.py` from 2,187 lines to 88 lines while guaranteeing 100% bit-exact notebook structural parity. Cross-synchronized modular cell architecture to `lemgendary-datasets/tools/notebooks/` and modularized `core/docs/`.
- **Phase 8: Model Export Subsystem** — Implemented dedicated export subpackage `training/export/` orchestrating concurrent FP32, FP16, standalone PyTorch (`.pt`), fixed-shape WebGPU (Opset 17), and Forex MT5 signal targets via unified `export_all()` coordinator. Verified byte-identical parity against reference exports.
- **Phase 7: Unified Cloud Management** — Implemented structured `CloudManager` protocol in `training/cloud/` with multi-provider credential discovery, stealth secret masking, timeout-bounded git-lfs pushes with auto-rebase, zero-copy hardlink staging for Kaggle Hub, and FUSE/REST Google Drive synchronization with SHA256 verification.
- **Phase 6: Governance, Curriculum, Thermal & Metric Registry** — Encapsulated optimization intelligence into `training/governance/` (`metrics.py`, `curriculum.py`, `thermal.py`, `sota.py`, `governor.py`). Reduced `optimization_engine.py` from 1,117 lines to 82 lines while preserving full backward compatibility for `audit_epoch()` and loop-breaker policies.
- **Phase 5: Checkpoint Management, Recovery BFS & SOTA Rollback** — Built atomic checkpoint manager with `.tmp` file staging, disk space headroom evaluation, emergency pruning, typed `ResumeState`, proportional scheduler runway stretching, and Kaggle BFS recovery.
- **Phase 4: Modern Container Readers & Lossless Decoders** — Implemented zero-copy container readers in `training/data/containers/` for Directory (lossless WebP/PNG/JPEG), Parquet (PyArrow LRU caching), MDS (MosaicML Streaming with Zstd), LitData (tensor streaming), and WebDataset (sharded tar), resolved via `dataset_info.yaml` manifests.
- **Phase 3: Data Loaders, Worker Topologies & Dynamic Degradation** — Unified cross-platform `ManifoldResolver`, hardware-tuned worker topology calculation (`WorkerTopology`), leak-free loader disposal, and dynamic synthetic degradation routines linking cleanly with `lemgendary-datasets`.
- **Phase 2: Hardware Discovery, Execution Policy & SentinelGuard** — Implemented hardware auto-discovery probing CUDA, MPS, XPU, DML, and CPU architectures with execution policies for cuDNN benchmarking, TF32, channels-last memory formats, and proactive VRAM headroom protection via `SentinelGuard`.
- **Phase 1: Core Utilities, Structured Secrets & Delegation Adapters** — Established repository root auto-discovery (`get_project_root()`), `ForceTTY` stream logging, safe subprocess execution, signal traps and active process tracking, cross-repo delegation adapters (`EnvManagerDelegate`, `DatasetCompilerDelegate`), and dotfile secret parsers.
- **Phase 0: Baseline Freeze & Binary Blobs Eviction** — Created git baseline tag `pre-refactor-v2`, evicted binary files from root, configured `pyproject.toml` and strict `pyrightconfig.json`, and implemented deterministic fixtures.

---

## Output Topology

```text
checkpoints/                                       # Model weights (.gitignore protected)
|-- <model_key>/
|   |-- best.pth                                   # SOTA metric checkpoint
|   |-- progress.pth                               # Intra-epoch recovery checkpoint
|   `-- history.csv                                # Training validation metric trajectory
export/                                            # Model compilation artifacts
|-- <model_key>_fp32.onnx                          # FP32 ONNX inference graph
|-- <model_key>_fp16.onnx                          # FP16 half-precision ONNX
|-- <model_key>_webgpu.onnx                        # Fixed-shape WebGPU Opset 17 graph
`-- <model_key>_standalone.pt                      # Standalone PyTorch artifact
.lemtrain_server/                                  # Sidecar daemon state root (.gitignore)
|-- jobs.db                                        # SQLite persistent job queue (WAL mode)
|-- token                                          # Master local authentication token
|-- server.pid                                     # Daemon process identifier
`-- logs/<job_id>.log                              # Buffered job execution logs
```

---

## Project Ecosystem

| # | Project | Folder | Description |
| :--- | :--- | :--- | :--- |
| 1 | LemGendary Environment Manager | `.\lemgendary-env-manager\` | Ecosystem CLI (`lem-env`), virtual environment lifecycle, multi-gate validation |
| 2 | LemGendary Dataset Compiler Suite | `.\lemgendary-datasets\` | Manifold compiler, container format transcoders, dataset sidecar daemon on port 8100 |
| 3 | LemGendary Model Training Suite | `.\lemgendary-training-suite\` | Universal model training, SOTA governance, Typer CLI (`lemtrain`), sidecar daemon on port 8200 |
| 4 | LemGendary AI Studio GUI | `.\lemgendary-ai-studio-gui\` | Tauri v2 Desktop GUI interface consuming sidecar REST and WebSocket endpoints |
| 5 | LemGendary AI Documentation Hub | `.\lemgendary-docs\` | Comprehensive research whitepapers, architecture specifications, API and CLI manuals |
| 6 | LemGendary Compiled Manifolds | `.\LemGendaryDatasets\` | Physical dataset manifolds, shards, Parquet tables, and dataset manifests |
| 7 | LemGendary Trained Models | `.\LemGendaryModels\` | Versioned neural model weights, ONNX binaries, and evaluation cards |

---

LemGendary AI Suite — Advanced Agentic Coding 2026
