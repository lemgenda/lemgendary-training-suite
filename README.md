# LemGendary Model Training Suite

> Industrial-standard training, evaluation, export, and telemetry orchestration suite for Vision, Restoration, and Financial Time-Series deep learning models.
>
> **Architecture details and technical whitepapers:** [Master Training Suite Guide (PAPER_TRAINING_SUITE.md)](../lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md) · [Master CLI Manual (MANUAL_CLI.md)](../lemgendary-docs/MD-Papers/MANUAL_CLI.md) · [Master API Manual (MANUAL_API.md)](../lemgendary-docs/MD-Papers/MANUAL_API.md) · [Documentation Hub](https://lemgenda.github.io/ai-training-whitepapers/index.html)

---

## Current Status

| Metric | Value |
| :--- | :--- |
| **Version** | `v2026.11.1` |
| **Phase** | Post-Modernization S.O.L.I.D. Architecture Complete (Phases 0-14 Complete) |
| **Next** | Production Modernization & S.O.L.I.D. Architecture Complete — Ecosystem Ready |
| **Verified Models** | 24+ production architectures, multi-platform execution (Local, Kaggle T4x2/P100, Google Colab) |
| **Roadmap** | [training_suite_refactoring_roadmap.md](../lemgendary-docs/roadmaps/training_suite_refactoring_roadmap.md) |

---

## Changelog

### v2026.11.1 — Parallel Strategy Audit, Modernized Manifold Sync & DDP Tooling

- **Parallel Strategies Audit & DDP Engine** — Completed full audit of Single, DataParallel (DP), and DistributedDataParallel (DDP) execution pathways. Upgraded `training/parallel/ddp.py` with multi-GPU initialization using PyTorch distributed process groups (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`), gradient accumulation via `no_sync()`, broadcast sync, and safe distributed cleanup. Added `preferred_parallel` field per model in `unified_models_v2.yaml` (`ddp` for large vision backbones, `dp` for multitask restoration, `single` for lightweight / sequential models).
- **Modernized Dataset Format Synchronization** — Added full lossless `.webp` format and container fallback decoding to `data/dataset.py`, ensuring native loader compatibility with all modernized manifold container standards (Directory, Parquet, MDS, LitData, WebDataset). Fixed bare exception handlers to eliminate silent failures.
- **YOLO Ultralytics Trainer Delegation** — Enforced delegation guard in `models/factory.py` directing YOLOv8 architectures to Ultralytics native trainer with dedicated multi-GPU DDP support.
- **Distributed Training Launcher Tool** — Introduced `tools/launch_ddp.py` wrapping `torchrun` for multi-GPU execution in local cluster, cloud, and Kaggle environments with dynamic worker configuration and automated port assignment.
- **Diagnostic & Verification Hardening** — Resolved all IDE type and parameter mismatch diagnostics in `training/core_loop.py` for Forex dataset loading, checkpoint recovery discovery, resume state handling, export parameters, and notebook generation paths. Validated with 100% pass rate under `lem-env validate`.

### v2026.11.0 — Full S.O.L.I.D. Modernization, Sidecar Daemon & Canonical Presets (Phases 0-14)

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
