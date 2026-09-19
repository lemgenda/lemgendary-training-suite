# LemGendary Model Training Suite

> Industrial-standard training, evaluation, export, and telemetry orchestration suite for Vision, Restoration, and Financial Time-Series deep learning models.
>
> **Function reference and architecture details:** [Whitepaper (PAPER_TRAINING_SUITE.md)](./lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md) · [Roadmap (training_suite_refactoring_roadmap.md)](./lemgendary-docs/roadmaps/training_suite_refactoring_roadmap.md)

---

## Current Status

| Metric | Value |
| :--- | :--- |
| **Version** | `v2026.11.0` |
| **Phase** | Post-Modernization S.O.L.I.D. Architecture Complete (Phases 0-14 Complete) |
| **Next** | Production Modernization & S.O.L.I.D. Architecture Complete — Ecosystem Ready |
| **Verified Models** | 24+ production architectures, multi-platform execution (Local, Kaggle T4x2/P100, Google Colab) |
| **Roadmap** | [training_suite_refactoring_roadmap.md](../lemgendary-docs/roadmaps/training_suite_refactoring_roadmap.md) |

---

## Changelog

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

## Getting Started

### 1. Canonical CLI (`lemtrain` / `python cli.py`)

The primary command-line interface for local training, evaluation, compilation, and cloud synchronization:

```bash
# Display global help and command groups
python cli.py --help

# In-process training run using a canonical preset
python cli.py train mirnet_exposure --preset quick-sota --epochs 10

# Deterministic validation under torch.no_grad()
python cli.py eval mirnet_exposure --checkpoint checkpoints/mirnet_exposure/best.pth

# Multi-target model compilation (FP32 ONNX, FP16 ONNX, PyTorch standalone, WebGPU)
python cli.py export mirnet_exposure

# Kaggle and Google Colab training notebook generation
python cli.py notebooks mirnet_exposure --platform all

# Inspect and prune checkpoints
python cli.py checkpoints list mirnet_exposure
python cli.py checkpoints prune mirnet_exposure --keep 3

# View and inspect canonical presets
python cli.py presets list
python cli.py presets show quick-sota

# System resource and model architecture audit
python cli.py audit system
python cli.py audit model mirnet_exposure

# Cloud checkpoint synchronization
python cli.py sync run mirnet_exposure --target gdrive --epoch 5
```

### 2. Sidecar Daemon Control

The training suite includes a background FastAPI service for headless automation and desktop GUI integration:

```bash
# Start sidecar daemon in background on port 8200
python cli.py server start --daemon

# Check daemon health and active hardware sensors
python cli.py server status

# Export OpenAPI 3.1 specification contract
python cli.py server openapi --output openapi.json

# Stop running daemon process
python cli.py server stop
```

### 3. Master Models Hub Console

The interactive PowerShell terminal console for rapid system bootstrapping, fleet smoke testing, and headless Kaggle cloud orchestration:

```powershell
./lemgendary_models_hub.ps1
```

---

## Canonical Presets Specification

The training suite standardizes training execution across CLI, API, and Desktop GUI through profiles defined in `presets.yaml`:

| Preset | Precision | Batch Size | Learning Rate | Optimizer | Scheduler | Epochs | Curriculum | SOTA Tracking | Sentinel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `quick-sota` | `amp_fp16` | 64 | 0.001 | `adamw` | Cosine Warm Restarts | 50 | Yes | Yes | Yes |
| `debug-tiny` | `fp32` | 4 | 0.0001 | `adam` | Constant | 2 | No | No | No |
| `walk-forward` | `amp_fp16` | 512 | 0.0005 | `adamw` | Plateau | 100 | Yes | Yes | Yes |
| `restoration-ultra` | `amp_fp16` | 16 | 0.0002 | `adamw` | Cosine | 80 | Yes | Yes | Yes |
| `detection-yolo` | `amp_fp16` | 32 | 0.001 | `sgd` | Linear | 100 | No | Yes | Yes |

---

## Sidecar API Architecture (`127.0.0.1:8200`)

The FastAPI sidecar daemon coordinates background training, evaluations, exports, and real-time telemetry:

- **State Root**: `.lemtrain_server/` (local tokens, process PID, SQLite database, per-job log files).
- **Persistence**: SQLite database (`jobs.db`) configured with `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;`.
- **Authentication**: Local security token in `.lemtrain_server/token` validated via `Authorization: Bearer <token>` or `X-LemTrain-Token` header.
- **REST Endpoints**:
  - `GET /api/health` — Daemon uptime, version, and GPU accelerator profile.
  - `GET /api/config` & `GET /api/presets` — Suite configuration and canonical training presets.
  - `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/logs` — Asynchronous job lifecycle.
  - `GET /api/models`, `GET /api/models/{key}`, `GET /api/models/{key}/audit` — Model parameter topologies and architecture audits.
  - `POST /api/training/train`, `/evaluate`, `/export` — In-process job dispatch.
  - `GET /api/datasets` — Discovered compiled manifolds and datasets.
  - `GET /api/env/telemetry` — Host hardware metrics (CPU, RAM, GPU, VRAM, Disk).
  - `GET /api/gui/state`, `GET /api/gui/models/with-stats`, `POST /api/gui/quick-train` — Desktop GUI aggregation and quick dispatch.
- **WebSocket Streaming**:
  - `/api/ws/jobs/{job_id}/logs` — Line-by-line live log streaming for specific jobs.
  - `/api/ws/logs` — Global daemon event broadcast stream.
- **Interactive Documentation**: Swagger UI at `http://127.0.0.1:8200/docs` and OpenAPI JSON at `http://127.0.0.1:8200/openapi.json`.

---

## Project Structure

```text
lemgendary-training-suite/
|-- cli.py                         # Canonical root CLI entrypoint
|-- lemgendary_models_hub.ps1      # Interactive PowerShell terminal orchestrator
|-- config.yaml                    # Global paths and runtime environment manifest
|-- unified_models_v2.yaml         # Master model architecture registry
|-- presets.yaml                   # Canonical training preset definitions
|-- openapi.json                   # Frozen OpenAPI 3.1 specification contract
|-- pyproject.toml                 # Package manifest and build metadata
|-- requirements.txt               # Pinned Python package dependencies
|-- pyrightconfig.json             # Strict static type checker configuration
|-- README.md                      # Architecture documentation and changelog
|-- models.md                      # Registered model descriptions and specifications
|
|-- training/                      # Core training suite package
|   |-- checkpoint/                # Atomic checkpointing, Kaggle BFS recovery, resume stretching
|   |-- cli/                       # Canonical Typer CLI command definitions
|   |-- cloud/                     # Multi-provider cloud sync (GitHub, Kaggle, Google Drive)
|   |-- config/                    # Structured secrets and environment configurations
|   |-- data/                      # Manifold resolver, worker topologies, container readers
|   |-- export/                    # ONNX (FP32/FP16), PyTorch standalone, WebGPU, MT5 exporters
|   |-- governance/                # Metrics registry, curriculum ladders, thermal cooling, SOTA tracker
|   |-- hardware/                  # Device discovery, execution policies, SentinelGuard
|   |-- notebooks/                 # Modular notebook cell generators and platform builders
|   |-- parallel/                  # Multi-GPU DataParallel and DistributedDataParallel strategies
|   |-- server/                    # FastAPI sidecar daemon, SQLite WAL queue, REST/WS routes
|   |-- services/                  # In-process S.O.L.I.D. services layer
|   |-- training/                  # Modular multi-epoch engine, AMP context, optimizer factories
|   `-- utils/                     # Paths auto-discovery, logging wrappers, subprocess runner
|
|-- models/                        # Deep learning architecture definitions and factory
|-- data/                          # Dataset containers and multi-task datasets
|-- export/                        # Export utilities and legacy shims
|-- tools/                         # Relocated auxiliary scripts and diagnostic tools
`-- tests/                         # Test suite
    |-- smoke/                     # Entrypoint smoke tests
    `-- unit/                      # 114 passing unit tests across all subsystems
```

---

## Ecosystem Quality & Compliance Standards

The LemGendary Model Training Suite adheres strictly to ecosystem governance standards:

- **Zero Suppressions**: 100% free of `# type: ignore`, `# noqa`, `# pylint: disable`, and bare `except:` blocks.
- **Zero Emojis**: Complete absence of Unicode emojis in source code, tests, docstrings, and documentation.
- **Strict Typing**: Fully typed with strict Pyright verification.
- **Full Test Coverage**: 114 unit tests passing cleanly across all subsystems.
- **Multi-Gate Compliance**: Verified with 100% `[PASS]` under `env_manager.cli validate -p lemgendary-training-suite`.
