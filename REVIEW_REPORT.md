# LemGendary Model Training Suite - Lead Architect Review

Review type: read-only review. No source, config, or existing document was
changed. Scope: `./` (this repository). Context: the `../ROBOTS.md` and
`../Instructions.md` governance files, the sibling repositories, and
`../lemgendary-docs/MD-Papers/PAPER_TRAINING_SUITE.md` and
`PAPER_TRAINING_PATHOLOGY.md`.

Code snapshot: HEAD `e598d4e`, clean working tree. Version strings disagree
(see Documentation Findings).

Severity scale: Critical, High, Medium, Low. Every finding cites
`path:line` relative to this repository unless another root is named.
Statements that could not be verified without executing torch code are
marked **(unverified at runtime)**.

## 1. Executive Findings

### 1.1 Strengths

- The services layer exists (`training/services/*`). The Typer CLI
  (`training/cli/lemtrain.py`) and the FastAPI routes call the same
  `TrainingService`, `CheckpointService`, `AuditService` and other service
  classes for most operations.
- Publication-grade FID is implemented as ROBOTS.md requires:
  `FrechetInceptionDistance(feature=2048)` with 2-sample micro-chunks
  (`training/training/validation.py:412-434`).
- The governor's retreat path cools the learning rate and holds the manifold
  instead of retreating (`training/governance/governor.py:842-866`). The
  explicit `can_spatial_retreat = False` comment documents the ratchet
  intent.
- The resume path reconciles the fraction upward against `metrics.csv`
  (`training/core_loop.py:539-563`). This matches the paper's
  `f_{t+1} = max(f_t, f_new)` claim.
- The sidecar has a job queue backed by SQLite in WAL mode
  (`training/server/state.py`), cancellation events, and log streaming over
  WebSockets.
- All 22 endpoints that the GUI calls on port 8200 exist in code (see 7.2).
- The secrets loader masks values and reads dotfiles relative to the project
  and workspace roots (`training/config/secrets.py:10-70`).

### 1.2 Weaknesses

- God modules. The highest radon cyclomatic complexities are
  `SmartTrainingGovernor.audit_epoch` with CC 171
  (`training/governance/governor.py:588`), `validate_one_epoch` with CC 168
  (`training/training/validation.py:75`), `get_models_with_stats` with
  CC 163 (`training/server/routes/gui.py:131`), and `build_training_context`
  with CC 128. Radon rates 18 functions at grade F.
- Largest files: `training/governance/yolo_governor.py` (1581 lines),
  `training/governance/governor.py` (1285), `training/server/jobs.py`
  (1062), `data/mt5_pipeline.py` (843), `data/dataset.py` (840),
  `data/forex_dataset.py` (836), `training/core_loop.py` (787),
  `training/cli/lemtrain.py` (754), `training/training/validation.py`
  (714), and `training/server/routes/gui.py` (653).
- Error swallowing is widespread. Ruff (isolated mode) reports 210 blind
  `except Exception` handlers (BLE001) and 39 try-except-pass blocks (S110).
  The worst files are `training/server/jobs.py` (25),
  `training/kaggle_monitor.py` (22), and
  `training/training/validation.py` (13).
- Several advertised governor features are dead code at runtime: the
  loop-breaker, Mini-SWA, the resolution veto, and early stopping (see 4.1).
- Inventories and defaults are duplicated in at least five places (see 7.1).

### 1.3 Risks

- Security: the sidecar serves every route without authentication by
  default and allows CORS from any origin with credentials (see D-05). Any
  web page open in the user's browser could start training, Kaggle pushes,
  or the planned bug-report e-mail.
- Scientific integrity: `tools/recalibrate_fid.py` rewrites historical
  `metrics.csv` FID values with heuristic formulas (D-03). The GUI reports
  SOTA targets as achieved values when no measurement exists (D-08).
- Unbounded training: the epoch loop never stops on budget or on the
  governor's early-stop signal (D-01). A 1-epoch fleet smoke test cannot
  finish.

### 1.4 Opportunities

- Make `unified_models_v2.yaml` the single model and metric registry. Expose
  it via `/api/models` and generate `models.md`, the GUI fallback data, and
  `canonical_capabilities` entries from it.
- Extract a typed `CurriculumPolicy` and `RatchetGuard` with unit tests that
  make the ROBOTS.md invariants machine-checked.
- Replace the `if/elif job_type` dispatch (`training/server/jobs.py:338-357`)
  with a job-handler registry (OCP).

## 2. Architecture Findings

### 2.1 Current Architecture

- Entry points: the `cli.py` shim leads to `training/cli/lemtrain.py`
  (Typer). `train_all.py`, `cloud_hub.py`, `judicial_audit_api.py`, and
  `sync_to_gdrive.py` are shims to `tools/*`. `training/train.py` is a
  legacy argparse entry into `training/core_loop.main`.
- Core training path: `TrainingService` leads to `core_loop.run_training` /
  `build_training_context` (`training/core_loop.py`). That leads to
  `training/training/engine.py` (epoch loop), which uses `epoch.py`,
  `validation.py`, and `governance/governor.py`, and persists through
  `checkpoint/lifecycle_manager.py`.
- Sidecar: `training/server/app.py` (factory) has 10 route modules under
  `/api`. `training/server/jobs.py` runs jobs in-process in a 2-worker thread
  pool. It replaces `sys.stdout` and `sys.stderr` globally with a tee
  (`training/server/jobs.py:66-160`).
- Compatibility shims stay at the package root: `training/sota_rollback.py`,
  `training/optimization_engine.py`, `training/model_registry.py`,
  `training/cloud_sync.py`, and `training/gdrive_cloud_manager.py`.
- Dependency direction is mostly inward (routes and CLI depend on services,
  services on core). There are exceptions. Routes import private helpers
  from other routes (`training/server/routes/gui.py:13` imports
  `_load_registry` from `routes/models.py`). Route modules also contain
  business logic: `gui.py` computes training completion and `models.py`
  computes soup paths.

### 2.2 Proposed Architecture

- `training/registry/` becomes the only reader of
  `unified_models_v2.yaml`. It provides a `MetricSpec` registry (direction,
  weight, CSV column, label) that replaces `governance/metrics.py`
  `DEFAULT_*`, `routes/gui.py:96` `METRIC_REGISTRY`, and the quality-score
  copies.
- Add `training/paths/hub.py` with one `ModelHubLocator` (resolved from
  `config.yaml: paths.*`). It replaces the 15+ ad-hoc
  `"LemGendaryModels"` joins (see 7.1).
- Add `training/services/status_service.py` (the 4-pillar completion
  evaluator). Both `engine.py:424-490` and `routes/gui.py:453-501` would
  call it.
- Add `training/governance/policy/` with `CurriculumPolicy`
  (fraction/resolution transitions plus a ratchet guard), `ThermalPolicy`,
  `JoltPolicy`, and `PlateauPolicy`. `audit_epoch` becomes an orchestrator
  under 100 lines.
- Add `training/server/jobs/handlers/` with one handler class per job type,
  registered by key.
- Add a `training/server/security.py` dependency that applies to all
  routers, including WebSockets. Restrict CORS to the configured GUI
  origins.

## 3. Refactoring Plan (Proposed, Not Applied)

1. Split `governor.py` into policies. Preserve `get_state` and `load_state`
   keys for checkpoint compatibility, and add ratchet property tests.
2. Extract `ModelHubLocator` and replace the direct joins listed in 7.1.
   Delete the test-only branch in `get_model_hub_dir`
   (`training/export/universal_exporter.py:32-42`), which compares roots,
   and remove its `mkdir` side effect.
3. Extract `StatusService` (the 4-pillar check) and reuse it in the engine
   and GUI routes.
4. Move `KNOWN_PARAMS_M` (`training/server/routes/gui.py:68-91`) into the
   registry as `parameters_m`, and derive the `PHASES` model lists in
   `tools/train_all.py:41-120` from registry `category` and `phase` fields.
5. Replace subprocess fan-out to in-repo scripts with in-process service
   calls. Affected call sites: `training/cli/lemtrain.py:196`,
   `training/server/jobs.py:725`, `tools/train_all.py:186`, and
   `training/train_forex_curriculum.py:254`.
6. Break `routes/gui.py:get_models_with_stats` into a CSV telemetry reader,
   ladder resolution from the registry, and `StatusService`.
7. Replace the `JobManager` `if/elif` dispatch with a handler registry.
8. Delete the dead modules: `training/prefetch_worker.py` (no callers), the
   root-level shims after deprecation, and the unused governor
   loop-breaker, or wire it in (see 4.1).
9. Remove the `.pylintrc` suppressions (`.pylintrc:6-54`, including
   `bare-except`, `broad-exception-caught`, and `used-before-assignment`).
   Then fix the findings. These disables are config-based suppression,
   which ROBOTS.md forbids.

## 4. Debug Report

Issues are ordered by severity. Each entry gives severity, root cause, and
proposed fix.

### D-01 High - Epoch loop is unbounded and ignores early stop

- Evidence: `training/training/engine.py:81` is `while True`. The loop
  exits only on 4-pillar completion (`:486-492`) or cancellation. At
  `:495-503` it extends `total_epochs` by 10 whenever the budget is reached.
  The governor's early-stop flag is unpacked as `_stop_trig` (`:256`) and
  never read. `training/core_loop.py:660` forces `total_epochs >= 20` on
  resume.
- Impact: `lemtrain test-all --epochs 1`, `/api/training/unit-test-all`, and
  the "1-epoch diagnostic" fleet run can never finish. Pillar 1 needs at
  least 20 epochs, so they keep running until they are cancelled or exhaust
  the Kaggle quota.
- Root cause: the "Open-Ended Horizon" feature (README v16.9.33) replaced
  the budget contract instead of being an opt-in.
- Fix: honour `total_epochs` as a hard cap unless
  `optimization.open_ended: true`. Act on `_stop_trig`. Add a
  `max_extension_epochs` ceiling. Add an engine test with a stub governor.

### D-02 High - Pareto Model Soup CLI and API are non-functional

- Evidence 1: `training/cli/lemtrain.py:83` and
  `training/server/routes/models.py:190` pass a directory `Path` to
  `create_pareto_model_soup`, which iterates over it as
  `list[Path]` (`training/export/soup.py:85`). This raises `TypeError`
  **(unverified at runtime; follows from types)**.
- Evidence 2: both callers look under `project_root/LemGendaryModels`
  (`lemtrain.py:74`, `models.py:183`). Vaults are written to
  `project_root.parent/LemGendaryModels` (`universal_exporter.py:40`,
  `core_loop.py:204`), so the directory check always fails.
- Evidence 3: `training/server/routes/models.py:186` uses `status`, which
  is never imported (ruff F821). The 404 branch therefore raises
  `NameError`, and the GUI's `blendModelSoup` call receives HTTP 500.
- Fix: resolve the hub through one locator and glob `vault_*.pth`. Accept
  explicit donors and weights in a Pydantic body. Import
  `fastapi.status`. Add route and CLI tests. `tests/unit/test_soup.py`
  covers only the library function.

### D-03 High - Fabricated FID history (`tools/recalibrate_fid.py`)

- Evidence: `tools/recalibrate_fid.py:20-100` overwrites `metrics.csv` FID
  values with formulas such as `fid * 45 + lpips * 26`. It hardcodes
  `approx_fid = 2.78` for epoch 2 and recomputes `Quality_Score`.
- Impact: this violates the ROBOTS.md Chart and Metrics Governance rule
  (metrics must match the physical checkpoint and `metrics.csv`) and the
  intent of the Publication-Grade FID Invariant. Charts and whitepapers that
  are generated after a run would publish non-measured numbers.
- Fix: remove the tool. Re-evaluate the historical checkpoints with real
  Inception-2048 FID, or mark legacy rows as `FID_legacy64` and exclude
  them from charts.

### D-04 High - 4-pillar completion uses post-audit resolution

- Evidence: `training/training/engine.py:435-440` reads
  `current_resolution` and `resolution` from the pre-audit state, but
  `get_state()` stores the key `input_size`
  (`training/governance/governor.py:1157`). The lookup therefore falls back
  to `ctx.governor.current_res`, which is the post-audit value after a
  possible spatial jump.
- Impact: a 100%-fraction epoch at 384 px that triggers a jump to the top
  rung (512 px) passes `ladder_passed`. The model can be declared
  "completed" without ever training at 512 px.
- Fix: read `pre_audit_gov_state["input_size"]` and share the key constants
  with `get_state`.

### D-05 High - Unauthenticated sidecar with wildcard CORS and credentials

- Evidence: `training/server/app.py:67` defaults to `enforce_auth=False`,
  and `get_app()` (`:152`) never enables it. `:121-127` sets
  `allow_origins=["*"]` together with `allow_credentials=True`. WebSocket
  routes are always open (`:139`). `:84` logs the first 8 characters of the
  token.
- Impact: any origin can trigger training, Kaggle pushes, notebook
  generation, and the planned SMTP bug-report.
- Fix: enforce the token in production. Restrict origins to the Tauri
  origins and Vite dev origins from config. Have the GUI send
  `X-LemTrain-Token`.

### D-06 Medium - Broken `lemtrain` console script

- Evidence: `pyproject.toml` declares `lemtrain = "cli:main"`, but neither
  `cli.py` nor `training/cli/lemtrain.py` defines `main`. `cli` is also not
  packaged: `packages.find` includes only `training*`, `models*`, `data*`,
  and `export*`. The tracked
  `lemgendary_training_suite.egg-info/entry_points.txt` carries the same
  broken target.
- Fix: `lemtrain = "training.cli.lemtrain:app"`. Untrack `*.egg-info`.

### D-07 Medium - Dead config keys and VRAM caps that are ignored

- Evidence: `config.yaml` defines
  `hardware.max_allowed_local_resolution` and
  `max_allowed_cloud_resolution`. The governor reads
  `hardware.max_allowed_resolution` (`training/governance/governor.py:143`),
  which does not exist in config. `governor.manifold.plateau_patience` is
  ignored; the code reads `governor.plateau_patience` (`:65`). Grep finds no
  consumer for `min_softmax_temp`, `safety_headroom_multiplier`,
  `vram_limit_gb`, `mask_tokens`, `stealth_mode`, or
  `inter_model_cooldown_seconds`. The VRAM tiers are hardcoded at
  `governor.py:132-141`.
- Fix: add a typed config schema (Pydantic) that rejects unknown keys, and
  move the VRAM tiers into config.

### D-08 Medium - GUI reports targets as achieved metrics

- Evidence: `training/server/routes/gui.py:406-408` sets
  `b_val = s["target"]` when no measurement exists. The entry is then
  reported as `passed`. Non-target metrics are always `passed: True`
  (`:436`). `parameters_m` falls back to a made-up `10.0` (`:549`).
- Fix: return `null` and `passed: false` when there is no measurement. Take
  parameter counts from the registry.

### D-09 Medium - Governor ignores the engine's SOTA state

- Evidence: `audit_epoch` ignores its `best_quality`, `regression_epochs`,
  `current_lr`, and `base_lr` arguments (ruff ARG002 at
  `governor.py:591-596`). It keeps its own `SotaTracker` while the engine
  keeps another (`training/core_loop.py:323`). The engine always passes
  `regression_epochs=0` (`engine.py:260`). `governor.py:961` overwrites
  `best_quality` on every fraction step.
- Fix: inject a single SOTA tracker. Remove the dead parameters.

### D-10 Medium - Advertised recovery features are never invoked

- Evidence: grep finds no production caller of
  `SmartTrainingGovernor.register_rollback` (`governor.py:1124`) or
  `veto_resolution_jump` (`:1097`). `trigger_mini_swa` is set
  (`governor.py:1021`) but never consumed. The papers describe all three as
  active, along with an `update_bn` rollback guard.
- Fix: implement them in the engine with tests, or remove them from code
  and papers.

### D-11 Medium - Secrets file not ignored

- Evidence: `training/config/secrets.py:15` maps `.GOOGLE_DRIVE_JSON`.
  `git check-ignore .GOOGLE_DRIVE_JSON` reports that the file is not
  ignored (`.gitignore` covers only `.GOOGLE_DRIVE`).
- Fix: add the entry, or use a single `.secrets/` directory that is
  ignored.

### D-12 Medium - Shell injection and a silent failure in a dead module

- Evidence: `training/prefetch_worker.py:52` runs
  `subprocess.run(cmd, shell=True)` with an interpolated `source_id`, then
  `except Exception: pass` (`:53`). The module has no callers.
- Fix: delete the module.

### D-13 Low - FID statistics can be inconsistent on error or resume

- Evidence: `validation.py:430-437` updates real features, then fake
  features. An exception between the two leaves the sets unbalanced. After
  an intra-validation resume, `fid_batches_accumulated` is restored
  (`:165`) but the Inception feature state is not, so FID covers only the
  batches after the resume.
- Fix: update a chunk pair atomically and drop the pair on failure. On
  resume, restart FID for that epoch.

### D-14 Low - Inconsistent preset precedence in quick-train

- Evidence: `training/server/routes/gui.py:604-630`. Epochs and batch size
  prefer the registry over the preset, but the learning rate prefers the
  preset over the registry.
- Fix: one documented precedence order: request, then registry, then
  preset, then defaults.

### D-15 Low - Magic quality anchor on resume

- Evidence: `training/core_loop.py:534-537` clamps `best_quality` to
  `555.0` for face and synthesis models.
- Fix: replace the constant with a config-driven, documented recalibration,
  or remove it.

### D-16 Low - Hardcoded dataset fallback ignores `paths.datasets_root`

- Evidence: `training/core_loop.py:366-367` uses
  `project_root.parent / "LemGendaryDatasets"` and the name
  `LemGendizedAIHelperLemGend`. `:330` correctly uses `config.paths`.

## 5. Training Invariants Compliance

### 5.1 Monotonic Ratchet Invariant - Partially compliant (ruling needed)

- Compliant: within a resolution rung the fraction only increases. Examples
  are `governor.py:941-957` and `curriculum.py:50-53`
  (`max(current, new)`). The retreat branch holds fraction and resolution
  and cools the learning rate (`governor.py:842-866`). Rollback escalation
  only promotes resolution (`training/governance/sota.py:90-105`). Resume
  takes the maximum fraction from `metrics.csv` (`core_loop.py:554-561`).
- Non-compliant under a literal reading: every spatial jump resets the
  fraction downward. `governor.py:981` resets to `initial_fraction`,
  `:674` to `0.5` on a forced jump, and `:1138` to a hardcoded `0.15` on a
  breakout. ROBOTS.md says fractions must "progress monotonically" and must
  "never step sample fraction ... backwards". The paper presents the reset
  as intended ("dataset fraction starts afresh from 15%",
  `PAPER_TRAINING_SUITE.md:99`). A lexicographic (resolution, fraction)
  ratchet is compliant; a literal fraction ratchet is not. The owner must
  decide which applies.
- Latent violation: `veto_resolution_jump` (`governor.py:1097-1110`)
  assigns `current_res = fallback_res` with no check that
  `fallback_res >= current_res`. It is currently dead code.
- Initialization can lower the resolution:
  `governor.py:150-165` clamps the configured `input_size` down to the
  hardware cap and to `res_ladder[0]`. This is acceptable before training
  starts, but `load_state` (`:1212-1213`) then restores an unclamped
  checkpoint resolution with no hardware re-cap.
- Documentation conflict: both papers describe "Intra-Resolution Data
  Recoil" that steps the fraction back (75% to 55%). See
  `PAPER_TRAINING_SUITE.md:135` and
  `PAPER_TRAINING_PATHOLOGY.md:313`. This contradicts ROBOTS.md, and the
  code no longer does it, so the papers must change.
- Tests: no test asserts the ratchet. Grep for `ratchet`, `monoton`, and
  `retreat` under `tests/` finds nothing.

### 5.2 Publication-Grade FID Invariant - Compliant (with caveats)

- `training/training/validation.py:412-416` uses `feature=2048`, and
  `:429-434` uses `chunk_size = 2` for both the real and fake updates.
- Caveats: D-13 (consistency), D-03 (fabricated legacy values), and no test
  that pins `feature=2048` or the chunk size.

### 5.3 Pareto Model Soup / SWA - Non-compliant

- The library function exists and has unit tests
  (`training/export/soup.py:19-115`, `tests/unit/test_soup.py`).
- Both user-facing paths, the CLI command and `POST /api/models/{key}/soup`,
  are broken (D-02).
- There is no SWA implementation. Grep finds no `AveragedModel`,
  `swa_utils`, or `update_bn`. The Mini-SWA trigger is never consumed
  (D-10), and MS-SWA (`PAPER_TRAINING_SUITE.md:592`) is not implemented.
- `soup.py:103` copies the first donor's metadata, including
  `governor_state`, and stores unnormalized weights in `soup_weights`. BN
  statistics are not re-estimated after blending.

## 6. Documentation Findings

- Version drift: `pyproject.toml`, `training/server/app.py`, and
  `training/cli/lemtrain.py:91` say `16.2.9`. `README.md:14` says
  `v16.9.61-STABLE`. `../lemgendary-docs/canonical_capabilities.json` says
  `v16.9.60-STABLE`. `manifest/documents.json` lists the training paper at
  `v16.9.44` and the pathology paper at `v16.9.27`. `config.yaml:7` says
  `16.0.0-nuclear`.
- `openapi.json` is stale. It lists 19 operations. The code exposes 39 HTTP
  routes plus 2 WebSockets. Missing from the spec are all of
  `/api/training/kaggle/*`, `/api/training/llm/*`,
  `/api/training/unit-test-*`, `/api/notebooks/*`, and
  `/api/models/{key}/{docs,export,variations,soup}`. Regenerate it with
  `lemtrain server openapi`.
- Suppressed linting in docs violates ROBOTS.md:
  `README.md:1` contains `<!-- markdownlint-disable MD013 -->` and
  `PAPER_TRAINING_SUITE.md:1` disables `MD051 MD013`. With inline config
  ignored, markdownlint-cli2 reports 4 broken TOC anchors (MD051 at lines
  32, 33, 34, and 38), 255 MD013 warnings in the paper, and 318 in the
  README. `PAPER_TRAINING_PATHOLOGY.md` has 175 MD013 warnings, and
  `models.md` has 48.
- Paper versus code contradictions:
  - The 4-pillar formula says `Epochs >= 30` (`PAPER_TRAINING_SUITE.md:108`)
    but Pillar 4 says 20 (`:121`). The code uses 20 (`engine.py:481`,
    `gui.py:497`). The HTML has both values.
  - The paper gives `maturity_soak = 3` (`:96`); `config.yaml` sets 5.
  - "Dynamic Validation Sharding 30%" (`:82`): no implementation was found
    (grep for `high_fidelity_fraction` finds nothing).
  - Hardware-aware resolution capping via
    `max_allowed_local_resolution` (`:77`) is not consumed (D-07).
  - The loop-breaker, Mini-SWA, `update_bn`, and MS-SWA are described as
    active (`:133-134`, `:204`, `:592`) but are dead or missing (D-10).
  - Intra-resolution fraction recoil (`:135`; pathology paper `:313`)
    contradicts ROBOTS.md (5.1).
- MD/HTML synchronization: `papers/training-suite-master.html` has 47
  h2-h4 headings, while the MD file has 49. The HTML headings drop the
  `(v16.9.25)`, `(v16.9.26)`, and `(v16.9.28)` suffixes, so the pair is not
  word-for-word. The pathology heading counts match (52 each).
- `canonical_capabilities.json` advertises an API route
  `POST /v16/train/start` for 22 models. That route does not exist; the
  real route is `POST /api/training/train`. It also advertises CLI commands
  `python -m training.cli.lemtrain --model X`, which is invalid because
  `train` takes a positional `MODEL`. 14 model `category` values differ
  from the registry (for example `quality_assessment` versus `enhance`).
- `models.md` omits `ai_helper_lemgend`, which is in the registry.
- The pathology paper's Group D table lists `diffusion_sdxl` and
  `diffusion_flux`, which are not in `unified_models_v2.yaml`.
- Whitepaper rigor: Section 1 is titled "1. Abstract" in both papers, as
  required. Neither paper states the ROBOTS.md Scalar Quality Score
  (`PSNR + 20*SSIM - 20*LPIPS`). The code uses a different weighted score
  (`training/telemetry.py:162-191`, `governor.py:225-249`), and
  `recalibrate_fid.py` uses a third formula. This needs a decision on the
  canonical formula.

## 7. Cross-Project Findings

### 7.1 SSOT and Inventory Duplication

Model inventory: `unified_models_v2.yaml` has 23 models, and its declared
`models_count` is 23.

- `../lemgendary-ai-studio-gui/src/components/training/fallbackModels.ts`
  has 23 keys, all matching. It is still a parallel copy of 603 lines.
- `training/server/routes/gui.py:68` `KNOWN_PARAMS_M` has 22 keys. It is
  missing `ai_helper_lemgend`.
- `tools/train_all.py:41` `PHASES` is missing `forex_predictor`. It
  contains 6 keys that are not registered: `diffusion_flux`,
  `diffusion_sdxl`, `retinaface_resnet`, `test_llm`, `vlm_blip2`, and
  `vlm_llava`.
- `models.md` has 22 keys. It is missing `ai_helper_lemgend`.
- `../lemgendary-docs/canonical_capabilities.json` has 23 keys, with field
  drift (see 6).

Other duplicates:

- Default ladders: `[256, 384, 512]` and `[1, 5, 15, 60, 240, 1440]` appear
  in `engine.py:450-455` and `gui.py:178-187`.
- Metric directions: `governance/metrics.py:41` and `gui.py:96`.
- Quality-score formulas: `telemetry.py:173`, `governor.py:225`,
  `governance/metrics.py:134`, and `validation.py:690`.
- Forex detection is copied at least 5 times with slightly different
  predicates: `governor.py:84-89`, `engine.py:153-157`, `engine.py:220`,
  `engine.py:429`, and `gui.py:149`.
- The `LemGendaryModels` path is joined directly in more than 15 places, for
  example `checkpoint/recovery.py:41-47`,
  `checkpoint/preflight_generator.py:78`, `routes/models.py:102`,
  `export/validation.py:268`, and `cloud/git_hub.py:38-40`.
- Requirements: `requirements.txt` differs from
  `../lemgendary-env-manager/requirements/requirements-training.txt`
  (`fastapi`, `diffusers`, `gdown`, `kagglehub`, `litdata`, and
  `MetaTrader5` pins). `pyproject.toml` dependencies are a third,
  incomplete list. For example, it lacks `torchmetrics`.

### 7.2 GUI Integration

GUI calls on port 8200, from
`../lemgendary-ai-studio-gui/src/api/client.ts`:

- `gui/models/with-stats` and `gui/quick-train`
- `jobs` and `jobs/{id}/cancel`
- `models/{k}/docs/generate`, `models/{k}/export/stage`, `models/{k}/soup`,
  and `models/{k}/variations/validate`
- `notebooks/generate` and `notebooks/generate-all`
- `training/evaluate` and `training/export`
- `training/kaggle/{status,kernels,train,monitor,pull,push,push-model,`
  `push-all}`
- `training/llm/train` and `training/unit-test-all`

All of these exist in the route decorators. Sixteen are missing from
`openapi.json`. `models/{k}/soup` fails at runtime (D-02).

- `../lemgendary-ai-studio-gui/src/components/TestModal.tsx:49` hardcodes
  `http://127.0.0.1:8200/api/training/unit-test-run` instead of using
  `TRAINING_BASE`.
- The GUI never sends `X-LemTrain-Token`. Enabling authentication (D-05)
  requires a GUI change in the same release.

### 7.3 Absolute and Platform Paths

- `training/cloud/gdrive.py:49-50` contains two hardcoded `G:` drive-letter
  paths ("My Drive" and "Shared drives").
- The tracked file `.cache/yolo_configs/yolo_yolov8n_config.yaml:83`
  contains a `path:` value that is an absolute `C:` drive path to a
  developer machine's dataset directory.
- Cloud runtime roots (`/kaggle/working`, `/content`) are hardcoded in
  `training/data/manifold.py:60-66`, `checkpoint/recovery.py:68-73`,
  `cloud/git_hub.py:36-38`, `data/dataset.py:53,347,366`, and
  `data/forex_dataset.py:241,249`. These are platform mount points rather
  than developer paths, but they should come from one environment-profile
  registry.

### 7.4 Suppressions

- `training/core_loop.py:426` contains `# type: ignore[arg-type]`.
- `training/cloud/metadata_generator.py:384` contains
  `# pragma: no cover` on a dummy function.
- `.pylintrc` disables 47 checks, including `bare-except`,
  `broad-exception-caught`, and `used-before-assignment`.
- `training/utils/interrupt.py:84-88` installs an excepthook that hides
  `BrokenPipeError` and `ConnectionResetError` traces.
- The generated notebook cells contain bare `except:`
  (`training/notebooks/cells/model.py:114,218,311`).

### 7.5 Subprocess to In-Repo Python Scripts

ROBOTS.md SSOT governance forbids these:

- `training/cli/lemtrain.py:196` (`test-all` runs `tools/train_all.py`)
- `training/server/jobs.py:725` (API `unit-test-all` runs the same script)
- `tools/train_all.py:186` (runs `training/train.py` per model)
- `training/train_forex_curriculum.py:254`
- `cloud_jobs/*/train_cloud.py:7` (runs `training.core_loop` inside the
  Kaggle kernel; acceptable as a remote entry point)

## 8. Bug-Report Backend Integration Notes

These notes cover sections 5-6 of `../bug_report_implementation_plan.md`.

How the codebase is structured today:

- Routes are thin `APIRouter` modules mounted with prefix `/api`
  (`training/server/app.py:131-140`). The GUI router has prefix `/gui`, so
  the effective path is **`POST /api/gui/bug-report`**, not
  `/gui/bug-report`. The plan's frontend call `fetch("/gui/bug-report")`
  (plan section 3.8) is relative and would hit the Tauri or Vite origin. It
  must use `${TRAINING_BASE}/api/gui/bug-report`.
- Services are plain classes with `__init__(project_root: Path | None)`,
  for example `AuditService` (`training/services/audit_service.py:25-29`).
  Each is exported from `training/services/__init__.py`, and routes create
  them from `request.app.state.server_state.project_root`.
- No route accepts uploads today: grep for `UploadFile`, `File(`, and
  `Form(` finds none. `python-multipart` is **not** in `requirements.txt`,
  `pyproject.toml`, or `uv.lock`, and FastAPI refuses to register `Form` or
  `File` parameters without it. The plan's claim that no new Python
  packages are needed (plan section 12.2) is incorrect. Add
  `python-multipart` to this repo's `requirements.txt`, to `pyproject.toml`,
  and to `../lemgendary-env-manager/requirements/requirements-training.txt`
  together.
- Secrets: use `training/config/secrets.py`. Add `BUG_REPORT_SMTP_USER` and
  `BUG_REPORT_SMTP_PASS` to `_SECRET_MAPPING` (`:10-18`) with dotfiles
  such as `.BUG_REPORT_SMTP`, add those dotfiles to `.gitignore`, and read
  them via `get_secret()`. Do not use raw `os.environ`. Fail with HTTP 503
  and a clear `detail` when the secrets are missing; do not let
  `SMTPAuthenticationError` surface as a 500.
- Config (SSOT): move the recipient address, SMTP host and port, maximum
  attachment size, and maximum screenshot count into a `bug_report:`
  section of `config.yaml`. Do not use module constants (plan section 6).
- CORS and auth: with the current `allow_origins=["*"]`, credentials
  enabled, and no auth, any web page could send e-mail through the user's
  account. Ship the route only behind token auth or an origin allow-list
  (D-05).

Idiomatic shape of the integration:

- A thin route (`async def`) validates input with Pydantic. Use `EmailStr`
  (requires `email-validator`) or a regex. Use a `Literal` or enum for
  `module`, built from the GUI module registry. Use a `bool` form field
  rather than `"true"` string parsing.
- Accept screenshots as `list[UploadFile]` with a maximum length of 5,
  instead of five separate parameters.
- Enforce size and type limits before reading: `image/webp`, `video/webm`,
  and `text/csv`, with a total size under Gmail's 25 MB limit. Stream
  uploads to a `SpooledTemporaryFile` rather than holding everything in
  memory.
- Write `report.csv` with `csv.writer`. The plan's hand-built CSV escapes
  only the `description` field, so `email` and `module` can break the CSV.
- Use `datetime.now(timezone.utc)`. `datetime.utcnow()` emits a
  `DeprecationWarning` on Python 3.12, and ROBOTS.md does not permit
  warnings.
- `smtplib` is blocking. Run `send` through `run_in_threadpool` or
  `loop.run_in_executor`, or define the route as a sync `def`, so it does
  not block the event loop that serves log WebSockets.
- Inject an `SmtpTransport` protocol into `BugReportService` (DIP). Unit
  tests can then use a fake transport through the existing `TestClient`
  pattern (`tests/unit/test_server.py:12-23`).
- Consider placing the route in a new `routes/bug_report.py` that keeps
  the `/gui` prefix. `gui.py` is already 653 lines with CC 163.
- If the CLI must match the API (ROBOTS.md SSOT), add
  `lemtrain bug-report` on top of the same service.
- Regenerate `openapi.json`. Update `canonical_capabilities.*`,
  `manifest/documents.json`, and the README changelog in the same change.

## 9. Proposed Changelog Entries

These entries describe proposed changes. They are ready to merge only when
the matching fixes land.

```markdown
### v16.9.62 - Review Remediation (proposed)

- Fixed unbounded epoch loop; `total_epochs` is a hard cap unless
  `optimization.open_ended` is set; governor early-stop honoured.
- Fixed 4-pillar completion check reading post-audit resolution.
- Fixed Pareto Model Soup CLI/API (donor discovery, hub root, missing
  `status` import).
- Removed `tools/recalibrate_fid.py` synthetic FID rewriting.
- Secured sidecar: token auth enforced, CORS restricted to GUI origins.
- Fixed `lemtrain` console-script target.
- Wired `hardware.max_allowed_*_resolution` into the governor.
- Regenerated `openapi.json` (39 HTTP routes, 2 WebSockets).
- Removed markdownlint inline disables; fixed 4 broken TOC anchors.
```

## 10. Remaining Recommendations (Ranked)

1. Fix D-01, D-02, D-04, and D-05 together, with tests (High).
2. Retire `tools/recalibrate_fid.py` and audit the published charts for
   recalibrated rows (D-03).
3. Get an owner ruling on the fraction reset at a spatial jump. Encode the
   decision in a `RatchetGuard` with property tests, and fix both papers
   (5.1).
4. Implement SWA (`torch.optim.swa_utils`) and a vault-aware soup service
   with BN re-estimation, or retract the claims (5.3).
5. Consolidate the registry and metrics SSOT (7.1). Generate `models.md`,
   the GUI fallback data, and the canonical capabilities from
   `unified_models_v2.yaml`.
6. Add a typed config schema that fails on unknown or dead keys (D-07).
7. Remove the `.pylintrc` suppressions and the 249 blind or silent excepts,
   starting with `server/jobs.py` and `validation.py`.
8. Decompose the grade-F functions (2.1, 3).
9. Replace the in-repo subprocess fan-out with service calls (7.5).
10. Untrack `.cache/`, `*.egg-info`, and `.vscode/`, and gitignore
    `.GOOGLE_DRIVE_JSON` (D-11).

## Appendix: Tools Run

All tools ran in an isolated scratch venv. Nothing was installed into the
repository or globally. Torch was not installed, so findings that depend on
torch execution are static.

- `ruff 0.17.3 check --isolated` (F, E9, B, BLE, S110): 418
  findings. These include 1 `F821` (`routes/models.py:186`), 210 BLE001,
  39 S110, 9 F841, and 2 B023 (`training/training/epoch.py:214,218`).
- `radon cc`: 18 functions at grade F. The top five are listed in 1.2.
- `pytest` on `tests/unit/test_secrets.py` and `tests/unit/test_utils.py`
  (torch-free): 17 passed. The other suites need torch and were not run.
- `markdownlint-cli2 0.18.1`, run with both default config and
  `noInlineConfig`. Results are in section 6.
- A custom PyYAML script compared the inventories across the registry, the
  GUI, `gui.py`, `train_all.py`, `models.md`, and the docs capabilities
  (7.1).
- Route decorators were compared with `openapi.json` and with the GUI
  `client.ts` URLs (6, 7.2).
- `git check-ignore` and `git ls-files` were used for tracked artifacts.
  These commands are read-only.
