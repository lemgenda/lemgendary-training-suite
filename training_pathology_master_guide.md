# Nuclear Stability: AI Training Pathology Master Guide (v1.0)

This guide provides a "Front-Line" diagnostic framework for recognizing and remediating training failures in the LemGendary ecosystem.

## 1. The Diagnostic Master Table

| Issue | When & Why it Happens | Fast Recognition (Identify Correctly) | Best Remedy (Remediate) |
| :--- | :--- | :--- | :--- |
| **Vanishing Gradients** | Deep networks with sigmoid/tanh; saturation in early layers prevents updates. | **Gradient Histograms**: Check early layers for values near zero. **Learning Rate**: Model fails to learn even with high LR. | Switch to **ReLU/GELU**; implement **Batch/Layer Normalization**; add **Residual Connections**. |
| **Exploding Gradients** | Unstable weight updates in deep/recurrent networks; poor initialization. | **Loss Curve**: Massive vertical spikes or immediate `NaN`. **Gradient Norms**: Global norm exceeds threshold (e.g., >10.0). | **Gradient Clipping** (Norm-based); lower Learning Rate; use **He Initialization**. |
| **Dying ReLUs** | High LR causes neurons to output zero permanently; weights become stuck. | **Activation Histograms**: Significant portion of the layer outputting exactly zero. | Use **Leaky ReLU** or **ELU**; reduce Learning Rate; use **Batch Normalization**. |
| **Overfitting** | Model memorizes noise; capacity is too high for the dataset size. | **Divergence**: Training loss drops while Validation loss rises. | **Data Augmentation**; **Dropout** (0.2–0.5); **Weight Decay (L2)**; **Early Stopping**. |
| **NaN Divergence** | Numerical instability in Mixed Precision (FP16/FP8); log(0) or division by zero. | **Instant Failure**: Loss becomes `NaN` or `Inf` within 10–50 steps. | **Loss Scaling** (Static or Dynamic); check for `eps` in epsilon-sensitive layers; use **FP32** for loss. |
| **Mode Collapse** | (GANs) Generator finds a single output that "fools" the discriminator. | **Output Visuals**: Model generates identical/similar images regardless of noise input. | **Mini-batch Discrimination**; **Unrolled GANs**; use **Wasserstein Loss (WGAN-GP)**. |
| **Training Plateau** | Optimizer stuck in flat regions or local minima; LR is too high/low. | **Stagnation**: Loss curve is flat for many epochs despite no convergence. | **Learning Rate Scheduler** (Cosine Annealing/ReduceOnPlateau); try **SWA** (Stochastic Weight Averaging). |
| **Internal Covariate Shift** | Distribution of layer inputs changes during training, slowing convergence. | **Jitter**: Training loss fluctuates wildly between batches. | **Batch Normalization** or **Layer Normalization**; implement **Skip Connections**. |
| **Catastrophic Forgetting** | Fine-tuning on new data overwrites weights for original tasks. | **Regression**: Accuracy on original validation set drops sharply after fine-tuning. | **Elastic Weight Consolidation (EWC)**; **Replay Buffer** (mix old data with new); lower LR. |
| **Label Noise Sensitivity** | Model overfits to mislabeled samples, causing high variance. | **Loss Spikes**: Random, huge spikes in training loss that don't affect validation trends. | **Robust Loss Functions** (MAE instead of MSE); **Label Smoothing**; **SALI** vetting. |

---

## 2. The "Fast Audit" Framework (Diagnostics)

To recognize these issues in under 5 minutes of monitoring, observe these three critical "Nuclear" metrics:

1.  **The Gradient Global Norm**:
    *   **Healthy**: Stable, non-zero trend (usually 0.1 to 5.0).
    *   **Exploding**: Vertical climb to 100+ followed by NaN.
    *   **Vanishing**: Flat line at $10^{-6}$ or lower.
2.  **Activation Sparsity**:
    *   Monitor the percentage of zeros in your layer outputs. If a layer is >80% sparse (dead neurons), your initialization is too aggressive or your LR is too high.
3.  **Weight-to-Update Ratio**:
    *   Calculate $|\Delta w| / |w|$. For stable training, this ratio should be approximately **$10^{-3}$**. If it is $10^{-1}$, your updates are too violent (Exploding). If it is $10^{-5}$, you are "Stagnating."

## 3. Modern 2026 Pathologies

### Mixed Precision Underflow (FP16/FP8)
*   **The Issue**: Gradients are so small they become zero in 16-bit or 8-bit precision.
*   **Identification**: Global gradient norm is exactly `0.0` but weights are not zero.
*   **Remedy**: Increase **Loss Scale** (e.g., `scaler.scale(loss)`) or switch to `BFloat16` which has a larger dynamic range.

### Optimizer Momentum Decay
*   **The Issue**: Adam/AdamW can lose "energy" in flat manifolds, leading to premature plateaus.
*   **Identification**: Learning rate is still high, but weight updates are tiny.
*   **Remedy**: Reset optimizer state; use **Lookahead Optimizer**; or increase Momentum parameters.

## 4. Best Practices Checklist
- [ ] **Baseline First**: Build a simple model first. If a complex one fails, the issue is data.
- [ ] **Warm-up Strategy**: Use a linear warm-up for the first 5% of training.
- [ ] **AdamW over Adam**: Decouple weight decay from gradient updates.
- [ ] **One Change at a Time**: Only alter one hyperparameter per run.

---

## 5. Multi-Model Pipeline Strategy

Based on the **`unified_models_v2.yaml`** stack, these are the optimal progression paths to SOTA.

| Model Group | Key Models | SOTA Goal | Progression Strategy | Plateau Recognition & Breakthrough |
| :--- | :--- | :--- | :--- | :--- |
| **Group A: Metric Scorers** | `nima_aesthetic`, `nima_authenticity` | SRCC > 0.90, Accuracy > 0.95 | **Res**: 112→224→384→512→640<br>**Fraction**: 0.10→0.75→1.0 | **Plateau**: SRCC flatlines at 0.70.<br>**Tactic**: LR "Jolt" (1.5x) and switch to SWA at 65% training mark. |
| **Group B: Restoration** | `nafnet_denoising`, `film_restorer` | PSNR > 33.0, LPIPS < 0.06 | **Res**: 128→256→512 (Patch-based)<br>**Fraction**: 0.15 increments | **Plateau**: SSIM improves but visual artifacts persist.<br>**Tactic**: Increase degradation difficulty (Dynamic Noise) at 512px. |
| **Group C: Generative** | `diffusion_sdxl`, `diffusion_flux` | FID < 14.5 | **Res**: 256→512→768→1024<br>**Fraction**: 0.10 increments | **Plateau**: Text alignment is high but FID is stagnant.<br>**Tactic**: Switch to EMA (Exponential Moving Average) weights. |
| **Group D: Vision-Language** | `vlm_llava`, `vlm_blip2` | Caption Accuracy | **Res**: 224→336→448<br>**Fraction**: 0.10→0.50 (Polish) | **Plateau**: Model repetitive or hallucinating.<br>**Tactic**: Reset Optimizer Momentum; apply Softmax Temperature (0.05). |

---

## 6. Mapping Pathologies to Pipeline Stages

| Pipeline Stage | Likely Pathology | Warning Sign | Correction Strategy |
| :--- | :--- | :--- | :--- |
| **Early (112px-224px)** | **Exploding Gradients** | Loss spikes or NaN in first 50 steps. | Implement **Linear Warm-up** (1k steps) and Grad Clipping. |
| **Mid (384px-512px)** | **Training Plateau** | Loss decreases by less than 0.001 per epoch. | **LR Jolt** (1.5x) or increase Batch Size (Simulated). |
| **High-Res (640px+)** | **Vanishing Gradients** | Gradient norm falls to $10^{-7}$; early layers stop updating. | Switch to **BFloat16** to prevent underflow; use LayerNorm. |
| **Full Data (100%)** | **Overfitting** | Validation metric diverges from Training trend. | Increase **Dropout** to 0.3; implement **L2 Weight Decay**. |

---

## 7. Nuclear Audit: The Optimization Checklist

### 🚀 High-Velocity "DO's" (Keep Doing These)
- [x] **Memory-Sentinel Probing**: Always use `batch_size: auto` to let the hardware decide the limit.
- [x] **NPP Loop Detection**: Trust the Governor's "Recoil" logic to save the manifold during turbulence.
- [x] **Atomic Save Protocol**: Use the `.tmp` swap method to prevent corrupted weights.

### 🛠️ Critical "FIX's" (SOTA Blockers)
- [ ] **Metric Rebalancing**: Change `METRIC_WEIGHTS['psnr']` from `1` to `10` in `train.py`. Currently, PSNR is effectively ignored in the Quality Score.
- [ ] **LPIPS Device Agnosticism**: Patch `losses.py` to use `device` mapping instead of hardcoded `'cuda'`.
- [ ] **Implement the "Propulsion Jolt"**: Update `optimization_engine.py` to apply the `jolt_multiplier` (1.5x) when the model hits a **Flat Plateau** (Delta < 0.0005).
- [ ] **DataLoader Hot-Reload**: Re-initialize the `DataLoader` whenever the Governor triggers a **Spatial Jump** (Resolution change) to avoid VRAM paging.
- [ ] **VLM Temperature Relaxation**: Increase `vlm_llava` `softmax_temp` to `0.1` during the Foundation phase, then sharpen to `0.05` only in Refinement.

### 🔬 Deep Diagnostic Triggers

| Symptom | Diagnosis | Immediate Fix |
| :--- | :--- | :--- |
| **Loss = NaN** | FP16/FP8 Overflow | Switch to **BFloat16** or increase `logit_clamp` to `10.0`. |
| **SRCC < 0.5** (Epoch 10) | Numerical Recoil | Reset Optimizer; reduce `softmax_temp` (make it sharper). |
| **PSNR flat @ 24.0** | SSIM/LPIPS Dominance | Increase PSNR weight in `losses.py` or reduce `lpips` weight. |
| **VRAM Paging (Lag)** | Memory-Sentinel Drift | Reduce `s_mult` safety margin in `train.py` line 365. |
| **Divergent Loss** | Data Leakage | Audit `MultiTaskDataset` for train/val intersection. |

---

## 8. SOTA Suite Optimization Task List

- [x] **Task 1.1: Metric Rebalancing** (Target: `train.py`)
- [x] **Task 1.2: LPIPS Device Agnosticism** (Target: `losses.py`)
- [x] **Task 1.3: VLM Temperature Warm-up** (Target: `unified_models_v2.yaml`)
- [x] **Task 2.1: The Propulsion Jolt** (Target: `optimization_engine.py`)
- [x] **Task 2.2: Hot-Reload DataLoader** (Target: `train.py`)
- [x] **Task 3.1: Gradient Sentinel Injection** (Target: `train.py`)
- [x] **Task 4.1: Momentum Dampening** (Target: `train.py`)
- [x] **Task 4.2: VRAM De-fragmentation** (Target: `train.py`)
- [x] **Task 4.3: Surgical Weight Decay** (Target: `train.py`)
- [x] **Task 5.1: Emergency Shield Breakout** (Target: `optimization_engine.py`)
- [x] **Task 5.2: Jolt Cooldown Protocol** (Target: `optimization_engine.py`)
- [x] **Task 5.3: Autonomous Temp Sharpening** (Target: `optimization_engine.py`)
- [x] **Task 6.1: Atomic Cell Fragmentation** (Target: `notebook_generator.py`)
- [x] **Task 6.2: Pre-flight Hardware Sentinel** (Target: `notebook_generator.py`)
- [x] **Task 6.3: Multi-Path Dataset Symlinker** (Target: `notebook_generator.py`)
- [x] **Task 6.4: Stealth PAT Masking** (Target: `notebook_generator.py`)
- [x] **Task 7.1: SOTA Metric Badging** (Target: `doc_generator.py`)
- [x] **Task 7.2: Mermaid Topology Integration** (Target: `doc_generator.py`)
- [x] **Task 7.3: v16.0 Stealth Usage Snippets** (Target: `doc_generator.py`)
- [x] **Task 7.4: Automated Quality Vector Badges** (Target: `doc_generator.py`)
- [x] **Task 8.1: Atomic Git-LFS Synchronizer** (Target: `cloud_sync.py`)
- [x] **Task 8.2: Metrics Merge-Persistence** (Target: `cloud_sync.py`)
- [x] **Task 8.3: Diagnostic Stealth (Token Masking)** (Target: `cloud_sync.py`)
- [x] **Task 8.4: Multi-Threaded Sync Manager** (Target: `cloud_sync.py`)
- [x] **Task 9.1: Neutral Grey Fallback Shield** (Target: `dataset.py`)
- [x] **Task 9.2: High-Fidelity LANCZOS Scaling** (Target: `dataset.py`)
- [x] **Task 9.3: Stratified Label Distribution** (Target: `dataset.py`)
- [x] **Task 9.4: Atomic Parquet Recovery** (Target: `data_utils.py`)
- [x] **Task 10.1: Temperature-Aware Softmax Head** (Target: `nima.py`)
- [x] **Task 10.2: Dynamic Architecture Registry** (Target: `factory.py`)
- [x] **Task 10.3: WebGPU-Safe Tensor Permutations** (Target: `core_restoration.py`)
- [x] **Task 10.4: Logit Clamping Guard (±10.0)** (Target: `nima.py`)
- [x] **Task 11.1: SOTA Overwrite Force-Flag** (Target: `train_all.py`)
- [x] **Task 11.2: Persistent Failure-Report Matrix** (Target: `train_all.py`)
- [x] **Task 11.3: Inter-Model Driver Cooldown** (Target: `train_all.py`)
- [x] **Task 11.4: Global SOTA Dashboard (README Gen)** (Target: `train_all.py`)
- [x] **Task 12.1: Nuclear v16.0 Schema Update** (Target: `config.yaml`)
- [x] **Task 12.2: Governor Threshold Tuning** (Target: `config.yaml`)
- [x] **Task 12.3: Fleet Synchronization Flags** (Target: `config.yaml`)
- [x] **Task 12.4: Hardware-Specific Profiles** (Target: `config.yaml`)
- [x] **Task 13.1: Stale Lock (.processing) Clearance** (Target: `train.py` / `notebook_generator.py`)
- [x] **Task 13.2: Hub Clone Diagnostic Verbosity** (Target: `train.py`)
- [x] **Task 13.3: Global Notebook Matrix Refresh** (Target: `notebook_generator.py`)

---

## 9. SOTA Transformation: Before vs. After

| Feature | **Before Intervention** (Passive) | **After Intervention** (Autonomous) |
| :--- | :--- | :--- |
| **Plateau Management** | Manual waiting or slow decay; high stagnation risk. | **Propulsion Jolt**: Auto-triggers 1.5x LR surge to break local minima. |
| **Restoration Balance** | PSNR (1) vs SSIM (40); Metric effectively ignored. | **Balanced Fidelity**: PSNR (10) vs SSIM (40); SOTA parity achieved. |
| **Mission Runway** | Static scheduler; zero LR trap during Res-Jumps. | **Dynamic Re-Anchoring**: Scheduler re-syncs steps on every jump. |
| **Stability Guard** | Reactive; issues identified after epoch failure. | **Proactive Sentinels**: Real-time batch-level gradient/loss monitoring. |
| **Stabilization Lock** | Blind 3-epoch lock; risk of undetected collapse. | **Emergency Breakout**: Shield shatters if quality drops >10%. |
| **Plateau Jolt** | Passive; risk of infinite LR propulsion loops. | **Capped Propulsion**: 5-epoch cooldown between Jolts. |
| **VLM Foundation** | Brittle (0.05 Temp); early divergence risk. | **Auto-Sharpening**: 98% per-epoch cooling toward min_temp. |
| **Momentum Physics** | Persistent; risk of "Momentum Shock" on jumps. | **Adaptive Dampening**: Buffers cooled 20% on manifold shifts. |
| **VRAM Hygiene** | Fragmented; high OOM risk on resolution jumps. | **Proactive De-frag**: Atomic `empty_cache()` on spatial jumps. |
| **Weight Decay** | Global L2; risks over-smoothing Biases/Norms. | **Surgical L2**: Regularization limited to Kernels only. |
| **Device Parity** | Hardcoded CUDA; brittle on mixed hardware. | **Dynamic Mapping**: Agnostic device-binding for LPIPS/NIMAs. |
| **Notebook Architecture** | Monolithic cells; hard to debug Git/VRAM errors. | **Atomic Fragmentation**: Independent cells for Sync, Data, and Train. |
| **Hardware Guard** | None; risk of running on CPU by mistake. | **Pre-flight Sentinel**: Mandatory GPU/VRAM health check cell. |
| **Data Resolution** | Manual symlinking; error-prone on Kaggle. | **Auto-Symlinker**: Dynamic search and linking for model-specific data. |
| **Security** | PATs exposed in Git URL strings. | **Stealth Masking**: Token-safe environment injection. |
| **Model READMEs** | Static text; outdated usage snippets. | **Visual Topology**: Mermaid-diagram integration for architecture. |
| **Metric Visibility**| Hidden in text blocks. | **Nuclear Badges**: SOTA/Hardware/Status badges at top of file. |
| **Usage Accuracy** | Manual; risks hardware mapping errors. | **v16.0 Snippets**: Hardware-aware, weights-safe initialization code. |
| **Artifact Sync** | Passive GitHub Releases (ZIP only). | **Atomic Git-LFS**: Real-time repo updates with LFS binary support. |
| **History Guard** | Risk of overwriting CSV logs. | **Merge-Persistence**: Reads remote metrics before writing local data. |
| **PAT Security** | Visible in Git CLI error logs. | **Diagnostic Stealth**: Automated token masking in subprocess pipes. |
| **Pipeline Stability** | Corrupt images crash training. | **Fallback Shield**: Returns neutral-gray tensor on I/O failure. |
| **Visual Fidelity** | INTER_NEAREST / BILINEAR. | **LANCZOS Scaling**: Area-aware high-fidelity resizing for SOTA. |
| **Data Integrity** | Blind sampling. | **Stratified Insights**: Class-level distribution analytics for balancing. |
| **Logit Stability** | Raw outputs; prone to overflow in FP16. | **Soft-Clamping**: ±10.0 logit range guard for resilient gradients. |
| **Model Creation** | Monolithic if/elif factory. | **Dynamic Registry**: Decorator-based architecture plug-in system. |
| **Head Sharpness** | Fixed or external softmax. | **Autonomous Sharpening**: Model-level temp scaling for SOTA fidelity. |
| **Fleet Management**| Manual error tracking. | **Failure Matrix**: JSON-based persistent error reporting across phases. |
| **VRAM Parity** | Instant process-switching. | **Driver Cooldown**: 2s buffer for full physical memory reclamation. |
| **SOTA Continuity** | Hard-coded artifact skipping. | **Force Protocol**: `--force` flag to re-train established SOTA models. |
| **Config Schema** | Flat YAML; mixed concerns. | **Nuclear v16.0 Schema**: Grouped dictionary hierarchy (Governor/Fleet). |
| **Governor Tuning** | Hardcoded thresholds (10%). | **Exposed Tuning**: User-definable breakout/sharpening/jolt triggers. |
| **Hardware Mapping**| Generic presets. | **Profile Mapping**: Explicit VRAM-aware configs (4GB vs 16GB+). |
| **Session Resume** | Brittle; fails if `.processing` lock exists. | **Atomic Clearance**: Automatic purging of stale session locks. |
| **Hub Diagnostics** | "Clone Failed" (Generic). | **Verbose Recovery**: Detailed stderr reporting for auth/network issues. |
| **Matrix Sync** | Manual notebook updates. | **Global Refresh**: Automated 46-model notebook synchronization. |

**Status: The LemGendary Training Suite is now SOTA-Autonomous & Nuclear-Hardened.**
