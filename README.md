# LemGendary AI: Model Training & Deployment Guide
Professional Edition - v6.5

This document is the definitive guide for the LemGendary AI infrastructure. It covers the end-to-end workflow from setting up your environment to exporting production-ready WebGPU models.

---

## 1. Quickstart: The LemGendary Hub
The easiest way to manage your AI workflow is via the **LemGendary Hub**.

### How to Run
1. Open PowerShell in the project root (`lemgendary-training-suite`).
2. Run the Hub script:
   ```powershell
   .\lemgendary_hub.ps1
   ```

### Menu Options
**1.** **Initialize/Fix Environment**: Installs Python 3.12 natively and creates the `.venv` enclosure. Upgrades pip, then installs PyTorch 2.4.1+cu121 alongside all other mathematical dependencies for inference.

**2.** **Train Individual Model**: Launches the interactive LemGendary Unified Training Suite, empowering singular network selection for localized structural training loops natively.
   * **Submenu Categories & Targets**:
     * **1. Quality Assessment**
       * `nima_aesthetic`
       * `nima_technical`
     * **2. Face & Detection**
       * `codeformer`
       * `parsenet`
       * `retinaface_mobilenet`
       * `retinaface_resnet`
       * `yolov8n`
     * **3. Super-Resolution**
       * `ultrazoom_x2`
       * `ultrazoom_x3`
       * `ultrazoom_x4`
       * `ultrazoom_x8`
     * **4. Image Restoration**
       * `ffanet_indoor`
       * `ffanet_outdoor`
       * `mprnet_deraining`
       * `mirnet_lowlight`
       * `mirnet_exposure`
       * `nafnet_debluring`
       * `nafnet_denoising`
     * **5. Universal Hybrid**
       * `upn_v2`
       * `professional_multitask_restoration`
       * `film_restorer`

**3.** **Global Orchestration**: Automates sequential continuous execution of **all 21 models** on your local hardware architecture uninterrupted.

**4.** **Deploy to Kaggle Cloud**: Outputs physical instructions on how to push the pre-generated Jupyter topologies to Kaggle's T4 instances instantly.

**5.** **Smart Cloud Orchestration**: Uses your `C:\` local GPU to mathematically train all 21 models natively by streaming specific datasets through Kaggle explicitly via the API, executing Python loops, and instantly **purging the cache** iteratively. This strictly locks local peak SSD consumption below `~220GB`.
   * **Execution Phases**:
     * *Phase 1:* Deep Quality Assessment
     * *Phase 2A:* High-Fidelity Facial Analytics
     * *Phase 2B:* Massive Universal Detection
     * *Phase 3A:* Super-Resolution Synthesis
     * *Phase 3B:* Degradation Removal Arrays
     * *Phase 3C:* Low-Light Recovery
     * *Phase 3D:* Denoising Networks
     * *Phase 3E:* Universal Cross-Domain Restoration

**6.** **Single-Epoch Unit Test**: Initiates a massive forced override across all 21 core neural networks, executing precisely **1 automated functional epoch per model** to strictly validate local memory arrays and buffer structural constraints flawlessly without committing to a 50-hour cycle.

**9.** **Run Environmental Janitor**: Force-purges all orphaned Python/PowerShell processes and environment locks to structurally reset the project's memory state.

**7.** **Exit**: Terminate hub.

---

## 🚀 Technical Whitepaper: The Architecture of LemGendary AI
For a deep dive into the mathematical foundations, visual taxonomy, and hardware-aware optimizations of this suite, please refer to our official technical whitepaper:

*   **[Technical Whitepaper (HTML Version)](https://lemgenda.github.io/lemgendary-training-suite/papers/nima-quality.html)** - *Recommended: Premium Viewing Experience (Dark Mode)*
*   **[Technical Whitepaper (Markdown Version)](technical_paper/PAPER_LEMGENDARY_NIMA.md)** - *Documentation Source*

This paper details the **LemGendized Universal Quality Subset**, the **2026 Resonance Loss** ($Loss_{EMD} + 0.3 \times (1.0 - PLCC)$), and our specialized **GTX 1650** VRAM management strategies.

---

## 2. File Organization
- `configs/` - Master structural neural definitions
- `training/` - Local PyTorch execution matrices
- `data/datasets/` - Local caches for Kaggle streaming layers
- `trained-models/` - Dynamic WebGPU ONNX/FP32 architecture output arrays.

---

## 3. Training & Heterogeneous Compute

### Hardware Profiles
- **Extreme Mode**: Saturates all Logical Cores (12 threads) and utilizes **Dual-GPUs**.
  - **Primary (NVIDIA)**: Handles the heavy model training (Backpropagation).
  - **Auxiliary (AMD/Intel)**: Offloads the real-time image synthesis (Blur, Noise, Haze).

### Universal SOTA-Priority Engine
To ensure State-of-the-Art (SOTA) results without wasting compute:
1. **Metric-Driven Saving**: The engine now prioritizes specialized SOTA metrics (PSNR, PLCC, FID) over raw validation loss. `best.pth` is only updated when a new mathematical quality benchmark is achieved.
2. **Config-Driven Targets**: All success criteria (e.g., `PLCC > 0.95`, `PSNR > 32.5`) are modularized in `unified_models.yaml`, allowing architecture-specific termination logic.
3. **Reinforcement Countdown**: The suite automatically enforces exactly **1 final reinforcement epoch** instantly upon breaching all configured SOTA targets to stabilize the weights before ONNX compilation.
4. **Max 50 Epochs**: Absolute safety hard-cap.

---

## 4. Exporting to Production
All exports are consolidated in **`trained-models/`**.

### Standardized Formats:
- **`ModelName.onnx`** (FP16): Production-ready for WebGPU. Dynamically automatically routed directly into your Next.js `public/models/restoration/` web directory for instant UI rendering.
- **`ModelName_FP32.onnx`** (FP32): Reference model for precision debugging. Weights are stored in an external `.onnx.data` sidecar.

## 5. Kaggle Cloud Integration
The LemGendary Training Suite natively supports 100% cloud-based dataset training via Kaggle without downloading the 200GB topological arrays locally. The mathematical engine is perfectly tuned to support `--env kaggle` dataset interceptions.

### How to Launch a Cloud Training Node:
1. Navigate to your Kaggle profile and access the dedicated LemGendary Jupyter Notebooks. The deployment topologies are hosted directly on Kaggle to prevent localized repository bloat:
   - **Solo Notebooks**: Topologies reliant solely upon exactly one dataset environment.
   - **Multi-Mount Notebooks**: Cross-domain models demanding multiple concurrent dataset arrays.
2. Select your desired Notebook and click **Copy & Edit** to clone the environment into your active workspace.
3. Click **Add Data** on the right-hand interface. Search for and mount **exactly** the LemGendized dataset(s) explicitly requested via comments at the top of the code cell.
4. In **Session Options**, mathematically assign the **Accelerator** natively to **GPU T4 x2** or **GPU P100**. *Do not use CPU or TPU.*
5. Click **Run All**. The notebook will securely clone this repository, align pathing natively against `/kaggle/input/`, and begin executing the epochs directly into your browser!

---

## 6. Processing Pipeline
- **Diagnostic Eyes**: YOLOv8n (Detection, Pose, Classification), NIMA Aesthetic Scorer, NIMA Technical Scorer.
- **Core Hands**: Professional Multi-Task Restorer, UPN v2 (AI Auto-Fix Parameter Predictor), Universal Film Restorer.
- **Surgical Tools**: CodeFormer (Face), ParseNet (Face Parsing), NAFNet (Denoise/Deblur), MPRNet (Deraining), MIRNet v2 (Low-Light/Exposure), FFANet (Dehaze Indoor/Outdoor), RetinaFace (MobileNet/ResNet).
- **UltraZoom**: UltraZoom x2, UltraZoom x3, UltraZoom x4, UltraZoom x8.

---

## 7. Neural Convergence Benchmarks (SOTA Targets)
| Category / Task | Key Metrics | Acceptable | Excellent | State-of-the-Art (SOTA) |
| :--- | :--- | :--- | :--- | :--- |
| **Restoration** (SuperRes, Noise, LowLight, Degradation) | PSNR / SSIM / LPIPS | ~28.0 dB / ~0.82 / ~0.15 | >30.5 dB / >0.89 / <0.10 | **>32.5 dB / >0.94 / <0.06** |
| **Face Restoration** | PSNR / SSIM / FID | ~28.0 dB / ~0.80 / ~15.0 | >31.0 dB / >0.88 / <12.0 | **>33.0 dB / >0.92 / <8.0** |
| **Detection / Pose** | mAP@0.5:0.95 | ~0.300 | >0.450 | **>0.650** |
| **Quality Assessment** | PLCC / SRCC | ~0.85 / ~0.80 | >0.90 / >0.85 | **>0.95 / >0.90** |

## 8. Model Inventory Detail

| Model Name | Category | Base Architecture | Source Framework | Core Computational Purpose | Target Evaluation Topology |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LemGendary NIMA Aesthetic Scorer** | Enhance | NIMA | [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) | Aesthetic quality scorer trained on AVA dataset | [LemGendizedQualityDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-quality-dataset) |
| **LemGendary NIMA Technical Scorer** | Enhance | NIMA | [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) | Technical quality scorer trained on LIVE+TID2013 | [LemGendizedQualityDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-quality-dataset) |
| **LemGendary UPN v2 Parameter Predictor** | Enhance | LemGenda Native | [Internal Native Architecture] | Universal parameter predictor for image restoration | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset), [LemGendizedLowLightDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset), [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset), [LemGendizedNoiseDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset) |
| **LemGendary Universal Film Restorer** | Enhance | LemGenda Native | [Internal Native Architecture] | Universal image restoration autoencoder | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset), [LemGendizedLowLightDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset), [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset), [LemGendizedNoiseDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset) |
| **LemGendary CodeFormer Face Restoration** | Face | CodeFormer | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | Face restoration model | [LemGendizedFaceDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset) |
| **LemGendary ParseNet Face Parsing** | Face | ParseNet | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | Face parsing model for segmentation | [LemGendizedFaceDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset) |
| **LemGendary RetinaFace MobileNet Detection** | Face | RetinaFace | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MobileNet-based face detection | [LemGendizedFaceDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset), [LemGendizedDetectionDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-detection-dataset) |
| **LemGendary RetinaFace ResNet Detection** | Face | RetinaFace | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | ResNet-based face detection | [LemGendizedFaceDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset), [LemGendizedDetectionDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-detection-dataset) |
| **LemGendary FFANet Dehazing (Indoor)** | Restoration | FFANet | [zhhyy/FFA-Net](https://github.com/zhhyy/FFA-Net) | FFANet indoor dehazing restoration | [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset) |
| **LemGendary FFANet Dehazing (Outdoor)** | Restoration | FFANet | [zhhyy/FFA-Net](https://github.com/zhhyy/FFA-Net) | FFANet outdoor dehazing restoration | [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset) |
| **LemGendary MIRNet v2 Low-Light Enhancement** | Restoration | MIRNet | [swz30/MIRNet](https://github.com/swz30/MIRNet) | MIRNet v2 low-light image enhancement | [LemGendizedLowLightDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset) |
| **LemGendary MIRNet v2 Exposure Correction** | Restoration | MIRNet | [swz30/MIRNet](https://github.com/swz30/MIRNet) | MIRNet v2 for overexposure correction and dynamic range adjustment | [LemGendizedLowLightDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset) |
| **LemGendary MPRNet Deraining** | Restoration | MPRNet | [swz30/MPRNet](https://github.com/swz30/MPRNet) | MPRNet image deraining | [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset) |
| **LemGendary NAFNet Debluring** | Restoration | NAFNet | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | NAFNet image debluring | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset), [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset) |
| **LemGendary NAFNet Denoising** | Restoration | NAFNet | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | NAFNet image denoising | [LemGendizedNoiseDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset) |
| **LemGendary UltraZoom 2x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 2x super-resolution model | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset) |
| **LemGendary UltraZoom 3x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 3x super-resolution model | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset) |
| **LemGendary UltraZoom 4x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 4x super-resolution model | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset) |
| **LemGendary UltraZoom 8x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 8x super-resolution model | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset) |
| **LemGendary YOLOv8n Multi-Task Model** | Yolo | YOLOv8n | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | Unified YOLOv8n for classification, detection, and pose | [LemGendizedDetectionDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-detection-dataset), [LemGendizedFaceDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset) |
| **LemGendary Professional Multi-Task Restoration Model** | Restoration | LemGenda Native | [Internal Native Architecture] | Shared Encoder Multi-Task model for Denoise, Deblur, Derain, Dehaze, and Low-Light | [LemGendizedSuperResDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset), [LemGendizedLowLightDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset), [LemGendizedDegradationDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset), [LemGendizedNoiseDataset](https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset) |

## 9. Dataset Inventory Detail

| DataSetName | Category | OriginalDataSet Sources | Kaggle Source | Purpose | Associated Models |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LemGendizedQualityDataset** | Quality | AVA, LIVE, TID2013, PIPAL, DPED | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-quality-dataset) | Aesthetic/Technical mathematical evaluation | NIMA Aesthetic, NIMA Technical |
| **LemGendizedFaceDataset** | Face | FFHQ, CelebAMask, Helen, WFLW, AffectNet | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset) | High-fidelity face restoration, parsing, detection | YOLOv8n, CodeFormer, ParseNet, RetinaFace (MobileNet), RetinaFace (ResNet) |
| **LemGendizedDegradationDataset** | Restoration | HIDE, Rain100L, RealBlur, REDS, RESIDE | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset) | Image Dehaze, Derain, and Deblur | Professional MultiTask Restorer, Universal Film Restorer, UPN v2, FFANet (Indoor), FFANet (Outdoor), MPRNet, NAFNet (Debluring) |
| **LemGendizedSuperResDataset** | Restoration | DIV2K, Flickr2K, Urban100, LSDIR, REDS | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset) | High-Fidelity Super-Resolution | Professional MultiTask Restorer, Universal Film Restorer, UPN v2, NAFNet (Debluring), UltraZoom (x2, x3, x4, x8) |
| **LemGendizedLowLightDataset** | Restoration | LOL, SICE, ExposureCorrection, ImageExposure | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset) | Extreme Low-Light and Exposure manipulation | Professional MultiTask Restorer, Universal Film Restorer, UPN v2, MIRNet (Low-Light), MIRNet (Exposure) |
| **LemGendizedNoiseDataset** | Restoration | SIDD, DND, MultiNoises | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset) | Advanced Neural Denoising | Professional MultiTask Restorer, Universal Film Restorer, UPN v2, NAFNet (Denoising) |
| **LemGendizedDetectionDataset** | Detection | COCO, Objects365, OpenImages | [Source](https://www.kaggle.com/datasets/lemtreursi/lemgendized-detection-dataset) | Universal Object Detection, Pose, and Classification | YOLOv8n, RetinaFace (MobileNet), RetinaFace (ResNet) |
| ArtImages | Restoration | Art_Dataset_Clear (User Uploaded) | | Art restoration | Professional MultiTask Restorer |

<br>

---

## 10. Advanced V2.0 Engine Mechanics

The LemGendary Training Suite has been heavily upgraded to incorporate native architectural efficiencies for complex multi-model dataset routing:

### ⚡ Zero-Latency Kaggle Background Streams
When you execute Option 5 (**Smart Cloud Orchestration**), the Python loop breaks memory isolation to execute a massive 8-phase mathematical pipeline. During execution, the moment a mathematical model natively **breaches the SOTA Baseline threshold**, the core `train.py` hook structurally detaches an independent Python `subprocess` thread.
This invisible background worker (`prefetch_worker.py`) seamlessly streams the subsequent 20GB+ Kaggle Datasets directly over the Kaggle API and radically uncompresses them into your local `data/datasets/` cache while the active PyTorch Tensor structure performs its final 1-Epoch Cooldown. This strictly guarantees **100% Zero-Latency Data Handoffs** mathematically without stealing Focus or interrupting Local GPU arrays.

### 🛡️ Intelligent SOTA Checkpoint Recoveries
The LemGendary Training Suite (2026 Engine) incorporates a multi-layer state recovery architecture:

- **Metric Persistence**: We have structurally resolved the historical `-1.0` regression bug. `best_quality_score` (NIMA) and `best_val_loss` (Restoration) are now explicitly persisted inside the `.pth` state and restored correctly during both `Global Guardrail` probes and `Session Resumption`.
- **2026 Continuity Protocol**: If you resume a model from a checkpoint that has not yet hit its SOTA targets but has already reached its defining epoch limit, the orchestrator now triggers an **Automated Continuity Extension (+20 Epochs)** to ensure the mission doesn't stall until the benchmarks are breached.
- **SOTA Bypass (Fast-Forward)**: If a model already achieved its benchmarks in a previous session but the export was interrupted, re-running the Hub will now detect the SOTA flag and **fast-forward directly to the ONNX export phase**, bypassing the 90-minute training loop entirely to save GPU compute.

### 🧬 Resilience & Numerical Stability (v1.0.42)
The deployment chain is now natively optimized for unstable Windows file-system and high-precision convergence:

- **Modular Hyperparameter Injection**: Mathematical stabilizers (Softmax Temperature, EMD Epsilon, Logit Clamping) are now dynamically injected from the model registry (`unified_models.yaml`). This ensures multi-model safety, where hardening for NIMA doesn't interfere with restoration manifolds.
- **Serial Extraction Mutex**: To prevent SSD contention and CPU hangs, the dataset pipeline implements a **Global Named Mutex** (`Global\LemGendaryExtractionLock`). Downloads remain parallel, but ZIP extractions are serialized (one-at-a-time).
- **Infinite NaN Recovery**: The engine incorporates deep-state sanitization. Upon detecting a numerical explosion (NaN), the system re-initializes the `GradScaler`, performs a momentum flush, and rolls back to the historical SOTA baseline with an automatic **50% LR Cooling** phase.
- **Metric Recovery Engine (Zero-Guess Documentation)**: During SOTA fast-forwards, the script no longer uses "guestimate" placeholders. It dynamically scrapes your local `metrics.csv` to extract the real-world historical PLCC, SRCC, and PSNR values for the final production `README.md`.
- **Resilient Artifact Sync (Windows IO Guard)**: The final deployment sync includes a mandatory **Settle-Period** and a **3-Attempt Retry Loop** to handle Windows `WinError 32` file locks.
