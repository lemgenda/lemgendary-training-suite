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
1. **Initialize/Fix Environment**: Installs Python 3.10 natively and creates the `.venv` enclosure. Upgrades pip, then installs PyTorch 2.4.1+cu121 alongside all other mathematical dependencies for inference.
2. **Train Individual Model**: Launches the interactive LemGendary Unified Training Suite, empowering singular network selection for localized structural training loops natively.
3. **Global Orchestration**: Automates sequential continuous execution of **all 21 models** on your local hardware architecture uninterrupted.
4. **Deploy to Kaggle Cloud**: Outputs physical instructions on how to push the pre-generated Jupyter topologies to Kaggle's T4 instances instantly.
5. **Exit**: Terminate hub.

---

## 3. Training & Heterogeneous Compute

### Hardware Profiles
- **Extreme Mode**: Saturates all Logical Cores (12 threads) and utilizes **Dual-GPUs**.
  - **Primary (NVIDIA)**: Handles the heavy model training (Backpropagation).
  - **Auxiliary (AMD/Intel)**: Offloads the real-time image synthesis (Blur, Noise, Haze).

### Smart Termination Logic
To ensure State-of-the-Art (SOTA) results without wasting compute:
1. **Target Reached**: If the strict mathematical SOTA threshold (`PLCC > 0.95 / SRCC > 0.90` for Quality, `PSNR > 32.5` for Restoration) is breached, the orchestrator triggers dynamic Early-Stopping.
2. **Cooldown Buffer**: The suite automatically mathematically enforces exactly **1 final cooldown epoch** instantly upon breaching the array threshold to perfectly stabilize the tensor weights before executing `.onnx` compilation.
3. **Max 50 Epochs**: Absolute safety hard-cap to violently terminate if convergence plateaus.

---

## 4. Exporting to Production
All exports are consolidated in **`trained-models/`**.

### Standardized Formats:
- **`ModelName.onnx`** (FP16): Production-ready for WebGPU. Dynamically automatically routed directly into your Next.js `public/models/restoration/` web directory for instant UI rendering.
- **`ModelName_FP32.onnx`** (FP32): Reference model for precision debugging. Weights are stored in an external `.onnx.data` sidecar.

## 5. Kaggle Cloud Integration
The LemGendary Training Suite natively supports 100% cloud-based dataset training via Kaggle without downloading the 200GB topological arrays locally. The mathematical engine is perfectly tuned to support `--env kaggle` dataset interceptions.

### How to Launch a Cloud Training Node:
1. Open your local repository folder. I dynamically generated all 9 `.ipynb` deployment Jupyter Notebooks based strictly on their internal `unified_models.yaml` dataset dependencies:
   - **Solo Notebooks**: Topologies reliant solely upon exactly one environment (e.g. `Kaggle_Train_Solo_LemGendizedQualityDataset.ipynb`).
   - **Multi-Mount Notebooks**: Cross-domain models demanding multiple concurrent topologies (e.g. `Kaggle_Train_Multi_Universal_Restoration.ipynb`).
2. Navigate to your Kaggle profile and click **Create -> Notebook**.
3. In the top toolbar, select **File -> Import Notebook**. Select the `.ipynb` file of your choice.
4. Click **Add Data** on the right-hand interface. Search for and mount **exactly** the dataset(s) explicitly requested via comments at the top of the imported code cell.
5. In **Session Options**, mathematically assign the **Accelerator** natively to **GPU T4 x2** or **GPU P100**. *Do not use CPU or TPU.*
6. Click **Run All**. The notebook will securely clone this repository, align pathing natively against `/kaggle/input/`, and begin executing the epochs directly into your browser!

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
| **LemGendary NIMA Aesthetic Scorer** | Enhance | NIMA | [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) | Aesthetic quality scorer trained on AVA dataset | LemGendizedQualityDataset |
| **LemGendary NIMA Technical Scorer** | Enhance | NIMA | [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) | Technical quality scorer trained on LIVE+TID2013 | LemGendizedQualityDataset |
| **LemGendary UPN v2 Parameter Predictor** | Enhance | LemGenda Native | [Internal Native Architecture] | Universal parameter predictor for image restoration | LemGendizedSuperResDataset, LemGendizedLowLightDataset, LemGendizedDegradationDataset, LemGendizedNoiseDataset |
| **LemGendary Universal Film Restorer** | Enhance | LemGenda Native | [Internal Native Architecture] | Universal image restoration autoencoder | LemGendizedSuperResDataset, LemGendizedLowLightDataset, LemGendizedDegradationDataset, LemGendizedNoiseDataset |
| **LemGendary CodeFormer Face Restoration** | Face | CodeFormer | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | Face restoration model | LemGendizedFaceDataset |
| **LemGendary ParseNet Face Parsing** | Face | ParseNet | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | Face parsing model for segmentation | LemGendizedFaceDataset |
| **LemGendary RetinaFace MobileNet Detection** | Face | RetinaFace | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MobileNet-based face detection | LemGendizedFaceDataset, LemGendizedDetectionDataset |
| **LemGendary RetinaFace ResNet Detection** | Face | RetinaFace | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | ResNet-based face detection | LemGendizedFaceDataset, LemGendizedDetectionDataset |
| **LemGendary FFANet Dehazing (Indoor)** | Restoration | FFANet | [zhhyy/FFA-Net](https://github.com/zhhyy/FFA-Net) | FFANet indoor dehazing restoration | LemGendizedDegradationDataset |
| **LemGendary FFANet Dehazing (Outdoor)** | Restoration | FFANet | [zhhyy/FFA-Net](https://github.com/zhhyy/FFA-Net) | FFANet outdoor dehazing restoration | LemGendizedDegradationDataset |
| **LemGendary MIRNet v2 Low-Light Enhancement** | Restoration | MIRNet | [swz30/MIRNet](https://github.com/swz30/MIRNet) | MIRNet v2 low-light image enhancement | LemGendizedLowLightDataset |
| **LemGendary MIRNet v2 Exposure Correction** | Restoration | MIRNet | [swz30/MIRNet](https://github.com/swz30/MIRNet) | MIRNet v2 for overexposure correction and dynamic range adjustment | LemGendizedLowLightDataset |
| **LemGendary MPRNet Deraining** | Restoration | MPRNet | [swz30/MPRNet](https://github.com/swz30/MPRNet) | MPRNet image deraining | LemGendizedDegradationDataset |
| **LemGendary NAFNet Debluring** | Restoration | NAFNet | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | NAFNet image debluring | LemGendizedSuperResDataset, LemGendizedDegradationDataset |
| **LemGendary NAFNet Denoising** | Restoration | NAFNet | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | NAFNet image denoising | LemGendizedNoiseDataset |
| **LemGendary UltraZoom 2x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 2x super-resolution model | LemGendizedSuperResDataset |
| **LemGendary UltraZoom 3x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 3x super-resolution model | LemGendizedSuperResDataset |
| **LemGendary UltraZoom 4x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 4x super-resolution model | LemGendizedSuperResDataset |
| **LemGendary UltraZoom 8x Super-Resolution** | Ultrazoom | UltraZoom | [andrewdalpino/MewZoom](https://github.com/andrewdalpino/MewZoom/) | 8x super-resolution model | LemGendizedSuperResDataset |
| **LemGendary YOLOv8n Multi-Task Model** | Yolo | YOLOv8n | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | Unified YOLOv8n for classification, detection, and pose | LemGendizedDetectionDataset, LemGendizedFaceDataset |
| **LemGendary Professional Multi-Task Restoration Model** | Restoration | LemGenda Native | [Internal Native Architecture] | Shared Encoder Multi-Task model for Denoise, Deblur, Derain, Dehaze, and Low-Light | LemGendizedSuperResDataset, LemGendizedLowLightDataset, LemGendizedDegradationDataset, LemGendizedNoiseDataset |

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
