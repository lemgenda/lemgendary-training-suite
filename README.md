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
1. **Prepare Python Environment**: Automatically creates a `.venv_training` enclosure and installs all dependencies (PyTorch 2.4.1+cu121, DirectML, YOLO, etc.).
2. **Optimize Hardware Utilization**: Profiles your CPU threads and GPUs. Configures "Extreme" mode to use **NVIDIA (CUDA)** for training and **AMD/Intel (DirectML)** for real-time synthesis.
3. **Standardize Models**: Guided conversion of raw `.pth` or `.onnx` weights into the LemGendary format.
4. **Standardize Datasets**: Guided processing of raw images into clean/degraded training pairs.
5. **Train Models**: Launches the unified training suite to select models and begin the learning process.

---

## 3. Training & Heterogeneous Compute

### Hardware Profiles
- **Extreme Mode**: Saturates all Logical Cores (12 threads) and utilizes **Dual-GPUs**.
  - **Primary (NVIDIA)**: Handles the heavy model training (Backpropagation).
  - **Auxiliary (AMD/Intel)**: Offloads the real-time image synthesis (Blur, Noise, Haze).

### Smart Termination Logic
To ensure "Excellent" results without wasting compute:
1. **Target Reached**: If the strict mathematical threshold (`PLCC > 0.85 / SRCC > 0.82` for Quality, `PSNR > 32.0` for Restoration) is breached, the orchestrator triggers dynamic Early-Stopping.
2. **Reinforcement Buffer**: The suite automatically mathematically enforces exactly **2 final reinforcement epochs** instantly upon breaching the array threshold to perfectly stabilize the tensor weights before executing `.onnx` compilation.
3. **Max 50 Epochs**: Absolute safety hard-cap to violently terminate if convergence plateaus.

---

## 4. Exporting to Production
All exports are consolidated in **`trained-models/`**.

### Standardized Formats:
- **`ModelName.onnx`** (FP16): Production-ready for WebGPU. Dynamically automatically routed directly into your Next.js `public/models/restoration/` web directory for instant UI rendering.
- **`ModelName_FP32.onnx`** (FP32): Reference model for precision debugging. Weights are stored in an external `.onnx.data` sidecar.

---

## 5. Model Inventory
- **Diagnostic Eyes**: YOLOv8n (Detection), NIMA (Aesthetic/Technical Scoring).
- **Core Hands**: Professional Multi-Task Restorer, UPN v2 (AI Auto-Fix), Universal Film Restorer.
- **Surgical Tools**: CodeFormer (Face), NAFNet (Denoise), FFANet (Dehaze), MIRNet (Exposure).
- **UltraZoom**: Real-ESRGAN based x2/x3/x4/x8 upscalers.

---

## 6. GPU Benchmark (NVIDIA GTX 1650)
| Metric | Baseline | Optimized (Extreme) |
| :--- | :--- | :--- |
| **Throughput** | ~4.5 hrs / epoch | **~45 mins / epoch** |
| **Speedup** | 1x | **5.3x faster** |
| **VRAM Usage** | ~2.5 GB | **3.5 GB (87%)** |
| **Precision** | 32-bit (FP32) | **Mixed Precision (AMP)** |

*Stay Lemgendary.* verified syntax OK.

## 7. Model Inventory Detail

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

## 8. Dataset Inventory Detail

| DataSetName | Category | OriginalDataSet Sources | Purpose | Associated Models |
| :--- | :--- | :--- | :--- | :--- |
| LemGendizedQualityDataset | Quality | AVA, LIVE, TID2013, PIPAL | Aesthetic/Technical mathematical evaluation | NIMA Aesthetic, NIMA Technical |
| LemGendizedFaceDataset | Face | CelebAMask-HQ, FFHQ, Helen, WiderFace | High-fidelity face restoration, parsing, detection | YOLOv8n Unified |
| LemGendizedDegradationDataset | Restoration | GoPro, HIDE, RESIDE, Rain100 | Image Dehaze, Derain, and Deblur | Professional MultiTask Restorer |
| LemGendizedSuperResDataset | Restoration | DIV2K, Flickr2K, Urban100, LSDIR, REDS | High-Fidelity Super-Resolution | UltraZoom, Professional MultiTask Restorer |
| LemGendizedLowLightDataset | Restoration | DarkFace, ExposureCorrection, LOL, SIDD, SICE, ImageExposure | Extreme Low-Light and Exposure manipulation | Professional MultiTask Restorer |
| LemGendizedNoiseDataset | Restoration | DND, SIDD, Synthetic | Advanced Neural Denoising | Professional MultiTask Restorer |
| LemGendizedDetectionDataset | Detection | COCO, Objects365, OpenImages | Universal Object Detection, Pose, and Classification | YOLOv8n Unified |
| ArtImages | Restoration | Art_Dataset_Clear (User Uploaded) | Art restoration | Professional MultiTask Restorer |

