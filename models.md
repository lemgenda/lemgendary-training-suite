# LemGendary Models Architecture & Domain Matrix

This document provides a unified architectural taxonomy and operational index for all production models within the **LemGendary Training Suite**.

Models are categorized across three operational domains:

1. **Image Manipulation & Restoration**: SOTA computer vision architectures targeting pixel reconstruction, artifact removal, facial fidelity, and parameter regression.
2. **Financial & Time-Series**: Multi-asset, multi-timeframe temporal predictors processing causal OHLCV sequence manifolds.
3. **Image Generation & Multimodal**: High-capacity generative pipelines and multimodal vision-language architectures.

---

## 1. Master Models Architecture Matrix

| Model Key | Model Name | Domain | Category | Dataset Type | Architecture & Backbone | Res Ladder | Target Manifold | Recommended Accelerator | Target SOTA Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nima_aesthetic_mobile` | LemGendary NIMA Aesthetic Scorer (Mobile) | Image Manipulation & Restoration | `enhance` | `quality` | MobileNetV2 (Global Composition) | [224] | `LemGendizedNimaAestheticLarge` | P100 | PLCC: 0.65, SRCC: 0.65 |
| `nima_aesthetic_efficientnet` | LemGendary NIMA Aesthetic Scorer (EfficientNetV2-S) | Image Manipulation & Restoration | `enhance` | `quality` | EfficientNetV2-S (Global Composition) | [224, 256, 384] | `LemGendizedNimaAestheticLarge` | P100 | PLCC: 0.70, SRCC: 0.70 |
| `nima_aesthetic_pro` | LemGendary NIMA Aesthetic Scorer (Pro ViT) | Image Manipulation & Restoration | `enhance` | `quality` | Swin-v2-T (Global Multi-Scale Attention) | [256, 384, 512] | `LemGendizedNimaAestheticLarge` | Dual T4 | PLCC: 0.75, SRCC: 0.75 |
| `nima_technical` | LemGendary NIMA Technical Scorer | Image Manipulation & Restoration | `enhance` | `quality` | EfficientNetV2-S (Spatial Integrity) | [256, 384, 512] | `LemGendizedNimaTechnicalLarge` | P100 | PLCC: 0.91, SRCC: 0.91, Rank Margin: 0.05 |
| `nima_authenticity` | LemGendary Authenticity Scorer (AI vs Human) | Image Manipulation & Restoration | `enhance` | `quality` | EfficientNetV2-S (Distribution Scorer) | [256, 384, 512, 768] | `LemGendizedNimaAuthenticityLarge` | P100 | Accuracy: 0.96 |
| `upn_v2` | LemGendary UPN v2 Parameter Predictor | Image Manipulation & Restoration | `enhance` | `parameter_prediction` | UPN_v2 (MobileNet-Lite Parameter Regressor) | [128, 192, 256] | `LemGendizedUpnV2Large` | P100 | MAE: 0.05 |
| `film_restorer` | LemGendary Universal Film Restorer | Image Manipulation & Restoration | `restoration` | `restoration` | UniversalFilmRestorer (Residual Dense Autoencoder) | [256, 384, 512] | `LemGendizedFilmRestorerLarge` | Dual T4 | PSNR: 24.0, SSIM: 0.80, LPIPS: 0.25, FID: 12.0 |
| `codeformer` | LemGendary CodeFormer Face Restoration | Image Manipulation & Restoration | `face` | `face` | CodeFormer (Transformer-Based Face Restoration) | [512] | `LemGendizedCodeFormerLarge` | Dual T4 | PSNR: 30.5, SSIM: 0.93, LPIPS: 0.08, FID: 5.2 |
| `parsenet` | LemGendary ParseNet Face Parsing | Image Manipulation & Restoration | `face` | `segmentation` | ParseNet (Bilateral Face Segmentation Network) | [512] | `LemGendizedParseNetLarge` | Dual T4 | mIoU: 0.86 |
| `retinaface` | LemGendary RetinaFace Detection | Image Manipulation & Restoration | `face` | `face_detection` | RetinaFace (MobileNetV1-0.25 FPN Backbone) | [640] | `LemGendizedRetinaFaceMobileNetLarge` | P100 | mAP Easy: 0.915, Med: 0.890, Hard: 0.750 |
| `ffanet_indoor` | LemGendary FFANet Dehazing (Indoor) | Image Manipulation & Restoration | `restoration` | `restoration` | BranchedFFANet (Feature Fusion Attention) | [256, 384, 512] | `LemGendizedFfaNetIndoorLarge` | Dual T4 | PSNR: 36.5, SSIM: 0.990, LPIPS: 0.08, FID: 12.0 |
| `ffanet_outdoor` | LemGendary FFANet Dehazing (Outdoor) | Image Manipulation & Restoration | `restoration` | `restoration` | BranchedFFANet (Feature Fusion Attention) | [256, 384, 512] | `LemGendizedFfaNetOutdoorLarge` | Dual T4 | PSNR: 33.7, SSIM: 0.986, LPIPS: 0.08, FID: 12.0 |
| `mirnet_lowlight` | LemGendary MIRNet v2 Low-Light Enhancement | Image Manipulation & Restoration | `restoration` | `restoration` | MIRNet_v2 (Multi-Scale Residual Network) | [256, 384, 512] | `LemGendizedMirNetLowLightLarge` | Dual T4 | PSNR: 24.3, SSIM: 0.840, LPIPS: 0.08, FID: 12.0 |
| `mirnet_exposure` | LemGendary MIRNet v2 Exposure Correction | Image Manipulation & Restoration | `restoration` | `restoration` | MIRNet_v2 (Multi-Scale Residual Network) | [256, 384, 512] | `LemGendizedMirNetExposureLarge` | Dual T4 | PSNR: 24.3, SSIM: 0.840, LPIPS: 0.08, FID: 12.0 |
| `mprnet_deraining` | LemGendary MPRNet Deraining | Image Manipulation & Restoration | `restoration` | `restoration` | MPRNet (Multi-Stage Progressive Network) | [256, 384, 512] | `LemGendizedMprNetDerainingLarge` | Dual T4 | PSNR: 30.6, SSIM: 0.900, LPIPS: 0.07, FID: 12.0 |
| `nafnet_debluring` | LemGendary NAFNet Debluring | Image Manipulation & Restoration | `restoration` | `restoration` | NAFNet (Nonlinear Activation-Free Network) | [256, 384, 512] | `LemGendizedNafNetDebluringLarge` | Dual T4 | PSNR: 33.9, SSIM: 0.970, LPIPS: 0.04, FID: 6.0 |
| `nafnet_denoising` | LemGendary NAFNet Denoising | Image Manipulation & Restoration | `restoration` | `restoration` | NAFNet (Nonlinear Activation-Free Network) | [256, 384, 512] | `LemGendizedNafNetDenoisingLarge` | Dual T4 | PSNR: 40.2, SSIM: 0.965, LPIPS: 0.02, FID: 4.0 |
| `yolov8n` | LemGendary YOLOv8n Multi-Task Model | Image Manipulation & Restoration | `yolo` | `detection` | YOLOv8n (CSPDarknet53 + PANet) | [320, 480, 640] | `LemGendizedYoloV8n` | P100 | mAP50: 0.540, mAP50-95: 0.390 |
| `professional_multitask_restoration` | LemGendary Professional Multi-Task Restoration Model | Image Manipulation & Restoration | `restoration` | `restoration` | MultiTaskRestorer (Shared Encoder Multi-Task MoE) | [256, 384, 512] | `LemGendizedProfessionalMultitaskRestorationLarge` | Dual T4 | PSNR: 32.0, SSIM: 0.930, LPIPS: 0.07, FID: 12.0 |
| `ultrazoom` | LemGendary UltraZoom Master Model | Image Manipulation & Restoration | `restoration` | `restoration` | UltraZoomMaster (Sub-Pixel ESPCN Super-Resolution) | [256, 384, 512] | `LemGendizedUltraZoomLarge` | Dual T4 | PSNR: 34.0, SSIM: 0.950, LPIPS: 0.04, FID: 10.0 |
| `universal_nsfw_classification` | LemGendary Universal NSFW Classifier | Image Manipulation & Restoration | `enhance` | `classification` | EfficientNetV2-S (Multi-Class Categorical Head) | [224] | `LemGendizedClassificationMasterManifoldLarge` | P100 | Accuracy: 0.98 |
| `forex_predictor` | LemGendary Forex Predictor | Financial & Time-Series | `forex` | `forex` | Multi-Scale CNN-Transformer (Causal TCN + Attention) | [1, 5, 15, 60, 240, 1440] (TFs) | `LemGendizedForexUniverseLarge` | P100 | Dir Acc: 58.5%, Win Rate: 56.0%, PF: 1.65, Sharpe: 1.85 |

---

## 2. Domain Architectural Breakdowns

### 2.1 Image Manipulation & Restoration Domain

The **Image Manipulation & Restoration** domain encompasses specialized image restoration, aesthetic scoring, biometric feature parsing, and photographic adjustment models.

1. **Pixel-Level Restoration Autoencoders**:
   - **NAFNet (Deblurring & Denoising)**: Employs Nonlinear Activation-Free blocks utilizing Simplified Channel Attention (SCA) and SimpleGate elementwise multipliers to eliminate non-linear activations, boosting training throughput by up to 2.4x while maintaining SOTA perceptual recovery.
   - **BranchedFFANet (Indoor & Outdoor Dehazing)**: Features Feature Fusion Attention with spatial-channel dual attention mechanisms, resolving complex non-uniform atmospheric scattering and haze gradients.
   - **MIRNet_v2 (Low-Light & Exposure)**: Dual-stream multi-scale residual blocks maintain spatially fine details alongside contextual semantics to correct high dynamic range and severe underexposure without noise blow-up.
   - **MPRNet (Deraining)**: Multi-Stage Progressive network that blends contextual representations at early stages with fine detail recovery at later stages via supervised attention modules.
   - **UniversalFilmRestorer**: Deep Residual Dense Block bottleneck coupled with progressive upsampling to eliminate severe vintage film scratches, dust, chemical degradation, and color dye fading.
   - **MultiTaskRestorer**: Shared encoder architecture with task-routed mixture-of-experts (MoE) heads trained on the comprehensive `LemGendizedProfessionalMultitaskRestorationLarge` manifold.

2. **Quality & Authenticity Distribution Scorers**:
   - **NIMA Models (Mobile, EfficientNetV2-S, Pro ViT)**: Implements Earth Mover's Distance (EMD) loss over 10-bin quality score distributions, learning human visual preference, composition, and micro-texture integrity.
   - **AuthenticityScorer**: EfficientNetV2-S backbone with spatial statistical pooling, trained on high-entropy blends of real photographs and diffusion/synthetic artifacts.
   - **UniversalClassifier**: High-throughput multi-class categorical engine designed for rapid content moderation and NSFW safety filtering.

3. **Facial Restoration & Geometry Parsing**:
   - **CodeFormer**: Expressive vector-quantized codebook lookup network with transformer feature binding for blind face restoration.
   - **ParseNet**: 19-class bilateral semantic segmentation network resolving discrete anatomical face boundaries.
   - **RetinaFace**: Single-shot feature pyramid network with MobileNetV1-0.25 backbone optimized for robust multi-scale facial landmarking and bounding box localization.

### 2.2 Financial & Time-Series Domain

The **Financial & Time-Series** domain targets multi-asset, multi-timeframe financial prediction models operating under causal constraints.

- **ForexPredictor**:
  - **Temporal Convolutional Network (TCN)**: Dilated causal Conv1D stacks extract temporal feature hierarchies without future information lookahead.
  - **Cross-Timeframe Attention (CTFA)**: Multi-head cross-attention mechanism projects short-horizon microstructure volatility (1m, 5m, 15m) into macro-trend structural manifolds (1h, 4h, Daily).
  - **Loss Engine**: Dual-objective optimization balancing directional cross-entropy classification (`SELL`, `HOLD`, `BUY`) and Take-Profit/Stop-Loss pips boundary regression.
  - **Data Source**: Unified Parquet manifold `LemGendizedForexUniverseLarge` containing over 26.8 million sequence samples across 16 global currency and commodity symbols (2019-2026).

### 2.3 Image Generation & Multimodal Domain

The **Image Generation & Multimodal** domain accommodates massive generative models and vision-language token aligners.

- **Master Generative Pipelines**: Latent Diffusion Models (SDXL, Flux-1) mapping textual embeddings to high-resolution latent spatial tensors.
- **Master Multimodal Reasoning**: Autoregressive conversational models (LLaVA-1.5, BLIP-2) combining visual patch encoders with causal language decoder heads.
