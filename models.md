# LemGendary Models Architecture & Domain Matrix

This document provides a unified architectural taxonomy and operational index
for all production models within the **LemGendary Training Suite**.

Models are categorized across three operational domains:

1. **Image Manipulation & Restoration**: SOTA computer vision architectures
   targeting pixel reconstruction, artifact removal, facial fidelity, and
   parameter regression.
2. **Financial & Time-Series**: Multi-asset, multi-timeframe temporal predictors
   processing causal OHLCV sequence manifolds.
3. **Image Generation & Multimodal**: High-capacity generative pipelines and
   multimodal vision-language architectures.

---

## 1. Master Models Architecture Matrix

- `nima_aesthetic_mobile`: LemGendary NIMA Aesthetic Scorer (Mobile)
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `quality`
  - Architecture & Backbone: MobileNetV3-Small (Global Composition)
  - Res Ladder: [224]
  - Target Manifold: `LemGendizedNimaAesthetic`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: PLCC: 0.65, SRCC: 0.65
- `nima_aesthetic_efficientnet`: LemGendary NIMA Aesthetic Scorer
  (EfficientNetV2-S)
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `quality`
  - Architecture & Backbone: EfficientNetV2-S (Global Composition)
  - Res Ladder: [224, 256, 384]
  - Target Manifold: `LemGendizedNimaAesthetic`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: PLCC: 0.70, SRCC: 0.70
- `nima_aesthetic_pro`: LemGendary NIMA Aesthetic Scorer (Pro ViT)
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `quality`
  - Architecture & Backbone: Swin-v2-T (Global Multi-Scale Attention)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedNimaAesthetic`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PLCC: 0.75, SRCC: 0.75
- `nima_technical`: LemGendary NIMA Technical Scorer
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `quality`
  - Architecture & Backbone: EfficientNetV2-S (Spatial Integrity)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedNimaTechnical`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: PLCC: 0.91, SRCC: 0.91, Rank Margin: 0.05
- `nima_authenticity`: LemGendary Authenticity Scorer (AI vs Human)
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `quality`
  - Architecture & Backbone: EfficientNetV2-S (Distribution Scorer)
  - Res Ladder: [256, 384, 512, 768]
  - Target Manifold: `LemGendizedNimaAuthenticity`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: Accuracy: 0.96
- `upn_v2`: LemGendary UPN v2 Parameter Predictor
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `parameter_prediction`
  - Architecture & Backbone: UPN_v2 (MobileNetV3-Small Parameter Regressor)
  - Res Ladder: [128, 192, 256]
  - Target Manifold: `LemGendizedUpnV2`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: MAE: 0.05
- `film_restorer`: LemGendary Universal Film Restorer
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: UniversalFilmRestorer (Residual Dense Autoencoder)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedFilmRestorer`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 24.0, SSIM: 0.80, LPIPS: 0.25, FID: 12.0
- `codeformer`: LemGendary CodeFormer Face Restoration
  - Domain: Image Manipulation & Restoration
  - Category: `face`
  - Dataset Type: `face`
  - Architecture & Backbone: CodeFormer (Transformer-Based Face Restoration)
  - Res Ladder: [512]
  - Target Manifold: `LemGendizedCodeFormer`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 30.5, SSIM: 0.93, LPIPS: 0.08, FID: 5.2
- `parsenet`: LemGendary ParseNet Face Parsing
  - Domain: Image Manipulation & Restoration
  - Category: `face`
  - Dataset Type: `segmentation`
  - Architecture & Backbone: ParseNet (Bilateral Face Segmentation Network)
  - Res Ladder: [512]
  - Target Manifold: `LemGendizedParseNet`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: mIoU: 0.86
- `retinaface`: LemGendary RetinaFace Detection
  - Domain: Image Manipulation & Restoration
  - Category: `face`
  - Dataset Type: `face_detection`
  - Architecture & Backbone: RetinaFace (MobileNetV3-Small Backbone)
  - Res Ladder: [640]
  - Target Manifold: `LemGendizedRetinaFace`
  - Recommended Accelerator: T4
  - Target SOTA Metrics (provisional): Box IoU: 0.80, AP50: 0.95,
    AP50:95: 0.60, Landmark NME: 0.10
- `ffanet_indoor`: LemGendary FFANet Dehazing (Indoor)
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: BranchedFFANet (Feature Fusion Attention)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedFfaNetIndoor`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 36.5, SSIM: 0.990, LPIPS: 0.08, FID: 12.0
- `ffanet_outdoor`: LemGendary FFANet Dehazing (Outdoor)
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: BranchedFFANet (Feature Fusion Attention)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedFfaNetOutdoor`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 33.7, SSIM: 0.986, LPIPS: 0.08, FID: 12.0
- `mirnet_lowlight`: LemGendary MIRNet v2 Low-Light Enhancement
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: MIRNet_v2 (Multi-Scale Residual Network)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedMirNetLowLight`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 24.3, SSIM: 0.840, LPIPS: 0.08, FID: 12.0
- `mirnet_exposure`: LemGendary MIRNet v2 Exposure Correction
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: MIRNet_v2 (Multi-Scale Residual Network)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedMirNetExposure`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 24.3, SSIM: 0.840, LPIPS: 0.08, FID: 12.0
- `mprnet_deraining`: LemGendary MPRNet Deraining
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: MPRNet (Multi-Stage Progressive Network)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedMprNetDeraining`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 30.6, SSIM: 0.900, LPIPS: 0.07, FID: 12.0
- `nafnet_debluring`: LemGendary NAFNet Debluring
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: NAFNet (Nonlinear Activation-Free Network)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedNafNetDebluring`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 33.9, SSIM: 0.970, LPIPS: 0.04, FID: 6.0
- `nafnet_denoising`: LemGendary NAFNet Denoising
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: NAFNet (Nonlinear Activation-Free Network)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedNafNetDenoising`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 40.2, SSIM: 0.965, LPIPS: 0.02, FID: 4.0
- `yolov8n`: LemGendary YOLOv8n Multi-Task Model
  - Domain: Image Manipulation & Restoration
  - Category: `yolo`
  - Dataset Type: `detection`
  - Architecture & Backbone: YOLOv8n (CSPDarknet53 + PANet)
  - Res Ladder: [320, 480, 640]
  - Target Manifold: `LemGendizedYoloV8n`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: mAP50: 0.540, mAP50-95: 0.390
- `professional_multitask_restoration`: LemGendary Professional Multi-Task
  Restoration Model
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: MultiTaskRestorer (Shared Encoder Multi-Task MoE)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedMultitaskRestorationPro`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 32.0, SSIM: 0.930, LPIPS: 0.07, FID: 12.0
- `ultrazoom`: LemGendary UltraZoom Master Model
  - Domain: Image Manipulation & Restoration
  - Category: `restoration`
  - Dataset Type: `restoration`
  - Architecture & Backbone: UltraZoomMaster (Sub-Pixel ESPCN Super-Resolution)
  - Res Ladder: [256, 384, 512]
  - Target Manifold: `LemGendizedUltraZoom`
  - Recommended Accelerator: Dual T4
  - Target SOTA Metrics: PSNR: 34.0, SSIM: 0.950, LPIPS: 0.04, FID: 10.0
- `universal_nsfw_classification`: LemGendary Universal NSFW Classifier
  - Domain: Image Manipulation & Restoration
  - Category: `enhance`
  - Dataset Type: `classification`
  - Architecture & Backbone: EfficientNetV2-S (Multi-Class Categorical Head)
  - Res Ladder: [224]
  - Target Manifold: `LemGendizedClassificationMaster`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: Accuracy: 0.98
- `forex_predictor`: LemGendary Forex Predictor
  - Domain: Financial & Time-Series
  - Category: `forex`
  - Dataset Type: `forex`
  - Architecture & Backbone: Multi-Scale CNN-Transformer (Causal TCN +
    Attention)
  - Res Ladder: [1, 5, 15, 60, 240, 1440] (TFs)
  - Target Manifold: `LemGendizedForexUniverse`
  - Recommended Accelerator: T4
  - Target SOTA Metrics: Dir Acc: 58.5%, Win Rate: 56.0%, PF: 1.65, Sharpe: 1.85
- `ai_helper_lemgend`: LemGendary AI Studio Assistant (Causal LLM)
  - Domain: Image Generation & Multimodal (language assistant)
  - Category: `llm`
  - Dataset Type: `text_instruction`
  - Architecture & Backbone: Causal Decoder Transformer (QLoRA on
    Qwen2.5-Coder-1.5B-Instruct)
  - Context Length: 2048 tokens
  - Target Manifold: `LemGendizedAIHelperLemGend`
  - Target SOTA Metrics: Eval Loss: 1.15, Perplexity: 3.15, Token Acc: 72.5%

---

## 2. Domain Architectural Breakdowns

### 2.1 Image Manipulation & Restoration Domain

The **Image Manipulation & Restoration** domain encompasses specialized image
restoration, aesthetic scoring, biometric feature parsing, and photographic
adjustment models.

1. **Pixel-Level Restoration Autoencoders**:
   - **NAFNet (Deblurring & Denoising)**: Employs Nonlinear Activation-Free
     blocks utilizing Simplified Channel Attention (SCA) and SimpleGate
     elementwise multipliers to eliminate non-linear activations, boosting
     training throughput by up to 2.4x while maintaining SOTA perceptual
     recovery.
   - **BranchedFFANet (Indoor & Outdoor Dehazing)**: Features Feature Fusion
     Attention with spatial-channel dual attention mechanisms, resolving complex
     non-uniform atmospheric scattering and haze gradients.
   - **MIRNet_v2 (Low-Light & Exposure)**: Dual-stream multi-scale residual
     blocks maintain spatially fine details alongside contextual semantics to
     correct high dynamic range and severe underexposure without noise blow-up.
   - **MPRNet (Deraining)**: Multi-Stage Progressive network that blends
     contextual representations at early stages with fine detail recovery at
     later stages via supervised attention modules.
   - **UniversalFilmRestorer**: Deep Residual Dense Block bottleneck coupled
     with progressive upsampling to eliminate severe vintage film scratches,
     dust, chemical degradation, and color dye fading.
   - **MultiTaskRestorer**: Shared encoder architecture with task-routed
     mixture-of-experts (MoE) heads trained on the comprehensive
     `LemGendizedMultitaskRestorationPro` manifold.

2. **Quality & Authenticity Distribution Scorers**:
   - **NIMA Models (Mobile, EfficientNetV2-S, Pro ViT)**: Implements Earth
     Mover's Distance (EMD) loss over 10-bin quality score distributions,
     learning human visual preference, composition, and micro-texture integrity.
   - **AuthenticityScorer**: EfficientNetV2-S backbone with spatial statistical
     pooling, trained on high-entropy blends of real photographs and
     diffusion/synthetic artifacts.
   - **UniversalClassifier**: High-throughput multi-class categorical engine
     designed for rapid content moderation and NSFW safety filtering.

3. **Facial Restoration & Geometry Parsing**:
   - **CodeFormer**: Expressive vector-quantized codebook lookup network with
     transformer feature binding for blind face restoration.
   - **ParseNet**: 19-class bilateral semantic segmentation network resolving
     discrete anatomical face boundaries.
   - **RetinaFace**: Single-stage face and landmark detection network with
     unified MobileNetV3-Small backbone optimized for robust facial landmarking
     and bounding box localization.

### 2.2 Financial & Time-Series Domain

The **Financial & Time-Series** domain targets multi-asset, multi-timeframe
financial prediction models operating under causal constraints.

- **ForexPredictor**:
  - **Temporal Convolutional Network (TCN)**: Dilated causal Conv1D stacks
    extract temporal feature hierarchies without future information lookahead.
  - **Cross-Timeframe Attention (CTFA)**: Multi-head cross-attention mechanism
    projects short-horizon microstructure volatility (1m, 5m, 15m) into
    macro-trend structural manifolds (1h, 4h, Daily).
  - **Loss Engine**: Dual-objective optimization balancing directional
    cross-entropy classification (`SELL`, `HOLD`, `BUY`) and
    Take-Profit/Stop-Loss pips boundary regression.
  - **Data Source**: Unified Parquet manifold `LemGendizedForexUniverse`
    containing over 26.8 million sequence samples across 16 global currency and
    commodity symbols (2019-2026).

### 2.3 Image Generation & Multimodal Domain

The **Image Generation & Multimodal** domain accommodates massive generative
models and vision-language token aligners.

- **Master Generative Pipelines**: Latent Diffusion Models (SDXL, Flux-1)
  mapping textual embeddings to high-resolution latent spatial tensors.
- **Master Multimodal Reasoning**: Autoregressive conversational models
  (LLaVA-1.5, BLIP-2) combining visual patch encoders with causal language
  decoder heads.
