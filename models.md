# LemGendary Models & Manifolds Matrix

| Model Key | Model Name | Backbone | Res Ladder | Manifold Avg (Std) Res | Our Target Metrics | Official SOTA Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nima_aesthetic_mobile` | LemGendary NIMA Aesthetic Scorer (Mobile) | MobileNetV2 (Global Composition) | [224] | 640x505 | PLCC: 0.60, SRCC: 0.60 | SRCC: ~0.610, PLCC: ~0.638 |
| `nima_aesthetic_efficientnet` | LemGendary NIMA Aesthetic Scorer (EfficientNetV2-S) | EfficientNetV2-S (Global Composition) | [224, 384] | 640x505 | PLCC: 0.7, SRCC: 0.7 | SRCC: ~0.650, PLCC: ~0.650 (Est.) |
| `nima_aesthetic_pro` | LemGendary NIMA Aesthetic Scorer (Pro ViT) | Swin-v2-T (Global Multi-Scale Attention) | [256, 384, 512] | 640x505 | PLCC: 0.75, SRCC: 0.75 | SRCC: ~0.650, PLCC: ~0.650 (Est.) |
| `nima_technical` | LemGendary NIMA Technical Scorer | EfficientNetV2-S (Spatial Integrity) | [384, 512] | 512x512 | PLCC: 0.91, SRCC: 0.91, RANK_MARGIN: 0.05 | Accuracy/PLCC: ~0.80 - 0.90 |
| `nima_authenticity` | LemGendary Authenticity Scorer (AI vs Human) | EfficientNetV2-S (Distribution Scorer) | [256, 384, 512, 768] | 1608x1212 | ACCURACY: 0.96 | Accuracy: ~0.85 - 0.95 (AI Detection SOTA) |
| `upn_v2` | LemGendary UPN v2 Parameter Predictor | MobileNet-Lite | [128, 192, 256] | 2810x3381 | MAE: 0.05 | *N/A (Custom Architecture)* |
| `film_restorer` | LemGendary Universal Film Restorer | NAFNet-Derived Base | [256, 384, 512] | 2040x1536 | PSNR: 24.0, SSIM: 0.8, LPIPS: 0.25 | *N/A (Custom Architecture)* |
| `codeformer` | LemGendary CodeFormer Face Restoration | UNet (Latent Autoencoder) | 512 | 1024x1024 | FID: 5.2, PSNR: 30.5, SSIM: 0.93 | FID: ~5.3-6.0, PSNR: ~28-30 |
| `parsenet` | LemGendary ParseNet Face Parsing | UNet (19-class Seg) | [512] | 1024x1024 | MIOU: 0.86 | mIoU: ~0.75 - 0.85 |
| `retinaface_mobilenet` | LemGendary RetinaFace MobileNet Detection | MobileNet | [640] | 1024x1024 | MAP_EASY: 0.915, MAP_MEDIUM: 0.89, MAP_HARD: 0.75 | Easy: 90.7%, Med: 88.1%, Hard: 73.8% |
| `ffanet_indoor` | LemGendary FFANet Dehazing (Indoor) | Feature Fusion Attention | [256, 384, 512] | 530x399 | PSNR: 36.50, SSIM: 0.990, LPIPS: 0.08, FID: 12.0 | PSNR: 36.50, SSIM: 0.9906 |
| `ffanet_outdoor` | LemGendary FFANet Dehazing (Outdoor) | Feature Fusion Attention | [256, 384, 512] | 1600x1200 | PSNR: 33.70, SSIM: 0.986, LPIPS: 0.08, FID: 12.0 | PSNR: 33.70, SSIM: 0.9860 |
| `mirnet_lowlight` | LemGendary MIRNet v2 Low-Light Enhancement | Multi-Scale Residual Block | [256, 384, 512] | 500x363 | PSNR: 24.30, SSIM: 0.840, LPIPS: 0.08, FID: 12.0 | PSNR: ~24.14, SSIM: 0.830 |
| `mirnet_exposure` | LemGendary MIRNet v2 Exposure Correction | Multi-Scale Residual Block | [256, 384, 512] | 2810x3381 | PSNR: 24.30, SSIM: 0.840, LPIPS: 0.08, FID: 12.0 | PSNR: ~24.14, SSIM: 0.830 |
| `mprnet_deraining` | LemGendary MPRNet Deraining | Multi-Stage Progressive | [256, 384, 512] | 512x512 | PSNR: 30.60, SSIM: 0.900, LPIPS: 0.07, FID: 12.0 | PSNR: ~30.41, SSIM: 0.89 |
| `nafnet_debluring` | LemGendary NAFNet Debluring | Nonlinear Activation Free | [256, 384, 512] | Not Found | PSNR: 33.90, SSIM: 0.970, LPIPS: 0.04, FID: 6.0 | PSNR: 33.71, SSIM: 0.967 |
| `nafnet_denoising` | LemGendary NAFNet Denoising | Nonlinear Activation Free | [256, 384, 512] | Not Found | PSNR: 40.20, SSIM: 0.965, LPIPS: 0.02, FID: 4.0 | PSNR: 39.96, SSIM: 0.960 |
| `yolov8n` | LemGendary YOLOv8n Multi-Task Model | CSPDarknet53 | [320, 480, 640] | 586x479 | MAP50: 0.540, MAP50_95: 0.390 | mAP50-95: 37.3 |
| `professional_multitask_restoration` | LemGendary Professional Multi-Task Restoration Model | SharedEncoder (MoE) | [256, 384, 512] | 703x584 | PSNR: 32.0, SSIM: 0.93, LPIPS: 0.07, FID: 12.0 | *N/A (Custom Architecture)* |
| `ultrazoom` | LemGendary UltraZoom Master Model | NAFNet-Derived Base | [256, 384, 512] | 2040x1368 | PSNR: 34.0, SSIM: 0.95, LPIPS: 0.04, FID: 10.0 | *N/A (Custom Architecture)* |
| `universal_nsfw_classification` | LemGendary Universal NSFW Classifier | MobileNetV2 (Categorical Anchor) | [224] | 693x672 | ACCURACY: 0.98 | Accuracy: ~0.95+ |
