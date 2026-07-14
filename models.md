# LemGendary Models & Manifolds Matrix

| Model Key | Model Name | Model Class | Backbone | Min Train Res | Max Train Res | Training Res Ladder | Val Res | Manifold Name | Manifold Min Res | Manifold Max Res | Manifold Avg (Std) Res |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `nima_aesthetic_mobile` | LemGendary NIMA Aesthetic Scorer (Mobile) | `NIMA_Model` | MobileNetV2 (Global Composition) | 224 | 224 | [224] | 224 | `LemGendizedNimaAesthetic` | 640x395 | 640x640 | 640x505 |
| `nima_aesthetic_efficientnet` | LemGendary NIMA Aesthetic Scorer (EfficientNetV2-S) | `NIMA_Model` | EfficientNetV2-S (Global Composition) | 224 | 384 | [224, 384] | 384 | `LemGendizedNimaAesthetic` | 640x395 | 640x640 | 640x505 |
| `nima_aesthetic_pro` | LemGendary NIMA Aesthetic Scorer (Pro ViT) | `NIMA_Model` | Swin-v2-T (Global Multi-Scale Attention) | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedNimaAesthetic` | 640x395 | 640x640 | 640x505 |
| `nima_technical` | LemGendary NIMA Technical Scorer | `NIMA_Model` | EfficientNetV2-S (Spatial Integrity) | 384 | 512 | [384, 512] | 512 | `LemGendizedNimaTechnical` | 512x512 | 512x512 | 512x512 |
| `nima_authenticity` | LemGendary Authenticity Scorer (AI vs Human) | `NIMA_Model` | EfficientNetV2-S (Distribution Scorer) | 256 | 768 | [256, 384, 512, 768] | 768 | `LemGendizedNimaAuthenticity` | 853x853 | 2948x1760 | 1316x1118 |
| `upn_v2` | LemGendary UPN v2 Parameter Predictor | `UPN_v2` | MobileNet-Lite (Parameter Predictor) | 128 | 256 | [128, 192, 256] | 256 | `LemGendizedUpnV2` | 2592x2000 | 3008x4256 | 2810x3381 |
| `film_restorer` | LemGendary Universal Film Restorer | `UniversalFilmRestorer` | NAFNet-Derived Base | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedFilmRestorer` | 2040x1356 | 2040x1848 | 2040x1536 |
| `codeformer` | LemGendary CodeFormer Face Restoration | `CodeFormer` | UNet (Latent Autoencoder) | 512 | 512 | [512] | 512 | `LemGendizedCodeFormer` | 1024x1024 | 1024x1024 | 1024x1024 |
| `parsenet` | LemGendary ParseNet Face Parsing | `ParseNet` | UNet (19-class Segmentation) | 512 | 512 | [512] | 512 | `LemGendizedParseNet` | 1024x1024 | 1024x1024 | 1024x1024 |
| `retinaface_mobilenet` | LemGendary RetinaFace MobileNet Detection | `RetinaFace` | MobileNet (Detection Anchors) | 640 | 640 | [640] | 640 | `LemGendizedRetinaFaceMobileNet` | 1024x1024 | 1024x1024 | 1024x1024 |
| `ffanet_indoor` | LemGendary FFANet Dehazing (Indoor) | `FFANet` | Feature Fusion Attention | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedFfaNetIndoor` | 530x399 | 530x399 | 530x399 |
| `ffanet_outdoor` | LemGendary FFANet Dehazing (Outdoor) | `FFANet` | Feature Fusion Attention | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedFfaNetOutdoor` | 1600x1200 | 1600x1200 | 1600x1200 |
| `mirnet_lowlight` | LemGendary MIRNet v2 Low-Light Enhancement | `MIRNet` | Multi-Scale Residual Block | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedMirNetLowLight` | 500x341 | 500x375 | 500x363 |
| `mirnet_exposure` | LemGendary MIRNet v2 Exposure Correction | `MIRNet` | Multi-Scale Residual Block | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedMirNetExposure` | 2592x2000 | 3008x4256 | 2810x3381 |
| `mprnet_deraining` | LemGendary MPRNet Deraining | `MPRNet` | Multi-Stage Progressive | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedMprNetDeraining` | 512x512 | 512x512 | 512x512 |
| `nafnet_debluring` | LemGendary NAFNet Debluring | `NAFNet` | Nonlinear Activation Free | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedNafNetDebluring` | 1280x720 | 1280x720 | 1280x720 |
| `nafnet_denoising` | LemGendary NAFNet Denoising | `NAFNet` | Nonlinear Activation Free | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedNafNetDenoising` | 600x464 | 600x464 | 600x464 |
| `yolov8n` | LemGendary YOLOv8n Multi-Task Model | `YOLO` | CSPDarknet53 | 320 | 640 | [320, 480, 640] | 640 | `LemGendizedYoloV8n` | 480x313 | 640x640 | 586x479 |
| `professional_multitask_restoration` | LemGendary Professional Multi-Task Restoration Model | `MultiTaskRestorer` | SharedEncoder (MoE Soft Routing) | 256 | 512 | [256, 384, 512] | 512 | `ProfessionalMultitaskRestoration` | Not Found | Not Found | Not Found |
| `ultrazoom` | LemGendary UltraZoom Master Model | `UltraZoomMaster` | NAFNet-Derived Base | 256 | 512 | [256, 384, 512] | 512 | `LemGendizedUltraZoom` | 2040x1344 | 2040x1404 | 2040x1368 |
| `universal_nsfw_classification` | LemGendary Universal NSFW Classifier | `UniversalClassifier` | MobileNetV2 (Categorical Anchor) | 224 | 224 | [224] | 224 | `ClassificationMasterManifold` | 640x480 | 800x936 | 693x672 |
