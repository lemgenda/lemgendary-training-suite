"""Dynamic image restoration degradation augmentations and on-the-fly degradation engine."""

import io
import logging
import math
import random
import sys
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch

from training.utils.paths import get_workspace_root

logger = logging.getLogger("lemtrain.degrade")

# Attempt dynamic linkage with lemgendary-datasets degrade engine
_HAS_DEGRADE_CORE = False
CoreDynamicDegrader: Any = None
core_parse_profile: Any = None

try:
    import importlib
    datasets_root = get_workspace_root() / "lemgendary-datasets"
    if datasets_root.exists() and str(datasets_root) not in sys.path:
        sys.path.append(str(datasets_root))
    degrade_pkg = importlib.import_module("degrade")
    CoreDynamicDegrader = getattr(degrade_pkg, "DynamicDegrader", None)
    core_parse_profile = getattr(degrade_pkg, "parse_profile", None)
    _HAS_DEGRADE_CORE = CoreDynamicDegrader is not None and core_parse_profile is not None
except (ImportError, ModuleNotFoundError, AttributeError) as exc:
    logger.debug("Core degrade engine from lemgendary-datasets not linked: %s", exc)
    _HAS_DEGRADE_CORE = False


class JpegCompressionGuard:
    """Simulates real-world JPEG artifacts on PIL images to shield against compression overfitting."""

    def __init__(self, probability: float = 1.0) -> None:
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            quality = random.randint(65, 95)
            buffer = io.BytesIO()
            rgb_img = img if img.mode == "RGB" else img.convert("RGB")
            rgb_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img


def apply_synthetic_degradation(
    img_tensor: torch.Tensor,
    deg: float,
    theta: float,
    conf: float,
) -> torch.Tensor:
    """Apply synthetic degradation pipeline (Gaussian blur, noise, JPEG, and scale) to image tensor [C, H, W]."""
    # Convert to numpy HWC for OpenCV operations
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().copy()
    h, w = img_np.shape[:2]

    # Degradation 1: Gaussian Blur
    blur_sigma = deg * 4.0 + 0.1
    ksize = int(blur_sigma * 3) * 2 + 1
    ksize = max(3, min(ksize, 31))
    img_np = cv2.GaussianBlur(img_np, (ksize, ksize), blur_sigma)

    # Degradation 2: Additive Gaussian Noise
    noise_sigma = conf * 0.08
    if noise_sigma > 0.001:
        noise = np.random.randn(*img_np.shape).astype(np.float32) * noise_sigma
        img_np = img_np + noise

    # Degradation 3: JPEG Compression
    jpeg_quality = int(95 - (theta / math.pi) * 80)
    jpeg_quality = max(10, min(jpeg_quality, 95))
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    encoded_ok, enc = cv2.imencode(".jpg", img_uint8, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if encoded_ok:
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if decoded is not None:
            img_np = decoded.astype(np.float32) / 255.0

    # Degradation 4: Downscale + Upscale
    scale_factor = max(0.25, 1.0 - deg * conf * 0.75)
    if scale_factor < 0.95:
        small_h = max(8, int(h * scale_factor))
        small_w = max(8, int(w * scale_factor))
        img_np = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)

    img_np = np.clip(img_np, 0.0, 1.0)
    return torch.from_numpy(img_np).permute(2, 0, 1).float()


def apply_film_degradation(img_tensor: torch.Tensor) -> torch.Tensor:
    """Synthesizes vintage film degradation (sepia/grayscale, grain, defocus blur, scratches)."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().copy()

    # Sepia / Grayscale tint
    if np.random.rand() < 0.8:
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        gray_stack = np.stack([gray, gray, gray], axis=-1)
        if np.random.rand() < 0.5:
            sepia = np.zeros_like(img_np)
            sepia[..., 0] = gray_stack[..., 0] * 1.07
            sepia[..., 1] = gray_stack[..., 1] * 0.74
            sepia[..., 2] = gray_stack[..., 2] * 0.43
            img_np = sepia
        else:
            img_np = gray_stack

    # Film Grain
    noise_level = float(np.random.uniform(0.02, 0.15))
    noise = np.random.normal(0, noise_level, img_np.shape).astype(np.float32)
    img_np = img_np + noise

    # Defocus Blur
    if np.random.rand() < 0.7:
        blur_radius = float(np.random.uniform(0.5, 2.0))
        img_np = cv2.GaussianBlur(img_np, (0, 0), blur_radius)

    # Film Scratches
    if np.random.rand() < 0.6:
        h, w = img_np.shape[:2]
        num_scratches = np.random.randint(1, 5)
        for _ in range(num_scratches):
            x1 = int(np.random.randint(0, w))
            y1 = 0
            x2 = x1 + np.random.randint(-20, 20)
            y2 = h
            color = float(np.random.choice([0.0, 1.0]))
            thickness = np.random.randint(1, 3)
            cv2.line(img_np, (x1, y1), (x2, y2), (color, color, color), thickness)

    img_np = np.clip(img_np, 0.0, 1.0)
    return torch.from_numpy(img_np).permute(2, 0, 1).float()


def synthesize_degradation(target_img: Image.Image) -> Image.Image:
    """Synthesize PIL-level degradation (blur, noise, JPEG compression, median filtering)."""
    degraded = target_img.copy()
    r = random.random()
    if r < 0.25:
        return degraded.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    if r < 0.50:
        return degraded.filter(ImageFilter.MedianFilter(size=random.choice([3, 5])))
    if r < 0.75:
        buf = io.BytesIO()
        if degraded.mode != "RGB":
            degraded = degraded.convert("RGB")
        degraded.save(buf, format="JPEG", quality=random.randint(20, 60))
        buf.seek(0)
        return Image.open(buf)

    arr = np.array(degraded, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5.0, 20.0), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class DynamicOnTheFlyDegrader:
    """On-The-Fly Degradation Engine for dynamic training augmentation."""

    def __init__(self, mode: str = "motion-blur+iso-noise", intensity: str = "medium") -> None:
        self.mode = mode
        self.intensity = intensity
        self._core_degrader = None

        if _HAS_DEGRADE_CORE and callable(core_parse_profile) and callable(CoreDynamicDegrader):
            try:
                prof = core_parse_profile(mode, intensity=str(intensity))
                self._core_degrader = CoreDynamicDegrader(prof)
            except Exception as exc:
                logger.debug("Fallback to standalone degradation: %s", exc)
                self._core_degrader = None

    def __call__(
        self,
        img_tensor: torch.Tensor,
        sample_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply dynamic degradation to [C, H, W] tensor in [0, 1]."""
        if self._core_degrader is not None:
            img_hwc = img_tensor.permute(1, 2, 0).cpu().numpy()
            deg_arr, meta = self._core_degrader.degrade_array(img_hwc, sample_seed=sample_seed)
            deg_tensor = torch.from_numpy(deg_arr).permute(2, 0, 1).float()
            return deg_tensor, meta

        deg = random.uniform(0.1, 0.9)
        theta = random.uniform(0.0, math.pi)
        conf = random.uniform(0.1, 0.8)
        degraded = apply_synthetic_degradation(img_tensor, deg=deg, theta=theta, conf=conf)
        params = {
            "mode": self.mode,
            "deg_intensity": round(deg, 4),
            "theta_blend": round(theta, 4),
            "noise_conf": round(conf, 4),
        }
        return degraded, params
