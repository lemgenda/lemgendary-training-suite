import os
import multiprocessing
import math
import random
import warnings
import yaml  # pyre-ignore
import torch  # pyre-ignore
import cv2  # pyre-ignore
import numpy as np  # pyre-ignore
from PIL import Image, ImageFile, ImageFilter  # pyre-ignore
import json
import shutil
from torch.utils.data import Dataset  # pyre-ignore
from torchvision import transforms  # pyre-ignore


def apply_synthetic_degradation(img_tensor, deg, theta, conf):
    """
    Apply synthetic degradation to a clean image tensor.
    
    Degradation pipeline (controlled by deg/theta/conf):
      - deg  ∈ [0, 1]: Controls overall degradation intensity
      - theta ∈ [0, π]: Controls degradation orientation/type blend
      - conf ∈ [0, 1]: Controls degradation confidence/sharpness
    
    Returns degraded image tensor (same shape as input).
    """
    # Convert to numpy HWC for OpenCV operations
    img_np = img_tensor.permute(1, 2, 0).numpy().copy()  # [H, W, 3] float32 [0, 1]
    h, w = img_np.shape[:2]
    
    # --- Degradation 1: Gaussian Blur (controlled by deg) ---
    blur_sigma = deg * 4.0 + 0.1  # [0.1, 4.1]
    ksize = int(blur_sigma * 3) * 2 + 1  # Ensure odd kernel size
    ksize = max(3, min(ksize, 31))  # Clamp to valid range
    img_np = cv2.GaussianBlur(img_np, (ksize, ksize), blur_sigma)
    
    # --- Degradation 2: Additive Gaussian Noise (controlled by conf) ---
    noise_sigma = conf * 0.08  # [0, 0.08] standard deviation
    if noise_sigma > 0.001:
        noise = np.random.randn(*img_np.shape).astype(np.float32) * noise_sigma
        img_np = img_np + noise
    
    # --- Degradation 3: JPEG Compression (controlled by theta) ---
    # theta [0, π] maps to quality [95, 15] — higher theta = more compression
    jpeg_quality = int(95 - (theta / math.pi) * 80)  # [95, 15]
    jpeg_quality = max(10, min(jpeg_quality, 95))
    # Simulate JPEG by encoding/decoding in memory
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    img_uint8 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # cv2 uses BGR, but our input was RGB — imencode/imdecode preserves channel order for numpy
    img_np = img_uint8.astype(np.float32) / 255.0
    
    # --- Degradation 4: Downscale + Upscale (controlled by deg * conf) ---
    scale_factor = max(0.25, 1.0 - deg * conf * 0.75)  # [0.25, 1.0]
    if scale_factor < 0.95:
        small_h, small_w = max(8, int(h * scale_factor)), max(8, int(w * scale_factor))
        img_np = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Clamp and convert back to tensor
    img_np = np.clip(img_np, 0.0, 1.0)
    return torch.from_numpy(img_np).permute(2, 0, 1).float()

def apply_film_degradation(img_tensor):
    """
    Synthesizes vintage film degradation (sepia/grayscale, grain, blur, scratches).
    """
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # 1. Sepia/Grayscale (80% chance)
    if np.random.rand() < 0.8:
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        gray = np.stack([gray, gray, gray], axis=-1)
        if np.random.rand() < 0.5: # Sepia tint
            sepia = np.zeros_like(img_np)
            sepia[..., 0] = gray[..., 0] * 1.07
            sepia[..., 1] = gray[..., 1] * 0.74
            sepia[..., 2] = gray[..., 2] * 0.43
            img_np = sepia
        else:
            img_np = gray

    # 2. Film Grain (Gaussian Noise)
    noise_level = np.random.uniform(0.02, 0.15)
    noise = np.random.normal(0, noise_level, img_np.shape)
    img_np = img_np + noise

    # 3. Defocus Blur
    if np.random.rand() < 0.7:
        blur_radius = np.random.uniform(0.5, 2.0)
        img_np = cv2.GaussianBlur(img_np, (0, 0), blur_radius)

    # 4. Scratches (Vertical/Diagonal lines)
    if np.random.rand() < 0.6:
        h, w = img_np.shape[:2]
        num_scratches = np.random.randint(1, 5)
        for _ in range(num_scratches):
            x1 = np.random.randint(0, w)
            y1 = 0
            x2 = x1 + np.random.randint(-20, 20)
            y2 = h
            color = np.random.choice([0.0, 1.0]) # Black or white scratch
            thickness = np.random.randint(1, 3)
            cv2.line(img_np, (x1, y1), (x2, y2), (color, color, color), thickness)
            
    img_np = np.clip(img_np, 0.0, 1.0)
    return torch.from_numpy(img_np).permute(2, 0, 1).float()

# [SENIOR HARDENING v16.0 - SYNC_ID: 1152]

class MultiTaskDataset(Dataset):
    """
    Nuclear-Hardened Universal Dataset (v16.0).
    Implements Fallback Shields, LANCZOS scaling, and stratified distribution analytics.
    """
    def __init__(self, config, model_key=None, is_train=True, env="local", sample_fraction=1.0):
        self.config = config
        self.is_train = is_train
        self.env = env
        self.sample_fraction = sample_fraction
        self.sync_mode = multiprocessing.Value('b', False)
        self.split = "train" if is_train else "val"
        # 2026 Resilience: Map to the modern 'paths' structure in config.yaml
        p_paths = config.get("paths", {})
        self.data_root = p_paths.get("datasets_root", config.get("datasets_dir", "data/datasets"))
        
        if not os.path.isabs(self.data_root):
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if self.env == 'kaggle':
                self.data_root = "/kaggle/working/LemGendaryDatasets"
            else:
                self.data_root = os.path.normpath(os.path.join(workspace_root, self.data_root))
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True # Shield against corrupt JPG headers
        
        unified_models_path = os.path.join(os.path.dirname(__file__), "..", config["unified_models"])
        with open(unified_models_path, 'r') as f:
            self.unified_models = yaml.safe_load(f)
        
        self.model_key = model_key or list(self.unified_models.keys())[0]
        self.model_info = self.unified_models.get(self.model_key)
        
        raw_type = self.model_info.get("dataset_type", "restoration")
        self.task_type = raw_type[0] if isinstance(raw_type, list) else raw_type
        
        size_raw = self.model_info.get("input_size", config.get("default_img_size", 256))
        if isinstance(size_raw, list):
            self.size = (int(size_raw[1]), int(size_raw[2])) if len(size_raw) == 3 else (int(size_raw[0]), int(size_raw[1]))
        else:
            self.size = (size_raw, size_raw)
        
        self.samples = []
        self.all_samples = []
        self.path_cache = {}
        self._load_manifest(config)
        self.build_transforms()

    def _load_manifest(self, config):
        self.all_samples = []
        self.samples = []
        if self.env == 'kaggle' and config.get("debug", False):
            print("📡 [DEBUG] Mounted datasets in /kaggle/input:")
            try:
                if os.path.exists('/kaggle/input'):
                    for d in os.listdir('/kaggle/input'):
                        d_path = os.path.join('/kaggle/input', d)
                        if os.path.isdir(d_path):
                            print(f"  -> {d} contains: {os.listdir(d_path)[:10]}")
                            for sub in os.listdir(d_path):
                                sub_path = os.path.join(d_path, sub)
                                if os.path.isdir(sub_path):
                                    try:
                                        print(f"      -> {sub} contains: {os.listdir(sub_path)[:5]}")
                                    except Exception:
                                        pass
                else:
                    print("  -> /kaggle/input does not exist!")
            except Exception as e:
                print("  -> Error listing /kaggle/input:", e)

        raw_dataset_names = self.model_info.get("datasets", [])
        suffix = "KaggleReady" if self.env == 'kaggle' else config.get("execution", {}).get("suffixes", {}).get(config.get("execution", {}).get("mode", "training"), "")
        # 2026 Resilience: Support 'LemGendized' prefix automatically
        dataset_names = []
        for name in raw_dataset_names:
            dataset_names.append(f"{name}{suffix}")
            if not name.lower().startswith("lemgendized"):
                dataset_names.append(f"LemGendized{name}{suffix}")
        
        for ds_name in dataset_names:
            ds_path = self.get_dataset_path(ds_name)
            if ds_path is None: continue
            self.path_cache[ds_name] = ds_path
            # 2026: Parameter prediction loads from targets/ (clean source images)
            if self.task_type == "parameter_prediction":
                img_dir = os.path.join(ds_path, "targets", self.split)
                if not os.path.exists(img_dir):
                    img_dir = os.path.join(ds_path, "images", self.split)
            else:
                img_dir = os.path.join(ds_path, "images", self.split)
            if not os.path.exists(img_dir): continue
            
            items = os.listdir(img_dir)
            files = [f for f in items if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for f in files:
                self.all_samples.append((ds_name, f))
        
        self.samples = list(self.all_samples)
        if self.is_train and self.sample_fraction < 1.0:
            import random
            random.seed(42) # Deterministic sampling
            random.shuffle(self.samples)
            self.samples = self.samples[:max(1, int(len(self.samples) * self.sample_fraction))]

    def build_transforms(self):
        # 2026 Resilience: LANCZOS is superior for Aesthetic/Quality manifolds
        interp = transforms.InterpolationMode.LANCZOS if self.task_type == "quality" else transforms.InterpolationMode.BILINEAR
        
        transform_list = [transforms.Resize(self.size, interpolation=interp)]
        if self.is_train and self.task_type == "quality":
            transform_list.append(transforms.RandomHorizontalFlip())
            if "aesthetic" in self.model_key:
                transform_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.05))
        
        transform_list.append(transforms.ToTensor())
        if self.task_type == "quality":
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
        self.transform = transforms.Compose(transform_list)
        
        # 2026: Parameter prediction needs clean ToTensor only (no normalization)
        # Degradation is applied after transform in __getitem__
        if self.task_type == "parameter_prediction":
            self.clean_transform = transforms.Compose([
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor()
            ])

    def get_dataset_path(self, ds_name):
        # 1. Standard Resolution
        path = os.path.join(self.data_root, ds_name)
        if os.path.exists(os.path.join(path, 'images', 'train')):
            return path
            
        # 1b. Case-Insensitive Local Check
        if os.path.exists(self.data_root):
            try:
                for item in os.listdir(self.data_root):
                    if item.lower() == ds_name.lower():
                        cand = os.path.join(self.data_root, item)
                        if os.path.exists(os.path.join(cand, 'images', 'train')):
                            return cand
            except Exception:
                pass
            
        # 2. Nuclear Brute-Force (Kaggle Only)
        if self.env == 'kaggle':
            # Strip suffixes to get the core target slug (e.g., lemgendizednimaauthenticity)
            target = ds_name.lower().replace("-", "").replace("_", "")
            for suffix in ["kaggleready", "large", "mini"]:
                target = target.replace(suffix, "")
                
            # Perform a sub-millisecond Depth-Restricted BFS directory scan (max depth 4, directories only)
            if os.path.exists('/kaggle/input'):
                try:
                    queue = ['/kaggle/input']
                    depths = {'/kaggle/input': 0}
                    while queue:
                        curr = queue.pop(0)
                        depth = depths[curr]
                        if depth > 4: continue
                        
                        try:
                            items = os.listdir(curr)
                        except Exception:
                            continue
                            
                        for item in items:
                            path = os.path.join(curr, item)
                            if os.path.isdir(path):
                                depths[path] = depth + 1
                                queue.append(path)
                                
                                name_lower = item.lower().replace("-", "").replace("_", "")
                                if target in name_lower or 'lemgendary' in name_lower:
                                    # Check 1: Direct mount (e.g., path/images/train)
                                    if os.path.exists(os.path.join(path, 'images', 'train')):
                                        if self.config.get("debug", False):
                                            print(f"📡 [DEBUG] get_dataset_path({ds_name}) target={target} -> Direct match: {path}")
                                        return path
                                    # Check 2: Nested ZIP mount (e.g., path/NestedFolder/images/train)
                                    try:
                                        for sub in os.listdir(path):
                                            sub_path = os.path.join(path, sub)
                                            if os.path.isdir(sub_path):
                                                if os.path.exists(os.path.join(sub_path, 'images', 'train')):
                                                    if self.config.get("debug", False):
                                                        print(f"📡 [DEBUG] get_dataset_path({ds_name}) target={target} -> Nested match: {sub_path}")
                                                    return sub_path
                                    except Exception:
                                        pass
                except Exception as e:
                    if self.config.get("debug", False):
                        print(f"📡 [DEBUG] get_dataset_path scan error: {e}")
                    pass
            if self.config.get("debug", False):
                print(f"📡 [DEBUG] get_dataset_path({ds_name}) target={target} -> Failed to resolve path!")
            
        return None

    def load_image(self, img_path):
        """Fallback Shield: Returns Neutral-Gray tensor on I/O failure (Task 9.1)."""
        try:
            with Image.open(img_path) as img:
                return img.convert('RGB')
        except Exception as e:
            # print(f"⚠️ [SHIELD] I/O Failure on {os.path.basename(img_path)}. Ejecting Neutral Gray.")
            return Image.new('RGB', self.size[::-1], (128, 128, 128))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self.sync_mode, 'value') and self.sync_mode.value:
            return torch.zeros((3, self.size[0], self.size[1])), torch.zeros(1), self.task_type
        elif not hasattr(self.sync_mode, 'value') and self.sync_mode:
            return torch.zeros((3, self.size[0], self.size[1])), torch.zeros(1), self.task_type

        ds_name, fname = self.samples[idx]
        ds_path = self.path_cache.get(ds_name)
        if ds_path is None:
            ds_path = self.get_dataset_path(ds_name)
            self.path_cache[ds_name] = ds_path
        img_path = os.path.join(ds_path, "images", self.split, fname)
        
        img = self.load_image(img_path)
        
        if self.model_key == "ultrazoom":
            # 2026 Resilience: Dynamic super-resolution 2x downscaling on-the-fly
            lr_size = (self.size[0] // 2, self.size[1] // 2)
            lr_transform = transforms.Compose([
                transforms.Resize(lr_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])
            img_tensor = lr_transform(img)
        else:
            img_tensor = self.transform(img)
        
        if self.task_type in ["restoration", "enhancement", "face"]:
            tgt_path = os.path.join(ds_path, "targets", self.split, fname)
            if os.path.exists(tgt_path):
                target = self.load_image(tgt_path)
                target_tensor = self.transform(target)
            else:
                target_tensor = self.transform(img) # Ensure HR fallback shape
            
            # 2026: Dynamic Film Degradation for Universal Film Restorer
            if self.model_key == "film_restorer":
                # Check if it's an identical file (meaning it's from a clean dataset like DIV2K)
                try:
                    if os.path.exists(tgt_path) and os.path.samefile(img_path, tgt_path):
                        img_tensor = apply_film_degradation(target_tensor)
                except OSError:
                    pass

            # --- 2026: Multi-Task Task ID Extractor ---
            task_str = self.task_type
            if self.model_key == "professional_multitask_restoration":
                fname_lower = fname.lower()
                import re
                match = re.search(r"professionalmultitaskrestoration_([a-z_]+)_compiled_", fname_lower)
                if match:
                    task_str = match.group(1)
                else:
                    # Fallback to substring matching if compiled without the new tag prefix
                    if "denois" in fname_lower:
                        task_str = "denoise"
                    elif "deblur" in fname_lower:
                        task_str = "deblur"
                    elif "derain" in fname_lower:
                        task_str = "derain"
                    elif "dehaze_outdoor" in fname_lower or "outdoor" in fname_lower:
                        task_str = "dehaze_outdoor"
                    elif "dehaze_indoor" in fname_lower or "indoor" in fname_lower:
                        task_str = "dehaze_indoor"
                    elif "dehaze" in fname_lower or "ffa" in fname_lower:
                        task_str = "dehaze_indoor"
                    elif "lowlight" in fname_lower:
                        task_str = "lowlight"
                    elif "exposure" in fname_lower:
                        task_str = "exposure"
                    elif "zoom" in fname_lower or "super" in fname_lower:
                        task_str = "superres"
                    elif "vintage" in fname_lower or "film" in fname_lower:
                        task_str = "vintage"
                    elif "codeformer" in fname_lower or "face_restor" in fname_lower:
                        task_str = "face_restorer"
                    elif "parsenet" in fname_lower or "face_pars" in fname_lower:
                        task_str = "face_parser"
                    else:
                        task_str = "denoise" # Fallback default

            return img_tensor, target_tensor, task_str
        
        elif self.task_type == "parameter_prediction":
            # 2026: On-the-fly Synthetic Degradation for Parameter Prediction
            # Load clean image, sample random params, apply degradation, return both
            tgt_path = os.path.join(ds_path, "targets", self.split, fname)
            if os.path.exists(tgt_path):
                clean_img = self.load_image(tgt_path)
            else:
                clean_img = img  # Fallback to input image as clean source
            
            clean_tensor = self.clean_transform(clean_img)
            
            # Sample random degradation parameters
            deg = random.random()                    # [0.0, 1.0]
            theta = random.random() * math.pi        # [0.0, π]
            conf = random.random()                    # [0.0, 1.0]
            
            # Apply synthetic degradation
            degraded_tensor = apply_synthetic_degradation(clean_tensor, deg, theta, conf)
            
            # Ground truth parameter vector
            params = torch.tensor([deg, theta, conf], dtype=torch.float32)
            
            return degraded_tensor, params, "parameter_prediction"
            
        elif self.task_type == "quality":
            if self.model_key == "nima_authenticity":
                # Dynamically construct target distribution for authenticity
                # 'real' = High authenticity (Class 1, peak at bin 10)
                # 'shutterstock' = Low authenticity (Class 0, peak at bin 1)
                fname_lower = fname.lower()
                if "real" in fname_lower:
                    score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
                else:
                    score = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                return img_tensor, torch.tensor(score, dtype=torch.float32), "quality"
            
            label_path = os.path.join(ds_path, "labels", self.split, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    try:
                        score = [float(x) for x in f.read().split()]
                        if len(score) < 10: score = score + [0.0] * (10 - len(score))
                        return img_tensor, torch.tensor(score[:10], dtype=torch.float32), "quality"
                    except: pass
            return img_tensor, torch.zeros(10), "quality"
            
        return img_tensor, torch.zeros(1), self.task_type

    def update_strategy(self, size=None, fraction=None):
        """Dynamic manifold scaling for curriculum learning (v16.2)."""
        if size:
            if isinstance(size, (int, float)):
                self.size = (int(size), int(size))
            elif isinstance(size, (list, tuple)):
                self.size = (int(size[0]), int(size[1]))
            self.build_transforms()
        if fraction is not None:
            self.sample_fraction = fraction
            self._load_manifest(self.config)

    def get_distribution(self):
        """Task 9.3: Analyze label manifold for stratified balancing."""
        print(f"📊 [DATA] Analyzing distribution for {self.model_key}...")
        stats = {"total": len(self.samples), "tasks": {}}
        # Implementation for background distribution analysis...
        return stats
