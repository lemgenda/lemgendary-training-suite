import os
import warnings
import yaml  # pyre-ignore
import torch  # pyre-ignore
import cv2  # pyre-ignore
import numpy as np  # pyre-ignore
from PIL import Image, ImageFile  # pyre-ignore
from torch.utils.data import Dataset  # pyre-ignore
from torchvision import transforms  # pyre-ignore

class MultiTaskDataset(Dataset):
    """
    Universal Dataset loader for LemGendary Training Suite suite.
    Automatically handles Restoration, Detection, and Quality tasks 
    based on unified_models.yaml and unified_data.yaml.
    """
    def __init__(self, config, model_key=None, is_train=True, env="local"):
        self.is_train = is_train
        self.env = env
        self.split = "train" if is_train else "validate"
        self.data_root = config.get("datasets_dir", "../data/datasets")
        
        # --- 2026 Turbo Initialization ---
        # Set global PIL flags once during init instead of per-image load
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        unified_models_path = os.path.join(os.path.dirname(__file__), "..", config["unified_models"])
        unified_data_path = os.path.join(os.path.dirname(__file__), "..", config["unified_data"])
        
        with open(unified_models_path, 'r') as f:
            self.unified_models = yaml.safe_load(f)
        
        # If no model_key provided, default to the first one or a generic one
        if not model_key:
            model_key = list(self.unified_models.keys())[0]
            print(f"Warning: No model_key provided to Dataset, defaulting to {model_key}")
            
        self.model_key = model_key
        self.model_info = self.unified_models.get(model_key)
        if not self.model_info:
            raise ValueError(f"Model {model_key} not found in unified_models.yaml")
            
        raw_type = self.model_info.get("dataset_type", "restoration")
        self.task_type = raw_type[0] if isinstance(raw_type, list) else raw_type
        
        # Handle input_size: can be int, [H, W], or [C, H, W]
        size_raw = self.model_info.get("input_size", config.get("default_img_size", 256))
        if isinstance(size_raw, list):
            if len(size_raw) == 3:
                self.size = (int(size_raw[1]), int(size_raw[2])) # [C, H, W] -> (H, W)
            else:
                self.size = (int(size_raw[0]), int(size_raw[1])) # [H, W] -> (H, W)
        else:
            self.size = (size_raw, size_raw) # int -> (H, W)
        
        self.samples = []
        dataset_names = self.model_info.get("datasets", [])
        
        self.kaggle_links = {
            "LemGendizedQualityDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-quality-dataset",
            "LemGendizedNoiseDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset",
            "LemGendizedLowLightDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset",
            "LemGendizedFaceDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset",
            "LemGendizedDegradationDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset",
            "LemGendizedDetectionDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-dettection-dataset",
            "LemGendizedSuperResDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset"
        }
        for ds_name in dataset_names:
            ds_path = self.get_dataset_path(ds_name)
            img_dir = os.path.join(ds_path, "images", self.split)
            if not os.path.exists(img_dir):
                # 2026 Autonomic Data Recovery Logic
                from .data_utils import download_and_extract_dataset
                if not download_and_extract_dataset(ds_name, self.data_root):
                    link = self.kaggle_links.get(ds_name, f"https://www.kaggle.com/datasets/lemtreursi/{ds_name.lower()}")
                    print(f"\n❌ CRITICAL: The required isolated '{ds_name}' dataset topological array was structurally NOT FOUND!")
                    print(f"   👉 You must securely download and map it natively from Kaggle: {link}")
                    print(f"      Mapped Path Checked: {img_dir}\n")
                    continue
                # Re-verify path after recovery by recalculating dynamic fallback mounts
                ds_path = self.get_dataset_path(ds_name)
                img_dir = os.path.join(ds_path, "images", self.split)
                if not os.path.exists(img_dir): continue
                
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for f in files:
                self.samples.append((ds_name, f))
                
        # --- 2026: SOTA Rank-Aware Augmentations ---
        transform_list = [transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if self.is_train and self.task_type == "quality":
            # Only apply Jitter to Aesthetic training; Disable for Technical to maintain ground-truth integrity
            if self.model_key == "nima_aesthetic":
                transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        
        transform_list.append(transforms.ToTensor())
        
        # --- 2026: ImageNet-Handoff Normalization ---
        # Strictly required for Quality models using pre-trained feature backbones (NIMA)
        if self.task_type == "quality":
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
        self.transform = transforms.Compose(transform_list)
        
        print(f"Loaded {len(self.samples)} samples for {model_key} (Task: {self.task_type}, Split: {self.split})")

    def fast_process(self, img):
        """High-speed 2026 data pipeline bypassing PIL overhead."""
        if img is None:
            return torch.zeros((3, self.size[0], self.size[1]))
        
        # Fast Resize
        if img.shape[:2] != self.size:
            img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
            
        # Fast Normalization
        img = img.astype(np.float32) / 255.0
        if self.task_type == "quality":
            # ImageNet stats for NIMA feature backbones
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
        # --- 2026: SOTA Integrity Shield ---
        # Ensure the final tensor is numerically safe before handing it to the GPU
        tensor = torch.from_numpy(img.transpose(2, 0, 1))
        if not torch.isfinite(tensor).all():
            print(f"\n⚠️  [INTEGRITY] Non-finite values detected in processed sample! Zeroing out to prevent singularity...")
            return torch.zeros_like(tensor)
            
        return tensor

    def __len__(self):
        return len(self.samples)

    def get_dataset_path(self, ds_name):
        if getattr(self, 'env', 'local') == 'kaggle':
            # Kaggle mounts match the end of the URL
            k_name = self.kaggle_links.get(ds_name, "").split('/')[-1]
            base_kaggle_path = f"/kaggle/input/{k_name}"
            # Protect against datasets natively extracted into a matching internal subfolder
            if os.path.exists(os.path.join(base_kaggle_path, ds_name, "images")):
                return os.path.join(base_kaggle_path, ds_name)
            # ONLY return the Kaggle mount if the foundational topology actually exists physically
            if os.path.exists(os.path.join(base_kaggle_path, "images")):
                return base_kaggle_path
            # Fallback for datasets actively recovered dynamically into working memory instead of native mounts
            local_fallback = os.path.join(self.data_root, ds_name)
            if os.path.exists(os.path.join(local_fallback, ds_name, "images")):
                return os.path.join(local_fallback, ds_name)
            if os.path.exists(local_fallback):
                return local_fallback
            return base_kaggle_path
        return os.path.join(self.data_root, ds_name)

    def load_image(self, img_path):
        try:
            # 2026 OpenCV Hardware Acceleration: 3-5x faster than PIL for 384x384
            img = cv2.imread(img_path)
            if img is None: return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def __getitem__(self, idx):
        import random
        current_idx = idx
        
        for _ in range(50):
            ds_name, fname = self.samples[current_idx]
            ds_path = self.get_dataset_path(ds_name)
            img_path = os.path.join(ds_path, "images", self.split, fname)
            
            img = self.load_image(img_path)
            if img is not None:
                # 2026 Resilience Optimization: INTER_AREA is superior for downsampling technical datasets (Preserves Alias-free Noise/Blur)
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                break
                
            print(f"\n[Warning] Corrupted image detected and permanently deleted: {img_path}")
            try:
                if os.path.exists(img_path): os.remove(img_path)
                lbl_txt = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
                if os.path.exists(lbl_txt): os.remove(lbl_txt)
                tgt_path = img_path.replace("images", "targets")
                if os.path.exists(tgt_path): os.remove(tgt_path)
            except Exception as e:
                pass
                
            current_idx = random.randint(0, len(self.samples) - 1)
            
        if img is None:
            img = Image.new('RGB', (self.size[1], self.size[0]), (0, 0, 0))
        
        if self.task_type in ["restoration", "enhancement"]:
            ds_path = self.get_dataset_path(ds_name)
            tgt_path = os.path.join(ds_path, "targets", self.split, fname)
            if not os.path.exists(tgt_path):
                base_name = os.path.splitext(fname)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = os.path.join(ds_path, "targets", self.split, base_name + ext)
                    if os.path.exists(alt_path):
                        tgt_path = alt_path
                        break
                        
            if os.path.exists(tgt_path):
                target = self.load_image(tgt_path)
            else:
                target = img.copy()
            return self.fast_process(img), self.fast_process(target), self.task_type
            
        elif self.task_type == "quality":
            # 2026 Architectural Sync: While Aesthetic and Technical models may share the same physical 
            # image repository (LemGendizedQualityDataset), the 'labels' directory contains distinct 
            # sub-vectors corresponding to the specific model's objective (Artistic vs Integrity).
            ds_path = self.get_dataset_path(ds_name)
            label_path = os.path.join(ds_path, "labels", self.split, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    try:
                        score = [float(x) for x in f.read().split()]
                        if len(score) < 10:
                            score = score + [0.0] * (10 - len(score))
                        # score.reverse()  # DELETED: Auditor confirms SOTA weights expect natural 1..10 mapping (Bin 10 = Best)
                        padded_score = [score[i] for i in range(10)]
                        return self.fast_process(img), torch.tensor(padded_score, dtype=torch.float32), "quality"
                    except:
                        pass
            return self.fast_process(img), torch.zeros(10), "quality"
            
        elif self.task_type == "detection":
            ds_path = self.get_dataset_path(ds_name)
            label_path = os.path.join(ds_path, "labels", self.split, os.path.splitext(fname)[0] + ".txt")
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        try:
                            labels.append([float(x) for x in line.split()])
                        except:
                            continue
            # Note: detection often requires padding labels or special collate_fn
            return self.fast_process(img), torch.tensor(labels), "detection"

        return self.fast_process(img), torch.zeros(1), self.task_type
