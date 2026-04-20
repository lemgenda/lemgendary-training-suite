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
    v5.7: Added class-level _file_cache for lightning-fast multi-worker initialization.
    """
    _file_cache = {} # Static cache to share dataset scanned state across workers

    def __init__(self, config, model_key=None, is_train=True, env="local", sample_fraction=1.0):
        self.is_train = is_train
        self.env = env
        self.sync_mode = False # --- 2026 Resiliency: Fast-Skip Sync ---
        self.split = "train" if is_train else "validate"
        self.data_root = config.get("datasets_dir", "../data/datasets")
        
        # --- 2026 Turbo Initialization ---
        # Set global PIL flags once during init instead of per-image load
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        unified_models_path = os.path.join(os.path.dirname(__file__), "..", config["unified_models"])
        
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
        self.all_samples = []
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
                
            cache_key = f"{ds_name}_{self.split}"
            if cache_key in MultiTaskDataset._file_cache:
                files = MultiTaskDataset._file_cache[cache_key]
                # print(f"✨ [CACHE] Restored {len(files)} files for {ds_name} from primary manifold memory")
            else:
                files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                MultiTaskDataset._file_cache[cache_key] = files
                
            for f in files:
                self.all_samples.append((ds_name, f))
        
        self.samples = list(self.all_samples)
        self.sample_fraction = sample_fraction
                
        # --- 2026: Mission Velocity Acceleration (Subsampling) ---
        if is_train and 0.0 < self.sample_fraction < 1.0:
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:int(len(self.samples) * self.sample_fraction)]
            print(f"🚀 [VELOCITY] Stochastic Subsampling ACTIVE: Using {len(self.samples)} samples ({self.sample_fraction*100:.1f}%)")

        self.build_transforms()
        print(f"Loaded {len(self.samples)} samples for {model_key} (Task: {self.task_type}, Split: {self.split})")

    def build_transforms(self):
        # --- 2026: SOTA Rank-Aware Augmentations ---
        transform_list = [transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if self.is_train and self.task_type == "quality":
            if self.model_key == "nima_aesthetic":
                transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        
        transform_list.append(transforms.ToTensor())
        if self.task_type == "quality":
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
        self.transform = transforms.Compose(transform_list)

    def update_strategy(self, fraction=None, size=None):
        """Autonomous recalibration of dataset parameters without disc re-scan."""
        if size is not None:
            if isinstance(size, list):
                self.size = (int(size[1]), int(size[2])) if len(size)==3 else (int(size[0]), int(size[1]))
            else:
                self.size = (size, size)
            self.build_transforms()
            
        if fraction is not None:
            self.sample_fraction = fraction
            self.samples = list(self.all_samples)
            if self.is_train:
                import random
                random.shuffle(self.samples)
                self.samples = self.samples[:int(len(self.samples) * self.sample_fraction)]
        
        print(f"📡 [DATASET RECALIBRATED] Fraction: {self.sample_fraction*100:.1f}% | Size: {self.size}")

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
            
        # 2026 Shift: Check Universal Shared Repository first
        shared_path = os.path.join(self.data_root, "_shared", ds_name)
        if os.path.exists(shared_path):
            return shared_path
            
        return os.path.join(self.data_root, ds_name)

    def load_image(self, img_path):
        try:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            # Safely handle physical file headers with Pillow, averting OpenCV C++ access violations
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                import numpy as np
                return np.array(img)
        except Exception as e:
            return None

    def __getitem__(self, idx):
        # --- 2026 Resiliency: Fast-Skip Synchronization ---
        # When synchronizing mid-epoch iterations, we bypass all I/O and processing
        # logic to instantly satisfy the DataLoader's iterator.
        if self.sync_mode:
            return torch.zeros((3, self.size[0], self.size[1])), torch.zeros(1), self.task_type

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
            ds_path = self.get_dataset_path(ds_name)
            label_path = os.path.join(ds_path, "labels", self.split, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    try:
                        # 2026 Resilience Protocol (v3.6): Source-level restoration confirmed.
                        # The disk format is now natively 1=Worst -> 10=Best.
                        score = [float(x) for x in f.read().split()]
                        if len(score) < 10:
                            score = score + [0.0] * (10 - len(score))
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
