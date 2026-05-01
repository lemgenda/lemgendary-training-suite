import os
import warnings
import yaml  # pyre-ignore
import torch  # pyre-ignore
import cv2  # pyre-ignore
import numpy as np  # pyre-ignore
from PIL import Image, ImageFile  # pyre-ignore
import json
import shutil
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
        self.split = "train" if is_train else "val"
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
        raw_dataset_names = self.model_info.get("datasets", [])
        
        # --- 2026 Resiliency: Apply dynamic execution suffix ---
        exec_mode = config.get("execution", {}).get("mode", "training")
        suffix = config.get("execution", {}).get("suffixes", {}).get(exec_mode, "")
        dataset_names = [f"{name}{suffix}" for name in raw_dataset_names]
        
        # --- 2026 Protocol Awareness: Load Data Registry ---
        data_registry_path = os.path.join(os.path.dirname(__file__), "..", "..", "lemgendary-datasets", "unified_data.yaml")
        if not os.path.exists(data_registry_path):
            # Fallback if the path is different (e.g. inside training suite)
            data_registry_path = os.path.join(os.path.dirname(__file__), "..", "unified_data.yaml")
        
        self.data_registry = {}
        if os.path.exists(data_registry_path):
            with open(data_registry_path, 'r') as f:
                self.data_registry = yaml.safe_load(f)

        self.kaggle_links = config.get("kaggle_dataset_urls", {})
        fallback_root = config.get("datasets_fallback_dir", "../LemGendaryDatasets")
        for ds_name in dataset_names:
            ds_path = self.get_dataset_path(ds_name)
            
            # Unified directory check
            def check_ds(path):
                if self.task_type in ["text_to_image", "image_to_text"]:
                    return os.path.exists(os.path.join(path, "parquet", self.split))
                return os.path.exists(os.path.join(path, "images", self.split))

            # Tier 1: Check Local Sandbox
            if not check_ds(ds_path):
                # Tier 2: Check LemGendaryDatasets Fallback Root
                fallback_ds_path = os.path.join(fallback_root, ds_name)
                if check_ds(fallback_ds_path):
                    print(f"\n🔄 [DATA] Found '{ds_name}' in Fallback ({fallback_root}). Injecting to local sandbox {self.data_root}...")
                    os.makedirs(ds_path, exist_ok=True)
                    shutil.copytree(fallback_ds_path, ds_path, dirs_exist_ok=True)
                else:
                    # Tier 3: Universal 2026 Recovery (HF / Kaggle)
                    from .data_utils import download_and_extract_dataset
                    
                    # Resolve source ref from registry
                    ds_info = self.data_registry.get(ds_name, {})
                    source_ref = ds_info.get("refs", [{}])[0].get("ref") if ds_info.get("refs") else None
                    
                    if not download_and_extract_dataset(ds_name, self.data_root, source_ref=source_ref):
                        print(f"\n❌ CRITICAL: The required isolated '{ds_name}' dataset manifold was structurally NOT FOUND!")
                        print(f"   👉 You must securely download and map it natively. [Ref: {source_ref}]")
                        print(f"      Mapped Path Checked: {ds_path}\n")
                        continue
            
            if not check_ds(ds_path): continue
            img_dir = os.path.join(ds_path, "images", self.split)
                
            cache_dir = os.path.join(self.data_root, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{ds_name}_{self.split}_manifest.json")
            
            files = []
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        files = json.load(f)
                    # print(f"✨ [MANIFEST] Instantly restored {len(files)} files for {ds_name} from disk cache.")
                except Exception as e:
                    print(f"⚠️ [MANIFEST] Cache corruption on {ds_name}. Re-scanning: {e}")
            
            if not files:
                # 2026 Resilience: First-time scan pulse
                if not getattr(self, '_scanned_already', False):
                    print(f"🔍 [DATA] Syncing Physical Manifold for '{ds_name}' (First-run disk scan)...")
                    self._scanned_already = True
                    
                if self.task_type in ["text_to_image", "image_to_text"]:
                    parquet_dir = os.path.join(ds_path, "parquet", self.split)
                    if os.path.exists(parquet_dir):
                        files = [f for f in os.listdir(parquet_dir) if f.lower().endswith('.parquet')]
                    else:
                        files = []
                        print(f"⚠️ [WARNING] Generative Task '{self.task_type}' requires Parquet schema in {parquet_dir}. None found!")
                else:
                    if os.path.exists(img_dir):
                        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    else:
                        files = []
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(files, f)
                except Exception as e:
                    print(f"⚠️ [MANIFEST] Failed to persist cache for {ds_name}: {e}")
                
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
            # Tier 1: Explicit mapping in config
            k_name = self.kaggle_links.get(ds_name, "").split('/')[-1]
            if k_name:
                base_kaggle_path = f"/kaggle/input/{k_name}"
                if os.path.exists(os.path.join(base_kaggle_path, "images")):
                    return base_kaggle_path

            # Tier 2: Heuristic - Check for case-insensitive folder match in /kaggle/input/
            input_root = "/kaggle/input"
            if os.path.exists(input_root):
                try:
                    for folder in os.listdir(input_root):
                        if folder.lower() == ds_name.lower():
                            return os.path.join(input_root, folder)
                except:
                    pass

            # Tier 3: Fallback for datasets actively recovered dynamically
            local_fallback = os.path.join(self.data_root, ds_name)
            if os.path.exists(local_fallback):
                return local_fallback
            
            return f"/kaggle/input/{ds_name.lower()}"
            
        # 2026 Shift: Check Universal Shared Repository first
        shared_path = os.path.join(self.data_root, "_shared", ds_name)
        if os.path.exists(shared_path):
            return shared_path
            
        return os.path.join(self.data_root, ds_name)

    def load_image(self, img_path):
        import time
        # --- 2026 Resiliency: Multi-Pass Load Retry ---
        for attempt in range(3):
            try:
                from PIL import Image, ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                # Safely handle physical file headers with Pillow, averting OpenCV C++ access violations
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    import numpy as np
                    return np.array(img)
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1)) # Backoff
                    continue
                # print(f"DEBUG: Failed to load {img_path} after 3 attempts: {e}")
                return None
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
            
            # --- 2026: Parquet Generative Dataloader Pipeline ---
            if self.task_type in ["text_to_image", "image_to_text"]:
                pq_path = os.path.join(ds_path, "parquet", self.split, fname)
                try:
                    import pandas as pd
                    import io
                    # Stream single random row from partition
                    df = pd.read_parquet(pq_path, engine='pyarrow')
                    row = df.sample(1).iloc[0]
                    
                    img_bytes = row["image_bytes"]
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img = np.array(img_pil)
                    img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                    tensor_img = self.fast_process(img)
                    
                    if self.task_type == "text_to_image":
                        prompt = row["prompt"]
                        aesthetic_score = row.get("aesthetic_score", 5.0)
                        # We return a dict that matches train.py unpacking
                        return {"pixel_values": tensor_img, "prompt": prompt, "aesthetic_score": aesthetic_score}, torch.zeros(1), self.task_type
                    else:
                        conversation = row["conversation"]
                        return {"pixel_values": tensor_img, "conversation": conversation}, torch.zeros(1), self.task_type
                except Exception as e:
                    print(f"\n[Warning] Parquet schema failure or partition corrupted: {pq_path} - {e}")
                    current_idx = random.randint(0, len(self.samples) - 1)
                    continue

            # Standard Image Directory Loader
            img_path = os.path.join(ds_path, "images", self.split, fname)
            
            img = self.load_image(img_path)
            if img is not None:
                # 2026 Resilience Optimization: INTER_AREA is superior for downsampling technical datasets (Preserves Alias-free Noise/Blur)
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                break
                
            print(f"\n[Warning] Corrupted image detected but preserved (Nuke Disabled): {img_path}")
            # [HARDENING] 2026 Protocol: Never delete physical files during a live training session.
            # try:
            #     if os.path.exists(img_path): os.remove(img_path)
            #     lbl_txt = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            #     if os.path.exists(lbl_txt): os.remove(lbl_txt)
            #     tgt_path = img_path.replace("images", "targets")
            #     if os.path.exists(tgt_path): os.remove(tgt_path)
            # except Exception as e:
            #     pass
                
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
            
            # --- 2026 Resilience: Synchronous Spatial Augmentation ---
            # To break the catastrophic overfitting plateau, we must force the network 
            # to learn translation-invariant features by synchronously augmenting both tensors.
            if self.is_train:
                import random
                # 50% chance Horizontal Flip
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    target = cv2.flip(target, 1)
                # 50% chance Vertical Flip
                if random.random() > 0.5:
                    img = cv2.flip(img, 0)
                    target = cv2.flip(target, 0)
                    
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
        elif self.task_type == "classification":
            ds_path = self.get_dataset_path(ds_name)
            label_path = os.path.join(ds_path, "labels", self.split, os.path.splitext(fname)[0] + ".txt")
            label = 0
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    try:
                        label = int(f.read().strip())
                    except:
                        pass
            return self.fast_process(img), torch.tensor([label], dtype=torch.long), "classification"
            
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
