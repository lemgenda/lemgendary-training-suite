"""Generates minimal deterministic test fixtures for LemGendary Model Training Suite."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

FIXTURES_DIR = Path(__file__).resolve().parent
DATASET_DIR = FIXTURES_DIR / "tiny_dataset"
IMAGES_DIR = DATASET_DIR / "images" / "train"
TARGETS_DIR = DATASET_DIR / "targets" / "train"
LABELS_DIR = DATASET_DIR / "labels" / "train"


def generate_fixtures() -> None:
    """Creates synthetic tiny dataset samples and a minimal torch checkpoint."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    # 1. Generate 8 synthetic 64x64 images, targets, and labels
    for idx in range(8):
        name = f"sample_{idx:03d}"
        img_arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        target_arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        img = Image.fromarray(img_arr)
        target = Image.fromarray(target_arr)

        img.save(IMAGES_DIR / f"{name}.png")
        target.save(TARGETS_DIR / f"{name}.png")

        meta = {
            "index": idx,
            "filename": f"{name}.png",
            "aesthetic_score": 5.5 + (idx * 0.2),
            "class_id": idx % 2,
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
        }
        (LABELS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Dataset info manifest
    dataset_info_content = """---
name: tiny_dataset
version: "1.0"
format: directory
container:
  primary: directory
total_samples: 8
splits:
  train: 8
tasks:
  - restoration
  - quality
  - detection
"""
    (DATASET_DIR / "dataset_info.yaml").write_text(dataset_info_content, encoding="utf-8")

    # 2. Generate minimal tiny checkpoint
    model_state = {
        "weight": torch.randn(2, 2),
        "bias": torch.zeros(2),
    }
    checkpoint = {
        "epoch": 1,
        "step": 8,
        "model_state_dict": model_state,
        "optimizer_state_dict": {},
        "sota_score": 0.85,
        "architecture": "tiny_net",
    }
    torch.save(checkpoint, FIXTURES_DIR / "tiny_checkpoint.pth")


if __name__ == "__main__":
    generate_fixtures()
