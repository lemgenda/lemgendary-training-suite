import torch
import os
import sys

from training.hardware.probe import audit_hardware_vram




def find_paths_pruned(root_path, target_sub, max_depth=8, is_dir=False):
    """
    2026 Resilience: Fast, depth-restricted BFS filesystem query that strictly prunes
    out massive dataset/image folders to prevent FUSE/network deadlocks.
    """
    if not os.path.exists(root_path):
        return []

    prune_dirs = {"datasets", "images", "train", "val", "test", "validation", "dataset", "val_images", "train_images"}
    results = []
    queue = [(root_path, 0)]

    while queue:
        curr, depth = queue.pop(0)
        if depth > max_depth:
            continue

        try:
            items = os.listdir(curr)
        except:
            continue

        for item in items:
            path = os.path.join(curr, item)
            item_lower = item.lower()

            # Prune massive dataset subfolders instantly
            if item_lower in prune_dirs:
                continue

            if os.path.isdir(path):
                queue.append((path, depth + 1))
                if is_dir and target_sub in item_lower:
                    results.append(path)
            elif not is_dir:
                # File matching (e.g. *.pth or metrics.csv)
                if target_sub.startswith("*"):
                    ext = target_sub.replace("*", "").lower()
                    if item_lower.endswith(ext):
                        results.append(path)
                elif target_sub.lower() in item_lower:
                    results.append(path)

    return results



def load_state_dict_robust(model, state_dict, strict=True):
    """Loads a state dict dynamically handling DataParallel 'module.' prefix mismatches."""
    is_model_dp = hasattr(model, 'module')
    is_state_dict_dp = any(k.startswith('module.') for k in state_dict.keys())

    new_state_dict = {}
    for k, v in state_dict.items():
        if is_model_dp and not is_state_dict_dp:
            new_key = 'module.' + k
        elif not is_model_dp and is_state_dict_dp:
            new_key = k[7:] if k.startswith('module.') else k
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=strict)