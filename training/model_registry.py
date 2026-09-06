import torch
import os
import sys

def audit_hardware_vram(model_key, model_info, config, device, model, res_override=None, mode='train', sample_fraction=1.0, fold=None, pairs=None):
    """
    2026 Memory-Sentinel: Atomic Hardware Probe (v17.0 Nuclear).
    Performs a real-world VRAM test at the specified resolution to find the
    absolute physical limit of the current GPU.
    """
    if model_info.get("dataset_type") == "forex" or "forex" in model_key.lower():
        configured_batch = model_info.get("batch_size", 64)
        if isinstance(configured_batch, str) and configured_batch.lower() == "auto":
            final_batch = 128 if mode == 'val' else 64
        else:
            final_batch = int(configured_batch) if configured_batch is not None else 64
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
        symbols_str = " | ".join(pairs) if pairs else "ALL"
        fold_str = fold if fold else "MAIN"
        print(f"[SIGNAL] [MEMORY-SENTINEL] {gpu_name} ({vram_gb:.1f}GB) | Phase: {mode.capitalize()} | Batch: {final_batch} | Fold: {fold_str} | Symbols: {symbols_str}")
        return final_batch

    # 2026 Resilience: Check for restoration models early to adjust VRAM margins and capabilities globally.
    is_restoration = any(x in model_key.lower() for x in ["nafnet", "mprnet", "mirnet", "ffanet", "codeformer", "film_restorer", "parsenet"])
    try:
        if device.type != 'cuda':
            fallback_val = config.get("defaults", {}).get("batch_size", 16)
            if isinstance(fallback_val, str) and fallback_val.lower() == "auto":
                return 16
            return int(fallback_val) if fallback_val is not None else 16

        # 2026 Resilience: Total VRAM Discovery (incorporating PyTorch's reserved pool)
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        vram_gb = total_vram / (1024**3)

        if device.type == 'cuda':
            unused_reserved = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            free_vram = free_vram + unused_reserved

        # 2026 Resilience: Paging Awareness (Shared Memory Guard)
        # If free VRAM is critically low (< 15% of total), we must assume
        # we are already paging into slow Shared Memory.
        is_exhausted = (free_vram / total_vram) < 0.15

        # Relaxed safety buffer: 4GB cards need all the VRAM they can get
        safety_multiplier = 0.90 if vram_gb < 4.5 else 0.95
        available_vram = free_vram * safety_multiplier

        # Use resolution from override or model_info
        res = res_override
        if res is None:
            res_raw = model_info.get("input_size", 224)
            res = res_raw[1] if isinstance(res_raw, (list, tuple)) else res_raw

        h = w = int(res)

        # --- The Probe (v17.2) ---
        # We instantiate a single-sample manifold to measure exact activation/gradient volume
        torch.cuda.empty_cache()

        try:
            # 2026 SOTA Resilience: The "Warmup" Pass
            # The absolute FIRST forward/backward pass in PyTorch initializes massive lazy buffers (CuDNN, etc.).
            # We must run a warmup pass FIRST so these lazy allocations don't inflate our peak memory reading!
            _dummy = {"pixel_values": torch.randn(1, 3, h, w).to(device)} if "diffusion" in model_key.lower() else torch.randn(1, 3, h, w).to(device)
            model.eval() if mode == 'val' else model.train()
            _out = model(_dummy)
            if mode == 'train':
                _loss = sum(v.mean() for v in _out.values()) if isinstance(_out, dict) else _out.mean()
                if isinstance(_loss, torch.Tensor): _loss.backward()
                model.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(0)
        before_probe = torch.cuda.memory_allocated(0)

        try:
            # 2026: Deterministic dummy input matching architecture requirements
            if "diffusion" in model_key.lower():
                dummy_input = {"pixel_values": torch.randn(1, 3, h, w).to(device)}
            else:
                dummy_input = torch.randn(1, 3, h, w).to(device)

            model.eval()
            if mode == 'train':
                model.train()
                output = model(dummy_input)
                if isinstance(output, dict):
                    loss = sum(v.mean() for v in output.values() if isinstance(v, torch.Tensor))
                else:
                    loss = output.mean()
                if isinstance(loss, torch.Tensor): loss.backward()
                # Use peak memory to capture activation volume during backward pass
                peak_probe = torch.cuda.max_memory_allocated(0) if device.type == 'cuda' else torch.cuda.memory_allocated(0)
                # Subtracting before_probe leaves peak activation + gradients. We apply a 1.2x multiplier for optimizer step spikes.
                sample_vram = (peak_probe - before_probe) * 1.2
            else:
                with torch.no_grad():
                    _ = model(dummy_input)
                # Use peak memory to capture activation volume during forward pass
                peak_probe = torch.cuda.max_memory_allocated(0) if device.type == 'cuda' else torch.cuda.memory_allocated(0)
                val_mult = 1.5 if is_restoration else 1.1
                sample_vram = (peak_probe - before_probe) * val_mult

            torch.cuda.empty_cache()
            if sample_vram <= 0: raise ValueError("Probe failed to measure manifold")

        except Exception as e:
            res_multiplier = (h * w) / (224 * 224)
            sample_vram = 60 * 1024 * 1024 * res_multiplier

        dynamic_batch = int(available_vram / sample_vram)

        # --- Pixel Volume Cap (v20.0 Strict) ---
        # v20.0: Validation allows for 2x pixel volume since no gradients are stored.
        val_mult = 2.0 if mode == 'val' else 1.0
        max_pixels = 12.0 * (1024**2) * val_mult
        if vram_gb < 4.5: max_pixels = 1.5 * (1024**2) * val_mult # Relaxed Sub-Nuclear 4GB Lockdown (Targets Batch 6 at 512px)
        elif vram_gb < 8.5: max_pixels = 5.0 * (1024**2) * val_mult
        elif vram_gb < 16.5: max_pixels = 12.0 * (1024**2) * val_mult
        else: max_pixels = 36.0 * (1024**2)

        pixel_cap = int(max_pixels / (h * w))
        system_cap = 256 if mode == 'val' else 128
        
        # 2026 Resilience: System RAM Safeguard against Dataloader Bloat
        # Kaggle instances have 30GB RAM. With multiple workers and large val batches, this spikes.
        sys_ram_gb = 64.0
        is_kaggle = False
        try:
            import psutil
            import os
            sys_ram_gb = psutil.virtual_memory().total / (1024**3)
            is_kaggle = os.path.exists('/kaggle/working') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
            # Containerized envs often misreport physical host RAM. Force clamp on Kaggle.
            if sys_ram_gb < 35.0 or is_kaggle:
                system_cap = min(system_cap, 32 if mode == 'val' else 24)
        except Exception as e:
            print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

        # 2026 Resilience: Restoration models (like NAFNet/MIRNet) use ConvTranspose2d which has CuDNN
        # workspace overheads. Scale workspace cap dynamically based on hardware VRAM tier.
        if is_restoration:
            if mode == 'train':
                dynamic_cap_train = 16 if vram_gb < 8.0 else (32 if vram_gb < 16.0 else 64)
                system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_train", dynamic_cap_train)
            else:
                dynamic_cap_val = 4 if vram_gb < 4.5 else (12 if vram_gb < 8.5 else (24 if vram_gb < 16.5 else 48))
                system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_val", dynamic_cap_val)

        # 2026: Diagnostic Telemetry (v18.7)
        if vram_gb < 4.5:
            pass # Silenced [SENTINEL-DEBUG]

        final_batch = max(1, min(dynamic_batch, pixel_cap, system_cap))

        # --- Exhaustion Emergency Clamp (v18.0) ---
        if is_exhausted and vram_gb < 6.0:
            final_batch = min(final_batch, 4) # Force-clamp to tiny batch if card is nearly full
            print(f" [WARNING] [MEMORY-SENTINEL] Dedicated VRAM exhausted ({free_vram/1e6:.1f}MB free). Hard-clamping Batch to {final_batch} to avoid Shared Memory paging.")
        elif vram_gb < 4.5 and free_vram < 500 * 1024 * 1024:
            # v19.0: Secondary safety clamp for 4GB cards with low headroom
            final_batch = min(final_batch, 8)
            print(f" [WARNING] [MEMORY-SENTINEL] Low Headroom Detected ({free_vram/1e6:.1f}MB free). Clamping to {final_batch}.")

        # 2026 Resilience: CuDNN bug mitigation (Odd batch sizes crash ConvTranspose2d in DataParallel)
        if final_batch > 1 and final_batch % 2 != 0:
            final_batch -= 1

        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 1
        if gpu_count > 1:
            final_batch = final_batch * gpu_count

        # 2026 Resilience: Multi-GPU Host RAM Guard
        # Multi-GPU scales VRAM capacity but shares the exact same host system RAM (30GB on Kaggle).
        # We must clamp total validation batch across all GPUs to prevent host RAM exhaustion.
        if mode == 'val' and (is_kaggle or sys_ram_gb < 35.0):
            max_host_val_batch = 16 if max(h, w) >= 512 else (24 if max(h, w) >= 384 else 32)
            final_batch = min(final_batch, max_host_val_batch)

        # --- 2026: Hardware Bottleneck Cloud Recommendation ---
        if final_batch <= 1 and vram_gb < 4.5:
            print("\n================================================================================")
            print(" [CRITICAL WARNING] HARDWARE BOTTLENECK REACHED")
            print(f" Your {gpu_name} ({vram_gb:.1f}GB) is physically struggling to train at {h}x{w}px.")

            if dynamic_batch < 1:
                print(" The required memory for a single image exceeds your available VRAM.")
                print(" RECOMMENDATION: Switch to CLOUD TRAINING (Kaggle/Colab) for this resolution.")
                print("================================================================================\n")
                sys.exit(1)
            else:
                print(" The Governor has dynamically forced Batch Size to 1 to prevent an Out-Of-Memory crash.")
                print(" If training becomes unstable at this resolution, the hardware limit is reached.")
                print(" RECOMMENDATION: Switch to CLOUD TRAINING (Kaggle/Colab) for higher resolutions.")
                print("================================================================================\n")

        if mode == 'train':
            print(f"[SIGNAL] [MEMORY-SENTINEL] {gpu_name} ({vram_gb:.1f}GB) | {mode.capitalize()} @ {h}px | Batch: {final_batch} (Pixels: {(h*w*final_batch)/1e6:.1f}M) | Dataset Fraction: {sample_fraction*100:.1f}%")
        else:
            is_quality = model_info.get("dataset_type") == "quality"
            # 2026 Resilience: Let the Governor report the actual final shard limit to prevent misleading 100% logs
            shard_str = f"100% (Quality)" if is_quality else "Governor Managed"
            print(f"[SIGNAL] [MEMORY-SENTINEL] {gpu_name} ({vram_gb:.1f}GB) | {mode.capitalize()} @ {h}px | Batch: {final_batch} (Pixels: {(h*w*final_batch)/1e6:.1f}M) | Dataset Fraction: {sample_fraction*100:.1f}% (Eval Shard: {shard_str})")
        return final_batch
    except Exception as e:
        print(f"[WARNING] [MEMORY-SENTINEL] Probe critical failure: {e}. Defaulting to safe baseline.")
        return 1



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
