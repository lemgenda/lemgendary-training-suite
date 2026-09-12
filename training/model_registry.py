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
        configured_batch = model_info.get("batch_size", "auto")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
        
        if isinstance(configured_batch, int) and configured_batch > 0:
            final_batch = configured_batch if mode == 'train' else configured_batch * 2
        else:
            if vram_gb >= 20.0:
                final_batch = 1024 if mode == 'train' else 2048
            elif vram_gb >= 14.0:
                final_batch = 512 if mode == 'train' else 1024
            elif vram_gb >= 7.0:
                final_batch = 384 if mode == 'train' else 768
            elif vram_gb >= 3.5:
                final_batch = 256 if mode == 'train' else 512
            else:
                final_batch = 64 if mode == 'train' else 128

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

        # Pre-flight CUDA sanity check (detects hardware uncorrectable ECC error and kernel image compatibility)
        try:
            _test = torch.ones(1, device=device) + 1.0
            torch.cuda.synchronize()
            del _test
        except Exception as ecc_err:
            err_str = str(ecc_err)
            if "ECC" in err_str or "uncorrectable" in err_str.lower():
                print("\n[CRITICAL ERROR] [HARDWARE SENTINEL] Uncorrectable ECC error detected on GPU!")
                print(" The accelerator memory is physically corrupted and locked by the NVIDIA driver.")
                print(" RECOMMENDATION: Restart your Kaggle session to acquire a healthy GPU node.")
                print(" Checkpoints remain intact; resume training with --model <model_name> on the new session.\n")
                sys.exit(1)
            if "no kernel image is available" in err_str or "cudaErrorNoKernelImageForDevice" in err_str:
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
                print("\n" + "=" * 80)
                print(f"[CRITICAL ERROR] [HARDWARE SENTINEL] No CUDA kernel image available for {gpu_name}!")
                print(" The active PyTorch installation lacks CUDA kernels for this GPU architecture.")
                print(" RECOMMENDATION: Switch Kaggle accelerator to 'GPU T4 x2' (sm_75 Turing) or install torch+cu118.")
                print("=" * 80 + "\n")
                sys.exit(1)
            raise ecc_err

        # Pre-probe surgical cache purge
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # 2026 Resilience: Total VRAM Discovery (incorporating PyTorch's reserved pool)
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        vram_gb = total_vram / (1024**3)

        unused_reserved = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        free_vram = free_vram + unused_reserved

        # 2026 Resilience: Paging Awareness (Shared Memory Guard)
        is_exhausted = (free_vram / total_vram) < 0.15

        # Use resolution from override or model_info
        res = res_override
        if res is None:
            res_raw = model_info.get("input_size", 224)
            res = res_raw[1] if isinstance(res_raw, (list, tuple)) else res_raw

        h = w = int(res)

        # Dynamic Headroom Tiering:
        # High resolution (>= 512px) and heavy restoration architectures require large headroom
        # for intermediate perceptual losses (LPIPS/VGG) and AdamW FP32 moment buffers.
        if h >= 512 or is_restoration:
            safety_multiplier = 0.70  # 30% free headroom safety margin
        elif h >= 384:
            safety_multiplier = 0.75  # 25% free headroom safety margin
        elif vram_gb < 4.5:
            safety_multiplier = 0.80  # 20% safety margin on 4GB cards
        else:
            safety_multiplier = 0.85  # 15% safety margin on standard workloads

        available_vram = free_vram * safety_multiplier

        # --- The Probe (v17.2) ---
        # We instantiate a single-sample manifold to measure exact activation/gradient volume
        torch.cuda.empty_cache()

        try:
            # 2026 SOTA Resilience: The "Warmup" Pass
            # The absolute FIRST forward/backward pass in PyTorch initializes massive lazy buffers (CuDNN, etc.).
            _dummy = {"pixel_values": torch.randn(1, 3, h, w).to(device)} if "diffusion" in model_key.lower() else torch.randn(1, 3, h, w).to(device)
            model.eval() if mode == 'val' else model.train()
            _out = model(_dummy)
            if mode == 'train':
                _loss = sum(v.mean() for v in _out.values()) if isinstance(_out, dict) else _out.mean()
                if isinstance(_loss, torch.Tensor): _loss.backward()
                model.zero_grad(set_to_none=True)
            del _dummy, _out
            torch.cuda.empty_cache()
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
                # Factor in perceptual loss (LPIPS/VGG) and optimizer step memory footprint
                probe_multiplier = 1.6 if is_restoration else 1.25
                sample_vram = (peak_probe - before_probe) * probe_multiplier
            else:
                with torch.no_grad():
                    _ = model(dummy_input)
                # Use peak memory to capture activation volume during forward pass
                peak_probe = torch.cuda.max_memory_allocated(0) if device.type == 'cuda' else torch.cuda.memory_allocated(0)
                val_mult = 1.6 if is_restoration else 1.15
                sample_vram = (peak_probe - before_probe) * val_mult

            del dummy_input
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if sample_vram <= 0: raise ValueError("Probe failed to measure manifold")

        except Exception as e:
            res_multiplier = (h * w) / (224 * 224)
            base_mb = 90 if is_restoration else 60
            sample_vram = base_mb * 1024 * 1024 * res_multiplier

        dynamic_batch = int(available_vram / sample_vram)

        # --- Pixel Volume Cap (v20.0 Strict) ---
        # v20.0: Validation allows for 2x pixel volume since no gradients are stored.
        val_mult = 2.0 if mode == 'val' else 1.0
        max_pixels = 12.0 * (1024**2) * val_mult
        if vram_gb < 4.5: max_pixels = 1.5 * (1024**2) * val_mult # Relaxed Sub-Nuclear 4GB Lockdown (Targets Batch 6 at 512px)
        elif vram_gb < 8.5: max_pixels = 5.0 * (1024**2) * val_mult
        elif vram_gb < 16.5: max_pixels = 10.0 * (1024**2) * val_mult # Hardened 16GB limit
        else: max_pixels = 32.0 * (1024**2)

        pixel_cap = int(max_pixels / (h * w))
        system_cap = 256 if mode == 'val' else 128
        
        # 2026 Resilience: System RAM Safeguard against Dataloader Bloat
        sys_ram_gb = 64.0
        is_kaggle = False
        try:
            import psutil
            sys_ram_gb = psutil.virtual_memory().total / (1024**3)
            is_kaggle = os.path.exists('/kaggle/working') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
            if sys_ram_gb < 35.0 or is_kaggle:
                system_cap = min(system_cap, 32 if mode == 'val' else 24)
        except Exception as e:
            print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

        # 2026 Resilience: Restoration models (like NAFNet/MIRNet) use ConvTranspose2d which has CuDNN
        # workspace overheads. Scale workspace cap dynamically based on hardware VRAM tier.
        if is_restoration:
            if mode == 'train':
                dynamic_cap_train = 8 if vram_gb < 8.0 else (16 if vram_gb < 16.5 else 32)
                system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_train", dynamic_cap_train)
            else:
                dynamic_cap_val = 4 if vram_gb < 4.5 else (8 if vram_gb < 8.5 else (16 if vram_gb < 16.5 else 32))
                system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_val", dynamic_cap_val)

        final_batch = max(1, min(dynamic_batch, pixel_cap, system_cap))

        # --- Exhaustion Emergency Clamp (v18.0) ---
        if is_exhausted and vram_gb < 6.0:
            final_batch = min(final_batch, 4) # Force-clamp to tiny batch if card is nearly full
            print(f" [WARNING] [MEMORY-SENTINEL] Dedicated VRAM exhausted ({free_vram/1e6:.1f}MB free). Hard-clamping Batch to {final_batch} to avoid Shared Memory paging.")
        elif vram_gb < 4.5 and free_vram < 500 * 1024 * 1024:
            final_batch = min(final_batch, 8)
            print(f" [WARNING] [MEMORY-SENTINEL] Low Headroom Detected ({free_vram/1e6:.1f}MB free). Clamping to {final_batch}.")

        # 2026 Resilience: CuDNN bug mitigation (Odd batch sizes crash ConvTranspose2d in DataParallel)
        if final_batch > 1 and final_batch % 2 != 0:
            final_batch -= 1

        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 1
        is_dataparallel = isinstance(model, torch.nn.DataParallel) or (hasattr(model, 'module') and gpu_count > 1)

        # 2026 Resilience: DataParallel Gathering Safeguard (v20.0)
        # Under PyTorch DataParallel, GPU 0 gathers all outputs and executes loss evaluation.
        # Linearly multiplying final_batch by gpu_count saturates GPU 0 memory at high resolutions.
        # For high resolutions (>= 512px) or restoration models, keep batch un-inflated and rely
        # on Universal Gradient Accumulation to hit target effective batch size.
        if gpu_count > 1:
            if is_dataparallel:
                if max(h, w) >= 512 or is_restoration:
                    # Keep per-device batch conservative; do not multiply by gpu_count
                    final_batch = min(final_batch, 4 if is_restoration else 8)
                else:
                    dp_factor = min(float(gpu_count), 1.5)
                    final_batch = max(1, int(final_batch * dp_factor))
                    if final_batch > 1 and final_batch % 2 != 0:
                        final_batch -= 1
            else:
                final_batch = final_batch * gpu_count

        # 2026 Resilience: Multi-GPU Host RAM Guard
        if mode == 'val' and (is_kaggle or sys_ram_gb < 35.0):
            max_host_val_batch = 12 if max(h, w) >= 512 else (16 if max(h, w) >= 384 else 24)
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
        err_msg = str(e)
        if "no kernel image is available" in err_msg or "cudaErrorNoKernelImageForDevice" in err_msg:
            print("\n" + "=" * 80)
            print(f"[CRITICAL ERROR] [HARDWARE SENTINEL] Probe aborted: CUDA error: no kernel image is available for execution on the device.")
            print(" The current PyTorch build does not support this GPU. Switch Kaggle accelerator to 'GPU T4 x2' or reinstall PyTorch with cu118.")
            print("=" * 80 + "\n")
            sys.exit(1)
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
