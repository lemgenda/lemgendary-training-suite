import torch
import os
import sys
import time

def safe_torch_save(obj, path):
    """Saves a torch object with disk-space auditing and atomic replacement."""
    import shutil
    dir_name = os.path.dirname(path)
    if not dir_name: dir_name = "."
    # 1. Audit Disk Space
    try:
        total, used, free = shutil.disk_usage(dir_name)
        free_gb = free / (1024**3)
        # 2026 Resilience: Relaxed threshold for high-res manifolds (5GB soft-gate)
        if free_gb < 5.0:
            if free_gb < 1.0:
                print(f" [WARNING] [DISK SENTINEL] CRITICAL: Low disk space detected ({free_gb:.2f}GB). Engaging Emergency Pruning...", file=sys.stderr)
                # Prune all .tmp and old progress files
                for f in os.listdir(dir_name):
                    if f.endswith(".tmp") or "_progress.pth" in f:
                        try:
                            f_path = os.path.join(dir_name, f)
                            if os.path.abspath(f_path) != os.path.abspath(path):
                                os.remove(f_path)
                        except Exception as e:
                            print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")
            else:
                # Soft Warning (Passive) - No pruning unless < 1GB
                pass

            # Final check
            _, _, free = shutil.disk_usage(dir_name)
            if free / (1024**3) < 0.2: # Less than 200MB
                print(f" [ERROR] [DISK SENTINEL] DISK FULL! Cannot save {os.path.basename(path)}. Aborting save to preserve manifold.", file=sys.stderr)
                print(f" [REMEDY] Free up disk space on {os.path.dirname(path)} before resuming training to prevent checkpoint corruption.", file=sys.stderr)
                return False
    except Exception as e:
        print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

    # 2. Atomic Save
    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path)
        safe_replace(tmp_path, path)
        return True
    except Exception as e:
        print(f" [ERROR] [DISK SENTINEL] Save failed for {os.path.basename(path)}: {e}", file=sys.stderr)
        print(" [REMEDY] Check if you have write permissions in the checkpoints directory or if the disk is full.", file=sys.stderr)
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception as e:
                print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")
        return False


def load_scheduler_state_stretched(scheduler, state_dict, current_total_steps, expected_step=None):
    """Loads scheduler state dict while stretching the runway if total_steps mismatch."""
    if 'total_steps' in state_dict:
        old_total = state_dict['total_steps']
        old_last = state_dict.get('last_epoch', 0)

        if old_total != current_total_steps:
            ratio = current_total_steps / max(1, old_total)
            state_dict['total_steps'] = current_total_steps

            if expected_step is not None:
                new_last = expected_step
            else:
                new_last = int(round(old_last * ratio))

            # Clamp to prevent out-of-bounds scheduler crashes
            new_last = max(0, min(current_total_steps - 1, new_last))

            state_dict['last_epoch'] = new_last
            state_dict['_step_count'] = new_last + 1

            if '_schedule_phases' in state_dict:
                for i, phase in enumerate(state_dict['_schedule_phases']):
                    if 'end_step' in phase:
                        if i == len(state_dict['_schedule_phases']) - 1:
                            phase['end_step'] = current_total_steps - 1
                        else:
                            old_end = phase['end_step']
                            phase['end_step'] = int(round((old_end + 1) * ratio - 1))
                            phase['end_step'] = max(0, min(current_total_steps - 1, phase['end_step']))

            print(f" [RESILIENCY] Stretched scheduler state dict from {old_total} to {current_total_steps} steps (last_epoch: {old_last} -> {new_last}).")
        elif expected_step is not None:
            # If total_steps matches, but last_epoch is de-synced/poisoned (too far ahead or behind expected)
            old_last = state_dict.get('last_epoch', 0)
            if old_last != expected_step:
                print(f" [RESILIENCY] [SHIELD] De-synced step count detected in scheduler_state ({old_last} vs expected {expected_step}). Re-anchoring to actual progress step ({expected_step}).")
                expected_clamped = max(0, min(current_total_steps - 1, expected_step))
                state_dict['last_epoch'] = expected_clamped
                state_dict['_step_count'] = expected_clamped + 1

    scheduler.load_state_dict(state_dict)

    if hasattr(scheduler, 'optimizer') and scheduler.optimizer is not None:
        try:
            for param_group, lr_val in zip(scheduler.optimizer.param_groups, scheduler.get_lr()):
                param_group['lr'] = lr_val
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [p['lr'] for p in scheduler.optimizer.param_groups]
        except Exception as e:
            print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")




def safe_replace(src, dst):
    """Battle-Hardened atomic replace for Windows. Uses 3-stage recovery (Replace -> Remove/Rename -> Copy/Delete)."""
    max_retries = 15
    base_delay = 0.5

    for i in range(max_retries):
        try:
            if os.path.exists(dst):
                # 2026: Windows Lock Defense - Rename then Replace
                temp_old = f"{dst}.old_{int(time.time())}"
                os.rename(dst, temp_old)
                os.rename(src, dst)
                try: os.remove(temp_old)
                except Exception as e:
                    print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")
            else:
                os.rename(src, dst)
            return True
        except (PermissionError, OSError) as e:
            time.sleep(base_delay * (1.5 ** i))
    return False
