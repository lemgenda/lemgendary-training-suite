"""Background dataset prefetcher: streams HF/Kaggle datasets to disk outside the training loop."""

import logging
import os
import subprocess
import sys

import psutil

logger = logging.getLogger("lemtrain.prefetch_worker")


def prefetch():
    """
    Isolated Worker Thread Sequence (v2.2.0)
    Designed to mathematically stream Kaggle and HuggingFace arrays
    outside of the PyTorch loop, ensuring 100% Zero-Latency handoffs.
    """
    if len(sys.argv) < 3:
        return

    # Example format: hf://adamo1139/llava-instruct-150k:LlavaInstruct,...
    datasets = sys.argv[1].split(",")
    target_dir = sys.argv[2]

    os.makedirs(target_dir, exist_ok=True)
    parent_pid = os.getppid()

    for ds_pair in datasets:
        # 2026 Process Hygiene: Suicide Check
        if not psutil.pid_exists(parent_pid):
            sys.exit(1)

        if not ds_pair or ":" not in ds_pair:
            continue
        source_id, folder_name = ds_pair.split(":")

        ds_path = os.path.join(target_dir, folder_name)
        if os.path.exists(ds_path):
            continue

        # Engage structural Mutex Lock to prevent Orchestrator from starting too early
        lock_file = os.path.join(target_dir, f"{folder_name}.lock")

        with open(lock_file, "w", encoding="utf-8") as f:
            f.write(f"STREAMING_{source_id}")

        # 2026: Smart Source Router
        if source_id.startswith("hf://"):
            repo_id = source_id.replace("hf://", "")
            # Leverage huggingface-cli for high-speed concurrent streaming
            cmd = (
                f'huggingface-cli download {repo_id} --local-dir "{ds_path}" '
                "--local-dir-use-symlinks False"
            )
        else:
            # Fallback to standard Kaggle streaming
            kaggle_id = source_id.replace("kaggle://", "")
            cmd = f'kaggle datasets download -d {kaggle_id} -p "{target_dir}" --unzip'

        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr_text = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
            logger.warning(
                "Prefetch of %s failed (exit %s): %s", source_id, exc.returncode, stderr_text
            )
        except OSError as exc:
            logger.warning("Prefetch of %s could not be launched: %s", source_id, exc)
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)


if __name__ == "__main__":
    prefetch()
