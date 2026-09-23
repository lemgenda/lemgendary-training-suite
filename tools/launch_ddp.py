"""LemGendary Model Training Suite - DDP Launch Wrapper.

Wraps ``torchrun`` to launch distributed training across all available GPUs
on a single node. All standard training arguments are forwarded transparently
to ``training/train.py``.

Usage (local, 2 GPUs):
    python tools/launch_ddp.py --model nima_aesthetic_mobile --epochs 100

Usage (Kaggle dual-T4):
    python tools/launch_ddp.py --model film_restorer --env kaggle --epochs 300

Note: yolov8n uses the Ultralytics native trainer which manages its own DDP
internally. Running this launcher with --model yolov8n is not required and will
be redirected automatically.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def _find_free_port(start: int = 29500, attempts: int = 100) -> int:
    """Find an available TCP port for the DDP master process."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start  # Last resort: use the default and let torchrun handle conflicts.


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/launch_ddp.py",
        description=(
            "LemGendary DDP Launcher: wraps torchrun to distribute training "
            "across all available GPUs on this node."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All unrecognized arguments are forwarded to training/train.py.\n"
            "Example:\n"
            "  python tools/launch_ddp.py --model nima_aesthetic_mobile --epochs 100 --env kaggle\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="professional_multitask_restoration",
        help="Model key from unified_models_v2.yaml (forwarded to train.py).",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="Master address for process group. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Master port for process group. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="Override GPU count. Defaults to torch.cuda.device_count().",
    )
    return parser


def main() -> None:
    # Split args: known (launcher) vs forwarded (train.py).
    launcher_parser = _build_arg_parser()
    launcher_args, train_argv = launcher_parser.parse_known_args()

    # Resolve the project root relative to this script.
    project_root = Path(__file__).resolve().parent.parent
    train_script = project_root / "training" / "train.py"
    if not train_script.exists():
        print(f"[ERROR] Training script not found at: {train_script}", file=sys.stderr)
        sys.exit(1)

    # GPU count resolution.
    try:
        import torch
        detected_gpus = torch.cuda.device_count()
    except ImportError:
        detected_gpus = 0

    nproc = launcher_args.nproc_per_node if launcher_args.nproc_per_node is not None else detected_gpus

    if nproc <= 0:
        print(
            "[WARNING] No CUDA GPUs detected. Falling back to --parallel single on a single process.",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(train_script),
            "--model", launcher_args.model,
            "--parallel", "single",
        ] + train_argv
        sys.exit(subprocess.call(cmd))

    if nproc == 1:
        print(
            "[WARNING] Only 1 GPU detected. DDP with 1 rank has no benefit. "
            "Launching as single-process (--parallel single).",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(train_script),
            "--model", launcher_args.model,
            "--parallel", "single",
        ] + train_argv
        sys.exit(subprocess.call(cmd))

    # Resolve master address and port.
    master_addr = launcher_args.master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = launcher_args.master_port or _find_free_port()

    # Effective batch size note for operator visibility.
    effective_batch_note = ""
    for i, arg in enumerate(train_argv):
        if arg == "--batch_size" and i + 1 < len(train_argv):
            try:
                per_proc_bs = int(train_argv[i + 1])
                effective_batch_note = f" (effective batch = {per_proc_bs * nproc} across {nproc} GPUs)"
            except ValueError:
                pass

    print("=" * 70, flush=True)
    print(" LemGendary DDP Launcher", flush=True)
    print(f"   GPUs          : {nproc}", flush=True)
    print(f"   Master addr   : {master_addr}:{master_port}", flush=True)
    print(f"   Model         : {launcher_args.model}", flush=True)
    if effective_batch_note:
        print(f"   Batch         : {effective_batch_note.strip()}", flush=True)
    print("=" * 70, flush=True)

    # Build the torchrun command.
    torchrun_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        str(train_script),
        "--model", launcher_args.model,
        "--parallel", "ddp",
    ] + train_argv

    # Ensure PYTHONPATH includes project root so imports resolve under torchrun.
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    project_root_str = str(project_root)
    if project_root_str not in pythonpath:
        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}".rstrip(os.pathsep)

    result = subprocess.call(torchrun_cmd, env=env)
    sys.exit(result)


if __name__ == "__main__":
    main()
