from __future__ import annotations
import argparse
import sys
import torch
import torch.nn as nn
from training.parallel import available_strategies, resolve_auto


def _cmd_list(_args):
    print("Available parallel strategies:")
    for name in available_strategies():
        print(f"  - {name}")
    return 0


def _cmd_resolve(args):
    import os
    import yaml

    # Load model_info from registry if available, so preferred_parallel is shown.
    model_info: dict = {}
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        registry_path = os.path.join(root, "unified_models_v2.yaml")
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}
            model_info = registry.get(args.model, {})
    except Exception:
        pass

    stub = nn.Linear(1, 1)
    if args.devices is not None:
        orig = torch.cuda.device_count
        torch.cuda.device_count = lambda: args.devices
        try:
            choice = resolve_auto(stub, args.model, {}, model_info=model_info)
        finally:
            torch.cuda.device_count = orig
    else:
        choice = resolve_auto(stub, args.model, {}, model_info=model_info)
    preferred = model_info.get("preferred_parallel", "(auto)")
    print(f"model         = {args.model}")
    print(f"devices       = {args.devices if args.devices is not None else torch.cuda.device_count()}")
    print(f"cuda          = {torch.cuda.is_available()}")
    print(f"preferred     = {preferred}")
    print(f"resolved      = {choice}")
    return 0


def main():
    p = argparse.ArgumentParser(prog="python -m training.parallel")
    sub = p.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("list")
    pl.set_defaults(func=_cmd_list)
    pr = sub.add_parser("resolve")
    pr.add_argument("--model", required=True)
    pr.add_argument("--devices", type=int, default=None)
    pr.set_defaults(func=_cmd_resolve)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())