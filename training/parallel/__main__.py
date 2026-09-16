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
    stub = nn.Linear(1, 1)
    if args.devices is not None:
        orig = torch.cuda.device_count
        torch.cuda.device_count = lambda: args.devices
        try:
            choice = resolve_auto(stub, args.model, {})
        finally:
            torch.cuda.device_count = orig
    else:
        choice = resolve_auto(stub, args.model, {})
    print(f"model         = {args.model}")
    print(f"devices       = {args.devices if args.devices is not None else torch.cuda.device_count()}")
    print(f"cuda          = {torch.cuda.is_available()}")
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