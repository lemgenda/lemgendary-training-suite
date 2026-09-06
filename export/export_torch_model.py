"""LemGendary SOTA Exporter: Checkpoint to Standalone PyTorch."""

import os
import argparse
import torch
from export.export_common import (
    init_export_environment,
    load_export_configs,
    resolve_export_paths,
    resolve_checkpoint_file,
    build_export_model,
    wrap_quality_model,
)

project_root = init_export_environment()


def main():
    parser = argparse.ArgumentParser(description="LemGendary SOTA Exporter: Checkpoint to Standalone PyTorch")
    parser.add_argument("--model", type=str, required=True, help="Model key from unified_models.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to specific .pth checkpoint to export")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts for automated pipelines")
    args = parser.parse_args()

    print(f"\nInitializing On-Demand SOTA PT Exporter for model: {args.model}")

    config, registry = load_export_configs(project_root)
    if not config or not registry:
        return

    model_info = registry.get(args.model)
    if not model_info:
        print(f"Error: Model '{args.model}' not found in registry.")
        print("[REMEDY] Verify the spelling of the model in unified_models.yaml.")
        return

    model = build_export_model(args.model, config, torch.device("cpu"))
    if model is None:
        return

    base_name, production_dir = resolve_export_paths(args.model, model_info, config, project_root)
    ckpt_path = resolve_checkpoint_file(args.model, production_dir, config, project_root, args.checkpoint)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f" Error: SOTA Checkpoint not found for {args.model}")
        return

    print(f" [LOAD] Extracting weights from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
            print(f"   -> Epoch: {ckpt.get('epoch', 'N/A')} | Quality: {ckpt.get('sota_metric', 'N/A')}")
        else:
            model.load_state_dict(ckpt)
            print("   -> Successfully loaded raw weights.")
    except Exception as e:
        print(f" Error: Failed to load state dictionary: {e}")
        return

    model.eval()
    model = wrap_quality_model(model, model_info)

    target_path = os.path.join(production_dir, f"{base_name}.pt")
    if os.path.exists(target_path):
        if args.yes:
            print(f"   -> [OVERWRITE] Non-interactive bypass active for {base_name}.pt.")
        else:
            print(f"\n  [WARNING] Production artifact '{target_path}' already exists.")
            ans = input("  Do you want to OVERWRITE this standalone model? [y/N]: ")
            if ans.lower().strip() != "y":
                print("  Export aborted by user.")
                return

    print(f" [EXPORT] Saving standalone PyTorch model object to {target_path}...")
    try:
        raw_model = model.model if hasattr(model, "model") and isinstance(model.model, torch.nn.Module) else model
        save_obj = {
            "model_state": raw_model.state_dict(),
            "model": raw_model,
        }
        torch.save(save_obj, target_path)
        print(" [SUCCESS] Standalone SOTA model is now production-ready.")
        print(f"   -> Usage: model = torch.load('{target_path}')")
    except Exception as e:
        print(f" Error during export: {e}")


if __name__ == "__main__":
    main()
