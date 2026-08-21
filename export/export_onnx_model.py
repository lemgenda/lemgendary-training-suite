import os
import sys
import argparse
import yaml
import torch
import time

# --- 2026 Unicode Windows Patch ---
# Force stdout/stderr to UTF-8 for clean cross-platform logging
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- 2026 Hardware Acceleration & Stability Patch ---
# Anchor the search path to the parent directory to allow root module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Increase recursion limit for exceptionally deep architectures (NIMA/Restorers)
sys.setrecursionlimit(2000)

def main():
    parser = argparse.ArgumentParser(description="LemGendary SOTA Exporter: Checkpoint to FP32/FP16 ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model key from unified_models.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to specific .pth checkpoint to export")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts for automated 2026 pipelines")
    args = parser.parse_args()

    print(f"\nInitializing On-Demand SOTA ONNX Exporter for model: {args.model}")
    
    # 1. Environment Discovery (Hierarchical Path Resolution)
    config_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        print(f" Error: config.yaml not found at {config_path}")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    unified_models_name = config.get("unified_models", "unified_models_v2.yaml")
    unified_models_path = os.path.join(project_root, unified_models_name)
    if not os.path.exists(unified_models_path):
        unified_models_path = os.path.join(project_root, "unified_models.yaml")
    if not os.path.exists(unified_models_path):
        print(f" Error: Unified models YAML not found.")
        return
        
    with open(unified_models_path, 'r', encoding='utf-8') as f:
        unified_models_registry = yaml.safe_load(f)

    model_info = unified_models_registry.get(args.model)
    if not model_info:
        print(f" Error: Model '{args.model}' not found in registry.")
        return

    # 2. Architecture Instantiation
    from models.factory import get_model
    # --- 2026 Resilience Patch ---
    # Force CPU for export subprocesses to prevent CUDA OOM when train.py holds VRAM
    device = torch.device("cpu")
    print(f" [ARCH] Instantiating architecture for {args.model} on {device}...")
    try:
        model = get_model(args.model, config).to(device)
    except Exception as e:
        print(f" Error during instantiation: {e}")
        return

    # 3. Checkpoint Discovery
    if args.checkpoint:
        ckpt_path = os.path.normpath(args.checkpoint)
    else:
        ckpt_dir_rel = config.get("checkpoint_dir", "checkpoints")
        ckpt_dir = os.path.normpath(os.path.join(project_root, ckpt_dir_rel))
        ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    
    if not os.path.exists(ckpt_path):
        print(f" Error: SOTA Checkpoint not found at {ckpt_path}")
        return

    print("\n [DEPLOY] Synchronizing SOTA models to Production Hub...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'], strict=False)
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f" Error during load: {e}")
        return
    
    model.eval()

    # 4. Production Synchronization
    model_filename = model_info.get("filename", args.model)
    base_name = f"LemGendary{model_filename}"
    production_dir_rel = os.path.join(config.get("export_dir", "../LemGendaryModels"), args.model)
    production_dir = os.path.normpath(os.path.join(project_root, production_dir_rel))
    os.makedirs(production_dir, exist_ok=True)

    if model_info.get("dataset_type") == "forex" or "forex" in args.model.lower():
        from export.mt5_signal import export_onnx
        ckpt_path = os.path.join(production_dir, "checkpoints", f"{args.model}_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(production_dir, "checkpoints", f"{args.model}_latest.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.normpath(os.path.join(project_root, "checkpoints", f"{args.model}_best.pth"))
        out_onnx = os.path.join(production_dir, f"{base_name}.onnx")
        if os.path.exists(ckpt_path):
            active_tfs = model_info.get("kwargs", {}).get("active_timeframes", [1, 5, 15, 60, 240, 1440])
            export_onnx(ckpt_path, out_onnx, active_timeframes=active_tfs)
        else:
            print(f" [WARNING] [EXPORT] No checkpoint found for {args.model} at {ckpt_path}.")
        return

    size_raw = model_info.get("input_size", config.get("default_img_size", 256))
    if size_raw is None:
        size_raw = 256
    if isinstance(size_raw, list):
        h, w = (int(size_raw[1]), int(size_raw[2])) if len(size_raw)==3 else (int(size_raw[0]), int(size_raw[1]))
    else:
        h, w = int(size_raw), int(size_raw)

    # --- 2026 Resilience: Cap ONNX Export Resolution ---
    h = min(h, 512)
    w = min(w, 512)
    
    # 3.5 Production Wrapping (Probabilities for Quality Manifolds)
    if model_info.get("dataset_type") == "quality":
        import torch.nn as nn
        class SoftmaxWrapper(nn.Module):
            def __init__(self, inner_model, temperature=1.0):
                super().__init__()
                self.inner_model = inner_model
                self.temperature = temperature
            def forward(self, x):
                logits = self.inner_model(x)
                return torch.nn.functional.softmax(logits / self.temperature, dim=1)
        
        # 2026 Resilience: Use temperature from model if anchored, else config default
        temp = getattr(model, "softmax_temp", torch.tensor(1.0)).item()
        if temp == 1.0:
            stab = model_info.get("stabilizers", {})
            temp = stab.get("softmax_temp", 1.0)
        
        print(f" [WRAP] Applying production Softmax wrapper with Temp={temp}")
        model = SoftmaxWrapper(model, temperature=temp)
        model.eval()

    dummy_input = torch.randn(1, 3, h, w).to(device)

    # 5. Export Matrix (FP32 & FP16)
    exports = [
        {"name": f"{base_name}_FP32.onnx", "half": False},
        {"name": f"{base_name}.onnx", "half": True}
    ]

    for export in exports:
        target_path = os.path.join(production_dir, str(export["name"]))
        
        # Overwrite Guardrail
        if os.path.exists(target_path):
            if args.yes:
                print(f"   -> [OVERWRITE] Non-interactive bypass active for {export['name']}.")
            else:
                print(f"\n [WARNING] Production artifact '{target_path}' already exists.")
                ans = input(f" Do you want to OVERWRITE this ONNX model? [y/N]: ")
                if ans.lower().strip() != 'y':
                    print(f" Skipping {export['name']}...")
                    continue

        print(f" [EXPORT] Synthesizing {'FP16' if export['half'] else 'FP32'} ONNX model to {export['name']}...")
        try:
            post_convert_fp16 = False
            if export["half"]:
                # --- 2026 Resilience: Self-Contained FP16 Calibration ---
                # Export to FP32 first to avoid PyTorch CPU Float16 trace limitations (e.g. ReflectionPad2d)
                # then convert directly using onnx tools.
                model.float()
                inp = dummy_input.float()
                # FP16 models are standalone and embedded for WebGPU performance
                save_ext = False
                post_convert_fp16 = True
            else:
                model.float()
                inp = dummy_input.float()
                # FP32 models use sidecar weighting as requested
                save_ext = True
                
            import logging
            import warnings
            
            onnx_logger = logging.getLogger("onnxscript")
            torch_logger = logging.getLogger("torch.onnx")
            old_onnx_level = onnx_logger.level
            old_torch_level = torch_logger.level
            onnx_logger.setLevel(logging.CRITICAL)
            torch_logger.setLevel(logging.CRITICAL)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.onnx.export(
                        model, (inp,), target_path,
                        export_params=True, opset_version=17,
                        do_constant_folding=True,
                        input_names=['input'], output_names=['output']
                    )
            except Exception as e17:
                onnx_logger.setLevel(old_onnx_level)
                torch_logger.setLevel(old_torch_level)
                print(f"   [RECOVER] Opset 17 export failed ({e17}). Escalating to Opset 18...")
                onnx_logger.setLevel(logging.CRITICAL)
                torch_logger.setLevel(logging.CRITICAL)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.onnx.export(
                        model, (inp,), target_path,
                        export_params=True, opset_version=18,
                        do_constant_folding=True,
                        input_names=['input'], output_names=['output']
                    )
            finally:
                onnx_logger.setLevel(old_onnx_level)
                torch_logger.setLevel(old_torch_level)

            # --- 2026 Resilience: Native FP16 Conversion ---
            if post_convert_fp16:
                try:
                    import onnx
                    import warnings
                    from onnxconverter_common import float16
                    print(f"   -> [CONVERSION] Converting synthesized FP32 graph to pure FP16 matrix...")
                    print(f"   -> [NOTICE] Exporting ONNX model to FP16 will safely truncate incredibly small FP32 values.")
                    onnx_model = onnx.load(target_path)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
                        
                    onnx.save(onnx_model_fp16, target_path)
                except Exception as e:
                    print(f"   [ERROR] Failed to convert ONNX to FP16: {e}")

            # Manual Weight Ejection for FP32 (External Data)
            if save_ext:
                try:
                    import onnx
                    onnx_model = onnx.load(target_path)
                    
                    # --- 2026 Resilience: SOTA Graph Sanitization ---
                    # 1. Shape Inference to normalize the graph
                    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                    
                    # 2. Pruning Orphaned Initializers (8x Bloat Fix)
                    # We only retain initializers that are physically referenced in the node inputs
                    referenced_initializers = set()
                    for node in onnx_model.graph.node:
                        for input_name in node.input:
                            referenced_initializers.add(input_name)
                    
                    # Filter the initializers list
                    initializers = [init for init in onnx_model.graph.initializer if init.name in referenced_initializers]
                    
                    # Clear and rebuild the initializer list
                    onnx_model.graph.ClearField("initializer")
                    onnx_model.graph.initializer.extend(initializers)
                    
                    data_loc = f"{export['name']}.data"
                    data_abs_path = os.path.join(production_dir, data_loc)
                    
                    # --- 2026 SOTA Resilience: Clean-Slate Synthesis (Additive Bloat Fix) ---
                    # We must delete the old .data file before saving, otherwise it may append
                    if os.path.exists(data_abs_path):
                        os.remove(data_abs_path)
                        print(f"   -> [CLEAN] Orphaned sidecar {data_loc} purged.")
                        
                    onnx.save_model(onnx_model, target_path, save_as_external_data=True, all_tensors_to_one_file=True, location=data_loc, size_threshold=1024)
                    print(f"   -> FP32 Weight Tensors sanitized and decoupled to {data_loc}")
                except ImportError:
                    print("   [WARNING] 'onnx' package missing. FP32 weights remain embedded.")
            else:
                print(f"   -> FP16 Weights physically EMBEDDED for standalone WebGPU deployment.")
                # --- 2026 Resilience: C++ API Ghost Purge ---
                # The PyTorch Legacy C++ exporter fallback sometimes forcefully ignores physical
                # embedding rules and spits out a sidecar. We surgically sever it here.
                ghost_data_loc = f"{export['name']}.data"
                ghost_abs_path = os.path.join(production_dir, ghost_data_loc)
                if os.path.exists(ghost_abs_path):
                    print(f"   -> [RECOVER] PyTorch C++ Fallback generated a sidecar. Re-embedding FP16 weights...")
                    try:
                        import onnx
                        onnx_model = onnx.load(target_path, load_external_data=True)
                        onnx.save_model(onnx_model, target_path, save_as_external_data=False)
                        os.remove(ghost_abs_path)
                        print(f"   -> [SUCCESS] FP16 sidecar successfully embedded and purged.")
                    except Exception as embed_err:
                        print(f"   -> [ERROR] Failed to embed sidecar: {embed_err}")

            print(f" [SUCCESS] {export['name']} generated.")
        except Exception as e:
            print(f" Error during ONNX export for {export['name']}: {e}")

    print("\n Export Suite Mission Complete.")

if __name__ == "__main__":
    main()
