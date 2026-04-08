import os
import sys
import argparse
import yaml
import torch
import time

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
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts for automated 2026 pipelines")
    args = parser.parse_args()

    print(f"\n🚀 Initializing On-Demand SOTA ONNX Exporter for model: {args.model}")
    
    # 1. Environment Discovery (Hierarchical Path Resolution)
    config_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Error: config.yaml not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    unified_models_name = config.get("unified_models", "unified_models.yaml")
    unified_models_path = os.path.join(project_root, unified_models_name)
    if not os.path.exists(unified_models_path):
        print(f"❌ Error: {unified_models_path} not found.")
        return
        
    with open(unified_models_path, 'r') as f:
        unified_models_registry = yaml.safe_load(f)

    model_info = unified_models_registry.get(args.model)
    if not model_info:
        print(f"❌ Error: Model '{args.model}' not found in registry.")
        return

    # 2. Architecture Instantiation
    from models.factory import get_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 [ARCH] Instantiating architecture for {args.model} on {device}...")
    try:
        model = get_model(args.model, config).to(device)
    except Exception as e:
        print(f"❌ Error during instantiation: {e}")
        return

    # 3. Checkpoint Discovery
    ckpt_dir_rel = config.get("checkpoint_dir", "trained-models/checkpoints")
    ckpt_dir = os.path.normpath(os.path.join(project_root, ckpt_dir_rel))
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: SOTA Checkpoint not found at {ckpt_path}")
        return

    print(f"📖 [LOAD] Extracting weights from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"❌ Error during load: {e}")
        return
    
    model.eval()

    # 4. Production Synchronization
    model_filename = model_info.get("filename", args.model)
    base_name = f"LemGendary{model_filename}"
    production_dir_rel = os.path.join(config.get("export_dir", "trained-models"), args.model)
    production_dir = os.path.normpath(os.path.join(project_root, production_dir_rel))
    os.makedirs(production_dir, exist_ok=True)
    
    size_raw = model_info.get("input_size", config.get("default_img_size", 256))
    if isinstance(size_raw, list):
        h, w = (int(size_raw[1]), int(size_raw[2])) if len(size_raw)==3 else (int(size_raw[0]), int(size_raw[1]))
    else:
        h, w = int(size_raw), int(size_raw)
        
    dummy_input = torch.randn(1, 3, h, w).to(device)

    # 5. Export Matrix (FP32 & FP16)
    exports = [
        {"name": f"{base_name}_FP32.onnx", "half": False},
        {"name": f"{base_name}.onnx", "half": True}
    ]

    for export in exports:
        target_path = os.path.join(production_dir, export["name"])
        
        # Overwrite Guardrail
        if os.path.exists(target_path):
            if args.yes:
                print(f"   -> [OVERWRITE] Non-interactive bypass active for {export['name']}.")
            else:
                print(f"\n⚠️  [WARNING] Production artifact '{target_path}' already exists.")
                ans = input(f"👉 Do you want to OVERWRITE this ONNX model? [y/N]: ")
                if ans.lower().strip() != 'y':
                    print(f"⏭  Skipping {export['name']}...")
                    continue

        print(f"✨ [EXPORT] Synthesizing {'FP16' if export['half'] else 'FP32'} ONNX model to {export['name']}...")
        try:
            if export["half"]:
                # --- 2026 Resilience: Self-Contained FP16 Calibration ---
                model.half()
                inp = dummy_input.half()
                # FP16 models are standalone and embedded for WebGPU performance
                save_ext = False
            else:
                model.float()
                inp = dummy_input.float()
                # FP32 models use sidecar weighting as requested
                save_ext = True
                
            torch.onnx.export(
                model, inp, target_path,
                export_params=True, opset_version=17,
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
            )

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
                    print("   ⚠️  [WARNING] 'onnx' package missing. FP32 weights remain embedded.")
            else:
                print(f"   -> FP16 Weights physically EMBEDDED for standalone WebGPU deployment.")
                # --- 2026 Resilience: C++ API Ghost Purge ---
                # The PyTorch Legacy C++ exporter fallback sometimes forcefully ignores physical
                # embedding rules and spits out a sidecar. We surgically sever it here.
                ghost_data_loc = f"{export['name']}.data"
                ghost_abs_path = os.path.join(production_dir, ghost_data_loc)
                if os.path.exists(ghost_abs_path):
                    os.remove(ghost_abs_path)
                    print(f"   -> [PURGE] PyTorch C++ Fallback ghost sidecar {ghost_data_loc} physically severed.")

            print(f"✅ [SUCCESS] {export['name']} generated.")
        except Exception as e:
            print(f"❌ Error during ONNX export for {export['name']}: {e}")

    print("\n🏁 Export Suite Mission Complete.")

if __name__ == "__main__":
    main()
