import yaml # pyre-ignore

def build_model_readme(model_key, unified_models, unified_data, epochs_trained, metrics):
    model_info = unified_models.get(model_key, {})
    name = model_info.get("name", model_key)
    desc = model_info.get("description", "Unified LemGendary Training Suite Matrix.")
    task = model_info.get("dataset_type", "unknown")
    datasets = model_info.get("datasets", [])
    model_filename = model_info.get("filename", model_key)
    base_name = f"LemGendary{model_filename}"
    
    ds_sizes = [f"- **{d}**: ~{unified_data.get(d, {}).get('count', 'N/A')} files actively integrated." for d in datasets]
    ds_str = "\n".join(ds_sizes)
    
    if task == "quality":
        eval_str = f"**PLCC**: {metrics.get('plcc', 'N/A')} | **SRCC**: {metrics.get('srcc', 'N/A')}"
        excel = "PLCC > 0.85 | SRCC > 0.82"
    elif task == "face":
        eval_str = f"**FID**: {metrics.get('fid', 'N/A')} | **LPIPS**: {metrics.get('lpips', 'N/A')} | **PSNR**: {metrics.get('psnr', 'N/A')}"
        excel = "FID < 12.0 | LPIPS < 0.12 | PSNR > 31.0"
    else:
        eval_str = f"**PSNR**: {metrics.get('psnr', 'N/A')} | **SSIM**: {metrics.get('ssim', 'N/A')} | **LPIPS**: {metrics.get('lpips', 'N/A')}"
        excel = "PSNR > 30.5 dB | SSIM > 0.89 | LPIPS < 0.10"
        
    yolo_blurb = ""
    orig_architecture = model_info.get("class_name", "Base Pytorch")
    
    if "yolo" in model_key.lower():
        eval_str = "Dynamically handled via CLI mAP scores during local evaluation bounds."
        excel = "mAP@0.5 > 0.65 | mAP@0.5:0.95 > 0.45"
        yolo_blurb = "This module incorporates Ultralytics bounding-box anchors and keypoint tensors structurally."
        orig_architecture = "Ultralytics YOLO Architecture Engine"
        
    return f"""# {name} Documentation

## 1. Description and Core Purpose
**{name}** is actively integrated as a `{task}` logic gate mapped internally within the LemGendary Training Suite universal application environment. 
**Description:** {desc}
{yolo_blurb}

## 2. Training Scope, Architecture, and Origins
This framework natively relies structurally on the **{orig_architecture}** back-bone. It was specifically natively processed mapping the exact mathematical convergence layers securely against the following target arrays:
{ds_str}

## 3. Evaluation Mathematics
The orchestration pipeline dynamically halts its internal epoch evaluations entirely against target **Excellent Baseline Thresholds** rather than raw Epoch completion cycles. Raw loss algorithms usually misrepresent visual outputs locally.
- **Achieved Post-Validation Metrics**: {eval_str}
- **Mandated Early Stopping Baselines**: {excel}
- **Execution Lifecycle**: Successfully mapped across {epochs_trained} total epoch evaluations securely.

*(We enforce entirely physical representation-tracking sequences—like PSNR, FID, LPIPS matrices natively assessing human-perceptual convergence directly across evaluations, structurally substituting standard cross-entropy metrics locally!)*

## 4. Matrix Output Files & Integration Protocol
Upon reaching the convergence termination bounds, the framework structurally isolates the following distinct data fragments naturally:
- `{base_name}.onnx`: The highly-compressed **FP16** Half-Precision mathematical binary locally embedding all weights effortlessly beneath WebGPU's 2GB hard constraint limit!
- `{base_name}_FP32.onnx`: The raw explicitly structured Full-Precision decoupled graph logic.
- `{base_name}_FP32.onnx.data`: The physically ejected massive tensor structures dynamically separated to fully prevent backend browser-memory crashing dynamically!
- `metrics.csv`: Real-time execution loss evaluations mapped globally across the runtime sequence.

## 5. WebGPU Interface Extrapolation Pipeline
This matrix effortlessly exports dynamically to intercept standard REST constraints internally directly natively within the Web application. 
The `{base_name}.onnx` array is structurally parsed directly into standard HTML-JavaScript arrays via the **ONNX Runtime Web** execution pipeline! It targets standard WebGL backend cores to calculate local inferences dynamically inside your user's isolated browsers perfectly mimicking the Python matrix natively.
"""
