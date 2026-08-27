import os
import yaml
import csv
import sys
from datetime import datetime

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from training.telemetry import TelemetryEngine

def generate_hub_readme(project_root=workspace_root):
    yaml_path = os.path.join(project_root, "unified_models_v2.yaml")
    hub_dir = os.path.join(os.path.dirname(project_root), "LemGendaryModels")
    readme_path = os.path.join(hub_dir, "README.md")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        unified_models = yaml.safe_load(f)
        
    engine = TelemetryEngine(".", ".")
    
    completed = []
    in_progress = []
    not_started = []
    
    for model_key, model_info in unified_models.items():
        if model_key == "_registry_metadata":
            continue
            
        model_name = model_info.get("name", model_key)
        task_type = model_info.get("dataset_type", "restoration")
        sota_targets = model_info.get("sota_targets", {})
        
        # Max resolution
        max_res = model_info.get("val_resolution", 0)
        if not max_res:
            opt = model_info.get("optimization", {})
            ladder = opt.get("res_ladder", [])
            if ladder: max_res = max(ladder)
        if not max_res: max_res = 256 # Fallback
        
        model_dir = os.path.join(hub_dir, model_key)
        model_readme = os.path.join(model_dir, "README.md")
        metrics_csv = os.path.join(model_dir, "metrics.csv")
        
        # 1. Parse Metrics Progress
        epoch = 0
        current_data = 1.0
        current_res = max_res
        current_qs = 0.0
        
        if os.path.exists(metrics_csv):
            try:
                with open(metrics_csv, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                    if reader:
                        last_row = reader[-1]
                        epoch_val = last_row.get("Epoch")
                        epoch = int(epoch_val) if epoch_val is not None else 0
                        data_val = last_row.get("Data")
                        current_data = float(data_val) if data_val is not None else 1.0
                        res_val = last_row.get("Res")
                        current_res = float(res_val) if res_val is not None else float(current_res)
                        qs_val = last_row.get("Quality_Score")
                        current_qs = float(qs_val) if qs_val is not None else 0.0
            except: pass
            
        completeness = 0.0
        if epoch > 1:
            data_pct = min(1.0, current_data)
            res_pct = min(1.0, current_res / max_res) if max_res > 0 else 1.0
            
            target_qs = 0.0
            if sota_targets:
                target_qs, _ = engine.compute_quality_score(sota_targets, sota_targets, task_type)
                
            if target_qs > 0:
                clamped_cqs = max(0.0, current_qs)
                qs_pct = min(1.0, clamped_cqs / target_qs)
            else:
                qs_pct = min(1.0, epoch / 100.0) # Fallback heuristic
                
            completeness = (data_pct + res_pct + qs_pct) / 3.0 * 100.0
            
        # 2. Check if SOTA achieved
        is_sota = (completeness >= 99.99)
        
        if is_sota:
            completed.append(model_name)
        elif epoch > 1:
            in_progress.append((model_name, completeness, epoch, current_data, current_res))
        else:
            not_started.append(model_name)
            
    # Build Markdown Dashboard
    lines = [
        "# LemGendary AI Training Matrix",
        "",
        f"Auto-generated live dashboard | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Completed (SOTA Targets Achieved)",
        "",
        "Models that have fully satisfied their strict target benchmarks and exported their architecture for production deployment.",
        ""
    ]
    
    if completed:
        for m in sorted(completed):
            lines.append(f"- {m} ![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen)")
    else:
        lines.append("- *No models have achieved SOTA targets yet.*")
        
    lines.extend([
        "",
        "## In Progress (Training)",
        "",
        "Models actively scaling the resolution ladder and optimizing weights across dataset manifolds.",
        ""
    ])
    
    if in_progress:
        lines.append("| Model | Completeness | Epochs | Data Fraction | Resolution |")
        lines.append("| --- | --- | --- | --- | --- |")
        # Sort descending by completeness
        in_progress.sort(key=lambda x: x[1], reverse=True)
        for m, pct, ep, df, res in in_progress:
            lines.append(f"| {m} | **{pct:.1f}%** | {ep} | {df*100:.0f}% | {int(res)}px |")
    else:
        lines.append("- *No models currently in active training.*")
        
    lines.extend([
        "",
        "## Not Started",
        "",
        "Registered matrix targets awaiting cluster allocation.",
        ""
    ])
    
    if not_started:
        for m in sorted(not_started):
            lines.append(f"- {m}")
    else:
        lines.append("- *All models have been deployed.*")
        
    lines.append("")
    
    # Save
    os.makedirs(hub_dir, exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"[HUB SYNC] Dashboard successfully updated at: {readme_path}")

if __name__ == "__main__":
    generate_hub_readme()
