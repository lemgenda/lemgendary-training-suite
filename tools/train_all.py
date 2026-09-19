"""Global LemGendary Fleet Orchestrator.

Orchestrates sequential model training runs across registered phases and manifolds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, List, TypedDict
import yaml

from training.utils.paths import get_project_root

try:
    import kagglesdk.kaggle_env as ke
    if not hasattr(ke, "get_web_endpoint"):
        def get_web_endpoint(env: Any) -> str:
            endpoint = ke.get_endpoint(env) if hasattr(ke, "get_endpoint") else "https://api.kaggle.com"
            if "api.kaggle.com" in endpoint:
                return "https://www.kaggle.com"
            return str(endpoint)
        ke.get_web_endpoint = get_web_endpoint
except ImportError:
    pass


class PhaseDef(TypedDict):
    name: str
    datasets: List[str]
    models: List[str]


PHASES: List[PhaseDef] = [
    {
        "name": "Phase 1: Deep Quality & Safety Assessment",
        "datasets": ["LemGendizedQualityDataset", "ClassificationMasterManifold"],
        "models": [
            "nima_aesthetic_mobile",
            "nima_aesthetic_efficientnet",
            "nima_aesthetic_pro",
            "nima_technical",
            "nima_authenticity",
            "universal_nsfw_classification",
        ],
    },
    {
        "name": "Phase 2A: High-Fidelity Facial Analytics",
        "datasets": ["LemGendizedFaceDataset"],
        "models": ["codeformer", "parsenet"],
    },
    {
        "name": "Phase 2B: Massive Universal Detection",
        "datasets": ["LemGendizedFaceDataset", "LemGendizedDetectionDataset"],
        "models": ["retinaface", "retinaface_resnet", "yolov8n"],
    },
    {
        "name": "Phase 3A: Master Super-Resolution Synthesis",
        "datasets": ["LemGendizedSuperResDataset"],
        "models": ["ultrazoom"],
    },
    {
        "name": "Phase 3B: Degradation Removal Arrays",
        "datasets": ["LemGendizedDegradationDataset"],
        "models": ["ffanet_indoor", "ffanet_outdoor", "mprnet_deraining"],
    },
    {
        "name": "Phase 3C: Low-Light Recovery",
        "datasets": ["LemGendizedLowLightDataset"],
        "models": ["mirnet_lowlight", "mirnet_exposure"],
    },
    {
        "name": "Phase 3D: Denoising Networks",
        "datasets": ["LemGendizedNoiseDataset"],
        "models": ["nafnet_denoising"],
    },
    {
        "name": "Phase 3E: Universal Cross-Domain Restoration",
        "datasets": [
            "LemGendizedSuperResDataset",
            "LemGendizedDegradationDataset",
            "LemGendizedLowLightDataset",
            "LemGendizedNoiseDataset",
        ],
        "models": [
            "nafnet_debluring",
            "film_restorer",
            "upn_v2",
            "professional_multitask_restoration",
        ],
    },
    {
        "name": "Phase 4: Master Generative Manifolds",
        "datasets": ["diffusion_master_manifold"],
        "models": ["diffusion_sdxl", "diffusion_flux"],
    },
    {
        "name": "Phase 5: Master Multimodal Reasoning",
        "datasets": ["vision_language_master_manifold"],
        "models": ["vlm_llava", "vlm_blip2"],
    },
]


def main() -> None:
    """Execute fleet orchestrator."""
    parser = argparse.ArgumentParser(description="Global LemGendary Fleet Orchestrator")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Bypass SOTA existence checks.")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    unified_models_name = config.get("unified_models", "unified_models_v2.yaml")
    unified_models_path = project_root / unified_models_name
    with open(unified_models_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    train_script = project_root / "training" / "train.py"
    failure_log_path = project_root / "fleet_failure_report.json"
    failure_report: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "failures": []}

    active_phases: list[PhaseDef] = []
    auto_accept = args.yes

    scheduled_models = set(m for p in PHASES for m in p["models"])
    unscheduled = [
        m for m, info in registry.items()
        if isinstance(info, dict) and m not in scheduled_models and not m.startswith("_")
    ]
    if unscheduled:
        PHASES.append({"name": "Phase 6: Dynamic Fleet Extensions", "datasets": [], "models": unscheduled})

    for phase in PHASES:
        approved_models: list[str] = []
        for model_key in phase["models"]:
            if model_key not in registry:
                continue

            final_pth = project_root.parent / "LemGendaryModels" / model_key / "checkpoints" / f"{model_key}_best.pth"
            if final_pth.exists() and not args.force:
                print(f"[SKIP] Model '{model_key}' has existing SOTA artifacts. Use --force to re-train.")
                continue

            if args.epochs == 1:
                export_dir = project_root.parent / "LemGendaryModels" / model_key
                if export_dir.exists():
                    files = os.listdir(export_dir)
                    has_onnx = any(f.endswith(".onnx") for f in files)
                    has_pt = any(f.endswith(".pt") or f.endswith(".pth") for f in files)
                    if has_onnx and has_pt:
                        print(f"[SKIP] Model '{model_key}' already has exported ONNX and PT binaries.")
                        continue

            if auto_accept or input(f"Train >> {model_key} << ? (y/n/all): ").strip().lower() in ["y", "all"]:
                approved_models.append(model_key)
                if not auto_accept and "all" in sys.stdin.readline():
                    auto_accept = True

        if approved_models:
            active_phases.append({"name": phase["name"], "datasets": phase["datasets"], "models": approved_models})

    for phase in active_phases:
        print(f"\n[FAST] Initiating {phase['name']}...")
        for model_key in phase["models"]:
            print(f"\nMatrix: {model_key}\n")
            cmd = [sys.executable, str(train_script), "--model", model_key, "--epochs", str(args.epochs), "--env", args.env]
            try:
                subprocess.check_call(cmd)
                print(f"[OK] {model_key} converged.")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {model_key} structural failure.")
                failure_report["failures"].append({"model": model_key, "phase": phase["name"], "code": e.returncode})
                with open(failure_log_path, "w", encoding="utf-8") as f:
                    json.dump(failure_report, f, indent=4)
                if not args.yes and input("Proceed to next? (y/n): ").lower() != "y":
                    sys.exit(1)

            time.sleep(2)

    print("\nFleet Orchestration Complete!")
    if failure_report["failures"]:
        print(f"Warning: {len(failure_report['failures'])} models failed. See {failure_log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining Suite Aborted by User.")
        sys.exit(0)
