"""Global LemGendary Fleet Orchestrator.

Orchestrates sequential model training runs across registered phases and manifolds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml

from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.tools.train_all")

try:
    import kagglesdk.kaggle_env as ke

    if not hasattr(ke, "get_web_endpoint"):

        def get_web_endpoint(env: Any) -> str:
            """Return the Kaggle web endpoint for ``env`` (shim for older kagglesdk)."""
            endpoint = (
                ke.get_endpoint(env) if hasattr(ke, "get_endpoint") else "https://api.kaggle.com"
            )
            if "api.kaggle.com" in endpoint:
                return "https://www.kaggle.com"
            return str(endpoint)

        ke.get_web_endpoint = get_web_endpoint
except ImportError as _kaggle_env_err:
    logger.debug("kagglesdk not installed; get_web_endpoint shim not applied: %s", _kaggle_env_err)


class PhaseDef(TypedDict):
    """One fleet training phase: display name, datasets it needs and models it trains."""

    name: str
    datasets: list[str]
    models: list[str]


PHASES: list[PhaseDef] = [
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
    {
        "name": "Phase 6: Foundation & Causal Assistant Intelligence",
        "datasets": ["LemGendizedAIHelperLemGend", "LemGendizedTestLLM"],
        "models": ["ai_helper_lemgend", "test_llm"],
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global LemGendary Fleet Orchestrator")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Bypass SOTA existence checks.")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_registry(project_root: Path) -> dict[str, Any]:
    """Load the model registry named by ``config.yaml`` (``unified_models``)."""
    config = _load_yaml(project_root / "config.yaml")
    unified_models_name = config.get("unified_models", "unified_models_v2.yaml")
    return _load_yaml(project_root / unified_models_name)


def _append_unscheduled_phase(registry: dict[str, Any]) -> None:
    """Add a phase for registry models that no phase in ``PHASES`` schedules."""
    scheduled_models = {m for p in PHASES for m in p["models"]}
    unscheduled = [
        m
        for m, info in registry.items()
        if isinstance(info, dict) and m not in scheduled_models and not m.startswith("_")
    ]
    if unscheduled:
        PHASES.append(
            {"name": "Phase 6: Dynamic Fleet Extensions", "datasets": [], "models": unscheduled}
        )


def _has_artifacts(project_root: Path, model_key: str, args: argparse.Namespace) -> bool:
    """True (with a ``[SKIP]`` line) when ``model_key`` needs no training run."""
    hub_dir = project_root.parent / "LemGendaryModels" / model_key
    final_pth = hub_dir / "checkpoints" / f"{model_key}_best.pth"
    if final_pth.exists() and not args.force:
        print(f"[SKIP] Model '{model_key}' has existing SOTA artifacts. Use --force to re-train.")
        return True
    if args.epochs == 1 and hub_dir.exists():
        files = os.listdir(hub_dir)
        has_onnx = any(f.endswith(".onnx") for f in files)
        has_pt = any(f.endswith((".pt", ".pth")) for f in files)
        if has_onnx and has_pt:
            print(f"[SKIP] Model '{model_key}' already has exported ONNX and PT binaries.")
            return True
    return False


def _approve(model_key: str, auto_accept: bool) -> tuple[bool, bool]:
    """Ask whether to train ``model_key``; return ``(approved, auto_accept)``."""
    if auto_accept:
        return True, True
    if input(f"Train >> {model_key} << ? (y/n/all): ").strip().lower() in ["y", "all"]:
        return True, "all" in sys.stdin.readline()
    return False, False


def _select_phases(
    registry: dict[str, Any], project_root: Path, args: argparse.Namespace
) -> list[PhaseDef]:
    """Return the phases with the models the operator approved for training."""
    active_phases: list[PhaseDef] = []
    auto_accept = args.yes
    for phase in PHASES:
        approved_models: list[str] = []
        for model_key in phase["models"]:
            if model_key not in registry or _has_artifacts(project_root, model_key, args):
                continue
            approved, auto_accept = _approve(model_key, auto_accept)
            if approved:
                approved_models.append(model_key)
        if approved_models:
            active_phases.append(
                {"name": phase["name"], "datasets": phase["datasets"], "models": approved_models}
            )
    return active_phases


@dataclass
class _Report:
    """Fleet failure report, rewritten to ``fleet_failure_report.json`` on each failure."""

    project_root: Path
    data: dict[str, Any] = field(
        default_factory=lambda: {"timestamp": datetime.now().isoformat(), "failures": []}
    )

    @property
    def path(self) -> Path:
        """Report file in the project root."""
        return self.project_root / "fleet_failure_report.json"

    def add_failure(self, failure: dict[str, Any]) -> None:
        """Record ``failure`` and rewrite the report file."""
        self.data["failures"].append(failure)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)


def _train_model(
    model_key: str, phase_name: str, args: argparse.Namespace, report: _Report
) -> None:
    """Run ``training/train.py`` for one model; record and confirm on failure."""
    print(f"\nMatrix: {model_key}\n")
    cmd = [
        sys.executable,
        str(report.project_root / "training" / "train.py"),
        "--model",
        model_key,
        "--epochs",
        str(args.epochs),
        "--env",
        args.env,
    ]
    try:
        subprocess.check_call(cmd)
        print(f"[OK] {model_key} converged.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {model_key} structural failure.")
        report.add_failure({"model": model_key, "phase": phase_name, "code": e.returncode})
        if not args.yes and input("Proceed to next? (y/n): ").lower() != "y":
            sys.exit(1)
    time.sleep(2)


def main() -> None:
    """Execute fleet orchestrator."""
    args = _parse_args()
    project_root = get_project_root()
    registry = _load_registry(project_root)
    report = _Report(project_root)
    _append_unscheduled_phase(registry)
    for phase in _select_phases(registry, project_root, args):
        print(f"\n[FAST] Initiating {phase['name']}...")
        for model_key in phase["models"]:
            _train_model(model_key, phase["name"], args, report)

    print("\nFleet Orchestration Complete!")
    if report.data["failures"]:
        failures = report.data["failures"]
        print(f"Warning: {len(failures)} models failed. See {report.path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining Suite Aborted by User.")
        sys.exit(0)
