"""Model factory: maps ``class_name`` entries in unified_models_v2.yaml to model classes."""

import inspect
import logging
import os

import yaml

from models.core_restoration import (
    BranchedFFANet,
    FFANet,
    GenericRestorationModel,
    MIRNetProxy,
    MPRNetProxy,
    NAFNet,
    UltraZoomModel,
    UniversalFilmRestorer,
    UPNv2Model,
)
from models.detection import RetinaFaceMobileNet
from models.face_restoration import CodeFormer, ParseNet
from models.forex_predictor import ForexPredictor
from models.generic_causal_llm import AIHelperLemGend, GenericCausalLLM
from models.master_generative import Flux1Master, StableDiffusionXL
from models.master_multimodal import BLIP2, LLaVAv15
from models.multitask_restorer import MultiTaskRestorer
from models.nima import AuthenticityScorer, NIMAModel, UniversalClassifier
from training.utils.lazy_imports import is_installed

# [SENIOR HARDENING v16.0 - SYNC_ID: 1412]

logger = logging.getLogger("lemtrain.factory")

# --- Dynamic Architecture Registry (Task 10.2) ---
# Core models depend only on torch/torchvision and are registered eagerly.
_CORE_MODELS = {
    "MultiTaskRestorer": MultiTaskRestorer,
    "NIMA_Model": NIMAModel,
    "AuthenticityScorer": AuthenticityScorer,
    "UniversalClassifier": UniversalClassifier,
    "ForexPredictor": ForexPredictor,
    "CodeFormer": CodeFormer,
    "ParseNet": ParseNet,
    "RetinaFace": RetinaFaceMobileNet,
    "GenericRestoration": GenericRestorationModel,
    "NAFNet": NAFNet,
    "FFANet": FFANet,
    "BranchedFFANet": BranchedFFANet,
    "MIRNet": MIRNetProxy,
    "MPRNet": MPRNetProxy,
    "UltraZoom": UltraZoomModel,
    "UltraZoomMaster": UltraZoomModel,
    "UniversalFilmRestorer": UniversalFilmRestorer,
    "UPN_v2": UPNv2Model,
    "GenericCausalLLM": GenericCausalLLM,
    "AIHelperLemGend": AIHelperLemGend,
}
_MODEL_REGISTRY = {}


def register_model(name):
    """Class decorator that registers a model class under ``name``."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def _populate_registry():
    """Register core models, then the heavy optional ones whose dependencies are installed."""
    _MODEL_REGISTRY.update(_CORE_MODELS)

    # Heavy generative models need the optional 'diffusers' package; the wrappers load it
    # when they are constructed.
    if is_installed("diffusers"):
        _MODEL_REGISTRY["StableDiffusionXL"] = StableDiffusionXL
        _MODEL_REGISTRY["Flux1_Master"] = Flux1Master
    else:
        logger.debug("Generative models unavailable: 'diffusers' is not installed.")

    # Heavy multimodal models need the optional 'transformers' package.
    if is_installed("transformers"):
        _MODEL_REGISTRY["LLaVA_v1_5"] = LLaVAv15
        _MODEL_REGISTRY["BLIP_2"] = BLIP2
    else:
        logger.debug("Multimodal models unavailable: 'transformers' is not installed.")


def apply_registry_logit_cap(model, stabilizers):
    """Set the model's logit soft cap from the registry ``stabilizers.logit_clamp`` (R2-T6)."""
    cap = stabilizers.get("logit_clamp") if isinstance(stabilizers, dict) else None
    if cap is not None and hasattr(model, "set_logit_cap"):
        model.set_logit_cap(float(cap))
    return model


def get_model(model_key, config=None):
    """Factory function using Dynamic Architecture Registry."""
    if not _MODEL_REGISTRY:
        _populate_registry()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unified_name = (
        config.get("unified_models", "unified_models_v2.yaml")
        if config
        else "unified_models_v2.yaml"
    )
    unified_models_path = os.path.join(project_root, unified_name)

    model_class_name = config.get("class_name") if isinstance(config, dict) else None
    kwargs = config.get("kwargs", {}) if isinstance(config, dict) else {}
    stabilizers = config.get("stabilizers") if isinstance(config, dict) else None
    if os.path.exists(unified_models_path):
        with open(unified_models_path, "r", encoding="utf-8") as f:
            unified = yaml.safe_load(f)
            if model_key in unified:
                model_class_name = unified[model_key].get("class_name", model_class_name)
                kwargs = unified[model_key].get("kwargs", kwargs)
                stabilizers = unified[model_key].get("stabilizers", stabilizers)

    # YOLO models are delegated to the Ultralytics native trainer via
    # governance.yolo_governor.run_governed_yolo_training().
    # Reaching this point means the bypass in core_loop.main() was skipped, which is a bug.
    if model_class_name == "YOLO":
        raise ValueError(
            f"YOLO models must be trained via the Ultralytics native trainer. "
            f"core_loop.py should intercept '{model_key}' before calling get_model(). "
            f"Ensure main() checks args.model == 'yolov8n' before build_training_context()."
        )

    if model_class_name in _MODEL_REGISTRY:
        if os.environ.get("RANK", "0") == "0":
            print(f" [FACTORY] Instantiating {model_class_name} for key: {model_key}")
        cls = _MODEL_REGISTRY[model_class_name]
        kwargs = kwargs or {}
        try:
            sig = inspect.signature(cls.__init__)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            filtered_kwargs = (
                kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in sig.parameters}
            )
        except (ValueError, TypeError) as sig_err:
            logger.debug("Signature inspection failed for %s: %s", model_class_name, sig_err)
            filtered_kwargs = kwargs
        return apply_registry_logit_cap(cls(**filtered_kwargs), stabilizers)

    raise ValueError(
        f" [FACTORY ERROR] Model architecture '{model_class_name}' not found or implemented."
    )
