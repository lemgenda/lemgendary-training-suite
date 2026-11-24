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


_LLM_CURRICULUM_KEYS = (
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "load_in_4bit",
    "use_rslora",
    "target_modules",
    "chunked_cross_entropy",
    "ce_chunk_size",
)


def _merge_llm_entry_kwargs(entry: dict, kwargs: dict) -> dict:
    """Merge top-level LLM foundation model, curriculum, and active env profile keys."""
    explicit_keys = set(kwargs or {})
    merged = dict(kwargs or {})
    if entry.get("category") != "llm" and entry.get("class_name") not in (
        "GenericCausalLLM",
        "AIHelperLemGend",
    ):
        return merged
    base_model = (
        entry.get("base_model_name_or_path")
        or entry.get("base_model_name")
        or entry.get("base_foundation_model")
    )
    if base_model and "base_model_name_or_path" not in merged:
        merged["base_model_name_or_path"] = base_model
    curriculum = entry.get("curriculum")
    if isinstance(curriculum, dict):
        for key in _LLM_CURRICULUM_KEYS:
            if key in curriculum and key not in explicit_keys:
                merged[key] = curriculum[key]
    default_env = (
        "kaggle"
        if (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_TPU_NAME"))
        else "local"
    )
    env = str(entry.get("_active_env") or os.environ.get("LEMTRAIN_ENV") or default_env).lower()
    profile_key = "cloud" if env in ("kaggle", "colab", "cloud") else "local"
    profile_cfg = entry.get(profile_key)
    if isinstance(profile_cfg, dict):
        for key in _LLM_CURRICULUM_KEYS:
            if key in profile_cfg and key not in explicit_keys:
                merged[key] = profile_cfg[key]
    return merged


def _resolve_model_entry(
    model_key: str, config: dict | None, unified_models_path: str
) -> tuple[str | None, dict, dict | None]:
    """Resolve ``(class_name, kwargs, stabilizers)`` from ``config`` and the registry YAML."""
    cfg_dict = config if isinstance(config, dict) else {}
    model_class_name = cfg_dict.get("class_name")
    kwargs = dict(cfg_dict.get("kwargs", {}))
    stabilizers = cfg_dict.get("stabilizers")
    unified_entry: dict = {}
    if os.path.exists(unified_models_path):
        with open(unified_models_path, "r", encoding="utf-8") as f:
            unified = yaml.safe_load(f) or {}
            if model_key in unified and isinstance(unified[model_key], dict):
                unified_entry = unified[model_key]
                model_class_name = model_class_name or unified_entry.get("class_name")
                kwargs = {**dict(unified_entry.get("kwargs") or {}), **kwargs}
                if stabilizers is None:
                    stabilizers = unified_entry.get("stabilizers")
    merged_entry = {**unified_entry, **cfg_dict} if cfg_dict else unified_entry
    if merged_entry:
        kwargs = _merge_llm_entry_kwargs(merged_entry, kwargs)
    return model_class_name, kwargs, stabilizers


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
    model_class_name, kwargs, stabilizers = _resolve_model_entry(
        model_key, config, unified_models_path
    )

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
