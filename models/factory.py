import inspect
import logging
import os

import yaml

# [SENIOR HARDENING v16.0 - SYNC_ID: 1412]

logger = logging.getLogger("lemtrain.factory")

# Optional model modules can fail at import time because a dependency is missing
# (ImportError), a native library cannot be loaded (OSError), a backend refuses to
# initialise (RuntimeError) or an installed dependency has an incompatible API
# (AttributeError). Anything else is a real bug and must propagate.
_OPTIONAL_IMPORT_ERRORS = (ImportError, OSError, RuntimeError, AttributeError)

# --- Dynamic Architecture Registry (Task 10.2) ---
_MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# Lazily import models to populate registry (Robust dependency shielding)
def _populate_registry():
    # 1. Base Restoration & Core Models (Always available)
    try:
        from models.multitask_restorer import MultiTaskRestorer
        _MODEL_REGISTRY["MultiTaskRestorer"] = MultiTaskRestorer
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "MultiTaskRestorer", e)

    try:
        from models.nima import AuthenticityScorer, NIMA_Model, UniversalClassifier
        _MODEL_REGISTRY["NIMA_Model"] = NIMA_Model
        _MODEL_REGISTRY["AuthenticityScorer"] = AuthenticityScorer
        _MODEL_REGISTRY["UniversalClassifier"] = UniversalClassifier
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "NIMA models", e)

    try:
        from models.forex_predictor import ForexPredictor
        _MODEL_REGISTRY["ForexPredictor"] = ForexPredictor
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "ForexPredictor", e)


    try:
        from models.face_restoration import CodeFormer, ParseNet
        _MODEL_REGISTRY["CodeFormer"] = CodeFormer
        _MODEL_REGISTRY["ParseNet"] = ParseNet
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "Face Restoration models", e)

    try:
        from models.detection import RetinaFace_MobileNet
        _MODEL_REGISTRY["RetinaFace"] = RetinaFace_MobileNet
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "Detection models", e)

    try:
        from models.core_restoration import (
            BranchedFFANet,
            FFANet,
            GenericRestorationModel,
            MIRNet_Proxy,
            MPRNet_Proxy,
            NAFNet,
            UltraZoomModel,
            UniversalFilmRestorer,
            UPN_v2_Model,
        )
        _MODEL_REGISTRY.update({
            "GenericRestoration": GenericRestorationModel,
            "NAFNet": NAFNet,
            "FFANet": FFANet,
            "BranchedFFANet": BranchedFFANet,
            "MIRNet": MIRNet_Proxy,
            "MPRNet": MPRNet_Proxy,
            "UltraZoom": UltraZoomModel,
            "UltraZoomMaster": UltraZoomModel,
            "UniversalFilmRestorer": UniversalFilmRestorer,
            "UPN_v2": UPN_v2_Model
        })
    except _OPTIONAL_IMPORT_ERRORS as e:
        logger.warning("Failed to import %s: %s", "Core Restoration models", e)

    # 2. Heavy Generative Models (May lack 'diffusers')
    try:
        from models.master_generative import Flux1_Master, StableDiffusionXL
        _MODEL_REGISTRY["StableDiffusionXL"] = StableDiffusionXL
        _MODEL_REGISTRY["Flux1_Master"] = Flux1_Master
    except _OPTIONAL_IMPORT_ERRORS as generative_err:
        logger.debug("Generative models unavailable (missing 'diffusers'?): %s", generative_err)

    # 3. Heavy Multimodal Models (May lack 'transformers')
    try:
        from models.master_multimodal import BLIP_2, LLaVA_v1_5
        _MODEL_REGISTRY["LLaVA_v1_5"] = LLaVA_v1_5
        _MODEL_REGISTRY["BLIP_2"] = BLIP_2
    except _OPTIONAL_IMPORT_ERRORS as multimodal_err:
        logger.debug("Multimodal models unavailable (missing 'transformers'?): %s", multimodal_err)

    # 4. Causal Assistant Language Models
    try:
        from models.generic_causal_llm import GenericCausalLLM, AIHelperLemGend
        _MODEL_REGISTRY["GenericCausalLLM"] = GenericCausalLLM
        _MODEL_REGISTRY["AIHelperLemGend"] = AIHelperLemGend
    except _OPTIONAL_IMPORT_ERRORS as llm_err:
        logger.debug("Causal LLMs unavailable: %s", llm_err)


def get_model(model_key, config=None):
    """Factory function using Dynamic Architecture Registry."""
    if not _MODEL_REGISTRY:
        _populate_registry()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unified_name = config.get("unified_models", "unified_models_v2.yaml") if config else "unified_models_v2.yaml"
    unified_models_path = os.path.join(project_root, unified_name)

    model_class_name = config.get("class_name") if isinstance(config, dict) else None
    kwargs = config.get("kwargs", {}) if isinstance(config, dict) else {}
    if os.path.exists(unified_models_path):
        with open(unified_models_path, "r") as f:
            unified = yaml.safe_load(f)
            if model_key in unified:
                model_class_name = unified[model_key].get("class_name", model_class_name)
                kwargs = unified[model_key].get("kwargs", kwargs)

    # YOLO models are delegated to the Ultralytics native trainer via core_loop._run_yolo_native().
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
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            filtered_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (ValueError, TypeError) as sig_err:
            logger.debug("Signature inspection failed for %s: %s", model_class_name, sig_err)
            filtered_kwargs = kwargs
        return cls(**filtered_kwargs)

    raise ValueError(f" [FACTORY ERROR] Model architecture '{model_class_name}' not found or implemented.")

