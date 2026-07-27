import os
import yaml
import torch

# [SENIOR HARDENING v16.0 - SYNC_ID: 1412]

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
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import MultiTaskRestorer: {e}")

    try:
        from models.nima import NIMA_Model, AuthenticityScorer, UniversalClassifier
        _MODEL_REGISTRY["NIMA_Model"] = NIMA_Model
        _MODEL_REGISTRY["AuthenticityScorer"] = AuthenticityScorer
        _MODEL_REGISTRY["UniversalClassifier"] = UniversalClassifier
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import NIMA models: {e}")

    try:
        from models.forex_predictor import ForexPredictor
        _MODEL_REGISTRY["ForexPredictor"] = ForexPredictor
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import ForexPredictor: {e}")


    try:
        from models.face_restoration import CodeFormer, ParseNet
        _MODEL_REGISTRY["CodeFormer"] = CodeFormer
        _MODEL_REGISTRY["ParseNet"] = ParseNet
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import Face Restoration models: {e}")

    try:
        from models.detection import RetinaFace_MobileNet
        _MODEL_REGISTRY["RetinaFace"] = RetinaFace_MobileNet
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import Detection models: {e}")

    try:
        from models.core_restoration import (
            NAFNet, FFANet, BranchedFFANet, MIRNet_Proxy, MPRNet_Proxy,
            GenericRestorationModel, UltraZoomModel, UniversalFilmRestorer, UPN_v2_Model
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
    except Exception as e:
        print(f" [WARNING] [FACTORY] Failed to import Core Restoration models: {e}")

    # 2. Heavy Generative Models (May lack 'diffusers')
    try:
        from models.master_generative import StableDiffusionXL, Flux1_Master
        _MODEL_REGISTRY["StableDiffusionXL"] = StableDiffusionXL
        _MODEL_REGISTRY["Flux1_Master"] = Flux1_Master
    except Exception as e:
        pass # Silently bypass generative imports if dependencies are missing

    # 3. Heavy Multimodal Models (May lack 'transformers')
    try:
        from models.master_multimodal import LLaVA_v1_5, BLIP_2
        _MODEL_REGISTRY["LLaVA_v1_5"] = LLaVA_v1_5
        _MODEL_REGISTRY["BLIP_2"] = BLIP_2
    except Exception as e:
        pass # Silently bypass multimodal imports if dependencies are missing


def get_model(model_key, config=None):
    """Factory function using Dynamic Architecture Registry."""
    if not _MODEL_REGISTRY: _populate_registry()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unified_name = config.get("unified_models", "unified_models_v2.yaml") if config else "unified_models_v2.yaml"
    unified_models_path = os.path.join(project_root, unified_name)
    
    model_class_name = None
    kwargs = {}
    if os.path.exists(unified_models_path):
        with open(unified_models_path, "r") as f:
            unified = yaml.safe_load(f)
            if model_key in unified:
                model_class_name = unified[model_key].get("class_name")
                kwargs = unified[model_key].get("kwargs", {})

    if model_class_name in _MODEL_REGISTRY:
        print(f" [FACTORY] Instantiating {model_class_name} for key: {model_key}")
        return _MODEL_REGISTRY[model_class_name](**kwargs)
    
    raise ValueError(f" [FACTORY ERROR] Model architecture '{model_class_name}' not found or implemented.")
