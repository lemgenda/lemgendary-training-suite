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

# Lazily import models to populate registry
def _populate_registry():
    from models.multitask_restorer import MultiTaskRestorer
    from models.nima import NIMA_Model, AuthenticityScorer
    from models.face_restoration import CodeFormer, ParseNet
    from models.detection import RetinaFace_MobileNet
    from models.core_restoration import (
        NAFNet, FFANet, MIRNet_Proxy, MPRNet_Proxy,
        GenericRestorationModel, UltraZoomModel, UniversalFilmRestorer, UPN_v2_Model
    )
    from models.master_generative import StableDiffusionXL, Flux1_Master
    from models.master_multimodal import LLaVA_v1_5, BLIP_2
    
    # Manual Map (until all models use @register_model)
    registry_map = {
        "MultiTaskRestorer": MultiTaskRestorer,
        "NIMA_Model": NIMA_Model,
        "AuthenticityScorer": AuthenticityScorer,
        "CodeFormer": CodeFormer,
        "ParseNet": ParseNet,
        "RetinaFace": RetinaFace_MobileNet,
        "GenericRestoration": GenericRestorationModel,
        "NAFNet": NAFNet,
        "FFANet": FFANet,
        "MIRNet": MIRNet_Proxy,
        "MPRNet": MPRNet_Proxy,
        "UltraZoom": UltraZoomModel,
        "UniversalFilmRestorer": UniversalFilmRestorer,
        "UPN_v2": UPN_v2_Model,
        "StableDiffusionXL": StableDiffusionXL,
        "Flux1_Master": Flux1_Master,
        "LLaVA_v1_5": LLaVA_v1_5,
        "BLIP_2": BLIP_2
    }
    _MODEL_REGISTRY.update(registry_map)

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
