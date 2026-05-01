import os
import yaml
import torch
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

def get_model(model_key, config=None):
    """
    Factory function to instantiate real high-fidelity models.
    Mocks have been purged as of v2.7.0.
    """
    # Load unified models if possible to resolve class_name
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

    # Routing to Real Architectures
    if model_class_name == "MultiTaskRestorer":
        return MultiTaskRestorer(num_tasks=6)
    elif model_class_name == "NIMA_Model":
        return NIMA_Model(**kwargs)
    elif model_class_name == "AuthenticityScorer":
        return AuthenticityScorer()
    elif model_class_name == "CodeFormer":
        return CodeFormer()
    elif model_class_name == "ParseNet":
        return ParseNet()
    elif model_class_name == "RetinaFace":
        return RetinaFace_MobileNet()
    elif model_class_name == "GenericRestoration":
        return GenericRestorationModel()
    elif model_class_name == "NAFNet":
        return NAFNet(**kwargs)
    elif model_class_name == "FFANet":
        return FFANet()
    elif model_class_name == "MIRNet":
        return MIRNet_Proxy()
    elif model_class_name == "MPRNet":
        return MPRNet_Proxy()
    elif model_class_name == "UltraZoom":
        return UltraZoomModel(**kwargs)
    elif model_class_name == "UniversalFilmRestorer":
        return UniversalFilmRestorer()
    elif model_class_name == "UPN_v2":
        return UPN_v2_Model()
    elif model_class_name == "YOLO":
        # Handled natively in train.py, but fallback throw if called
        raise ValueError("YOLO model initialization should be explicitly routed to Ultralytics native loop.")
    elif model_class_name == "StableDiffusionXL":
        return StableDiffusionXL(**kwargs)
    elif model_class_name == "Flux1_Master":
        return Flux1_Master(**kwargs)
    elif model_class_name == "LLaVA_v1_5":
        return LLaVA_v1_5(**kwargs)
    elif model_class_name == "BLIP_2":
        return BLIP_2(**kwargs)
    
    # Final Validation
    if not model_class_name:
        raise ValueError(f"❌ [FACTORY ERROR] Model key '{model_key}' was not found in the unified models registry ({unified_name}). Please check your config.")
        
    raise ValueError(f"❌ [FACTORY ERROR] Model architecture '{model_class_name}' for key '{model_key}' is not implemented in the factory routing logic.")
