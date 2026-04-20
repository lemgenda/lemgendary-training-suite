import os
import yaml
import torch
from models.multitask_restorer import MultiTaskRestorer
from models.nima import NIMA_Model
from models.face_restoration import CodeFormer, ParseNet
from models.detection import RetinaFace_MobileNet
from models.core_architectures import GenericRestorationModel, UltraZoomModel, UniversalFilmRestorer, UPN_v2_Model
from models.core_restoration import NAFNet, FFANet, MIRNet_Proxy, MPRNet_Proxy

def get_model(model_key, config=None):
    """
    Factory function to instantiate real high-fidelity models.
    Mocks have been purged as of v2.7.0.
    """
    # Load unified models if possible to resolve class_name
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unified_models_path = os.path.join(project_root, "unified_models.yaml")
    
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
    
    # Fallback for Multi-Task
    if model_key in ["multi_task_restorer", "professional_multitask_restoration"]:
        return MultiTaskRestorer(num_tasks=6)
        
    raise ValueError(f"Model architecture '{model_class_name}' for key '{model_key}' not implemented in factory.")
