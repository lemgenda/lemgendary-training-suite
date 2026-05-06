import sys
import os

# Add root to path so we can test local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_torch_import():
    import torch
    print(f"Torch version: {torch.__version__}")
    assert torch is not None

def test_pillow_import():
    from PIL import Image
    assert Image is not None

def test_training_modules():
    from training import losses, optimization_engine
    assert losses is not None
    assert optimization_engine is not None

def test_environment_vars():
    # Simple check to see if we can read environment
    import os
    assert 'PYTHONPATH' not in os.environ or os.environ['PYTHONPATH'] is not None
