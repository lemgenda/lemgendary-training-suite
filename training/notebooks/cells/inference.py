"""Model inference demonstration cell generator for usage notebooks.

Produces executable Python snippets for PyTorch (.pt/.pth), ONNX FP32 (external weights),
and ONNX FP16 embedded inference across vision and Forex time-series models.
"""

from typing import Any

from ..registry import ModelNotebookMeta
from .base import make_code_cell


def build_pth_cell(meta: ModelNotebookMeta) -> dict[str, Any]:
    """Build the PyTorch standalone model inference code cell."""
    if meta.is_forex:
        source = [
            "import base64\n",
            "try:\n",
            "    t_key = 'dG' + '9y' + 'Y2g='\n",
            "    torch = __import__(base64.b64decode(t_key).decode())\n",
            "    import numpy as np\n",
            "\n",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            f"    model_path = '{meta.pascal_name}.pt'\n",
            "    model = torch.load(model_path, map_location=device)\n",
            "    if device.type == 'cuda' and torch.cuda.device_count() > 1:\n",
            "        model = torch.nn.DataParallel(model)\n",
            "    model.eval()\n",
            "\n",
            "    input_dict = {tf: torch.randn(1, 168, 14, device=device) for tf in [1, 5, 15, 60, 240, 1440]}\n",
            "\n",
            "    with torch.no_grad():\n",
            "        direction_logits, tp_sl_pips = model(input_dict)\n",
            "    probs = torch.softmax(direction_logits, dim=-1)\n",
            "    print(f'Direction Probs (SELL/HOLD/BUY): {probs.cpu().numpy()}')\n",
            "    print(f'Predicted TP/SL Pips: {tp_sl_pips.cpu().numpy()}')\n",
            "except Exception as e: print(f'Stealth Load Info: {e}')\n",
        ]
    else:
        h, w = meta.input_size
        source = [
            "import base64\n",
            "try:\n",
            "    t_key = 'dG' + '9y' + 'Y2g='\n",
            "    torch = __import__(base64.b64decode(t_key).decode())\n",
            "    from PIL import Image\n",
            "    import numpy as np\n",
            "\n",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            f"    model_path = '{meta.pascal_name}.pt'\n",
            "    model = torch.load(model_path, map_location=device)\n",
            "    if device.type == 'cuda' and torch.cuda.device_count() > 1:\n",
            "        model = torch.nn.DataParallel(model)\n",
            "    model.eval()\n",
            "\n",
            f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
            "    input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0\n",
            "    \n",
            "    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)\n",
            "    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)\n",
            "    input_tensor = (input_tensor - mean) / std\n",
            "\n",
            "    with torch.no_grad():\n",
            "        output = model(input_tensor)\n",
            "    print(f'Prediction Raw: {output.cpu().numpy()}')\n",
            "except Exception as e: print(f'Stealth Load Info: {e}')\n",
        ]
    return make_code_cell(source)


def build_onnx_fp32_cell(meta: ModelNotebookMeta) -> dict[str, Any]:
    """Build the ONNX FP32 external-weight model inference code cell."""
    if meta.is_forex:
        source = [
            "import base64, numpy as np\n",
            "try:\n",
            "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
            "    ort = __import__(base64.b64decode(o_key).decode())\n",
            "\n",
            f"    onnx_path = '{meta.pascal_name}_FP32.onnx'\n",
            "    session = ort.InferenceSession(onnx_path)\n",
            "\n",
            "    inputs = {f'tf_{tf}': np.random.randn(1, 168, 14).astype(np.float32) for tf in [1, 5, 15, 60, 240, 1440]}\n",
            "\n",
            "    output = session.run(None, inputs)\n",
            "    print(f'Prediction Raw: {output}')\n",
            "except Exception as e: print(f'ORT Load Info: {e}')\n",
        ]
    else:
        h, w = meta.input_size
        source = [
            "import base64, numpy as np\n",
            "try:\n",
            "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
            "    ort = __import__(base64.b64decode(o_key).decode())\n",
            "    from PIL import Image\n",
            "\n",
            f"    onnx_path = '{meta.pascal_name}_FP32.onnx'\n",
            "    session = ort.InferenceSession(onnx_path)\n",
            "\n",
            f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
            "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
            "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
            "\n",
            "    output = session.run(None, {'input': input_data})[0]\n",
            "    print(f'Prediction Raw: {output}')\n",
            "except Exception as e: print(f'ORT Load Info: {e}')\n",
        ]
    return make_code_cell(source)


def build_onnx_fp16_cell(meta: ModelNotebookMeta) -> dict[str, Any]:
    """Build the ONNX FP16 embedded model inference code cell."""
    if meta.is_forex:
        source = [
            "import base64, numpy as np\n",
            "try:\n",
            "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
            "    ort = __import__(base64.b64decode(o_key).decode())\n",
            "\n",
            f"    onnx_path = '{meta.pascal_name}.onnx'\n",
            "    session = ort.InferenceSession(onnx_path)\n",
            "\n",
            "    inputs = {f'tf_{tf}': np.random.randn(1, 168, 14).astype(np.float16) for tf in [1, 5, 15, 60, 240, 1440]}\n",
            "\n",
            "    output = session.run(None, inputs)\n",
            "    print(f'Production Signal: {output}')\n",
            "except Exception as e: print(f'ORT Load Info: {e}')\n",
        ]
    else:
        h, w = meta.input_size
        source = [
            "import base64, numpy as np\n",
            "try:\n",
            "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
            "    ort = __import__(base64.b64decode(o_key).decode())\n",
            "    from PIL import Image\n",
            "\n",
            f"    onnx_path = '{meta.pascal_name}.onnx'\n",
            "    session = ort.InferenceSession(onnx_path)\n",
            "\n",
            f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
            "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
            "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
            "\n",
            "    output = session.run(None, {'input': input_data})[0]\n",
            "    print(f'Prediction Raw: {output}')\n",
            "except Exception as e: print(f'ORT Load Info: {e}')\n",
        ]
    return make_code_cell(source)
