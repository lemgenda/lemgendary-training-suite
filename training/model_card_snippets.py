"""Usage snippets and topology diagrams embedded in generated model cards."""

from __future__ import annotations

from training.utils.model_payload import pt_load_source

_PT_LOAD = "".join(pt_load_source("model_path", "device"))
"""Generated loader: release payload, checkpoint or bare state dict (one policy)."""
_PT_LOAD_INDENTED = "".join("    " + line for line in pt_load_source("model_path", "device"))


def _quality_usage(model_key: str, h: int, w: int) -> str:
    """Return the NIMA quality-scoring usage snippet."""
    return (
        "```"
        + f"""python
import torch, base64
from PIL import Image

# 1. Hardware-Agnostic Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Stealth Load (v16.0)
model_path = "{model_key}_latest.pth"
{_PT_LOAD}
# 3. Initialization
from models.nima import NIMAModel
model = NIMAModel().to(device)
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state)
model.eval()

# 4. Forward Pass
img = Image.open("photo.jpg").convert('RGB').resize(({h}, {w}))
arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
input_tensor = arr.unsqueeze(0).to(device) / 255.0
with torch.no_grad():
    probs = model(input_tensor)

# 5. Score Calculation
scores = torch.arange(1, 11).float().to(device)
mean_score = torch.sum(probs * scores).item()
print(f"Quality Score: {{mean_score:.2f}}")
```"""
    )


def _restoration_usage(model_key: str) -> str:
    """Return the image restoration usage snippet."""
    return (
        "```"
        + f"""python
import torch, base64
from PIL import Image

# 1. Hardware-Agnostic Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Stealth Load (v16.0)
model_path = "{model_key}_latest.pth"
{_PT_LOAD}
# 3. Initialization
from models.factory import create_model
model = create_model("{model_key}").to(device)
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state)
model.eval()

# 4. Restoration Pass
img = Image.open("degraded.jpg").convert('RGB')
arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
input_tensor = arr.unsqueeze(0).to(device) / 255.0
with torch.no_grad():
    restored = model(input_tensor)

# 5. Ejection
out_arr = restored.squeeze().permute(1, 2, 0).cpu().numpy()
restored_img = Image.fromarray((out_arr * 255).astype('uint8'))
restored_img.save("restored.png")
```"""
    )


def _forex_usage(model_key: str) -> str:
    """Return the Forex predictor usage snippet."""
    return (
        "```"
        + f"""python
import torch, os

# 1. Hardware-Agnostic Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Stealth Load (v16.0)
from models.forex_predictor import ForexPredictor
model = ForexPredictor().to(device)
model_path = "{model_key}_latest.pth"
if os.path.exists(model_path):
{_PT_LOAD_INDENTED}    model.load_state_dict(state)
model.eval()

# 3. Multi-Timeframe Sequence Inference [B, 168, 14]
# Active timeframes: 1m, 5m, 15m, 60m, 240m, 1440m
sample_input = {{
    1: torch.randn(1, 168, 14, device=device),
    5: torch.randn(1, 168, 14, device=device),
    15: torch.randn(1, 168, 14, device=device),
    60: torch.randn(1, 168, 14, device=device),
    240: torch.randn(1, 168, 14, device=device),
    1440: torch.randn(1, 168, 14, device=device),
}}
with torch.no_grad():
    out = model(sample_input)
    direction_logits = out["direction_logits"]
    tp_sl_pips = out["magnitude"]
    probs = torch.softmax(direction_logits, dim=-1)
    # Signal: 0=SELL, 1=HOLD, 2=BUY
    signal = torch.argmax(probs, dim=-1).item()
    pips = tp_sl_pips.cpu().numpy()
    print(f"Trade Signal: {{signal}}, Predicted TP/SL Pips: {{pips}}")
```"""
    )


_GENERIC_USAGE = "```python\n# Premium CLI Integration provided for generative/VLM tasks.\n```"


def usage_snippet(task: str, model_key: str, h: int, w: int) -> str:
    """Return the fenced usage snippet of ``task`` for ``model_key`` at ``h`` x ``w``."""
    if task == "quality":
        return _quality_usage(model_key, h, w)
    if task in ["restoration", "enhancement"]:
        return _restoration_usage(model_key)
    if task == "forex":
        return _forex_usage(model_key)
    return _GENERIC_USAGE


def topology_mermaid(task: str, res_str: str, arch: str) -> str:
    """Return the fenced Mermaid topology diagram of the model."""
    if task == "forex":
        return """```mermaid
graph TD
    Input[OHLCV Sequence] --> Backbone[Causal TCN]
    Backbone --> Attention[Cross-Timeframe Attention]
    Attention --> Head[Directional & Magnitude Head]
    Head --> Output[TP/SL & Trade Signal]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#00ff00,stroke:#333,stroke-width:4px
```"""
    return f"""```mermaid
graph TD
    Input[RGB Input {res_str}] --> Backbone[{arch}]
    Backbone --> Manifold[Latent Manifold]
    Manifold --> Head[{task.capitalize()} Head]
    Head --> Output[Predictive Array]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#00ff00,stroke:#333,stroke-width:4px
```"""
