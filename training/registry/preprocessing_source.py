"""Python source lines that apply a model's image preprocessing (generated cells and snippets).

The notebook inference cells and the model-card usage snippets render these lines, so the
code users copy applies the same :class:`~training.registry.preprocessing.ImagePreprocessing`
as training. Each line ends with ``\\n`` and has no indentation; ``img`` is a PIL RGB image
already resized to the model input, ``np`` is NumPy and ``torch`` / ``device`` exist for the
PyTorch form.
"""

from __future__ import annotations

from training.registry.preprocessing import ImagePreprocessing


def torch_input_source(preprocessing: ImagePreprocessing) -> list[str]:
    """``img`` to ``input_tensor`` (``[1, 3, H, W]`` float on ``device``)."""
    lines = [
        "input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()\n",
        "input_tensor = input_tensor.unsqueeze(0).to(device) / 255.0\n",
    ]
    if preprocessing.normalizes:
        lines += [
            f"mean = torch.tensor({list(preprocessing.mean)}).to(device).view(1, 3, 1, 1)\n",
            f"std = torch.tensor({list(preprocessing.std)}).to(device).view(1, 3, 1, 1)\n",
            "input_tensor = (input_tensor - mean) / std\n",
        ]
    return lines


def numpy_input_source(preprocessing: ImagePreprocessing) -> list[str]:
    """``img`` to ``input_data`` (``[1, 3, H, W]`` float32, the ONNX input)."""
    lines = ["input_data = np.array(img).astype(np.float32) / 255.0\n"]
    if preprocessing.normalizes:
        lines += [
            f"mean = np.array({list(preprocessing.mean)}, dtype=np.float32)\n",
            f"std = np.array({list(preprocessing.std)}, dtype=np.float32)\n",
            "input_data = (input_data - mean) / std\n",
        ]
    return [*lines, "input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n"]


def indented(lines: list[str], prefix: str = "    ") -> list[str]:
    """``lines`` with ``prefix`` in front of each."""
    return [prefix + line for line in lines]
