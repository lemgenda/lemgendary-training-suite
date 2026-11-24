"""Regression tests for training, export, DataLoader hardening, and permissive transforms."""

import os

import kornia.augmentation as K
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

from training.data.loaders import (
    TrainLoaderOptions,
    build_train_loader,
)


class DummyDataset(Dataset):
    """Synthetic dataset for deterministic training and DataLoader tests."""

    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.randn(3, 32, 32),
            "y": torch.tensor(idx % 2, dtype=torch.long),
        }


def test_forkserver_dataloader_multiprocessing() -> None:
    """Verify DataLoader with multiprocessing_context and worker_init_fn under Python 3.14."""
    ds = DummyDataset(size=12)
    opts = TrainLoaderOptions()
    loader = build_train_loader(ds, batch_size=4, options=opts)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0]["x"].shape == (4, 3, 32, 32)


def test_permissive_transforms_kornia_torchvision_v2() -> None:
    """Verify torchvision.transforms.v2 and kornia joint pipelines without albumentations."""
    # 1. torchvision v2 transform
    tv_transform = transforms_v2.Compose([
        transforms_v2.RandomHorizontalFlip(p=1.0),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ])
    inp = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    out_tv = tv_transform(inp)
    assert out_tv.shape == (3, 64, 64)

    # 2. kornia augmentation
    k_transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        data_keys=["input"],
    )
    batch_inp = torch.rand(2, 3, 64, 64)
    out_k = k_transform(batch_inp)
    assert out_k.shape == (2, 3, 64, 64)


def test_deterministic_training_loop_safetensors(tmp_path) -> None:
    """Run a deterministic 2-step synthetic training loop and assert safetensors parity."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    ds = DummyDataset(size=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    losses = []
    for step, batch in enumerate(loader):
        if step >= 2:
            break
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        assert torch.isfinite(loss).item()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert len(losses) == 2

    # Save and reload via safetensors
    st_path = str(tmp_path / "model.safetensors")
    save_file(model.state_dict(), st_path)
    loaded_state = load_file(st_path)

    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded_state[k])


def test_onnx_export_and_runtime_parity(tmp_path) -> None:
    """Test ONNX model export and inference parity against PyTorch eager output."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    model.eval()
    dummy_input = torch.randn(1, 3, 16, 16)

    onnx_path = str(tmp_path / "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    assert os.path.exists(onnx_path)
    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)

    # Inference parity
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    assert torch.allclose(torch.tensor(ort_outs[0]), torch.tensor(torch_out), rtol=1e-4, atol=1e-5)
