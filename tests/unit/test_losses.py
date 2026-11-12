"""Tests for multitask restoration loss handling and task routing."""

import torch

from data.sample_labels import LANDMARK_ABSENT
from training.core_loop import uses_perceptual_loss
from training.losses import (
    MULTITASK_HEAD_TO_INDEX,
    CombinedLoss,
    _face_detection_loss,
    _resolve_task_index_tensor,
    autocast_device_type,
)


def test_resolve_task_index_tensor_various_representations() -> None:
    """Verify task representation resolution across tensor, tuple, str, int, None."""
    device = torch.device("cpu")

    # None
    assert _resolve_task_index_tensor(None, 4, device) is None

    # Empty tensor
    assert _resolve_task_index_tensor(torch.tensor([]), 4, device) is None

    # Tensor integer
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    res = _resolve_task_index_tensor(t, 3, device)
    assert res is not None
    assert res.dtype == torch.long
    assert torch.equal(res, torch.tensor([1, 2, 3], dtype=torch.long))

    # Tuple of strings (e.g. from DataLoader collation)
    str_tuple = ("dehaze_indoor", "denoise", "superres")
    res_tuple = _resolve_task_index_tensor(str_tuple, 3, device)
    assert res_tuple is not None
    expected = torch.tensor(
        [
            MULTITASK_HEAD_TO_INDEX["dehaze_indoor"],
            MULTITASK_HEAD_TO_INDEX["denoise"],
            MULTITASK_HEAD_TO_INDEX["superres"],
        ],
        dtype=torch.long,
    )
    assert torch.equal(res_tuple, expected)

    # Single string expanded to batch size
    res_str = _resolve_task_index_tensor("deblur", 4, device)
    assert res_str is not None
    assert res_str.shape == (4,)
    assert torch.all(res_str == MULTITASK_HEAD_TO_INDEX["deblur"])


def test_combined_loss_multitask_restoration_forward() -> None:
    """Verify CombinedLoss runs cleanly with multitask restoration tuple outputs."""
    loss_fn = CombinedLoss("restoration")
    pred_imgs = torch.randn(2, 3, 32, 32)
    pred_logits = torch.randn(2, 11)
    preds = (pred_imgs, pred_logits)
    targets = torch.randn(2, 3, 32, 32)

    # 1. With tuple of strings
    loss_tuple = loss_fn(preds, targets, ("dehaze_indoor", "lowlight"))
    assert isinstance(loss_tuple, torch.Tensor)
    assert not torch.isnan(loss_tuple)
    assert loss_tuple.item() > 0

    # 2. With single string
    loss_str = loss_fn(preds, targets, "denoise")
    assert isinstance(loss_str, torch.Tensor)
    assert not torch.isnan(loss_str)

    # 3. With tensor indices
    loss_tensor = loss_fn(preds, targets, torch.tensor([0, 3]))
    assert isinstance(loss_tensor, torch.Tensor)
    assert not torch.isnan(loss_tensor)

    # 4. With None task_idx
    loss_none = loss_fn(preds, targets, None)
    assert isinstance(loss_none, torch.Tensor)
    assert not torch.isnan(loss_none)


def test_face_detection_loss_masks_absent_landmarks() -> None:
    """Box-only face labels (landmarks ``LANDMARK_ABSENT``) add no landmark loss."""
    p_bbox = torch.full((2, 4), 0.5)
    p_conf = torch.full((2, 1), 0.9)
    p_landm = torch.full((2, 10), 0.5)
    box_only = torch.tensor([[1.0, 0.5, 0.5, 0.5, 0.5] + [LANDMARK_ABSENT] * 10] * 2)
    with_landmarks = box_only.clone()
    with_landmarks[:, 5:] = 0.5

    loss_box_only = _face_detection_loss((p_bbox, p_conf, p_landm), box_only)
    loss_exact = _face_detection_loss((p_bbox, p_conf, p_landm), with_landmarks)
    assert torch.allclose(loss_box_only, loss_exact)

    far_landmarks = with_landmarks.clone()
    far_landmarks[:, 5:] = 0.0
    assert _face_detection_loss((p_bbox, p_conf, p_landm), far_landmarks) > loss_exact


def test_autocast_device_type_follows_tensor_device() -> None:
    """Autocast regions use the tensor's device type (CPU, MPS, XPU), CUDA only on CUDA."""
    assert autocast_device_type(torch.device("cpu")) == "cpu"
    assert autocast_device_type(torch.device("cuda", 0)) == "cuda"
    assert autocast_device_type(torch.device("mps")) == "mps"
    assert autocast_device_type(torch.device("meta")) == "cpu"


def test_perceptual_loss_follows_registry_loss_fn() -> None:
    """Only ``loss_fn`` names with an LPIPS term enable the perceptual loss (audit 3.2.3)."""
    assert uses_perceptual_loss("l1_lpips")
    assert uses_perceptual_loss("L1_LPIPS")
    assert not uses_perceptual_loss("l1")
    assert not uses_perceptual_loss("")
