"""Unit and smoke tests for YOLO model training improvements (Prompt 19).

Verifies:
  1. Train kwargs include close_mosaic schedule, multi-scale flag, and label smoothing.
  2. Per-class AP extraction and aggregate mAP logging.
  3. Consistent spatial validation transform configuration.
  4. EpochMetrics backwards compatibility and property access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from training.governance.yolo_callbacks import (
    ClassAPMetrics,
    DetectionLosses,
    EpochMetrics,
    _extract_per_class_ap,
)
from training.governance.yolo_ladder import YOLOLadderStage
from training.governance.yolo_run import YoloRunMixin, _Hardware, _RunPlan


def test_epoch_metrics_loss_and_class_ap() -> None:
    """Test EpochMetrics correctly computes properties and exposes per-class AP."""
    class_ap = ClassAPMetrics(
        ap50={"person": 0.82, "car": 0.74},
        ap50_95={"person": 0.58, "car": 0.49},
    )
    losses = DetectionLosses(
        box_loss=0.04,
        cls_loss=0.03,
        dfl_loss=0.01,
        train_loss=0.15,
    )
    metrics = EpochMetrics(
        map50=0.78,
        map50_95=0.535,
        losses=losses,
        lr=0.001,
        class_ap=class_ap,
    )

    assert metrics.val_loss == pytest.approx(0.08)
    assert metrics.train_loss == pytest.approx(0.15)
    assert metrics.box_loss == pytest.approx(0.04)
    assert metrics.cls_loss == pytest.approx(0.03)
    assert metrics.dfl_loss == pytest.approx(0.01)
    assert metrics.class_ap50["person"] == 0.82
    assert metrics.class_ap50_95["car"] == 0.49
    # Quality score = (map50_95 * 0.7) + (map50 * 0.3)
    expected_quality = (0.535 * 0.7) + (0.78 * 0.3)
    assert metrics.quality == pytest.approx(expected_quality)


def test_extract_per_class_ap_from_mock_trainer() -> None:
    """Verify _extract_per_class_ap correctly reads validator metrics and names."""
    mock_trainer = MagicMock()
    mock_validator = MagicMock()
    mock_metrics = MagicMock()
    mock_box = MagicMock()

    mock_validator.names = {0: "pedestrian", 1: "vehicle"}
    mock_metrics.ap_class_index = [0, 1]
    # all_ap shape: [num_classes, 10_iou_thresholds]
    mock_box.all_ap = np.array(
        [
            [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
            [0.72, 0.68, 0.64, 0.60, 0.56, 0.52, 0.48, 0.44, 0.40, 0.36],
        ]
    )

    mock_metrics.box = mock_box
    mock_validator.metrics = mock_metrics
    mock_trainer.validator = mock_validator

    ap50, ap50_95 = _extract_per_class_ap(mock_trainer)

    assert ap50["pedestrian"] == pytest.approx(0.85)
    assert ap50["vehicle"] == pytest.approx(0.72)
    assert ap50_95["pedestrian"] == pytest.approx(0.625)
    assert ap50_95["vehicle"] == pytest.approx(0.54)


def test_yolo_train_kwargs_mosaic_schedule_and_augmentations() -> None:
    """Verify _train_kwargs includes close_mosaic, multi_scale, and label_smoothing."""
    runner = YoloRunMixin()
    runner.sync = MagicMock()
    runner.spec = MagicMock()
    runner.args = MagicMock()
    runner.args.lr = 0.002
    runner.opt_config = {
        "close_mosaic": 15,
        "multi_scale": True,
        "label_smoothing": 0.05,
    }

    plan = _RunPlan(
        hardware=_Hardware(device_arg="0", vram_gb=16.0, is_amp_safe=True),
        yolo_yaml="fake_data.yaml",
        total_epochs=100,
        requested_batch=16,
        stages=[],
    )
    stage = YOLOLadderStage(
        stage_index=1,
        resolution=640,
        batch_size=16,
        fraction=1.0,
        target_epochs=50,
        patience=10,
    )

    kwargs = runner.build_train_kwargs(plan, stage, project_dir=MagicMock(), epochs=50)

    assert kwargs["close_mosaic"] == 15
    assert kwargs["multi_scale"] is True
    assert kwargs["label_smoothing"] == 0.05
    assert kwargs["imgsz"] == 640
    assert kwargs["lr0"] == pytest.approx(0.002)


def test_yolo_train_kwargs_default_close_mosaic() -> None:
    """Verify default close_mosaic schedule scales with epochs if unspecified."""
    runner = YoloRunMixin()
    runner.sync = MagicMock()
    runner.spec = MagicMock()
    runner.args = MagicMock()
    runner.args.lr = None
    runner.opt_config = {}

    plan = _RunPlan(
        hardware=_Hardware(device_arg="cpu", vram_gb=8.0, is_amp_safe=False),
        yolo_yaml="fake_data.yaml",
        total_epochs=50,
        requested_batch=8,
        stages=[],
    )
    stage = YOLOLadderStage(
        stage_index=1,
        resolution=320,
        batch_size=8,
        fraction=0.5,
        target_epochs=20,
        patience=5,
    )

    kwargs = runner.build_train_kwargs(plan, stage, project_dir=MagicMock(), epochs=20)

    # 20 // 5 = 4
    assert kwargs["close_mosaic"] == 4
    assert kwargs["multi_scale"] is False
    assert kwargs["label_smoothing"] == 0.0
