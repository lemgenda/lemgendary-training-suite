"""Single-process multi-GPU ``nn.DataParallel`` strategy."""

from __future__ import annotations

import torch
from torch import nn

from training.parallel.base import ParallelStrategy, generic_prepare_batch


class DataParallelStrategy(ParallelStrategy):
    """Replicate the model across visible GPUs with ``nn.DataParallel``."""

    name = "dp"

    def __init__(self, model, device, *, device_ids=None, output_device=0, **base_options):
        """``base_options`` (``model_key``, ``config``) go to :class:`ParallelStrategy`."""
        super().__init__(model, device, **base_options)
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.output_device = output_device

    def setup(self):
        if not torch.cuda.is_available() or len(self.device_ids) <= 1:
            self._wrapped = self.raw_model.to(self.device)
            return self._wrapped
        self._wrapped = nn.DataParallel(
            self.raw_model, device_ids=self.device_ids, output_device=self.output_device
        )
        return self._wrapped

    def prepare_batch(self, inputs, targets, tasks):
        inputs, targets, _ = generic_prepare_batch(inputs, targets, self.device, self._task_type())
        if isinstance(tasks, torch.Tensor):
            tasks = tasks.to(self.device, non_blocking=True)
        return inputs, targets, tasks
