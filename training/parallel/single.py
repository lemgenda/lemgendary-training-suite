"""Single-device (one GPU or CPU) training strategy."""

from __future__ import annotations

from training.parallel.base import ParallelStrategy, generic_prepare_batch


class SingleGPUStrategy(ParallelStrategy):
    """Place the model on one device without any wrapping."""

    name = "single"

    def setup(self):
        self._wrapped = self.raw_model.to(self.device)
        return self._wrapped

    def prepare_batch(self, inputs, targets, tasks):
        inputs, targets, _ = generic_prepare_batch(inputs, targets, self.device, self._task_type())
        return inputs, targets, tasks
