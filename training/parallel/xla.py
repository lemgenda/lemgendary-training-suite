"""PyTorch/XLA (Google Cloud TPU) parallel training strategy."""

from __future__ import annotations

import logging
from typing import Any

from training.parallel.base import ParallelStrategy, generic_prepare_batch
from training.utils.lazy_imports import load_torch_xla_model

logger = logging.getLogger("lemtrain.parallel.xla")


class XlaParallelStrategy(ParallelStrategy):
    """Place the model on an XLA TPU device and synchronize via ``torch_xla``."""

    name = "xla"

    def setup(self) -> Any:
        self._wrapped = self.raw_model.to(self.device)
        return self._wrapped

    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        inputs, targets, _ = generic_prepare_batch(inputs, targets, self.device, self._task_type())
        return inputs, targets, tasks

    @property
    def is_main_process(self) -> bool:
        try:
            xm = load_torch_xla_model()
            return int(xm.get_ordinal()) == 0
        except (ImportError, RuntimeError, AttributeError):
            return True

    def barrier(self) -> None:
        """Synchronize all XLA TPU cores via ``xm.rendezvous``."""
        try:
            xm = load_torch_xla_model()
            xm.rendezvous("lemtrain_barrier")
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("XLA barrier skipped: %s", exc)
