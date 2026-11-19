"""Exponential Moving Average (EMA) shadow model tracker for visual models.

Maintains an exponential moving average copy of model parameters during training:
    \\theta_{ema} = \\beta \\cdot \\theta_{ema} + (1 - \\beta) \\cdot \\theta_t
Provides smoother parameter trajectories, eliminates gradient noise variance,
and acts as a regularizer preventing late-stage overfitting and degradation.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class ModelEMA:
    """Shadow model tracker maintaining an Exponential Moving Average of parameters.

    Args:
        model: PyTorch module whose parameters should be tracked.
        decay: Base EMA decay factor (typically 0.999 or 0.9999).
        warmup_steps: Number of initial steps with accelerated EMA decay.
        use_buffers: Whether to also copy model buffers into the shadow state.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 2000,
        use_buffers: bool = True,
    ) -> None:
        self.decay = float(decay)
        self.warmup_steps = max(0, warmup_steps)
        self.use_buffers = use_buffers
        self.step = 0

        # Create shadow module without computation graph
        self.module: nn.Module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad = False

        self._backup_params: dict[str, torch.Tensor] = {}

    def get_shadow_module(self) -> nn.Module:
        """Return the tracked shadow module."""
        return self.module

    def get_decay(self) -> float:
        """Compute step-dependent decay rate with optional warmup."""
        if self.warmup_steps <= 0:
            return self.decay
        # Dynamic warmup ramping decay smoothly to target decay factor
        # decay_t = min(decay, (1 + step) / (10 + step))
        ramp = (1.0 + float(self.step)) / (10.0 + float(self.step))
        return min(self.decay, ramp)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters using current online model parameters."""
        self.step += 1
        current_decay = self.get_decay()
        one_minus_decay = 1.0 - current_decay

        online_params = dict(model.named_parameters())
        for name, shadow_param in self.module.named_parameters():
            if name in online_params:
                online_p = online_params[name].detach()
                if shadow_param.dtype.is_floating_point:
                    shadow_param.mul_(current_decay).add_(
                        online_p.to(dtype=shadow_param.dtype, device=shadow_param.device),
                        alpha=one_minus_decay,
                    )
                else:
                    shadow_param.copy_(online_p)

        if self.use_buffers:
            online_buffers = dict(model.named_buffers())
            for name, shadow_buf in self.module.named_buffers():
                if name in online_buffers:
                    shadow_buf.copy_(online_buffers[name].detach())

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap online model parameters with shadow parameters for evaluation or export."""
        self._backup_params.clear()
        for name, param in model.named_parameters():
            self._backup_params[name] = param.detach().clone()

        shadow_params = dict(self.module.named_parameters())
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow_params:
                    param.copy_(shadow_params[name])

    def restore(self, model: nn.Module) -> None:
        """Restore backed-up online model parameters after evaluation."""
        if not self._backup_params:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._backup_params:
                    param.copy_(self._backup_params[name])
        self._backup_params.clear()

    def state_dict(self) -> dict[str, Any]:
        """Serialize EMA state including shadow weights and iteration step."""
        return {
            "step": self.step,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "use_buffers": self.use_buffers,
            "shadow_state_dict": self.module.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state from dictionary."""
        self.step = int(state_dict.get("step", 0))
        self.decay = float(state_dict.get("decay", self.decay))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.use_buffers = bool(state_dict.get("use_buffers", self.use_buffers))
        if "shadow_state_dict" in state_dict:
            self.module.load_state_dict(state_dict["shadow_state_dict"])
