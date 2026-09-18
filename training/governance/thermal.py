"""Thermal state managing softmax temperature decay, logit clamping, and turbulence dampening."""

from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger("lemtrain.governance.thermal")


@dataclass
class ThermalState:
    """Controls model confidence calibration, temperature cooling, and output clamping."""

    temperature: float = 1.0
    cooling_factor: float = 0.98
    min_temperature: float = 0.10
    clamp_range: list[float] = field(default_factory=lambda: [15.0, 45.0])
    turbulence_dampening: bool = True

    def step_epoch(self) -> float:
        """Apply cooling factor to decay softmax temperature toward min_temperature."""
        old_temp = self.temperature
        self.temperature = max(self.min_temperature, round(self.temperature * self.cooling_factor, 4))
        logger.debug("Thermal step: temperature %.4f -> %.4f", old_temp, self.temperature)
        return self.temperature

    def dampen_turbulence(self, gradient_norm: float, threshold: float = 10.0) -> bool:
        """Temporarily boost temperature to soften sharp logits when excessive gradient volatility is detected."""
        if self.turbulence_dampening and gradient_norm > threshold:
            old_temp = self.temperature
            self.temperature = min(2.0, round(self.temperature * 1.25, 4))
            logger.warning(
                "Turbulence detected (grad_norm=%.2f > %.2f). Elevated temperature: %.4f -> %.4f",
                gradient_norm,
                threshold,
                old_temp,
                self.temperature,
            )
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for checkpoint payload."""
        return {
            "temperature": self.temperature,
            "cooling_factor": self.cooling_factor,
            "min_temperature": self.min_temperature,
            "clamp_range": list(self.clamp_range),
            "turbulence_dampening": self.turbulence_dampening,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermalState":
        """Deserialize state from dictionary."""
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            cooling_factor=float(data.get("cooling_factor", 0.98)),
            min_temperature=float(data.get("min_temperature", 0.10)),
            clamp_range=list(data.get("clamp_range", [15.0, 45.0])),
            turbulence_dampening=bool(data.get("turbulence_dampening", True)),
        )
