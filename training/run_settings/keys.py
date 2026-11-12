"""Names of the epoch settings of a run (shared by capture, pinning, policy and appliers)."""

from __future__ import annotations

LOCALITY = "data.locality"
BATCH = "batch"
LR_SCHEDULE = "lr_schedule"
WEIGHT_DECAY = "weight_decay"
GRAD_CLIP = "grad_clip"
LOADER = "loader"
EPOCH_SETTING_KEYS: tuple[str, ...] = (
    LOCALITY,
    BATCH,
    LR_SCHEDULE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    LOADER,
)
"""Settings that may change between epochs of a run."""

HARDWARE_SECTION = "hardware"
LOADER_ARGS: tuple[str, ...] = ("num_workers", "val_num_workers")
"""CLI loader options; ``config.yaml`` ``hardware`` holds the same names."""
