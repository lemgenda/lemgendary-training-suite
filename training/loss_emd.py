"""Earth Mover's Distance loss of the NIMA score distributions (R4-C4).

NIMA (Talebi and Milanfar, 2018, Eq. 1) defines, for ``N`` ordered score bins,
``EMD = (1/N * sum_k |CDF_p(k) - CDF_t(k)|^r)^(1/r)`` with ``r = 2``. The registry option
``stabilizers.emd_form`` selects the per-sample form, averaged over the batch:

* ``nima`` (default): ``sqrt(mean_k (CDF_p - CDF_t)^2 + 1e-8)``, the paper form; the
  epsilon keeps the square-root gradient finite at a perfect match;
* ``squared``: ``mean_k (CDF_p - CDF_t)^2``, the squared EMD of Hou, Yu and Samaras (2016).

Both sit on the scale of the other loss terms (soft Spearman, rank). Runs that started before
the option existed trained with ``sum_k`` (the squared EMD times ``N``); their checkpoints
carry no form, so a resume keeps that :data:`LEGACY_EMD_FORM`. The form is part of the
metric definition: fixed for a run and recorded in its checkpoints (:data:`EMD_FORM_KEY`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

EMD_FORM_KEY = "emd_form"
"""Stabilizer option and checkpoint payload key of the EMD form."""
NIMA_EMD_FORM = "nima"
SQUARED_EMD_FORM = "squared"
LEGACY_EMD_FORM = "legacy_sum"
"""The pre-R4 form (``sum_k``), kept only for runs resumed from checkpoints without a form."""
REGISTRY_EMD_FORMS: tuple[str, ...] = (NIMA_EMD_FORM, SQUARED_EMD_FORM)
EMD_FORMS: tuple[str, ...] = (*REGISTRY_EMD_FORMS, LEGACY_EMD_FORM)
DEFAULT_EMD_FORM = NIMA_EMD_FORM
SQRT_EPSILON = 1e-8


def registry_emd_form(stabilizers: Mapping[str, Any] | None) -> str:
    """The validated ``stabilizers.emd_form`` (``nima`` when absent)."""
    form = str((stabilizers or {}).get(EMD_FORM_KEY, DEFAULT_EMD_FORM))
    if form not in REGISTRY_EMD_FORMS:
        raise ValueError(f"stabilizers.emd_form must be one of {REGISTRY_EMD_FORMS}, got {form!r}")
    return form


def emd_loss(p_probs: torch.Tensor, t_probs: torch.Tensor, form: str) -> torch.Tensor:
    """Batch mean of the per-sample EMD between ``[B, N]`` distributions in ``form``."""
    squared = (torch.cumsum(p_probs, dim=-1) - torch.cumsum(t_probs, dim=-1)) ** 2
    if form == NIMA_EMD_FORM:
        per_sample = torch.sqrt(squared.mean(dim=-1) + SQRT_EPSILON)
    elif form == SQUARED_EMD_FORM:
        per_sample = squared.mean(dim=-1)
    elif form == LEGACY_EMD_FORM:
        per_sample = squared.sum(dim=-1)
    else:
        raise ValueError(f"Unknown EMD form {form!r}; expected one of {EMD_FORMS}")
    return per_sample.mean()
