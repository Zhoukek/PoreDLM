"""Standalone cosine classification bank for frozen signal representations.

The ``hidden_bank`` and ``log_temperature`` state names, initialization order,
and scoring arithmetic are compatible with the original V003 event readout.
The trainable rows are class weights, not necessarily empirical cluster means.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import torch
from torch import nn


_MIN_NORM = 1e-12


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _finite_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _strict_normalize(values: torch.Tensor, name: str) -> torch.Tensor:
    if not values.is_floating_point() or not bool(torch.isfinite(values).all().detach()):
        raise ValueError(f"{name} must contain finite floating-point values")
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all().detach()) or bool((norms <= _MIN_NORM).any().detach()):
        raise ValueError(f"{name} contains a zero-norm row")
    return values / norms


class KmerHiddenBank(nn.Module):
    """Classify hidden vectors using normalized class rows and temperature.

    Input shape is ``[..., hidden_dim]`` with at least two dimensions; output
    shape is ``[..., num_classes]``. Invalid or near-zero rows are rejected
    instead of silently normalized. No encoder or project-local module is
    imported, so cached hidden vectors can be classified independently.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1024,
        hidden_dim: int = 768,
        initial_temperature: float = 0.1,
        learnable_temperature: bool = True,
        centroids: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_integer(num_classes, "num_classes")
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        temperature = _finite_positive_scalar(initial_temperature, "initial_temperature")
        if not isinstance(learnable_temperature, (bool, np.bool_)):
            raise ValueError("learnable_temperature must be boolean")
        self.learnable_temperature = bool(learnable_temperature)

        self.hidden_bank = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        nn.init.normal_(self.hidden_bank, mean=0.0, std=self.hidden_dim ** -0.5)
        log_temperature = torch.tensor(math.log(temperature), dtype=torch.float32)
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature, persistent=True)
        if centroids is not None:
            self.initialize_from_centroids(centroids)

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive scalar temperature without detaching gradients."""
        value = self.log_temperature.exp()
        if not bool(torch.isfinite(value).detach()) or not bool((value > 0).detach()):
            raise ValueError("temperature became non-finite or non-positive")
        return value

    def initialize_from_centroids(
        self, centroids: np.ndarray | torch.Tensor
    ) -> "KmerHiddenBank":
        """Copy unit-normalized centroids into trainable class weights."""
        values = torch.as_tensor(
            centroids, dtype=self.hidden_bank.dtype, device=self.hidden_bank.device
        ).detach()
        if values.shape != self.hidden_bank.shape:
            raise ValueError(
                f"centroids must have shape {(self.num_classes, self.hidden_dim)}"
            )
        normalized = _strict_normalize(values, "centroids")
        with torch.no_grad():
            self.hidden_bank.copy_(normalized)
        return self

    def normalized_bank(self) -> torch.Tensor:
        """Return unit class rows while retaining gradients to the weights."""
        return _strict_normalize(self.hidden_bank, "hidden bank")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
            raise ValueError("hidden must have shape [...,hidden_dim]")
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden final dimension must equal {self.hidden_dim}")
        observations = _strict_normalize(hidden, "hidden")
        cosine = observations @ self.normalized_bank().transpose(0, 1)
        return cosine / self.temperature.to(dtype=cosine.dtype)
