from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class TrainingObjective(ABC):
    """Lifecycle used by objectives with optional auxiliary model state."""

    def setup(self, model: torch.nn.Module, device: torch.device) -> None:
        del model, device

    @abstractmethod
    def __call__(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> ObjectiveOutput:
        raise NotImplementedError

    def after_optimizer_step(self, model: torch.nn.Module) -> None:
        del model

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            raise ValueError(f"{self.__class__.__name__} does not define checkpoint state")

