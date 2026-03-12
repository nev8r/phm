"""
Model base module

this file is for defining shared model behavior and registry

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from USTC.SSE.BearingPrediction.common.registry import ComponentRegistry

MODEL_REGISTRY = ComponentRegistry("model")


class BaseBearingModel(nn.Module):
    """
    Shared interface for bearing models.
    """

    input_kind = "sequence"
    task_type = "regression"

    def __init__(self, output_size: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_size = output_size
        self.dropout_probability = dropout
        self.latest_attention_weights: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_monitor_state(self) -> dict[str, Any]:
        """
        return model structure metadata for experiment logging

        Returns
        -------
        dict[str, Any]
            metadata dictionary
        """

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        return {
            "model_name": self.__class__.__name__,
            "model_structure": repr(self),
            "parameter_count": int(parameter_count),
            "dropout_probability": self.dropout_probability,
            "output_size": self.output_size,
            "task_type": self.task_type,
            "input_kind": self.input_kind,
        }

    def maybe_get_attention(self) -> torch.Tensor | None:
        """
        return latest attention weights when available

        Returns
        -------
        torch.Tensor | None
            attention weights
        """

        return self.latest_attention_weights

