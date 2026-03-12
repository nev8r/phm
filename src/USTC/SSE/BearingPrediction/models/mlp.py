"""
MLP model module

this file is for defining feature based multilayer perceptron models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY


@MODEL_REGISTRY.register("mlp")
class MLP(BaseBearingModel):
    """
    Multilayer perceptron for feature-based modeling.
    """

    input_kind = "feature"

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        hidden_size: int = 64,
        dropout: float = 0.2,
        task_type: str = "regression",
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.task_type = task_type
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions = self.network(inputs)
        return {"prediction": predictions}

