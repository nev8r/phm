"""
CNN model module

this file is for defining convolutional bearing prediction models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY


@MODEL_REGISTRY.register("cnn")
class CNN(BaseBearingModel):
    """
    One-dimensional convolutional regression or classification network.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        hidden_channels: int = 32,
        dropout: float = 0.2,
        task_type: str = "regression",
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.task_type = task_type
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((hidden_channels * 2) * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )
        self.input_size = input_size
        self.hidden_channels = hidden_channels

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        predictions = self.network(inputs)
        return {"prediction": predictions}

