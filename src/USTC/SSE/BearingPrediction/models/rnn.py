"""
RNN model module

this file is for defining recurrent bearing prediction models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY


@MODEL_REGISTRY.register("rnn")
class RNN(BaseBearingModel):
    """
    Gated recurrent unit model for sequence prediction.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task_type: str = "regression",
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.task_type = task_type
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.sequence_length = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        encoded_output, hidden_state = self.encoder(inputs)
        del encoded_output
        last_hidden_state = hidden_state[-1]
        predictions = self.head(last_hidden_state)
        return {"prediction": predictions}

