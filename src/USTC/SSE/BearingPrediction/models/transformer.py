"""
Transformer model module

this file is for defining attention based bearing prediction models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY


class AttentionBlock(nn.Module):
    """
    Single transformer attention block that keeps attention weights.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.latest_attention_weights: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention_output, attention_weights = self.attention(inputs, inputs, inputs, need_weights=True, average_attn_weights=False)
        self.latest_attention_weights = attention_weights.detach()
        output_values = self.norm1(inputs + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(output_values)
        return self.norm2(output_values + self.dropout(feed_forward_output))


@MODEL_REGISTRY.register("transformer")
class Transformer(BaseBearingModel):
    """
    Lightweight transformer encoder for regression or classification.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        task_type: str = "regression",
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.task_type = task_type
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )
        self.sequence_length = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        encoded_values = self.input_projection(inputs) + self.position_embedding[:, : inputs.size(1)]
        for block in self.blocks:
            encoded_values = block(encoded_values)
            self.latest_attention_weights = block.latest_attention_weights
        pooled_values = encoded_values.mean(dim=1)
        predictions = self.head(pooled_values)
        return {"prediction": predictions}

