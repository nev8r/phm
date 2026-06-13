"""
xLSTM Transformer model module

this file is for implementing paper-style feature-sequence RUL models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import math

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY
from USTC.SSE.BearingPrediction.models.transformer import AttentionBlock


class ScalarMemoryLSTMBlock(nn.Module):
    """
    Exponential-gated scalar-memory recurrent block inspired by sLSTM.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, hidden_size * 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        hidden_state = inputs.new_zeros(batch_size, self.hidden_size)
        cell_state = inputs.new_zeros(batch_size, self.hidden_size)
        outputs: list[torch.Tensor] = []
        for step_index in range(sequence_length):
            gate_values = self.gates(torch.cat([inputs[:, step_index], hidden_state], dim=-1))
            input_logits, forget_logits, output_logits, candidate_values = gate_values.chunk(4, dim=-1)
            input_gate = torch.exp(torch.clamp(input_logits, min=-8.0, max=8.0))
            forget_gate = torch.exp(torch.clamp(forget_logits, min=-8.0, max=8.0))
            gate_sum = input_gate + forget_gate + 1e-6
            input_gate = input_gate / gate_sum
            forget_gate = forget_gate / gate_sum
            cell_state = forget_gate * cell_state + input_gate * torch.tanh(candidate_values)
            hidden_state = torch.sigmoid(output_logits) * torch.tanh(cell_state)
            outputs.append(hidden_state)
        return torch.stack(outputs, dim=1)


class MatrixMemoryLSTMBlock(nn.Module):
    """
    Lightweight matrix-memory recurrent block inspired by mLSTM.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.input_gate = nn.Linear(input_size, hidden_size)
        self.forget_gate = nn.Linear(input_size, hidden_size)
        self.output_gate = nn.Linear(input_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        memory_state = inputs.new_zeros(batch_size, self.hidden_size, self.hidden_size)
        normalizer_state = inputs.new_zeros(batch_size, self.hidden_size)
        outputs: list[torch.Tensor] = []
        scale = math.sqrt(float(self.hidden_size))
        for step_index in range(sequence_length):
            step_values = inputs[:, step_index]
            query_values = torch.tanh(self.query(step_values))
            key_values = torch.tanh(self.key(step_values))
            value_values = torch.tanh(self.value(step_values))
            input_gate = torch.exp(torch.clamp(self.input_gate(step_values), min=-8.0, max=8.0))
            forget_gate = torch.sigmoid(self.forget_gate(step_values))
            gated_value = input_gate * value_values
            memory_update = torch.einsum("bi,bj->bij", gated_value, key_values) / scale
            memory_state = forget_gate.unsqueeze(-1) * memory_state + memory_update
            normalizer_state = forget_gate * normalizer_state + input_gate * torch.abs(key_values)
            memory_read = torch.einsum("bij,bj->bi", memory_state, query_values)
            normalizer = torch.clamp(torch.sum(normalizer_state * torch.abs(query_values), dim=-1, keepdim=True), min=1.0)
            output_gate = torch.sigmoid(self.output_gate(step_values))
            outputs.append(output_gate * self.output_projection(memory_read / normalizer))
        return torch.stack(outputs, dim=1)


class XLSTMEncoderBlock(nn.Module):
    """
    Combine scalar and matrix xLSTM-inspired memory streams.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.scalar_memory = ScalarMemoryLSTMBlock(input_size, hidden_size)
        self.matrix_memory = MatrixMemoryLSTMBlock(input_size, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scalar_output = self.scalar_memory(inputs)
        matrix_output = self.matrix_memory(inputs)
        fused_output = self.fusion(torch.cat([scalar_output, matrix_output], dim=-1))
        return self.norm(fused_output)


class _FeatureSequenceHeadMixin:
    def _build_head(self, hidden_size: int, output_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )


@MODEL_REGISTRY.register("feature_sequence_transformer")
class FeatureSequenceTransformer(BaseBearingModel, _FeatureSequenceHeadMixin):
    """
    Transformer baseline for feature-sequence RUL regression.
    """

    input_kind = "sequence"

    def __init__(
        self,
        feature_size: int,
        output_size: int = 1,
        *,
        sequence_length: int = 10,
        hidden_size: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.randn(1, sequence_length, hidden_size) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.head = self._build_head(hidden_size, output_size, dropout)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_values = self.input_projection(inputs) + self.position_embedding[:, : inputs.size(1)]
        for block in self.blocks:
            encoded_values = block(encoded_values)
            self.latest_attention_weights = block.latest_attention_weights
        pooled_values = encoded_values.mean(dim=1)
        return {"prediction": self.head(pooled_values)}


@MODEL_REGISTRY.register("lstm_transformer")
class LSTMTransformer(BaseBearingModel, _FeatureSequenceHeadMixin):
    """
    LSTM-Transformer baseline for feature-sequence RUL regression.
    """

    input_kind = "sequence"

    def __init__(
        self,
        feature_size: int,
        output_size: int = 1,
        *,
        sequence_length: int = 10,
        hidden_size: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.recurrent_encoder = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.blocks = nn.ModuleList([AttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.head = self._build_head(hidden_size, output_size, dropout)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_values, _ = self.recurrent_encoder(inputs)
        for block in self.blocks:
            encoded_values = block(encoded_values)
            self.latest_attention_weights = block.latest_attention_weights
        pooled_values = encoded_values.mean(dim=1)
        return {"prediction": self.head(pooled_values)}


@MODEL_REGISTRY.register("xlstm_transformer")
class XLSTMTransformer(BaseBearingModel, _FeatureSequenceHeadMixin):
    """
    xLSTM-inspired Transformer for feature-sequence RUL regression.
    """

    input_kind = "sequence"

    def __init__(
        self,
        feature_size: int,
        output_size: int = 1,
        *,
        sequence_length: int = 10,
        hidden_size: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.xlstm_encoder = XLSTMEncoderBlock(hidden_size, hidden_size, dropout)
        self.blocks = nn.ModuleList([AttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.head = self._build_head(hidden_size, output_size, dropout)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_values = self.input_projection(inputs)
        encoded_values = self.xlstm_encoder(encoded_values)
        for block in self.blocks:
            encoded_values = block(encoded_values)
            self.latest_attention_weights = block.latest_attention_weights
        pooled_values = encoded_values.mean(dim=1)
        return {"prediction": self.head(pooled_values)}
