"""
CNN-LSTM attention model module

this file is for implementing a CNN-LSTM-AM RUL prediction model

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from USTC.SSE.BearingPrediction.models.base import BaseBearingModel, MODEL_REGISTRY


@MODEL_REGISTRY.register("cnn_lstm_attention")
class CNNLSTMAttention(BaseBearingModel):
    """
    CNN-LSTM model with temporal attention for feature-sequence RUL prediction.
    """

    input_kind = "sequence"

    def __init__(
        self,
        feature_size: int,
        output_size: int = 1,
        *,
        cnn_channels: int | tuple[int, ...] = (32, 64, 64),
        lstm_hidden_size: int = 64,
        lstm_layers: int = 3,
        fc_hidden_sizes: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        use_attention: bool = True,
        task_type: str = "regression",
    ) -> None:
        super().__init__(output_size=output_size, dropout=dropout)
        self.task_type = task_type
        self.feature_size = feature_size
        channel_sizes = self._normalize_channel_sizes(cnn_channels)
        self.cnn_channels = channel_sizes[-1]
        self.cnn_channel_sizes = tuple(channel_sizes)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.fc_hidden_sizes = fc_hidden_sizes
        self.use_attention = use_attention

        encoder_layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channel_sizes:
            encoder_layers.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_channels),
                    nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                ]
            )
            input_channels = output_channels
        encoder_layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        self.feature_encoder = nn.Sequential(*encoder_layers)
        self.temporal_encoder = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention_score = nn.Linear(lstm_hidden_size, 1)
        first_hidden_size, second_hidden_size = fc_hidden_sizes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, first_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() != 3:
            raise ValueError("CNNLSTMAttention expects inputs with shape (batch, sequence_length, feature_size)")

        batch_size, sequence_length, feature_size = inputs.shape
        if feature_size != self.feature_size:
            raise ValueError(f"expected feature_size={self.feature_size}, got {feature_size}")

        encoded_features = self.feature_encoder(inputs.reshape(batch_size * sequence_length, 1, feature_size))
        encoded_sequence = encoded_features.reshape(batch_size, sequence_length, self.cnn_channels)
        temporal_output, _ = self.temporal_encoder(encoded_sequence)
        if self.use_attention:
            attention_logits = self.attention_score(temporal_output).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1)
            context_vector = torch.sum(temporal_output * attention_weights.unsqueeze(-1), dim=1)
            self.latest_attention_weights = attention_weights.detach()
        else:
            context_vector = temporal_output[:, -1, :]
            self.latest_attention_weights = None
        return {"prediction": self.head(context_vector)}

    def get_monitor_state(self) -> dict[str, Any]:
        """
        return model structure metadata for experiment logging

        Returns
        -------
        dict[str, Any]
            metadata dictionary
        """

        state = super().get_monitor_state()
        state.update(
            {
                "cnn_channel_sizes": list(self.cnn_channel_sizes),
                "lstm_hidden_size": self.lstm_hidden_size,
                "lstm_layers": self.lstm_layers,
                "fc_hidden_sizes": list(self.fc_hidden_sizes),
                "use_attention": self.use_attention,
            }
        )
        return state

    @staticmethod
    def _normalize_channel_sizes(cnn_channels: int | tuple[int, ...]) -> tuple[int, int, int]:
        if isinstance(cnn_channels, int):
            return (cnn_channels, cnn_channels, cnn_channels)
        if len(cnn_channels) != 3:
            raise ValueError("cnn_channels must be an int or a tuple with three channel sizes")
        return cnn_channels
