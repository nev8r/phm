"""
Paper model module

this file is for defining CBAM-CNN-LSTM and ResCNN-LSTM paper models

created by zyj

copyright USTC

2026
"""

import torch
from torch import nn


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        padding = kernel_size // 2
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weights = self.channel_attention(x).unsqueeze(-1)
        x = x * channel_weights
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True).values
        temporal_weights = self.temporal_attention(torch.cat([avg_pool, max_pool], dim=1))
        return x * temporal_weights


class CBAMCNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        conv_channels: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            CBAM1D(conv_channels),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1])


class PaperCBAMCNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=10, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=10, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            CBAM1D(64, reduction=cbam_reduction, kernel_size=cbam_kernel_size),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {feature_dim}")
        x = x.reshape(batch_size * sequence_length, 1, feature_dim)
        x = self.encoder(x).squeeze(-1)
        x = x.reshape(batch_size, sequence_length, -1)
        output, _ = self.lstm(x)
        return self.regressor(output[:, -1])


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResCNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        conv_channels: int = 64,
        lstm_layers: int = 1,
        residual_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.residual_encoder = nn.Sequential(
            *[
                ResidualTemporalBlock(conv_channels, kernel_size=3, dropout=dropout)
                for _ in range(residual_blocks)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.residual_encoder(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1])


class CNNLSTMMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 64,
        conv_channels: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1])
