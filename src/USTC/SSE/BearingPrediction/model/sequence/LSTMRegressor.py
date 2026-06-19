"""
LSTM sequence model with a configurable output head.
"""

from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            output_dim: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        head_input_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(head_input_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)
