"""
Mlp model module

Purpose: define model components for bearing PHM tasks
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
