"""
RMSE LOSS module

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):  # 防止除以0
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='mean')
        return torch.sqrt(mse + self.eps)
