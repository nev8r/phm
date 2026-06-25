"""
Task-aware loss registry.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import torch.nn as nn
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.engine.loss.RMSELoss import RMSELoss
from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, REGRESSION


class LossRegistry:
    @staticmethod
    def build(cfg, task_type: str):
        name = str(OmegaConf.select(cfg, "name", default="auto"))
        if name == "auto":
            if task_type == REGRESSION:
                name = "mse"
            elif task_type in CLASSIFICATION_TYPES:
                name = "cross_entropy"
        if name == "mse":
            return nn.MSELoss()
        if name == "rmse":
            return RMSELoss()
        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported loss: {name}")
