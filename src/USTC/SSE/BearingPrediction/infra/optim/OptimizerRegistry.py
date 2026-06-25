"""
Optimizer registry.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import torch.optim as optim
from omegaconf import OmegaConf


class OptimizerRegistry:
    @staticmethod
    def build(cfg, parameters):
        name = str(OmegaConf.select(cfg, "name", default="adam"))
        lr = float(OmegaConf.select(cfg, "lr", default=0.001))
        weight_decay = float(OmegaConf.select(cfg, "weight_decay", default=0.0))
        if name == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            momentum = float(OmegaConf.select(cfg, "momentum", default=0.0))
            return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
        raise ValueError(f"Unsupported optimizer: {name}")
