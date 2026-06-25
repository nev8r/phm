"""
Learning-rate scheduler registry.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from omegaconf import OmegaConf


class SchedulerRegistry:
    @staticmethod
    def build(cfg, optimizer):
        del optimizer
        name = str(OmegaConf.select(cfg, "name", default="none"))
        if name in {"none", "null"}:
            return None
        raise ValueError(f"Unsupported scheduler: {name}")
