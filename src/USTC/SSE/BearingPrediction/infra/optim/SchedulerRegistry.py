"""
Learning-rate scheduler registry.
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
