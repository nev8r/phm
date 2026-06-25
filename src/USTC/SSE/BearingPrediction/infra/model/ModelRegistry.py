"""
Model registry.

Purpose: define model components for bearing PHM tasks
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Callable, Dict


class ModelRegistry:
    def __init__(self):
        self._builders: Dict[str, Callable] = {}

    def register(self, name: str, builder: Callable) -> None:
        self._builders[name] = builder

    def build(self, name: str, **kwargs):
        if name not in self._builders:
            raise ValueError(f"Unknown model: {name}")
        return self._builders[name](**kwargs)


MODEL_REGISTRY = ModelRegistry()
