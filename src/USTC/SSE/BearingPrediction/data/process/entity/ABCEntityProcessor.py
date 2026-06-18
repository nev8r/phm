"""
Abc entity processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

from abc import ABC, abstractmethod

from USTC.SSE.BearingPrediction.data.Entity import Entity


class ABCEntityProcessor(ABC):
    """
    所有对实体的处理器的抽象基类
    """

    @abstractmethod
    def run(self, entity: Entity, key: str) -> Entity:
        raise NotImplementedError

    def __call__(self, entity: Entity, key: str) -> Entity:
        return self.run(entity, key)
