"""
Tester base module

this file is for defining common testing workflow behavior

created by zdh

copyright USTC

2026
"""

from abc import ABC, abstractmethod

from USTC.SSE.BearingPrediction.data import Dataset
from USTC.SSE.BearingPrediction.engine.Result import Result


class ABCTester(ABC):
    """
    所有测试器的接口
    """

    def __init__(self, config: dict = None):
        # 初始化训练配置
        self.config = config if config else {}

    def __call__(self, model, test_set: Dataset) -> Result:
        return self.test(model, test_set)

    @abstractmethod
    def test(self, model, test_set: Dataset) -> Result:
        pass
