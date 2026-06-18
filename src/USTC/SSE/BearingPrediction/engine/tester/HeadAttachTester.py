"""
Head attach tester module

this file is for defining project module behavior

created by zdh

copyright USTC

2026
"""

import copy
from typing import Union, Tuple

import torch
from torch import nn

from USTC.SSE.BearingPrediction.data import Dataset
from USTC.SSE.BearingPrediction.engine.Result import Result
from USTC.SSE.BearingPrediction.engine.tester.ABCTester import ABCTester
from USTC.SSE.BearingPrediction.util.Device import select_torch_device


class HeadAttachTester(ABCTester):
    """
    迁移学习测试器
    当域适应的时候不是完整模型，需要拼接预测头形成完整的模型
    """

    def test(self, model, test_set: Dataset) -> Union[Result, Tuple[Result, ...]]:
        # 使用默认配置补充缺少项
        default_config = {
            'device': select_torch_device(),
            'dtype': torch.float32,
            'head': None,
            'norm': None
        }
        for k, v in default_config.items():
            self.config.setdefault(k, v)

        # 输入数据类型转换
        input_data = torch.from_numpy(test_set.x).to(dtype=self.config['dtype'], device=self.config['device'])

        # 拼接模型
        model = nn.Sequential(model, self.config['head'])

        # 输入模型
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        model.train()

        results = []
        if isinstance(output, tuple):
            for o in output:
                result = Result(y_hat=o.cpu().numpy(),
                                name=test_set.name,
                                entity_map=copy.deepcopy(test_set.entity_map))
                # 若配置了归一化器则反归一化结果
                if self.config.get('norm') is not None:
                    result = self.config['norm'].denorm_result(result)
                results.append(result)
            return tuple(results)

        # 结果打包
        result = Result(y_hat=output.cpu().numpy(),
                        name=test_set.name,
                        entity_map=copy.deepcopy(test_set.entity_map))

        # 若配置了归一化器则反归一化结果
        if self.config.get('norm') is not None:
            result = self.config['norm'].denorm_result(result)

        return result
