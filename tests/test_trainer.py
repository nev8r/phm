"""
Test trainer test module

this file is for verifying test trainer behavior

created by zy

copyright USTC

2026
"""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from USTC.SSE.BearingPrediction.data import Dataset
from USTC.SSE.BearingPrediction.engine.trainer.BaseTrainer import BaseTrainer
from USTC.SSE.BearingPrediction.util.Device import select_torch_device


class TrainerTest(unittest.TestCase):
    def test_base_trainer_moves_external_dataloader_batches_to_config_device(self):
        device = select_torch_device()
        model = nn.Linear(2, 1)
        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        data_loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        train_set = Dataset(name="external-loader-train")

        trainer = BaseTrainer(config={
            "device": device,
            "epochs": 1,
            "criterion": nn.MSELoss(),
            "data_loader": data_loader,
            "lr": 1e-3,
        })

        losses = trainer(model, train_set)

        self.assertEqual(next(model.parameters()).device.type, device.type)
        self.assertEqual(len(losses["MSELoss"]), 1)
        self.assertTrue(np.isfinite(losses["MSELoss"][0]))


if __name__ == "__main__":
    unittest.main()
