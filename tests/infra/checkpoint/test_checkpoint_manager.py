"""
Test Stage 5 checkpoint manager.

Purpose: verify test stage 5 checkpoint manager behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import torch
from torch import nn, optim

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.checkpoint.CheckpointManager import CheckpointManager


def test_checkpoint_manager_saves_and_loads_last_and_best(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    manager = CheckpointManager(ArtifactManager(tmp_path), enabled=True)
    payload = {
        "epoch": 1,
        "global_step": 3,
        "model": model,
        "optimizer": optimizer,
        "scheduler": None,
        "best_metric": 0.5,
        "best_epoch": 1,
        "model_spec": {"name": "linear"},
        "task_spec": {"name": "toy"},
        "trainer_config": {"max_epochs": 1},
        "feature_columns": ["f1", "f2"],
        "target_columns": ["y"],
        "history": [{"epoch": 1}],
    }

    manager.save_last(payload)
    manager.save_best(payload)
    loaded = manager.load(tmp_path / "checkpoints" / "best.ckpt")

    assert (tmp_path / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "checkpoints" / "best.ckpt").exists()
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert loaded["epoch"] == 1
    assert loaded["model_spec"]["name"] == "linear"
