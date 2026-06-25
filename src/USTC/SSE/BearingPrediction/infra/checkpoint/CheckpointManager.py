"""
Checkpoint manager.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from pathlib import Path
from typing import Dict, Union

import torch

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class CheckpointManager:
    def __init__(self, artifacts: ArtifactManager, enabled: bool = True):
        self.artifacts = artifacts
        self.enabled = enabled

    def save_last(self, payload: Dict) -> Path:
        return self._save("last.ckpt", payload)

    def save_best(self, payload: Dict) -> Path:
        return self._save("best.ckpt", payload)

    def _save(self, filename: str, payload: Dict) -> Path:
        if not self.enabled:
            return self.artifacts.path(f"checkpoints/{filename}")
        self.artifacts.mkdir("checkpoints")
        path = self.artifacts.path(f"checkpoints/{filename}")
        torch.save(_checkpoint_payload(payload), path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> Dict:
        return torch.load(Path(path), map_location="cpu")


def _checkpoint_payload(payload: Dict) -> Dict:
    model = payload["model"]
    optimizer = payload["optimizer"]
    scheduler = payload.get("scheduler")
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": payload.get("best_metric"),
        "best_epoch": payload.get("best_epoch"),
        "model_spec": payload["model_spec"],
        "task_spec": payload["task_spec"],
        "trainer_config": payload["trainer_config"],
        "feature_columns": payload["feature_columns"],
        "target_columns": payload["target_columns"],
        "history": payload["history"],
    }
