"""
Test Stage 5 configurable trainer.

Purpose: verify test stage 5 configurable trainer behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.model.ModelFactory import ModelFactory
from USTC.SSE.BearingPrediction.infra.task.TaskBuilder import TaskBuilder
from USTC.SSE.BearingPrediction.engine.trainer.ConfigurableTrainer import ConfigurableTrainer


def _datamodule():
    rows = []
    for bearing_id, split_name in [("Bearing1_1", "train"), ("Bearing1_2", "val"), ("Bearing1_3", "test")]:
        for timestep in range(2):
            rows.append({
                "sample_uid": f"{bearing_id}_{timestep}",
                "dataset": "XJTU-SY",
                "bearing_id": bearing_id,
                "condition_id": "35Hz12kN",
                "source_group": None,
                "sample_id": f"{timestep:06d}",
                "timestep": timestep,
                "split": split_name,
                "f1": float(timestep),
                "f2": float(timestep + 1),
                "piecewise_rul_norm": float(1 - timestep),
            })
    data = pd.DataFrame(rows)
    features = data[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep", "f1", "f2"]]
    labels = data[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep", "piecewise_rul_norm"]]

    class Split:
        train_sample_uids = data[data["split"] == "train"]["sample_uid"].tolist()
        val_sample_uids = data[data["split"] == "val"]["sample_uid"].tolist()
        test_sample_uids = data[data["split"] == "test"]["sample_uid"].tolist()

    task_cfg = OmegaConf.create({
        "name": "rul_tabular",
        "version": "v1",
        "task_type": "regression",
        "input_mode": "tabular",
        "feature_source": "cleaned",
        "feature_columns": {"include": "all", "exclude_columns": []},
        "target": {"columns": ["piecewise_rul_norm"], "dtype": "float32"},
    })
    return TaskBuilder(task_cfg).build(features, labels, split_result=Split()), task_cfg


def test_configurable_trainer_writes_checkpoints_metrics_and_predictions(tmp_path):
    datamodule, task_cfg = _datamodule()
    model_cfg = OmegaConf.create({"name": "mlp", "class_name": "MLP", "params": {"hidden_size": 8}})
    model, model_spec = ModelFactory(model_cfg).build(datamodule=datamodule, task_cfg=task_cfg)
    cfg = OmegaConf.create({
        "project": {"seed": 42},
        "trainer": {
            "device": "cpu",
            "dtype": "float32",
            "seed": 42,
            "max_epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "optimizer": {"name": "adam", "lr": 0.01, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "loss": {"name": "auto"},
            "metrics": {"name": "auto"},
            "monitor": {"split": "val", "metric": "loss", "mode": "min"},
            "gradient": {"clip_norm": 5.0},
            "eval": {"every_n_epochs": 1, "evaluate_train": True, "evaluate_test_at_end": True},
            "checkpoint": {"enabled": True, "save_best": True, "save_last": True, "resume_from": None},
            "prediction": {"save_train": True, "save_val": True, "save_test": True, "write_csv": False},
        },
    })
    context = SimpleNamespace(artifacts=ArtifactManager(tmp_path), run_dir=tmp_path)

    history = ConfigurableTrainer(cfg, context, datamodule, model, model_spec).train()

    assert len(history) == 2
    assert (tmp_path / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "checkpoints" / "best.ckpt").exists()
    assert (tmp_path / "metrics" / "history.json").exists()
    assert (tmp_path / "metrics" / "val_metrics.json").exists()
    assert (tmp_path / "predictions" / "val_predictions.parquet").exists()
    assert (tmp_path / "trainer" / "trainer_state.json").exists()
