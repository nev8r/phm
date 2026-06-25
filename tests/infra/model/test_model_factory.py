"""
Test Stage 5 model factory.

Purpose: verify test stage 5 model factory behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import pytest
import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

from USTC.SSE.BearingPrediction.infra.model.ModelFactory import ModelFactory


def _datamodule(input_dim=3, output_dim=1, input_mode="tabular", task_type="regression"):
    return SimpleNamespace(
        feature_columns=[f"f{i}" for i in range(input_dim)],
        target_columns=[f"y{i}" for i in range(output_dim)],
        task_spec={
            "input_mode": input_mode,
            "task_type": task_type,
            "sequence": {"length": 4} if input_mode == "feature_sequence" else None,
        },
        task_report={},
        input_dim=input_dim,
        output_dim=output_dim,
    )


def test_model_factory_builds_mlp_for_tabular_regression():
    cfg = OmegaConf.create({"name": "mlp", "class_name": "MLP", "params": {"hidden_size": 8}})
    datamodule = _datamodule(input_dim=3, output_dim=1)

    model, spec = ModelFactory(cfg).build(datamodule=datamodule, task_cfg=OmegaConf.create({"task_type": "regression"}))
    output = model(torch.randn(2, 3))

    assert output.shape == (2, 1)
    assert spec["input_dim"] == 3
    assert spec["output_dim"] == 1
    assert spec["hash"]


def test_model_factory_builds_lstm_for_sequence_regression():
    cfg = OmegaConf.create({"name": "lstm", "class_name": "LSTMRegressor", "params": {"hidden_size": 8, "num_layers": 1}})
    datamodule = _datamodule(input_dim=3, output_dim=1, input_mode="feature_sequence")

    model, spec = ModelFactory(cfg).build(datamodule=datamodule, task_cfg=OmegaConf.create({"task_type": "regression"}))
    output = model(torch.randn(2, 4, 3))

    assert output.shape == (2, 1)
    assert spec["class_name"] == "LSTMRegressor"


def test_model_factory_sets_classification_output_dim_from_num_classes():
    cfg = OmegaConf.create({"name": "mlp", "class_name": "MLP", "params": {"hidden_size": 8}})
    task_cfg = OmegaConf.create({"task_type": "multiclass_classification", "target": {"num_classes": 4}})
    datamodule = _datamodule(input_dim=3, output_dim=1, task_type="multiclass_classification")

    model, spec = ModelFactory(cfg).build(datamodule=datamodule, task_cfg=task_cfg)
    output = model(torch.randn(2, 3))

    assert output.shape == (2, 4)
    assert spec["output_dim"] == 4


def test_model_factory_rejects_lstm_for_tabular_task():
    cfg = OmegaConf.create({"name": "lstm", "class_name": "LSTMRegressor", "params": {"hidden_size": 8}})
    datamodule = _datamodule(input_dim=3, output_dim=1, input_mode="tabular")

    with pytest.raises(ValueError, match="sequence"):
        ModelFactory(cfg).build(datamodule=datamodule, task_cfg=OmegaConf.create({"task_type": "regression"}))
