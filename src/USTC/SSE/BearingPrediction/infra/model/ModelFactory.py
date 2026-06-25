"""
Build models from task-aware configuration.

Purpose: define model components for bearing PHM tasks
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from omegaconf import DictConfig, OmegaConf
from torch import nn

from USTC.SSE.BearingPrediction.infra.model.ModelSpec import ModelSpec
from USTC.SSE.BearingPrediction.infra.task.types import (
    BINARY_CLASSIFICATION,
    FEATURE_SEQUENCE,
    MULTICLASS_CLASSIFICATION,
    TABULAR,
)
from USTC.SSE.BearingPrediction.model.basic.MLP import MLP
from USTC.SSE.BearingPrediction.model.sequence.GRURegressor import GRURegressor
from USTC.SSE.BearingPrediction.model.sequence.LSTMRegressor import LSTMRegressor


class FlattenIfSequence(nn.Module):
    """Adapt sequence feature tensors for tabular models when configs request MLP."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.model(x)


class ModelFactory:
    """Create task-aware model instances and the matching serializable model spec."""

    def __init__(self, cfg: DictConfig):
        """Store the model config chosen by Hydra."""
        self.cfg = cfg

    def build(self, datamodule, task_cfg: DictConfig):
        """Build a model whose input/output dimensions match the constructed task."""
        name = str(OmegaConf.select(self.cfg, "name", default="mlp"))
        class_name = str(OmegaConf.select(self.cfg, "class_name", default=_default_class(name)))
        params = OmegaConf.to_container(OmegaConf.select(self.cfg, "params", default={}), resolve=True)
        input_mode = str(datamodule.task_spec.get("input_mode", OmegaConf.select(task_cfg, "input_mode", default=TABULAR)))
        task_type = str(datamodule.task_spec.get("task_type", OmegaConf.select(task_cfg, "task_type", default="regression")))
        output_dim = _output_dim(datamodule, task_cfg, task_type)

        if class_name == "MLP":
            input_dim = _mlp_input_dim(datamodule, input_mode)
            model = MLP(
                input_size=input_dim,
                hidden_size=int(params.get("hidden_size", 64)),
                output_size=output_dim,
            )
            if input_mode == FEATURE_SEQUENCE:
                model = FlattenIfSequence(model)
        elif class_name == "LSTMRegressor":
            if input_mode != FEATURE_SEQUENCE:
                raise ValueError("LSTMRegressor requires a feature_sequence task")
            input_dim = int(datamodule.input_dim)
            model = LSTMRegressor(
                input_dim=input_dim,
                hidden_size=int(params.get("hidden_size", 64)),
                output_dim=output_dim,
                num_layers=int(params.get("num_layers", 1)),
                dropout=float(params.get("dropout", 0.0)),
                bidirectional=bool(params.get("bidirectional", False)),
            )
        elif class_name == "GRURegressor":
            if input_mode != FEATURE_SEQUENCE:
                raise ValueError("GRURegressor requires a feature_sequence task")
            input_dim = int(datamodule.input_dim)
            model = GRURegressor(
                input_dim=input_dim,
                hidden_size=int(params.get("hidden_size", 64)),
                output_dim=output_dim,
                num_layers=int(params.get("num_layers", 1)),
                dropout=float(params.get("dropout", 0.0)),
                bidirectional=bool(params.get("bidirectional", False)),
            )
        else:
            raise ValueError(f"Unsupported model class_name: {class_name}")

        spec = ModelSpec(
            name=name,
            class_name=class_name,
            input_dim=input_dim,
            output_dim=output_dim,
            params=params,
            input_mode=input_mode,
            task_type=task_type,
        ).to_dict()
        return model, spec


def _default_class(name: str) -> str:
    return {
        "mlp": "MLP",
        "lstm": "LSTMRegressor",
        "gru": "GRURegressor",
    }.get(name, name)


def _mlp_input_dim(datamodule, input_mode: str) -> int:
    if input_mode != FEATURE_SEQUENCE:
        return int(datamodule.input_dim)
    sequence = datamodule.task_spec.get("sequence") or {}
    return int(datamodule.input_dim) * int(sequence.get("length", 1))


def _output_dim(datamodule, task_cfg: DictConfig, task_type: str) -> int:
    if task_type == BINARY_CLASSIFICATION:
        return 2
    if task_type == MULTICLASS_CLASSIFICATION:
        configured = OmegaConf.select(task_cfg, "target.num_classes", default=None)
        if configured is not None and str(configured) != "auto":
            return int(configured)
        return _infer_num_classes(datamodule)
    return int(datamodule.output_dim)


def _infer_num_classes(datamodule) -> int:
    distribution = datamodule.task_report.get("class_distribution", {})
    labels = set()
    for split_counts in distribution.values():
        labels.update(int(key) for key in split_counts.keys())
    if not labels:
        return 1
    return max(labels) + 1
