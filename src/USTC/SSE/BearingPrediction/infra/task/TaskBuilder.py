"""
Task data module builder.
"""

from fnmatch import fnmatch
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS
from USTC.SSE.BearingPrediction.infra.task.DataModule import DataModule
from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset
from USTC.SSE.BearingPrediction.infra.task.TaskRegistry import normalize_input_mode, normalize_task_type
from USTC.SSE.BearingPrediction.infra.task.TaskReport import build_task_report
from USTC.SSE.BearingPrediction.infra.task.TaskSpec import TaskSpec
from USTC.SSE.BearingPrediction.infra.task.TaskValidator import TaskValidator
from USTC.SSE.BearingPrediction.infra.task.WindowBuilder import WindowBuilder
from USTC.SSE.BearingPrediction.infra.task.types import FEATURE_SEQUENCE


class TaskBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.validator = TaskValidator()

    def build(self, features: pd.DataFrame, labels: pd.DataFrame, split_result=None) -> DataModule:
        self.validator.validate_sample_alignment(features, labels)
        target_columns = list(OmegaConf.select(self.cfg, "target.columns", default=[]))
        self.validator.validate_target_columns(labels, target_columns)

        feature_columns = self._select_feature_columns(features)
        self.validator.validate_feature_columns(feature_columns)

        task_type = normalize_task_type(str(OmegaConf.select(self.cfg, "task_type", default="regression")))
        input_mode = normalize_input_mode(str(OmegaConf.select(self.cfg, "input_mode", default="tabular")))
        feature_source = str(OmegaConf.select(self.cfg, "feature_source", default="cleaned"))
        manifest = WindowBuilder().build(features, labels, split_result=split_result, cfg=self.cfg)
        sequence = _to_plain(OmegaConf.select(self.cfg, "sequence", default=None))
        spec = TaskSpec(
            name=str(OmegaConf.select(self.cfg, "name", default="task")),
            version=str(OmegaConf.select(self.cfg, "version", default="v1")),
            task_type=task_type,
            input_mode=input_mode,
            feature_source=feature_source,
            feature_columns=feature_columns,
            target_columns=target_columns,
            sequence=sequence if input_mode == FEATURE_SEQUENCE else None,
        ).to_dict()
        report = build_task_report(
            task_name=spec["name"],
            task_type=task_type,
            input_mode=input_mode,
            feature_source=feature_source,
            feature_columns=feature_columns,
            target_columns=target_columns,
            manifest=manifest,
            labels=labels,
            sequence=sequence,
        )
        return DataModule(
            train=self._dataset_for_split("train", manifest, features, labels, feature_columns, target_columns, input_mode, task_type),
            val=self._dataset_for_split("val", manifest, features, labels, feature_columns, target_columns, input_mode, task_type),
            test=self._dataset_for_split("test", manifest, features, labels, feature_columns, target_columns, input_mode, task_type),
            all=self._dataset_for_split("all", manifest, features, labels, feature_columns, target_columns, input_mode, task_type),
            task_manifest=manifest,
            feature_columns=feature_columns,
            target_columns=target_columns,
            task_spec=spec,
            task_report=report,
        )

    def _select_feature_columns(self, features: pd.DataFrame) -> List[str]:
        cfg = OmegaConf.select(self.cfg, "feature_columns", default={})
        candidates = [column for column in features.columns if column not in FEATURE_INDEX_COLUMNS]
        include = str(OmegaConf.select(cfg, "include", default="all"))
        include_patterns = list(OmegaConf.select(cfg, "include_patterns", default=[]))
        exclude_patterns = list(OmegaConf.select(cfg, "exclude_patterns", default=[]))
        exclude_columns = set(OmegaConf.select(cfg, "exclude_columns", default=[]))

        if include == "all":
            selected = list(candidates)
        elif include == "patterns":
            selected = [column for column in candidates if _matches_any(column, include_patterns)]
        else:
            raise ValueError(f"Unsupported feature_columns.include: {include}")

        if include_patterns and include == "all":
            selected = [
                column for column in selected
                if _matches_any(column, include_patterns)
            ]
        selected = [column for column in selected if column not in exclude_columns]
        selected = [column for column in selected if not _matches_any(column, exclude_patterns)]
        return selected

    @staticmethod
    def _dataset_for_split(
            split_name: str,
            manifest: pd.DataFrame,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            feature_columns: List[str],
            target_columns: List[str],
            input_mode: str,
            task_type: str,
    ):
        subset = manifest[manifest["split"] == split_name].reset_index(drop=True)
        if subset.empty:
            return None
        return TaskDataset(features, labels, subset, feature_columns, target_columns, input_mode, task_type)


def _matches_any(column: str, patterns: List[str]) -> bool:
    return bool(patterns) and any(fnmatch(column, pattern) for pattern in patterns)


def _to_plain(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value
