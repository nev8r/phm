"""
Build task example manifests.
"""

from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS
from USTC.SSE.BearingPrediction.infra.task.TaskManifest import TASK_MANIFEST_COLUMNS, TaskManifest
from USTC.SSE.BearingPrediction.infra.task.types import FEATURE_SEQUENCE, TABULAR


class WindowBuilder:
    def build(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            split_result=None,
            cfg: Optional[DictConfig] = None,
    ) -> pd.DataFrame:
        cfg = cfg or OmegaConf.create({})
        data = self._aligned_samples(features, labels, split_result)
        input_mode = str(OmegaConf.select(cfg, "input_mode", default=TABULAR))
        if input_mode == TABULAR:
            manifest = self._build_tabular(data)
        elif input_mode == FEATURE_SEQUENCE:
            manifest = self._build_feature_sequence(data, cfg)
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        manifest = manifest[TASK_MANIFEST_COLUMNS]
        TaskManifest(manifest, TASK_MANIFEST_COLUMNS).validate()
        return manifest

    def _aligned_samples(self, features: pd.DataFrame, labels: pd.DataFrame, split_result) -> pd.DataFrame:
        _ensure_unique(features, "features")
        _ensure_unique(labels, "labels")
        missing = [column for column in FEATURE_INDEX_COLUMNS if column not in features.columns]
        if missing:
            raise ValueError(f"Missing feature metadata columns: {missing}")

        label_sample_uids = labels[["sample_uid"]].copy()
        data = features[list(FEATURE_INDEX_COLUMNS)].merge(label_sample_uids, on="sample_uid", how="inner")
        data["split"] = _split_names(data["sample_uid"], split_result)
        if split_result is not None:
            data = data[data["split"].notna()].copy()
        else:
            data["split"] = data["split"].fillna("all")
        return data.sort_values(["split", "dataset", "bearing_id", "timestep"]).reset_index(drop=True)

    def _build_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        for _, sample in data.iterrows():
            rows.append({
                "example_uid": f"{sample['split']}::{sample['bearing_id']}::{sample['sample_id']}",
                "split": sample["split"],
                "dataset": sample["dataset"],
                "bearing_id": sample["bearing_id"],
                "condition_id": sample["condition_id"],
                "source_group": sample["source_group"],
                "start_sample_uid": sample["sample_uid"],
                "end_sample_uid": sample["sample_uid"],
                "target_sample_uid": sample["sample_uid"],
                "start_timestep": int(sample["timestep"]),
                "end_timestep": int(sample["timestep"]),
                "target_timestep": int(sample["timestep"]),
                "num_timesteps": 1,
                "window_sample_uids": sample["sample_uid"],
            })
        return pd.DataFrame(rows)

    def _build_feature_sequence(self, data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
        sequence_cfg = OmegaConf.select(cfg, "sequence", default={})
        length = int(OmegaConf.select(sequence_cfg, "length", default=8))
        stride = int(OmegaConf.select(sequence_cfg, "stride", default=1))
        target_position = str(OmegaConf.select(sequence_cfg, "target_position", default="last"))
        drop_incomplete = bool(OmegaConf.select(sequence_cfg, "drop_incomplete", default=True))
        if length <= 0:
            raise ValueError("task.sequence.length must be > 0")
        if stride <= 0:
            raise ValueError("task.sequence.stride must be > 0")
        if target_position != "last":
            raise NotImplementedError("Only target_position=last is supported in Stage 4")

        rows: List[Dict] = []
        for (_, _, _), group in data.groupby(["split", "dataset", "bearing_id"], sort=False):
            group = group.sort_values("timestep").reset_index(drop=True)
            if len(group) < length and drop_incomplete:
                continue
            stop = len(group) - length + 1
            for start in range(0, max(stop, 0), stride):
                window = group.iloc[start:start + length]
                if len(window) != length:
                    continue
                target = window.iloc[-1]
                first = window.iloc[0]
                sample_uids = list(window["sample_uid"].astype(str))
                rows.append({
                    "example_uid": f"{target['split']}::{target['bearing_id']}::{first['sample_id']}-{target['sample_id']}",
                    "split": target["split"],
                    "dataset": target["dataset"],
                    "bearing_id": target["bearing_id"],
                    "condition_id": target["condition_id"],
                    "source_group": target["source_group"],
                    "start_sample_uid": first["sample_uid"],
                    "end_sample_uid": target["sample_uid"],
                    "target_sample_uid": target["sample_uid"],
                    "start_timestep": int(first["timestep"]),
                    "end_timestep": int(target["timestep"]),
                    "target_timestep": int(target["timestep"]),
                    "num_timesteps": int(len(window)),
                    "window_sample_uids": "|".join(sample_uids),
                })
        return pd.DataFrame(rows)


def _ensure_unique(frame: pd.DataFrame, name: str) -> None:
    if "sample_uid" not in frame.columns:
        raise ValueError(f"{name} must contain sample_uid")
    if frame["sample_uid"].duplicated().any():
        raise ValueError(f"{name}.sample_uid values must be unique")


def _split_names(sample_uids: pd.Series, split_result) -> pd.Series:
    if split_result is None:
        return pd.Series(["all"] * len(sample_uids), index=sample_uids.index)
    mapping: Dict[str, str] = {}
    mapping.update({sample_uid: "train" for sample_uid in split_result.train_sample_uids})
    mapping.update({sample_uid: "val" for sample_uid in split_result.val_sample_uids})
    mapping.update({sample_uid: "test" for sample_uid in split_result.test_sample_uids})
    return sample_uids.map(mapping)
