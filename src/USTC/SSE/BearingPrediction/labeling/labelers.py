"""
Dataset labeler module

this file is for constructing regression, classification and forecasting datasets

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.data.entities import BearingEntity, BearingWindowDataset
from USTC.SSE.BearingPrediction.feature import FeatureConfig, SignalFeatureExtractor
from USTC.SSE.BearingPrediction.preprocess import (
    DegradationStageResult,
    PREPROCESSOR_REGISTRY,
    PreprocessingPipeline,
    SlidingWindowConfig,
    SlidingWindowSegmenter,
    ThreeSigmaStageStrategy,
    RobustClip,
)


@dataclass
class LabelerConfig:
    """
    Shared labeler configuration.

    Parameters
    ----------
    window_size : int
        signal window size
    stride : int
        window stride
    normalization : str
        normalization name
    """

    window_size: int
    stride: int = 256
    normalization: str = "zscore"


class BearingRulLabeler:
    """
    Build end-to-end or feature-based RUL regression datasets.
    """

    def __init__(
        self,
        window_size: int,
        *,
        stride: int = 256,
        input_representation: str = "raw_signal",
        normalization: str = "zscore",
    ) -> None:
        self.config = LabelerConfig(window_size=window_size, stride=stride, normalization=normalization)
        self.input_representation = input_representation

    def _build_pipeline(self) -> PreprocessingPipeline:
        normalization_transform = PREPROCESSOR_REGISTRY.create(self.config.normalization)
        return PreprocessingPipeline([RobustClip(), normalization_transform])

    def label(self, entity: BearingEntity, channel_name: str) -> BearingWindowDataset:
        """
        construct a regression dataset from a bearing entity

        Parameters
        ----------
        entity : BearingEntity
            source entity
        channel_name : str
            channel name

        Returns
        -------
        BearingWindowDataset
            labeled dataset
        """

        segmenter = SlidingWindowSegmenter(SlidingWindowConfig(self.config.window_size, self.config.stride))
        pipeline = self._build_pipeline()
        feature_extractor = SignalFeatureExtractor(FeatureConfig(sample_rate=entity.sample_rate))

        input_windows: list[np.ndarray] = []
        target_values: list[float] = []
        metadata_records: list[dict[str, object]] = []
        feature_records: list[dict[str, float]] = []

        for sample_row in entity.samples.to_dict("records"):
            processed_signal = pipeline.apply(np.asarray(sample_row[channel_name], dtype=float))
            signal_windows = segmenter.segment(processed_signal)
            window_features = feature_extractor.extract(signal_windows)
            for window_index, signal_window in enumerate(signal_windows):
                input_windows.append(signal_window if self.input_representation == "raw_signal" else window_features.iloc[window_index].to_numpy(dtype=np.float32))
                target_values.append(float(sample_row["rul"]))
                metadata_records.append(
                    {
                        "entity_id": entity.entity_id,
                        "dataset_name": entity.dataset_name,
                        "sample_index": int(sample_row["sample_index"]),
                        "window_index": window_index,
                        "channel_name": channel_name,
                    }
                )
                feature_records.append(window_features.iloc[window_index].to_dict())

        feature_frame = pd.DataFrame.from_records(feature_records)
        return BearingWindowDataset(
            inputs=np.stack(input_windows).astype(np.float32),
            targets=np.asarray(target_values, dtype=np.float32),
            metadata_frame=pd.DataFrame.from_records(metadata_records),
            task_type="regression",
            target_name="rul",
            input_name=self.input_representation,
            feature_frame=feature_frame,
        )


class BearingStageLabeler:
    """
    Build degradation stage classification datasets.
    """

    def __init__(
        self,
        window_size: int,
        *,
        stride: int = 256,
        stage_strategy: object | None = None,
        normalization: str = "zscore",
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.stage_strategy = stage_strategy or ThreeSigmaStageStrategy()
        self.normalization = normalization

    def _build_pipeline(self) -> PreprocessingPipeline:
        normalization_transform = PREPROCESSOR_REGISTRY.create(self.normalization)
        return PreprocessingPipeline([RobustClip(), normalization_transform])

    def label(self, entity: BearingEntity, channel_name: str) -> tuple[BearingWindowDataset, DegradationStageResult]:
        """
        construct a stage classification dataset

        Parameters
        ----------
        entity : BearingEntity
            source entity
        channel_name : str
            channel name

        Returns
        -------
        tuple[BearingWindowDataset, DegradationStageResult]
            stage dataset and stage partition result
        """

        segmenter = SlidingWindowSegmenter(SlidingWindowConfig(self.window_size, self.stride))
        pipeline = self._build_pipeline()
        feature_extractor = SignalFeatureExtractor(FeatureConfig(sample_rate=entity.sample_rate))

        snapshot_indicator_records: list[float] = []
        snapshot_windows: list[list[np.ndarray]] = []
        snapshot_feature_tables: list[pd.DataFrame] = []
        for sample_row in entity.samples.to_dict("records"):
            processed_signal = pipeline.apply(np.asarray(sample_row[channel_name], dtype=float))
            signal_windows = segmenter.segment(processed_signal)
            feature_table = feature_extractor.extract(signal_windows)
            health_indicator = float(feature_extractor.build_health_indicator(feature_table).mean())
            snapshot_indicator_records.append(health_indicator)
            snapshot_windows.append(signal_windows)
            snapshot_feature_tables.append(feature_table)

        stage_result = self.stage_strategy.fit_predict(np.asarray(snapshot_indicator_records, dtype=float))
        inputs: list[np.ndarray] = []
        targets: list[int] = []
        metadata_records: list[dict[str, object]] = []
        feature_records: list[dict[str, float]] = []

        for sample_position, sample_row in enumerate(entity.samples.to_dict("records")):
            for window_index, signal_window in enumerate(snapshot_windows[sample_position]):
                inputs.append(signal_window)
                targets.append(int(stage_result.stage_labels[sample_position]))
                metadata_records.append(
                    {
                        "entity_id": entity.entity_id,
                        "dataset_name": entity.dataset_name,
                        "sample_index": int(sample_row["sample_index"]),
                        "window_index": window_index,
                        "channel_name": channel_name,
                        "health_indicator": stage_result.health_indicator[sample_position],
                    }
                )
                feature_records.append(snapshot_feature_tables[sample_position].iloc[window_index].to_dict())

        dataset = BearingWindowDataset(
            inputs=np.stack(inputs).astype(np.float32),
            targets=np.asarray(targets, dtype=np.int64),
            metadata_frame=pd.DataFrame.from_records(metadata_records),
            task_type="classification",
            target_name="stage_label",
            input_name="raw_signal",
            feature_frame=pd.DataFrame.from_records(feature_records),
        )
        return dataset, stage_result


class HealthIndicatorLabeler:
    """
    Build scalar-sequence forecasting datasets for rolling prediction experiments.
    """

    def __init__(self, history_size: int, horizon: int = 1) -> None:
        self.history_size = history_size
        self.horizon = horizon

    def label(self, health_indicator: np.ndarray) -> BearingWindowDataset:
        """
        construct a forecasting dataset from a health indicator series

        Parameters
        ----------
        health_indicator : np.ndarray
            health indicator values

        Returns
        -------
        BearingWindowDataset
            forecasting dataset
        """

        inputs: list[np.ndarray] = []
        targets: list[float] = []
        metadata_records: list[dict[str, object]] = []

        last_start = health_indicator.size - self.history_size - self.horizon + 1
        for start_index in range(max(last_start, 0)):
            end_index = start_index + self.history_size
            target_index = end_index + self.horizon - 1
            inputs.append(np.asarray(health_indicator[start_index:end_index], dtype=np.float32))
            targets.append(float(health_indicator[target_index]))
            metadata_records.append({"sample_index": start_index, "target_index": target_index})

        if not inputs:
            raise ValueError("health indicator sequence is too short for the requested history_size and horizon")

        return BearingWindowDataset(
            inputs=np.stack(inputs).astype(np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            metadata_frame=pd.DataFrame.from_records(metadata_records),
            task_type="regression",
            target_name="health_indicator",
            input_name="health_indicator",
            feature_frame=None,
        )
