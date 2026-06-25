"""
Manual processor feature backend.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.data.process.array.SpectralFeatureProcessor import SpectralFeatureProcessor
from USTC.SSE.BearingPrediction.data.process.array.TimeDomainFeatureProcessor import TimeDomainFeatureProcessor
from USTC.SSE.BearingPrediction.infra.feature.ABCFeatureBackend import ABCFeatureBackend
from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS, FeatureFrame
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader


class ManualProcessorFeatureBackend(ABCFeatureBackend):
    def __init__(self, cfg: DictConfig, reader: RawSampleReader = None):
        self.cfg = cfg
        self.reader = reader or RawSampleReader()
        self.name = str(OmegaConf.select(cfg, "name", default="manual_basic"))
        self.params = OmegaConf.select(cfg, "params", default={})

    def extract(self, index: pd.DataFrame) -> FeatureFrame:
        rows: List[Dict] = []
        for _, sample in index.iterrows():
            signal, channels = self.reader.read(sample)
            rows.append({**_index_values(sample), **self._extract_sample(sample, signal, channels)})

        data = pd.DataFrame(rows)
        feature_columns = [column for column in data.columns if column not in FEATURE_INDEX_COLUMNS]
        frame = FeatureFrame(
            data=data,
            index_columns=list(FEATURE_INDEX_COLUMNS),
            feature_columns=feature_columns,
            backend_name=self.name,
            feature_set_name=self.name,
            spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        frame.validate()
        return frame

    def _extract_sample(self, sample, signal: np.ndarray, channels: List[str]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        channel_signals = {channel: signal[:, idx] for idx, channel in enumerate(channels)}
        if bool(OmegaConf.select(self.params, "include_magnitude", default=False)):
            channel_signals["mag"] = np.sqrt(np.square(signal[:, 0]) + np.square(signal[:, 1]))

        time_cfg = OmegaConf.select(self.params, "time", default={})
        if bool(OmegaConf.select(time_cfg, "enabled", default=True)):
            time_features = list(OmegaConf.select(time_cfg, "features", default=[]))
            processor = TimeDomainFeatureProcessor(features=time_features)
            for channel, channel_signal in channel_signals.items():
                result = processor.run(channel_signal)
                for feature_name, feature_value in zip(time_features, result):
                    values[f"{channel}__time__{feature_name}"] = float(feature_value)

        spectral_cfg = OmegaConf.select(self.params, "spectral", default={})
        if bool(OmegaConf.select(spectral_cfg, "enabled", default=True)):
            spectral_features = list(OmegaConf.select(spectral_cfg, "features", default=[]))
            processor = SpectralFeatureProcessor(
                sampling_rate=float(sample["sampling_rate"]),
                features=spectral_features,
                include_dc=bool(OmegaConf.select(spectral_cfg, "include_dc", default=False)),
            )
            for channel, channel_signal in channel_signals.items():
                result = processor.run(channel_signal)
                for feature_name, feature_value in zip(spectral_features, result):
                    values[f"{channel}__spectral__{feature_name}"] = float(feature_value)

        return values


def _index_values(sample) -> Dict:
    return {column: sample[column] for column in FEATURE_INDEX_COLUMNS}
