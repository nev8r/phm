"""
tsfresh feature backend.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.ABCFeatureBackend import ABCFeatureBackend
from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS, FeatureFrame
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader


class TsfreshFeatureBackend(ABCFeatureBackend):
    def __init__(self, cfg: DictConfig, reader: RawSampleReader = None):
        self.cfg = cfg
        self.reader = reader or RawSampleReader()
        self.name = str(OmegaConf.select(cfg, "name", default="tsfresh_minimal"))
        self.params = OmegaConf.select(cfg, "params", default={})

    def extract(self, index: pd.DataFrame) -> FeatureFrame:
        from tsfresh import extract_features

        long_frame = self._to_long_frame(index)
        extracted = extract_features(
            long_frame,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=self._fc_parameters(),
            n_jobs=int(OmegaConf.select(self.params, "n_jobs", default=0)),
            chunksize=OmegaConf.select(self.params, "chunksize", default=None),
            disable_progressbar=bool(OmegaConf.select(self.params, "disable_progressbar", default=True)),
        )
        prefix = str(OmegaConf.select(self.params, "prefix", default="tsfresh"))
        extracted = extracted.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        extracted.columns = [f"{prefix}__{column}" for column in extracted.columns]
        extracted.index.name = "sample_uid"
        extracted = extracted.reset_index()

        data = index[list(FEATURE_INDEX_COLUMNS)].merge(extracted, on="sample_uid", how="left")
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

    def _to_long_frame(self, index: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        include_magnitude = bool(OmegaConf.select(self.params, "include_magnitude", default=False))
        for _, sample in index.iterrows():
            signal, channels = self.reader.read(sample)
            sample_uid = sample["sample_uid"]
            channel_signals = {channel: signal[:, idx] for idx, channel in enumerate(channels)}
            if include_magnitude:
                channel_signals["mag"] = np.sqrt(np.square(signal[:, 0]) + np.square(signal[:, 1]))
            for channel, values in channel_signals.items():
                for time_index, value in enumerate(values):
                    rows.append({
                        "id": sample_uid,
                        "time": time_index,
                        "kind": channel,
                        "value": float(value),
                    })
        return pd.DataFrame(rows)

    def _fc_parameters(self):
        from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

        name = str(OmegaConf.select(self.params, "fc_parameters", default="minimal"))
        if name == "minimal":
            return MinimalFCParameters()
        if name == "efficient":
            return EfficientFCParameters()
        if name == "comprehensive":
            return ComprehensiveFCParameters()
        raise ValueError(f"Unsupported tsfresh fc_parameters: {name}")
