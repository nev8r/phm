"""
Feature extractor orchestration.
"""

from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS
from USTC.SSE.BearingPrediction.infra.feature.FeatureRegistry import build_feature_backend
from USTC.SSE.BearingPrediction.infra.feature.FeatureSpec import FeatureSpec


class FeatureExtractor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def extract(self, index: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
        backends = list(OmegaConf.select(self.cfg, "backends", default=[]))
        if not backends:
            raise ValueError("feature.backends must contain at least one backend")

        merged = index[list(FEATURE_INDEX_COLUMNS)].copy()
        backend_reports: List[Dict] = []
        seen_features = set()

        for backend_cfg in backends:
            frame = build_feature_backend(backend_cfg).extract(index)
            duplicate = seen_features.intersection(frame.feature_columns)
            if duplicate:
                raise ValueError(f"Duplicate feature columns: {sorted(duplicate)}")
            seen_features.update(frame.feature_columns)
            merged = merged.merge(
                frame.data[["sample_uid", *frame.feature_columns]],
                on="sample_uid",
                how="left",
            )
            backend_reports.append({
                "name": frame.backend_name,
                "type": OmegaConf.select(backend_cfg, "type", default=""),
                "fc_parameters": OmegaConf.select(backend_cfg, "params.fc_parameters", default=None),
                "num_features": len(frame.feature_columns),
            })

        spec = FeatureSpec(
            name=str(OmegaConf.select(self.cfg, "name", default="features")),
            version=str(OmegaConf.select(self.cfg, "version", default="v1")),
            backends=OmegaConf.to_container(self.cfg.backends, resolve=True),
            cleaner=OmegaConf.to_container(OmegaConf.select(self.cfg, "cleaner", default={}), resolve=True),
        ).to_dict()
        return merged, spec, backend_reports
