"""
Feature artifact store.
"""

import pickle
from typing import Any, Dict

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class FeatureStore:
    def __init__(self, artifacts: ArtifactManager, write_csv: bool = True):
        self.artifacts = artifacts
        self.write_csv = write_csv

    def save(
            self,
            raw_features: pd.DataFrame,
            cleaned_features: pd.DataFrame,
            feature_spec: Dict[str, Any],
            feature_report: Dict[str, Any],
            cleaner_state: Any = None,
    ) -> None:
        self.artifacts.mkdir("features")
        raw_features.to_parquet(self.artifacts.path("features/raw_features.parquet"), index=False)
        cleaned_features.to_parquet(self.artifacts.path("features/cleaned_features.parquet"), index=False)
        if self.write_csv:
            raw_features.to_csv(self.artifacts.path("features/raw_features.csv"), index=False)
            cleaned_features.to_csv(self.artifacts.path("features/cleaned_features.csv"), index=False)
        self.artifacts.write_json("features/feature_spec.json", feature_spec)
        self.artifacts.write_json("features/feature_report.json", feature_report)
        if cleaner_state is not None:
            with self.artifacts.path("features/cleaner_state.pkl").open("wb") as fh:
                pickle.dump(cleaner_state, fh)
