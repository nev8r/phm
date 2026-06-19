"""
Prediction artifact store.
"""

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class PredictionStore:
    def __init__(self, artifacts: ArtifactManager, write_csv: bool = False):
        self.artifacts = artifacts
        self.write_csv = write_csv

    def save(self, split: str, predictions: pd.DataFrame) -> None:
        self.artifacts.mkdir("predictions")
        predictions.to_parquet(self.artifacts.path(f"predictions/{split}_predictions.parquet"), index=False)
        if self.write_csv:
            predictions.to_csv(self.artifacts.path(f"predictions/{split}_predictions.csv"), index=False)
