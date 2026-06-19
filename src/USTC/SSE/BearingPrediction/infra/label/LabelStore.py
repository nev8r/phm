"""
Label artifact store.
"""

from typing import Dict, Optional

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.degradation.DegradationStore import DegradationStore


class LabelStore:
    def __init__(self, artifacts: ArtifactManager, write_csv: bool = True):
        self.artifacts = artifacts
        self.write_csv = write_csv

    def save(
            self,
            labels: pd.DataFrame,
            label_spec: Dict,
            label_report: Dict,
            hi: Optional[pd.DataFrame] = None,
            fpt: Optional[Dict] = None,
    ) -> None:
        self.artifacts.mkdir("labels")
        labels.to_parquet(self.artifacts.path("labels/labels.parquet"), index=False)
        if self.write_csv:
            labels.to_csv(self.artifacts.path("labels/labels.csv"), index=False)
        self.artifacts.write_json("labels/label_spec.json", label_spec)
        self.artifacts.write_json("labels/label_report.json", label_report)
        DegradationStore(self.artifacts).save(hi=hi, fpt=fpt)
