"""
Task artifact store.
"""

from typing import Dict, List

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class TaskStore:
    def __init__(self, artifacts: ArtifactManager, write_csv: bool = True):
        self.artifacts = artifacts
        self.write_csv = write_csv

    def save(
            self,
            manifest: pd.DataFrame,
            task_spec: Dict,
            task_report: Dict,
            feature_columns: List[str],
            target_columns: List[str],
    ) -> None:
        self.artifacts.mkdir("task")
        manifest.to_parquet(self.artifacts.path("task/task_manifest.parquet"), index=False)
        if self.write_csv:
            manifest.to_csv(self.artifacts.path("task/task_manifest.csv"), index=False)
        self.artifacts.write_json("task/task_spec.json", task_spec)
        self.artifacts.write_json("task/task_report.json", task_report)
        self.artifacts.write_text("task/feature_columns.txt", "\n".join(feature_columns) + "\n")
        self.artifacts.write_text("task/target_columns.txt", "\n".join(target_columns) + "\n")
