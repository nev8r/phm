"""
Analysis artifact store.
"""

from typing import Dict

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class AnalysisStore:
    TABLE_KEYS = [
        "feature_summary",
        "rul_correlation",
        "degradation_scores",
        "health_state_separability",
        "early_fault_scores",
        "fault_type_scores",
        "feature_ranking",
    ]

    JSON_KEYS = [
        "analysis_spec",
        "analysis_report",
        "leakage_report",
    ]

    def __init__(self, artifacts: ArtifactManager, write_csv: bool = True, write_figures: bool = True):
        self.artifacts = artifacts
        self.write_csv = write_csv
        self.write_figures = write_figures

    def save(self, outputs: Dict) -> None:
        self.artifacts.mkdir("analysis")
        for key in self.JSON_KEYS:
            if key in outputs and outputs[key] is not None:
                self.artifacts.write_json(f"analysis/{key}.json", outputs[key])
        for key in self.TABLE_KEYS:
            frame = outputs.get(key)
            if isinstance(frame, pd.DataFrame):
                frame.to_parquet(self.artifacts.path(f"analysis/{key}.parquet"), index=False)
                if self.write_csv:
                    frame.to_csv(self.artifacts.path(f"analysis/{key}.csv"), index=False)
        if self.write_figures:
            for path in outputs.get("figures", []) or []:
                del path
