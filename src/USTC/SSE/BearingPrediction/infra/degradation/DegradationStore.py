"""
Degradation artifact store.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, Optional

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


class DegradationStore:
    def __init__(self, artifacts: ArtifactManager):
        self.artifacts = artifacts

    def save(self, hi: Optional[pd.DataFrame] = None, fpt: Optional[Dict] = None) -> None:
        if hi is None and fpt is None:
            return
        self.artifacts.mkdir("hi")
        if hi is not None:
            hi.to_parquet(self.artifacts.path("hi/hi.parquet"), index=False)
        if fpt is not None:
            self.artifacts.write_json("hi/fpt.json", fpt)
