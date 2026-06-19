"""
Test Stage 2 feature store.
"""

import json

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.feature.FeatureStore import FeatureStore


def test_feature_store_writes_feature_artifacts(tmp_path):
    store = FeatureStore(ArtifactManager(tmp_path), write_csv=True)
    raw = pd.DataFrame({"sample_uid": ["a"], "feature": [1.0]})
    cleaned = pd.DataFrame({"sample_uid": ["a"], "feature": [0.0]})
    spec = {"name": "manual_basic", "hash": "abc"}
    report = {"ok": True, "num_samples": 1}
    cleaner_state = {"feature_columns": ["feature"]}

    store.save(raw, cleaned, spec, report, cleaner_state)

    assert (tmp_path / "features" / "raw_features.parquet").exists()
    assert (tmp_path / "features" / "raw_features.csv").exists()
    assert (tmp_path / "features" / "cleaned_features.parquet").exists()
    assert (tmp_path / "features" / "cleaned_features.csv").exists()
    assert json.loads((tmp_path / "features" / "feature_spec.json").read_text())["name"] == "manual_basic"
    assert json.loads((tmp_path / "features" / "feature_report.json").read_text())["ok"] is True
    assert (tmp_path / "features" / "cleaner_state.pkl").exists()
