"""
Test Stage 3 label builder and store.
"""

import json

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.label.LabelBuilder import LabelBuilder
from USTC.SSE.BearingPrediction.infra.label.LabelStore import LabelStore


def _index():
    rows = []
    for bearing_id in ["Bearing1_1", "Bearing1_2"]:
        for timestep in range(5):
            rows.append({
                "sample_uid": f"{bearing_id}_{timestep}",
                "dataset": "XJTU-SY",
                "bearing_id": bearing_id,
                "condition_id": "35Hz12kN",
                "source_group": None,
                "sample_id": f"{timestep:06d}",
                "timestep": timestep,
                "sample_interval_seconds": 60,
                "fault_element": "outer",
            })
    return pd.DataFrame(rows)


def _features(index):
    data = index[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep"]].copy()
    data["mag__time__rms"] = [1, 1, 1.2, 3, 5, 0.5, 0.6, 0.8, 2, 4]
    return data


def test_label_builder_creates_degradation_basic_outputs():
    index = _index()
    cfg = OmegaConf.create({
        "name": "degradation_basic",
        "version": "v1",
        "requires_features": True,
        "hi": {
            "source_column_candidates": ["mag__time__rms"],
            "smooth": {"enabled": True, "window": 1},
            "normalize_per_bearing": True,
        },
        "fpt": {
            "healthy_ratio": 0.4,
            "sigma_ratio": 3.0,
            "consecutive_points": 1,
            "fallback": "healthy_ratio",
        },
        "outputs": [
            {"type": "linear_rul", "name": "linear_rul", "params": {"normalize": True}},
            {"type": "piecewise_rul", "name": "piecewise_rul", "params": {"normalize": True}},
            {"type": "health_state", "name": "health_state", "params": {"state_names": {0: "healthy", 1: "slight", 2: "moderate", 3: "severe"}, "post_fpt_boundaries": [0.4, 0.8]}},
            {"type": "early_fault", "name": "early_fault", "params": {"normal_value": 0, "abnormal_value": 1}},
            {"type": "fault_type_stage", "name": "fault_type_stage", "params": {"enabled_for_dataset": ["XJTU-SY"], "severe_state_id": 3, "normal_label": "normal", "degraded_label": "degraded_unknown"}},
        ],
    })

    labels, spec, report, hi, fpt = LabelBuilder(cfg).build(index=index, raw_features=_features(index))

    assert len(labels) == len(index)
    assert "linear_rul_norm" in labels
    assert "piecewise_rul_norm" in labels
    assert "health_state_id" in labels
    assert "early_fault" in labels
    assert "fault_type_stage_name" in labels
    assert report["ok"] is True
    assert spec["hash"]
    assert len(hi) == len(index)
    assert len(fpt["results"]) == 2


def test_label_store_writes_labels_hi_and_fpt(tmp_path):
    labels = pd.DataFrame({"sample_uid": ["a"], "linear_rul_norm": [0.0]})
    spec = {"name": "linear_rul", "hash": "abc"}
    report = {"ok": True}
    hi = pd.DataFrame({"sample_uid": ["a"], "hi_norm": [0.1]})
    fpt = {"results": [{"bearing_id": "Bearing1_1", "fpt_index": 0}]}

    LabelStore(ArtifactManager(tmp_path), write_csv=True).save(labels, spec, report, hi=hi, fpt=fpt)

    assert (tmp_path / "labels" / "labels.parquet").exists()
    assert (tmp_path / "labels" / "labels.csv").exists()
    assert json.loads((tmp_path / "labels" / "label_spec.json").read_text())["name"] == "linear_rul"
    assert json.loads((tmp_path / "labels" / "label_report.json").read_text())["ok"] is True
    assert (tmp_path / "hi" / "hi.parquet").exists()
    assert (tmp_path / "hi" / "fpt.json").exists()
