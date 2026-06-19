"""
Test Stage 3 sample labelers.
"""

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.label.EarlyFaultLabeler import EarlyFaultLabeler
from USTC.SSE.BearingPrediction.infra.label.FaultTypeStageLabeler import FaultTypeStageLabeler
from USTC.SSE.BearingPrediction.infra.label.HealthStateLabeler import HealthStateLabeler
from USTC.SSE.BearingPrediction.infra.label.LinearRulLabeler import LinearRulLabeler
from USTC.SSE.BearingPrediction.infra.label.PiecewiseRulLabeler import PiecewiseRulLabeler


def _index(n=5, dataset="XJTU-SY"):
    return pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(n)],
        "dataset": [dataset] * n,
        "bearing_id": ["Bearing1_1"] * n,
        "condition_id": ["35Hz12kN"] * n,
        "source_group": [None] * n,
        "sample_id": [f"{i:06d}" for i in range(n)],
        "timestep": list(range(n)),
        "sample_interval_seconds": [60] * n,
        "fault_element": ["outer"] * n,
    })


def _fpt(index, fpt_index):
    row = index.iloc[fpt_index]
    return {
        ("XJTU-SY", "Bearing1_1"): {
            "fpt_index": fpt_index,
            "fpt_sample_uid": row["sample_uid"],
            "fpt_timestep": int(row["timestep"]),
        }
    }


def test_linear_rul_labeler_outputs_steps_seconds_and_norm():
    index = _index(n=4)
    labels = LinearRulLabeler(OmegaConf.create({"normalize": True})).label(index)

    assert labels["linear_rul_steps"].tolist() == [3, 2, 1, 0]
    assert labels["linear_rul_seconds"].tolist() == [180, 120, 60, 0]
    assert labels["linear_rul_norm"].round(6).tolist() == [1.0, 0.666667, 0.333333, 0.0]


def test_piecewise_rul_labeler_uses_fpt_plateau():
    index = _index(n=5)
    labels = PiecewiseRulLabeler(OmegaConf.create({"normalize": True})).label(index, _fpt(index, 2))

    assert labels["piecewise_rul_norm"].tolist() == [1.0, 1.0, 1.0, 0.5, 0.0]
    assert labels["piecewise_rul_steps"].tolist() == [2.0, 2.0, 2.0, 1.0, 0.0]


def test_health_state_labeler_splits_post_fpt_progress():
    index = _index(n=10)
    labels = HealthStateLabeler(OmegaConf.create({
        "state_names": {0: "healthy", 1: "slight", 2: "moderate", 3: "severe"},
        "post_fpt_boundaries": [0.4, 0.8],
    })).label(index, _fpt(index, 2))

    assert labels.loc[0, "health_state_name"] == "healthy"
    assert labels.loc[2, "health_state_name"] == "slight"
    assert labels.loc[5, "health_state_name"] == "moderate"
    assert labels.loc[8, "health_state_name"] == "severe"
    assert set(labels["health_state_id"]).issubset({0, 1, 2, 3})


def test_early_fault_labeler_marks_after_fpt_as_abnormal():
    index = _index(n=5)
    labels = EarlyFaultLabeler(OmegaConf.create({"normal_value": 0, "abnormal_value": 1})).label(index, _fpt(index, 3))

    assert labels["early_fault"].tolist() == [0, 0, 0, 1, 1]


def test_fault_type_stage_labeler_only_uses_fault_element_in_severe_stage():
    index = _index(n=4)
    health = pd.DataFrame({
        "sample_uid": index["sample_uid"],
        "health_state_id": [0, 1, 2, 3],
        "health_state_name": ["healthy", "slight", "moderate", "severe"],
    })
    labels, mapping = FaultTypeStageLabeler(OmegaConf.create({
        "enabled_for_dataset": ["XJTU-SY"],
        "severe_state_id": 3,
        "normal_label": "normal",
        "degraded_label": "degraded_unknown",
    })).label(index, health)

    assert labels["fault_type_stage_name"].tolist() == ["normal", "degraded_unknown", "degraded_unknown", "outer"]
    assert mapping["normal"] == 0
    assert "outer" in mapping
