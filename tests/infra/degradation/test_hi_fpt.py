"""
Test Stage 3 health indicator and FPT behavior.
"""

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.degradation.FeatureColumnHICalculator import FeatureColumnHICalculator
from USTC.SSE.BearingPrediction.infra.degradation.ThreeSigmaFPTDetector import ThreeSigmaFPTDetector


def test_feature_column_hi_calculator_normalizes_per_bearing():
    features = pd.DataFrame({
        "sample_uid": [f"b1_{i}" for i in range(5)] + [f"b2_{i}" for i in range(5)],
        "dataset": ["XJTU-SY"] * 10,
        "bearing_id": ["Bearing1_1"] * 5 + ["Bearing1_2"] * 5,
        "condition_id": ["35Hz12kN"] * 10,
        "source_group": [None] * 10,
        "sample_id": [f"{i:06d}" for i in range(5)] * 2,
        "timestep": list(range(5)) * 2,
        "mag__time__rms": [1, 1, 1.2, 3, 5, 0.5, 0.6, 0.8, 2, 4],
    })
    cfg = OmegaConf.create({
        "source_column_candidates": ["mag__time__rms"],
        "smooth": {"enabled": True, "window": 3},
        "normalize_per_bearing": True,
        "direction": "bad_high",
    })

    frame = FeatureColumnHICalculator(cfg).calculate(features)

    assert len(frame.data) == len(features)
    assert frame.data["hi_source_column"].nunique() == 1
    assert frame.data["hi_source_column"].iloc[0] == "mag__time__rms"
    for _, group in frame.data.groupby("bearing_id"):
        assert group["hi_norm"].between(0, 1).all()


def test_three_sigma_fpt_detector_finds_consecutive_threshold_crossing():
    hi = pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(7)],
        "dataset": ["XJTU-SY"] * 7,
        "bearing_id": ["Bearing1_1"] * 7,
        "condition_id": ["35Hz12kN"] * 7,
        "source_group": [None] * 7,
        "sample_id": [f"{i:06d}" for i in range(7)],
        "timestep": list(range(7)),
        "hi_smooth": [1.0, 1.1, 1.0, 1.2, 5.0, 6.0, 7.0],
    })
    cfg = OmegaConf.create({
        "healthy_ratio": 0.4,
        "sigma_ratio": 3.0,
        "min_delta": 0.0,
        "consecutive_points": 2,
        "fallback": "healthy_ratio",
    })

    payload = ThreeSigmaFPTDetector(cfg).detect(hi, source_column="mag__time__rms")
    result = payload["results"][0]

    assert result["fpt_index"] == 4
    assert result["success"] is True
    assert result["fallback_used"] is False


def test_three_sigma_fpt_detector_uses_fallback_when_no_crossing():
    hi = pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(4)],
        "dataset": ["XJTU-SY"] * 4,
        "bearing_id": ["Bearing1_1"] * 4,
        "condition_id": ["35Hz12kN"] * 4,
        "source_group": [None] * 4,
        "sample_id": [f"{i:06d}" for i in range(4)],
        "timestep": list(range(4)),
        "hi_smooth": [1.0, 1.0, 1.0, 1.0],
    })
    cfg = OmegaConf.create({
        "healthy_ratio": 0.5,
        "sigma_ratio": 3.0,
        "consecutive_points": 2,
        "fallback": "healthy_ratio",
    })

    payload = ThreeSigmaFPTDetector(cfg).detect(hi, source_column="mag__time__rms")
    result = payload["results"][0]

    assert result["fpt_index"] == 2
    assert result["success"] is False
    assert result["fallback_used"] is True
