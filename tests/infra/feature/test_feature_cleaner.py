"""
Test Stage 2 feature cleaning.
"""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.FeatureCleaner import FeatureCleaner


def test_feature_cleaner_fits_only_train_samples_and_drops_constant_columns():
    features = pd.DataFrame({
        "sample_uid": ["train_1", "train_2", "test_1"],
        "dataset": ["XJTU-SY"] * 3,
        "bearing_id": ["Bearing1_1", "Bearing1_2", "Bearing1_5"],
        "condition_id": ["35Hz12kN"] * 3,
        "source_group": [None] * 3,
        "sample_id": ["000001"] * 3,
        "timestep": [0, 0, 0],
        "feature_signal": [8.0, 12.0, 10000.0],
        "feature_missing": [1.0, np.nan, 10000.0],
        "constant_feature": [5.0, 5.0, 999.0],
    })
    cfg = OmegaConf.create({
        "enabled": True,
        "imputer": "median",
        "scaler": "standard",
        "drop_constant": True,
        "constant_threshold": 1.0e-12,
    })

    cleaner = FeatureCleaner(cfg)
    cleaned = cleaner.fit_transform(features, train_sample_uids=["train_1", "train_2"])

    assert cleaner.scaler_mean["feature_signal"] == 10.0
    assert cleaner.imputer_values["feature_missing"] == 1.0
    assert "constant_feature" in cleaner.dropped_columns
    assert list(cleaned["sample_uid"]) == ["train_1", "train_2", "test_1"]
    assert np.isfinite(cleaned[cleaner.feature_columns].to_numpy()).all()
