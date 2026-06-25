"""
Test Stage 2 feature backends.

Purpose: verify test stage 2 feature backends behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np
from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.feature.ManualProcessorFeatureBackend import ManualProcessorFeatureBackend
from USTC.SSE.BearingPrediction.infra.feature.TsfreshFeatureBackend import TsfreshFeatureBackend
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder


def _xjtu_index(root):
    return IndexBuilder().build(OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}}))


def test_manual_processor_feature_backend_outputs_stable_columns(tmp_path):
    index = _xjtu_index(create_fake_xjtu_root(tmp_path / "xjtu"))
    cfg = OmegaConf.create({
        "name": "manual_basic",
        "type": "manual_processor",
        "params": {
            "include_magnitude": True,
            "time": {
                "enabled": True,
                "features": ["rms", "kurtosis"],
            },
            "spectral": {
                "enabled": True,
                "features": ["centroid", "entropy"],
                "include_dc": False,
            },
        },
    })

    frame = ManualProcessorFeatureBackend(cfg).extract(index)

    assert len(frame.data) == len(index)
    assert "sample_uid" in frame.data
    assert "h__time__rms" in frame.feature_columns
    assert "v__time__kurtosis" in frame.feature_columns
    assert "mag__time__rms" in frame.feature_columns
    assert "h__spectral__centroid" in frame.feature_columns
    assert np.isfinite(frame.data[frame.feature_columns].to_numpy()).all()


def test_tsfresh_feature_backend_outputs_prefixed_minimal_features(tmp_path):
    index = _xjtu_index(create_fake_xjtu_root(tmp_path / "xjtu")).head(2)
    cfg = OmegaConf.create({
        "name": "tsfresh_minimal",
        "type": "tsfresh",
        "params": {
            "fc_parameters": "minimal",
            "n_jobs": 0,
            "chunksize": None,
            "disable_progressbar": True,
            "include_magnitude": False,
            "prefix": "tsfresh",
        },
    })

    frame = TsfreshFeatureBackend(cfg).extract(index)

    assert len(frame.data) == len(index)
    assert any(column.startswith("tsfresh__h__") for column in frame.feature_columns)
    assert any(column.startswith("tsfresh__v__") for column in frame.feature_columns)
    assert np.isfinite(frame.data[frame.feature_columns].to_numpy()).all()
