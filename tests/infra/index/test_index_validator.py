"""
Test Stage 1 sample index validation.

Purpose: verify test stage 1 sample index validation behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.index.IndexValidator import IndexValidator


def test_index_validator_accepts_valid_index(tmp_path):
    root = create_fake_xjtu_root(tmp_path / "xjtu")
    cfg = OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}})
    index = IndexBuilder().build(cfg)

    report = IndexValidator().validate(index, strict=True)

    assert report["ok"] is True
    assert report["num_rows"] == 7
    assert report["num_bearings"] == 6
    assert {check["name"] for check in report["checks"]} >= {
        "required_columns_present",
        "sample_uid_unique",
        "file_path_exists",
        "timestep_monotonic_within_bearing",
    }


def test_index_validator_rejects_duplicate_sample_uid(tmp_path):
    root = create_fake_xjtu_root(tmp_path / "xjtu")
    cfg = OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}})
    index = IndexBuilder().build(cfg)
    broken = pd.concat([index, index.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="sample_uid_unique"):
        IndexValidator().validate(broken, strict=True)
