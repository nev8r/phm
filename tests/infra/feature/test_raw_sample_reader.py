"""
Test Stage 2 raw sample reading.

Purpose: verify test stage 2 raw sample reading behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder


def test_raw_sample_reader_reads_xjtu_csv_as_h_v(tmp_path):
    root = create_fake_xjtu_root(tmp_path / "xjtu")
    index = IndexBuilder().build(OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}}))
    row = index[index["bearing_id"] == "Bearing1_1"].iloc[0]

    signal, channels = RawSampleReader().read(row)

    assert signal.shape == (32, 2)
    assert channels == ["h", "v"]
    assert np.isfinite(signal).all()


def test_raw_sample_reader_reads_phm2012_headerless_and_semicolon(tmp_path):
    root = create_fake_phm2012_root(tmp_path / "phm2012")
    index = IndexBuilder().build(OmegaConf.create({"dataset": {"name": "PHM2012", "root": str(root)}}))
    comma_row = index[index["bearing_id"] == "Bearing1_1"].iloc[0]
    semicolon_row = index[index["bearing_id"] == "Bearing1_4"].iloc[0]

    comma_signal, comma_channels = RawSampleReader().read(comma_row)
    semicolon_signal, semicolon_channels = RawSampleReader().read(semicolon_row)

    assert comma_signal.shape == (32, 2)
    assert semicolon_signal.shape == (32, 2)
    assert comma_channels == ["h", "v"]
    assert semicolon_channels == ["h", "v"]


def test_raw_sample_reader_rejects_nan_or_inf(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Horizontal_vibration_signals,Vertical_vibration_signals\n1.0,inf\n", encoding="utf-8")
    row = {
        "dataset": "XJTU-SY",
        "file_path": str(path),
        "bearing_id": "Bearing1_1",
    }

    with pytest.raises(ValueError, match="NaN or Inf"):
        RawSampleReader().read(row)
