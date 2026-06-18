"""
Test Stage 1 sample index building.
"""

from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.index.SampleIndex import SAMPLE_INDEX_COLUMNS


def test_index_builder_creates_xjtu_sample_index(tmp_path):
    root = create_fake_xjtu_root(tmp_path / "xjtu")
    cfg = OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}})

    index = IndexBuilder().build(cfg)

    assert list(index.columns) == SAMPLE_INDEX_COLUMNS
    assert len(index) == 7
    first = index[index["sample_uid"] == "XJTU-SY::Bearing1_1::000001"].iloc[0]
    second = index[index["sample_uid"] == "XJTU-SY::Bearing1_1::000002"].iloc[0]
    assert first["condition_id"] == "35Hz12kN"
    assert first["sample_id"] == "000001"
    assert first["timestep"] == 0
    assert second["timestep"] == 1
    assert first["expected_points"] == 32768
    assert first["sample_interval_seconds"] == 60
    assert first["channel_names"] == "Horizontal Vibration,Vertical Vibration"
    assert first["fault_element"] == "outer"
    assert first["file_path"].endswith("35Hz12kN/Bearing1_1/1.csv")


def test_index_builder_creates_phm2012_sample_index(tmp_path):
    root = create_fake_phm2012_root(tmp_path / "phm2012")
    cfg = OmegaConf.create({"dataset": {"name": "PHM2012", "root": str(root)}})

    index = IndexBuilder().build(cfg)

    assert list(index.columns) == SAMPLE_INDEX_COLUMNS
    assert len(index) == 5
    assert "temp_00001" not in ",".join(index["file_path"])
    first = index[index["sample_uid"] == "PHM2012::Bearing1_1::000001"].iloc[0]
    test_row = index[index["bearing_id"] == "Bearing1_3"].iloc[0]
    assert first["source_group"] == "Learning_set"
    assert first["condition_id"] == "Condition1"
    assert first["expected_points"] == 2560
    assert first["sample_interval_seconds"] == 10
    assert first["fault_element"] is None
    assert test_row["source_group"] == "Full_Test_Set"
