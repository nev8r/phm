"""
Test Stage 1 bearing-level splitters.
"""

from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.split.CrossConditionSplitter import CrossConditionSplitter
from USTC.SSE.BearingPrediction.infra.split.LeaveOneBearingOutSplitter import LeaveOneBearingOutSplitter
from USTC.SSE.BearingPrediction.infra.split.OfficialPHM2012Splitter import OfficialPHM2012Splitter
from USTC.SSE.BearingPrediction.infra.split.SplitRegistry import build_splitter


def _xjtu_index(root):
    cfg = OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}})
    return IndexBuilder().build(cfg)


def _phm_index(root):
    cfg = OmegaConf.create({"dataset": {"name": "PHM2012", "root": str(root)}})
    return IndexBuilder().build(cfg)


def test_leave_one_bearing_out_splitter_uses_disjoint_bearings(tmp_path):
    index = _xjtu_index(create_fake_xjtu_root(tmp_path / "xjtu"))
    cfg = OmegaConf.create({
        "name": "xjtu_leave_one_bearing_out",
        "condition_id": "35Hz12kN",
        "test_bearing_id": "Bearing1_5",
        "val_bearing_id": "Bearing1_4",
    })

    result = LeaveOneBearingOutSplitter(cfg).split(index)

    assert result.test_bearings == ["Bearing1_5"]
    assert result.val_bearings == ["Bearing1_4"]
    assert "Bearing1_5" not in result.train_bearings
    assert "Bearing1_4" not in result.train_bearings
    assert set(result.train_sample_uids).isdisjoint(result.val_sample_uids)
    assert set(result.train_sample_uids).isdisjoint(result.test_sample_uids)
    assert result.report()["ok"] is True


def test_cross_condition_splitter_uses_condition_groups(tmp_path):
    index = _xjtu_index(create_fake_xjtu_root(tmp_path / "xjtu"))
    cfg = OmegaConf.create({
        "name": "xjtu_cross_condition",
        "train_condition_ids": ["35Hz12kN"],
        "val_condition_ids": ["37.5Hz11kN"],
        "test_condition_ids": ["40Hz10kN"],
    })

    result = CrossConditionSplitter(cfg).split(index)

    assert result.train_bearings == ["Bearing1_1", "Bearing1_2", "Bearing1_4", "Bearing1_5"]
    assert result.val_bearings == ["Bearing2_1"]
    assert result.test_bearings == ["Bearing3_1"]
    assert result.report()["ok"] is True


def test_phm2012_official_splitter_uses_source_group_and_val_bearings(tmp_path):
    index = _phm_index(create_fake_phm2012_root(tmp_path / "phm2012"))
    cfg = OmegaConf.create({
        "name": "phm2012_official",
        "mode": "source_group",
        "train_source_group": "Learning_set",
        "test_source_group": "Full_Test_Set",
        "val_bearings": ["Bearing2_2"],
    })

    result = OfficialPHM2012Splitter(cfg).split(index)

    assert result.train_bearings == ["Bearing1_1", "Bearing2_1"]
    assert result.val_bearings == ["Bearing2_2"]
    assert result.test_bearings == ["Bearing1_3"]
    assert result.report()["ok"] is True


def test_split_registry_builds_configured_splitter():
    cfg = OmegaConf.create({"name": "xjtu_leave_one_bearing_out"})

    splitter = build_splitter(cfg)

    assert isinstance(splitter, LeaveOneBearingOutSplitter)
