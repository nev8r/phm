"""
Test Stage 1 bearing-level splitters.

Purpose: verify test stage 1 bearing-level splitters behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.split.BearingIndexSplitter import BearingIndexSplitter
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


def _xjtu_multi_condition_index():
    rows = []
    for condition_id, condition_no in [("35Hz12kN", 1), ("37.5Hz11kN", 2), ("40Hz10kN", 3)]:
        for bearing_idx in range(1, 6):
            bearing_id = f"Bearing{condition_no}_{bearing_idx}"
            for sample_idx in range(2):
                rows.append({
                    "sample_uid": f"XJTU-SY::{bearing_id}::{sample_idx:06d}",
                    "dataset": "XJTU-SY",
                    "bearing_id": bearing_id,
                    "condition_id": condition_id,
                    "source_group": None,
                    "sample_id": f"{sample_idx:06d}",
                    "timestep": sample_idx,
                })
    import pandas as pd
    return pd.DataFrame(rows)


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


def test_bearing_index_splitter_crosses_conditions_by_bearing_suffix():
    index = _xjtu_multi_condition_index()
    cfg = OmegaConf.create({
        "name": "xjtu_bearing_index_split",
        "train_bearing_indices": [1, 2, 3],
        "val_bearing_indices": [4],
        "test_bearing_indices": [5],
    })

    result = BearingIndexSplitter(cfg).split(index)

    assert result.train_bearings == [
        "Bearing1_1", "Bearing1_2", "Bearing1_3",
        "Bearing2_1", "Bearing2_2", "Bearing2_3",
        "Bearing3_1", "Bearing3_2", "Bearing3_3",
    ]
    assert result.val_bearings == ["Bearing1_4", "Bearing2_4", "Bearing3_4"]
    assert result.test_bearings == ["Bearing1_5", "Bearing2_5", "Bearing3_5"]
    assert len(result.train_sample_uids) == 18
    assert len(result.val_sample_uids) == 6
    assert len(result.test_sample_uids) == 6
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

    assert result.train_bearings == ["Bearing1_1", "Bearing1_4", "Bearing2_1"]
    assert result.val_bearings == ["Bearing2_2"]
    assert result.test_bearings == ["Bearing1_3"]
    assert result.report()["ok"] is True


def test_phm2012_default_split_keeps_learning_set_for_train_and_takes_val_from_full_test():
    import pandas as pd

    index = pd.DataFrame({
        "sample_uid": [
            "s_b11", "s_b12", "s_b21", "s_b22", "s_b31", "s_b32",
            "s_b13", "s_b14", "s_b15", "s_b16", "s_b17",
            "s_b23", "s_b24", "s_b25", "s_b26", "s_b27", "s_b33",
        ],
        "bearing_id": [
            "Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2",
            "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
            "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3",
        ],
        "source_group": [
            "Learning_set", "Learning_set", "Learning_set", "Learning_set", "Learning_set", "Learning_set",
            "Full_Test_Set", "Full_Test_Set", "Full_Test_Set", "Full_Test_Set", "Full_Test_Set",
            "Full_Test_Set", "Full_Test_Set", "Full_Test_Set", "Full_Test_Set", "Full_Test_Set", "Full_Test_Set",
        ],
    })
    cfg = OmegaConf.create({
        "name": "phm2012_official",
        "mode": "explicit",
        "train_bearings": ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"],
        "val_bearings": ["Bearing1_3", "Bearing2_3"],
        "test_bearings": [
            "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
            "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3",
        ],
    })

    result = OfficialPHM2012Splitter(cfg).split(index)

    assert result.train_bearings == ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
    assert result.val_bearings == ["Bearing1_3", "Bearing2_3"]
    assert "Bearing3_3" in result.test_bearings
    assert set(result.train_sample_uids).isdisjoint(result.val_sample_uids)
    assert result.report()["ok"] is True


def test_split_registry_builds_configured_splitter():
    cfg = OmegaConf.create({"name": "xjtu_bearing_index_split"})

    splitter = build_splitter(cfg)

    assert isinstance(splitter, BearingIndexSplitter)
