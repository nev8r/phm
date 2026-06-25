"""
Test Stage 1 dataset metadata.

Purpose: verify test stage 1 dataset metadata behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.infra.metadata.PHM2012Meta import PHM2012Meta
from USTC.SSE.BearingPrediction.infra.metadata.XJTUSYMeta import XJTUSYMeta


def test_xjtu_sy_meta_exposes_conditions_and_fault_elements():
    meta = XJTUSYMeta()
    bearing = meta.get_bearing_meta("Bearing1_5")

    assert meta.dataset_name == "XJTU-SY"
    assert bearing.condition_id == "35Hz12kN"
    assert bearing.sampling_rate == 25600
    assert bearing.expected_points_per_sample == 32768
    assert bearing.sample_interval_seconds == 60
    assert bearing.speed_hz == 35.0
    assert bearing.load_n == 12000.0
    assert bearing.fault_element == ("inner", "outer")
    assert bearing.geometry.ball_count == 8
    assert len(list(meta.iter_bearing_meta())) == 15


def test_phm2012_meta_maps_bearing_prefix_to_condition():
    meta = PHM2012Meta()
    bearing = meta.get_bearing_meta("Bearing2_1")

    assert meta.dataset_name == "PHM2012"
    assert bearing.condition_id == "Condition2"
    assert bearing.sampling_rate == 25600
    assert bearing.expected_points_per_sample == 2560
    assert bearing.sample_interval_seconds == 10
    assert bearing.speed_hz == 27.5
    assert bearing.load_n == 4200.0
    assert bearing.fault_element is None
