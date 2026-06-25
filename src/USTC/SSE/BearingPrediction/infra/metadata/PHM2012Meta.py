"""
PHM2012 / PRONOSTIA dataset metadata.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, Iterable

from USTC.SSE.BearingPrediction.infra.metadata.BearingMeta import BearingMeta
from USTC.SSE.BearingPrediction.infra.metadata.DatasetMeta import DatasetMeta


class PHM2012Meta(DatasetMeta):
    DATASET_NAME = "PHM2012"
    SAMPLING_RATE = 25600
    EXPECTED_POINTS_PER_SAMPLE = 2560
    SAMPLE_INTERVAL_SECONDS = 10
    CHANNELS = ("Horizontal Vibration", "Vertical Vibration")

    CONDITIONS: Dict[str, Dict] = {
        "Bearing1_": {
            "condition_id": "Condition1",
            "speed_hz": 1800 / 60,
            "load_n": 4000.0,
        },
        "Bearing2_": {
            "condition_id": "Condition2",
            "speed_hz": 1650 / 60,
            "load_n": 4200.0,
        },
        "Bearing3_": {
            "condition_id": "Condition3",
            "speed_hz": 1500 / 60,
            "load_n": 5000.0,
        },
    }

    LEARNING_BEARINGS = (
        "Bearing1_1",
        "Bearing1_2",
        "Bearing2_1",
        "Bearing2_2",
        "Bearing3_1",
        "Bearing3_2",
    )
    TEST_BEARINGS = (
        "Bearing1_3",
        "Bearing1_4",
        "Bearing1_5",
        "Bearing1_6",
        "Bearing1_7",
        "Bearing2_3",
        "Bearing2_4",
        "Bearing2_5",
        "Bearing2_6",
        "Bearing2_7",
        "Bearing3_3",
    )

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def get_bearing_meta(self, bearing_id: str) -> BearingMeta:
        condition = self._condition_for_bearing(bearing_id)
        return BearingMeta(
            dataset=self.DATASET_NAME,
            bearing_id=bearing_id,
            condition_id=condition["condition_id"],
            sampling_rate=self.SAMPLING_RATE,
            sample_interval_seconds=self.SAMPLE_INTERVAL_SECONDS,
            expected_points_per_sample=self.EXPECTED_POINTS_PER_SAMPLE,
            channels=self.CHANNELS,
            speed_hz=condition["speed_hz"],
            load_n=condition["load_n"],
            fault_element=None,
        )

    def iter_bearing_meta(self) -> Iterable[BearingMeta]:
        for bearing_id in (*self.LEARNING_BEARINGS, *self.TEST_BEARINGS):
            yield self.get_bearing_meta(bearing_id)

    def _condition_for_bearing(self, bearing_id: str) -> Dict:
        for prefix, condition in self.CONDITIONS.items():
            if bearing_id.startswith(prefix):
                return condition
        raise KeyError(f"Unknown PHM2012 bearing id: {bearing_id}")
