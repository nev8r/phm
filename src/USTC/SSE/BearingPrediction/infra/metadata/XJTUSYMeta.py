"""
XJTU-SY dataset metadata.
"""

from typing import Dict, Iterable, Tuple

from USTC.SSE.BearingPrediction.infra.metadata.BearingGeometry import BearingGeometry
from USTC.SSE.BearingPrediction.infra.metadata.BearingMeta import BearingMeta
from USTC.SSE.BearingPrediction.infra.metadata.DatasetMeta import DatasetMeta


class XJTUSYMeta(DatasetMeta):
    DATASET_NAME = "XJTU-SY"
    SAMPLING_RATE = 25600
    EXPECTED_POINTS_PER_SAMPLE = 32768
    SAMPLE_INTERVAL_SECONDS = 60
    CHANNELS = ("Horizontal Vibration", "Vertical Vibration")
    GEOMETRY = BearingGeometry(
        ball_count=8,
        ball_diameter_mm=7.92,
        pitch_diameter_mm=34.55,
        contact_angle_deg=0.0,
    )

    CONDITIONS: Dict[str, Dict] = {
        "35Hz12kN": {
            "speed_hz": 35.0,
            "load_n": 12000.0,
            "bearing_prefix": "Bearing1_",
        },
        "37.5Hz11kN": {
            "speed_hz": 37.5,
            "load_n": 11000.0,
            "bearing_prefix": "Bearing2_",
        },
        "40Hz10kN": {
            "speed_hz": 40.0,
            "load_n": 10000.0,
            "bearing_prefix": "Bearing3_",
        },
    }

    FAULT_ELEMENTS: Dict[str, Tuple[str, ...]] = {
        "Bearing1_1": ("outer",),
        "Bearing1_2": ("outer",),
        "Bearing1_3": ("outer",),
        "Bearing1_4": ("cage",),
        "Bearing1_5": ("inner", "outer"),
        "Bearing2_1": ("inner",),
        "Bearing2_2": ("outer",),
        "Bearing2_3": ("cage",),
        "Bearing2_4": ("outer",),
        "Bearing2_5": ("outer",),
        "Bearing3_1": ("outer",),
        "Bearing3_2": ("inner", "outer", "cage", "ball"),
        "Bearing3_3": ("inner",),
        "Bearing3_4": ("inner",),
        "Bearing3_5": ("outer",),
    }

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def get_bearing_meta(self, bearing_id: str) -> BearingMeta:
        condition_id, condition = self._condition_for_bearing(bearing_id)
        return BearingMeta(
            dataset=self.DATASET_NAME,
            bearing_id=bearing_id,
            condition_id=condition_id,
            sampling_rate=self.SAMPLING_RATE,
            sample_interval_seconds=self.SAMPLE_INTERVAL_SECONDS,
            expected_points_per_sample=self.EXPECTED_POINTS_PER_SAMPLE,
            channels=self.CHANNELS,
            speed_hz=condition["speed_hz"],
            load_n=condition["load_n"],
            fault_element=self.FAULT_ELEMENTS.get(bearing_id),
            geometry=self.GEOMETRY,
        )

    def iter_bearing_meta(self) -> Iterable[BearingMeta]:
        for bearing_id in sorted(self.FAULT_ELEMENTS, key=_bearing_sort_key):
            yield self.get_bearing_meta(bearing_id)

    def _condition_for_bearing(self, bearing_id: str):
        for condition_id, condition in self.CONDITIONS.items():
            if bearing_id.startswith(condition["bearing_prefix"]):
                return condition_id, condition
        raise KeyError(f"Unknown XJTU-SY bearing id: {bearing_id}")


def _bearing_sort_key(bearing_id: str):
    prefix, number = bearing_id.split("_")
    return int(prefix.replace("Bearing", "")), int(number)
