"""
XJTU dataset loader module

this file is for loading XJTU-SY bearing dataset entities

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import re
from pathlib import Path

from USTC.SSE.BearingPrediction.dataset.base import BaseBearingLoader, DatasetResource


class XJTULoader(BaseBearingLoader):
    """
    Loader for the XJTU-SY bearing degradation dataset.
    """

    dataset_name = "XJTU-SY"
    resource = DatasetResource(
        name="xjtu-sy",
        homepage="https://biaowang.tech/xjtu-sy-bearing-datasets/",
        download_url="https://drive.google.com/uc?export=download&id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK",
        description="Xi'an Jiaotong University rolling bearing accelerated life dataset.",
        notes="官方页面同时提供 Google Drive、Dropbox、MediaFire 等多个镜像。若 Google Drive 直链触发确认页，请改用 homepage 中的镜像链接手动下载。",
    )

    def _infer_sample_rate(self, entity_path: Path) -> float:
        del entity_path
        return 25600.0

    def _build_entity_metadata(self, entity_path: Path) -> dict[str, object]:
        metadata = super()._build_entity_metadata(entity_path)
        operating_condition = entity_path.parent.name
        metadata["operating_condition"] = operating_condition
        metadata["sampling_points"] = 32768

        condition_values = self._parse_operating_condition(operating_condition)
        if condition_values is not None:
            rotating_speed_hz, radial_load_kn = condition_values
            metadata["rotating_speed_hz"] = rotating_speed_hz
            metadata["rotating_speed_rpm"] = rotating_speed_hz * 60.0
            metadata["radial_load_kn"] = radial_load_kn
        return metadata

    def _sample_period_seconds(self, entity_path: Path) -> float:
        del entity_path
        return 60.0

    def _snapshot_duration_seconds(self, entity_path: Path) -> float:
        del entity_path
        return 1.28

    @staticmethod
    def _parse_operating_condition(condition_name: str) -> tuple[float, float] | None:
        """
        parse XJTU-SY condition names such as 35Hz12kN

        Parameters
        ----------
        condition_name : str
            operating condition directory name

        Returns
        -------
        tuple[float, float] | None
            rotating speed in Hz and radial load in kN
        """

        match = re.search(
            r"(?P<speed>\d+(?:\.\d+)?)\s*hz\s*(?P<load>\d+(?:\.\d+)?)\s*kn",
            condition_name,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        return float(match.group("speed")), float(match.group("load"))
