"""
XJTU dataset loader module

this file is for loading XJTU-SY bearing dataset entities

created by cyy

copyright USTC

2026
"""

from __future__ import annotations

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
        condition_name = entity_path.parent.name.lower()
        if "37.5hz" in condition_name:
            return 25600.0
        if "40hz" in condition_name:
            return 25600.0
        return 25600.0

    def _build_entity_metadata(self, entity_path: Path) -> dict[str, object]:
        metadata = super()._build_entity_metadata(entity_path)
        metadata["operating_condition"] = entity_path.parent.name
        return metadata
