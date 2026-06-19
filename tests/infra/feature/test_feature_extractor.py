"""
Test Stage 2 feature extractor composition.
"""

from omegaconf import OmegaConf

from tests.infra.dataset_fixtures import create_fake_xjtu_root
from USTC.SSE.BearingPrediction.infra.feature.FeatureExtractor import FeatureExtractor
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder


def test_feature_extractor_merges_multiple_backends(tmp_path):
    root = create_fake_xjtu_root(tmp_path / "xjtu")
    index = IndexBuilder().build(OmegaConf.create({"dataset": {"name": "XJTU-SY", "root": str(root)}})).head(2)
    cfg = OmegaConf.create({
        "name": "manual_tsfresh_basic",
        "version": "v1",
        "backends": [
            {
                "name": "manual_basic",
                "type": "manual_processor",
                "params": {
                    "include_magnitude": False,
                    "time": {"enabled": True, "features": ["rms"]},
                    "spectral": {"enabled": False, "features": []},
                },
            },
            {
                "name": "tsfresh_minimal",
                "type": "tsfresh",
                "params": {
                    "fc_parameters": "minimal",
                    "n_jobs": 0,
                    "chunksize": None,
                    "disable_progressbar": True,
                    "include_magnitude": False,
                    "prefix": "tsfresh",
                },
            },
        ],
        "cleaner": {"enabled": True},
    })

    raw_features, spec, backend_reports = FeatureExtractor(cfg).extract(index)

    assert len(raw_features) == len(index)
    assert "h__time__rms" in raw_features.columns
    assert any(column.startswith("tsfresh__h__") for column in raw_features.columns)
    assert spec["name"] == "manual_tsfresh_basic"
    assert len(backend_reports) == 2
