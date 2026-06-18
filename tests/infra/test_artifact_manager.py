"""
Test artifact file management behavior.
"""

import json

from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


def test_artifact_manager_writes_and_reads_json(tmp_path):
    manager = ArtifactManager(tmp_path)

    output_path = manager.write_json("run.json", {"ok": True, "mode": "validate"})

    assert output_path == tmp_path / "run.json"
    assert output_path.exists()
    assert manager.read_json("run.json") == {"ok": True, "mode": "validate"}


def test_artifact_manager_writes_yaml_and_text(tmp_path):
    manager = ArtifactManager(tmp_path)

    yaml_path = manager.write_yaml("config/resolved.yaml", {"project": {"name": "BearingPrediction"}})
    text_path = manager.write_text("notes/stage.txt", "stage0\n")

    assert OmegaConf.load(yaml_path).project.name == "BearingPrediction"
    assert text_path.read_text() == "stage0\n"
    assert json.loads(manager.write_json("report.json", {"ok": True}).read_text())["ok"] is True
