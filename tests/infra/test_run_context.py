"""
Test experiment run context behavior.
"""

from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.experiment.RunContext import RunContext


def test_run_context_creates_run_directory_under_artifact_root(tmp_path):
    cfg = OmegaConf.create({
        "project": {
            "name": "BearingPrediction",
            "seed": 42,
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "run": {
            "name": "smoke",
            "tags": ["smoke", "infra"],
        },
    })

    context = RunContext.create(cfg)

    assert context.run_id
    assert "_smoke_" in context.run_id
    assert context.project_name == "BearingPrediction"
    assert context.seed == 42
    assert context.run_dir.parent == tmp_path / "artifacts" / "runs"
    assert context.run_dir.exists()


def test_run_context_saves_metadata_and_resolved_config(tmp_path):
    cfg = OmegaConf.create({
        "mode": "validate",
        "project": {
            "name": "BearingPrediction",
            "seed": 7,
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "run": {
            "name": "smoke",
            "tags": [],
        },
    })

    context = RunContext.create(cfg)
    context.save_metadata()
    context.save_resolved_config(cfg)

    assert (context.run_dir / "run.json").exists()
    assert (context.run_dir / "config" / "resolved.yaml").exists()
    assert context.artifacts.read_json("run.json")["run_id"] == context.run_id
