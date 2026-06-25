"""
test sequence task configs module.

Purpose: verify test sequence task configs module behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from hydra import compose, initialize_config_dir

from USTC.SSE.BearingPrediction.cli.main import find_conf_dir


def _compose_task(task_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(find_conf_dir())):
        return compose(config_name="smoke", overrides=[f"task={task_name}"])


def test_health_state_sequence_task_config_matches_gru_plan():
    cfg = _compose_task("health_state_sequence")

    assert cfg.task.name == "health_state_sequence"
    assert cfg.task.task_type == "multiclass_classification"
    assert cfg.task.input_mode == "feature_sequence"
    assert cfg.task.sequence.length == 8
    assert cfg.task.sequence.allow_cross_bearing is False
    assert cfg.task.target.columns == ["health_state_id"]
    assert cfg.task.target.num_classes == 4


def test_rul_linear_sequence_task_config_matches_gru_plan():
    cfg = _compose_task("rul_linear_sequence")

    assert cfg.task.name == "rul_linear_sequence"
    assert cfg.task.task_type == "regression"
    assert cfg.task.input_mode == "feature_sequence"
    assert cfg.task.sequence.length == 8
    assert cfg.task.sequence.allow_cross_bearing is False
    assert cfg.task.target.columns == ["linear_rul_norm"]
    assert cfg.task.target.dtype == "float32"


def test_early_fault_sequence_task_config_matches_gru_plan():
    cfg = _compose_task("early_fault_sequence")

    assert cfg.task.name == "early_fault_sequence"
    assert cfg.task.task_type == "binary_classification"
    assert cfg.task.input_mode == "feature_sequence"
    assert cfg.task.sequence.length == 8
    assert cfg.task.sequence.allow_cross_split is False
    assert cfg.task.target.columns == ["early_fault"]
    assert cfg.task.target.num_classes == 2
