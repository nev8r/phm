"""
test plot sequence results module.

Purpose: verify test plot sequence results module behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import json
from recipes.diagnostics.plot_sequence_results import (
    choose_primary_metric,
    history_metric_columns,
    latest_run_dir,
)


def test_choose_primary_metric_matches_task_type():
    assert choose_primary_metric("regression") == ("RMSE", "lower_is_better")
    assert choose_primary_metric("binary_classification") == ("WeightedF1", "higher_is_better")
    assert choose_primary_metric("multiclass_classification") == ("WeightedF1", "higher_is_better")


def test_history_metric_columns_prefers_val_primary_metric(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"epoch": 1, "train_loss": 0.5, "val_RMSE": 0.4, "val_loss": 0.3},
        {"epoch": 2, "train_loss": 0.4, "val_RMSE": 0.35, "val_loss": 0.25},
    ]))

    columns = history_metric_columns(path, primary_metric="RMSE")

    assert columns == ("train_loss", "val_RMSE", "val_loss")


def test_latest_run_dir_matches_exact_run_name_not_prefix(tmp_path):
    old_run = tmp_path / "20260101-exp_aaa"
    old_run.mkdir()
    (old_run / "trainer").mkdir()
    (old_run / "trainer" / "trainer_state.json").write_text("{}")
    (old_run / "run.json").write_text(json.dumps({"run_name": "exp"}))

    longer_run = tmp_path / "20260102-exp_200ep_bbb"
    longer_run.mkdir()
    (longer_run / "trainer").mkdir()
    (longer_run / "trainer" / "trainer_state.json").write_text("{}")
    (longer_run / "run.json").write_text(json.dumps({"run_name": "exp_200ep"}))

    assert latest_run_dir(tmp_path, "exp") == old_run
    assert latest_run_dir(tmp_path, "exp_200ep") == longer_run
