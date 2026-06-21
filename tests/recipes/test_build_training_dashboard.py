import json
from pathlib import Path

from recipes.demo.build_training_dashboard import (
    clean_json,
    copy_dashboard_asset,
    summarize_training_history,
)


def test_summarize_training_history_reports_best_epoch_and_losses(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.2},
        {"epoch": 3, "train_loss": 0.2, "val_loss": 0.25},
    ]), encoding="utf-8")

    summary = summarize_training_history(history_path)

    assert summary["last_epoch"] == 3
    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == 0.2
    assert summary["last_train_loss"] == 0.2
    assert summary["last_val_loss"] == 0.25


def test_copy_dashboard_asset_returns_relative_path(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "dashboard"
    source.write_bytes(b"not-a-real-png-for-copy-test")

    relative = copy_dashboard_asset(source, output, "figures/example/copied.png")

    assert relative == "figures/example/copied.png"
    assert (output / relative).read_bytes() == b"not-a-real-png-for-copy-test"
    assert str(tmp_path) not in relative


def test_clean_json_removes_non_finite_values_and_paths(tmp_path):
    payload = {
        "path": tmp_path / "example.json",
        "values": [1.0, float("nan"), float("inf"), -float("inf")],
    }

    cleaned = clean_json(payload)

    assert cleaned["path"] == str(tmp_path / "example.json")
    assert cleaned["values"] == [1.0, None, None, None]
