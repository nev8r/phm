"""
xLSTM-Transformer paper reproduction tests

this file is for testing Jiang et al. style RUL reproduction workflow

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from USTC.SSE.BearingPrediction.api import XLSTMTransformer
from USTC.SSE.BearingPrediction.examples.demo_workflows import (
    create_demo_phm2012_dataset,
    create_demo_xjtu_dataset,
    run_paper_xlstm_transformer_reproduction,
)


def test_xlstm_transformer_forward_returns_prediction_and_attention() -> None:
    model = XLSTMTransformer(feature_size=19, output_size=1, sequence_length=10, hidden_size=16, num_heads=2)
    inputs = torch.randn(4, 10, 19)

    output = model(inputs)
    attention_weights = model.maybe_get_attention()

    assert output["prediction"].shape == (4, 1)
    assert attention_weights is not None
    assert attention_weights.shape[:3] == (4, 2, 10)


def test_xlstm_reproduction_workflow_trains_two_datasets_and_baselines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BEARING_EXAMPLE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("BEARING_EXAMPLE_EPOCHS", "1")
    xjtu_root = create_demo_xjtu_dataset(tmp_path / "input_data", sample_count=16, signal_length=128)
    phm_root = create_demo_phm2012_dataset(tmp_path / "input_data", sample_count=16, signal_length=128)

    result = run_paper_xlstm_transformer_reproduction(
        xjtu_root=xjtu_root,
        phm2012_root=phm_root,
        max_samples_per_entity=16,
        prefer_real_data=True,
    )

    comparison_frame = pd.read_csv(result["comparison_path"])
    required_columns = {
        "dataset_name",
        "condition_name",
        "model_name",
        "rmse",
        "normalized_rmse",
        "r2",
        "phm2012_score",
        "huang_rul_score",
        "prediction_count",
        "epoch_count",
        "rmse_change_pct_vs_transformer",
        "score_change_pct_vs_transformer",
    }
    assert required_columns.issubset(comparison_frame.columns)
    assert set(comparison_frame["dataset_name"]) == {"XJTU-SY", "PHM2012"}
    assert {"XLSTM-Transformer", "Feature-Transformer", "LSTM-Transformer"}.issubset(
        set(comparison_frame["model_name"])
    )
    assert comparison_frame["huang_rul_score"].notna().all()
    assert comparison_frame["r2"].notna().all()
    assert (comparison_frame["epoch_count"] == 1).all()
    assert result["trained_model_count"] >= 6

    for run_summary in result["runs"]:
        assert Path(run_summary["prediction_path"]).exists()
        assert Path(run_summary["metrics_path"]).exists()
        metrics = json.loads(Path(run_summary["metrics_path"]).read_text(encoding="utf-8"))
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "huang_rul_score" in metrics


def test_xlstm_reproduction_require_real_data_rejects_demo_roots(tmp_path: Path) -> None:
    xjtu_root = create_demo_xjtu_dataset(tmp_path / "input_data", sample_count=16, signal_length=128)
    phm_root = create_demo_phm2012_dataset(tmp_path / "input_data", sample_count=16, signal_length=128)

    with pytest.raises(ValueError, match="official-scale real dataset root"):
        run_paper_xlstm_transformer_reproduction(
            xjtu_root=xjtu_root,
            phm2012_root=phm_root,
            max_samples_per_entity=16,
            prefer_real_data=True,
            require_real_data=True,
        )


def test_xlstm_reproduction_require_real_data_rejects_missing_roots(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_paper_xlstm_transformer_reproduction(
            xjtu_root=tmp_path / "missing-xjtu",
            phm2012_root=tmp_path / "missing-phm2012",
            max_samples_per_entity=16,
            prefer_real_data=True,
            require_real_data=True,
        )


def test_xlstm_transformer_notebook_exists_and_calls_workflow() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "examples" / "07_paper_xlstm_transformer_rul.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert "run_paper_xlstm_transformer_reproduction" in code_source
    assert "comparison_metrics.csv" in code_source
