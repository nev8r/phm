"""
test training gui demo module.

Purpose: verify test training gui demo module behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from recipes.demo.training_gui import (
    build_demo_export,
    build_training_replay_frame_specs,
    load_final_decisions,
    load_mlp_replay_runs,
    load_non_mlp_demo_runs,
)


def test_load_mlp_replay_runs_uses_chinese_presets_and_real_histories():
    runs = load_mlp_replay_runs()

    assert [run.title for run in runs] == [
        "XJTU-SY RUL 默认 MLP",
        "XJTU-SY EarlyFault 默认 MLP",
        "PHM2012 RUL 调参 MLP",
        "PHM2012 HealthState 调参 MLP",
    ]
    assert all(len(run.history) == 50 for run in runs)
    assert all(run.best_epoch >= 1 for run in runs)


def test_load_non_mlp_demo_runs_includes_regression_and_classification_figures():
    runs = load_non_mlp_demo_runs()
    by_title = {run.title: run for run in runs}

    assert "XJTU-SY RUL RandomForest" in by_title
    assert "XJTU-SY HealthState XGBoost" in by_title
    assert by_title["XJTU-SY RUL RandomForest"].figure_paths["预测值 vs 真实值"].exists()
    assert by_title["XJTU-SY RUL RandomForest"].figure_paths["残差图"].exists()
    assert by_title["XJTU-SY HealthState XGBoost"].figure_paths["混淆矩阵"].exists()
    assert by_title["XJTU-SY HealthState XGBoost"].figure_paths["特征重要性"].exists()


def test_load_final_decisions_returns_six_chinese_rows():
    decisions = load_final_decisions()

    assert len(decisions) == 6
    assert {row["数据集"] for row in decisions} == {"XJTU-SY", "PHM2012"}
    assert all("推荐模型" in row for row in decisions)


def test_build_demo_export_writes_chinese_docs_without_private_paths(tmp_path):
    build_demo_export(tmp_path)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    qa = (tmp_path / "VIDEO_QA.md").read_text(encoding="utf-8")
    screenshots = sorted((tmp_path / "screenshots").glob("*.png"))

    assert "训练过程中文 GUI 演示" in readme
    assert "视频验收记录" in qa
    assert len(screenshots) == 5
    assert "视频类型：自动逐 epoch 动画" in qa
    assert "epoch 1/50 → 50/50" in qa
    assert "曲线逐帧更新" in qa
    forbidden = [
        "/" + "Users/",
        "data/loader" + "_roots",
        "model" + ".pkl",
        "prediction " + "parquet",
        "check" + "point",
    ]
    combined = readme + qa
    assert all(item not in combined for item in forbidden)


def test_training_replay_frame_specs_include_each_epoch_and_diagnostics():
    mlp_run = load_mlp_replay_runs()[0]
    non_mlp_runs = load_non_mlp_demo_runs()
    decisions = load_final_decisions()

    specs = build_training_replay_frame_specs(mlp_run, non_mlp_runs, decisions)

    mlp_specs = [spec for spec in specs if spec["kind"] == "mlp_epoch"]
    assert len(mlp_specs) == 50
    assert mlp_specs[0]["epoch"] == 1
    assert mlp_specs[-1]["epoch"] == 50
    assert any(spec["kind"] == "non_mlp_regression" for spec in specs)
    assert any(spec["kind"] == "non_mlp_classification" for spec in specs)
    assert specs[-1]["kind"] == "final_decision"
