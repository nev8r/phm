"""
test demo training videos module.

Purpose: verify test demo training videos module behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import json

from recipes.demo.build_demo_training_videos import (
    build_demo_video_plans,
    build_video_frame_specs,
    write_video_docs,
)


def test_build_demo_video_plans_selects_rul_and_early_fault():
    plans = build_demo_video_plans()

    assert [plan.video_file for plan in plans] == [
        "demo_xjtu_rul_gru_50ep_accelerated.mp4",
        "demo_xjtu_early_gru_50ep_accelerated.mp4",
    ]
    assert plans[0].demo_run_name == "demo_video_xjtu_rul_linear_gru_sequence_50ep"
    assert plans[0].main_run_name == "xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep"
    assert plans[1].demo_run_name == "demo_video_xjtu_early_gru_sequence_50ep"
    assert plans[1].main_run_name == "xjtu_main_early_gru_sequence_compact_non_label_source_200ep"


def test_build_video_frame_specs_replays_each_epoch_then_main_figures():
    plan = build_demo_video_plans()[0]
    history = [
        {"epoch": epoch, "train_loss": 1.0 / epoch, "val_loss": 1.2 / epoch, "val_RMSE": 0.8 / epoch}
        for epoch in range(1, 51)
    ]

    specs = build_video_frame_specs(plan, history)

    epoch_specs = [spec for spec in specs if spec["kind"] == "training_epoch"]
    assert len(epoch_specs) == 50
    assert epoch_specs[0]["epoch"] == 1
    assert epoch_specs[-1]["epoch"] == 50
    assert specs[-1]["kind"] == "main_result_figures"
    assert specs[-1]["main_run_name"] == plan.main_run_name
    assert specs[-1]["figure_count"] == len(plan.main_figures)


def test_write_video_docs_records_demo_vs_main_policy(tmp_path):
    plans = build_demo_video_plans()
    demo_summaries = {
        plan.demo_run_name: {
            "completed": "是",
            "last_epoch": 50,
            "history_rows": 50,
            "best_epoch": 2,
            "test_primary": 0.1234,
        }
        for plan in plans
    }
    video_meta = {
        plan.video_file: {
            "duration": "12.0s",
            "resolution": "1280x720",
            "file_size": "1000 bytes",
        }
        for plan in plans
    }

    write_video_docs(tmp_path, plans, demo_summaries, video_meta)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    qa = (tmp_path / "VIDEO_QA.md").read_text(encoding="utf-8")
    manifest = (tmp_path / "MANIFEST.csv").read_text(encoding="utf-8")

    assert "50ep 训练只用于视频演示训练过程" in readme
    assert "200ep 结果才是主线实验结果" in readme
    assert "demo_xjtu_rul_gru_50ep_accelerated.mp4" in qa
    assert "demo_xjtu_early_gru_50ep_accelerated.mp4" in qa
    assert "逐 epoch 动画" in qa
    assert "50ep 是 demo training，200ep 是 main result" in qa
    assert "视频主画面不展示 val_loss" in qa
    assert "视频主画面不展示 validation primary metric" in qa
    assert "lr=0.0003" in qa
    assert "batch_size=256" in qa
    assert "weight_decay=0.0001" in qa
    assert "StepAC" in manifest
    forbidden = ["/Users/", "artifacts/demo_training/runs/", "test_predictions.parquet", "checkpoint", "model.pkl"]
    combined = readme + qa + manifest
    assert all(item not in combined for item in forbidden)
