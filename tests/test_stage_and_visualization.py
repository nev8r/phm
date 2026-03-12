"""
Stage and visualization tests

this file is for testing degradation stage partition and result visualization

created by zy

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

from USTC.SSE.BearingPrediction.api import (
    BaseTester,
    BaseTrainer,
    BearingStageLabeler,
    EarlyStopping,
    FPTStageStrategy,
    ResultVisualizer,
    SyntheticBearingFactory,
    Transformer,
    ThreeSigmaStageStrategy,
)


def test_stage_labeling_and_visualization_outputs(tmp_path: Path) -> None:
    factory = SyntheticBearingFactory(random_state=12)
    entity = factory.create_run_to_failure_entity("Bearing1_2", snapshot_count=24, signal_length=256)

    labeler = BearingStageLabeler(window_size=128, stride=128, stage_strategy=ThreeSigmaStageStrategy())
    dataset, stage_result = labeler.label(entity, "Horizontal Vibration")
    assert set(stage_result.stage_labels.tolist()) <= {0, 1, 2}

    fpt_result = FPTStageStrategy().fit_predict(stage_result.health_indicator)
    assert fpt_result.onset_index >= 0

    train_set, test_set = dataset.split_by_ratio(0.8)
    model = Transformer(input_size=128, output_size=3, d_model=16, nhead=2, num_layers=1, dropout=0.1, task_type="classification")
    trainer = BaseTrainer(callbacks=[EarlyStopping(patience=1)], batch_size=8, max_epochs=2, learning_rate=1e-3, weight_decay=1e-4)
    trainer.train(model, train_set, test_set)

    tester = BaseTester(batch_size=16)
    test_result = tester.test(model, test_set)

    visualizer = ResultVisualizer()
    visualizer.plot_confusion_matrix(test_result.targets, test_result.predictions, tmp_path / "confusion_matrix.png", labels=stage_result.stage_names)
    visualizer.plot_degradation_stages(stage_result.as_frame(), tmp_path / "degradation_stages.png")
    if test_result.attention_weights is not None:
        visualizer.plot_attention_heatmap(test_result.attention_weights, tmp_path / "attention_heatmap.png")

    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "degradation_stages.png").exists()

