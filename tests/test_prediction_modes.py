"""
Prediction mode tests

this file is for testing rolling prediction and uncertainty estimation

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import numpy as np

from USTC.SSE.BearingPrediction.api import (
    BaseTrainer,
    HealthIndicatorLabeler,
    MonteCarloDropoutPredictor,
    RollingPredictor,
    Transformer,
)


def test_rolling_and_uncertainty_prediction_modes() -> None:
    health_indicator = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    labeler = HealthIndicatorLabeler(history_size=8, horizon=1)
    dataset = labeler.label(health_indicator)
    train_set, test_set = dataset.split_by_ratio(0.8)

    model = Transformer(input_size=8, output_size=1, d_model=16, nhead=2, num_layers=1, dropout=0.2)
    trainer = BaseTrainer(batch_size=4, max_epochs=2, learning_rate=1e-3, weight_decay=1e-4)
    trainer.train(model, train_set, test_set)

    uncertainty_predictor = MonteCarloDropoutPredictor(passes=4)
    bundle = uncertainty_predictor.predict(model, test_set, device=trainer.device, batch_size=4)
    rolling_predictor = RollingPredictor()
    rolling_values = rolling_predictor.predict_sequence(model, dataset.inputs[0], steps=3, device=trainer.device)

    assert bundle["predictions"].shape == bundle["uncertainties"].shape
    assert rolling_values.shape == (3,)
