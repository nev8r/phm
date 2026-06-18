"""
Test phm training and benchmark workflows

this file is for verifying unified metrics, paper trainer smoke jobs, and baselines

created by zy

copyright USTC

2026
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from USTC.SSE.BearingPrediction.cli import run_cli
from USTC.SSE.BearingPrediction.workflow import (
    classification_metrics,
    regression_metrics,
)


class PhmWorkflowTest(unittest.TestCase):
    def test_regression_and_classification_metrics_are_complete(self):
        regression = regression_metrics(
            np.array([1.0, 0.5, 0.0], dtype=np.float32),
            np.array([0.9, 0.4, 0.1], dtype=np.float32),
        )
        classification = classification_metrics(
            np.array([0, 1, 1, 0], dtype=np.int64),
            np.array([0, 1, 0, 0], dtype=np.int64),
        )

        for key in ("mse", "rmse", "mae", "r2", "phm2012_score"):
            self.assertIn(key, regression)
            self.assertTrue(np.isfinite(regression[key]))
        for key in ("accuracy", "macro_f1", "weighted_f1", "fault_f1", "confusion_matrix"):
            self.assertIn(key, classification)

    def test_train_rul_sample_uses_trainer_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = run_cli([
                "train",
                "--task",
                "rul",
                "--preset",
                "smoke",
                "--sample",
                "--device",
                "cpu",
                "--output-dir",
                tmp,
            ])

            run_dir = next(Path(tmp).iterdir())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "model_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(metrics["trainer"], "BaseTrainer")
            self.assertEqual(metrics["task"], "rul")
            self.assertIn("mse", metrics["test"])
            self.assertGreater(summary["parameter_count"], 0)
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "figures" / "rul_prediction_curve.png").exists())

    def test_train_fault_sample_writes_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = run_cli([
                "train",
                "--task",
                "fault",
                "--preset",
                "smoke",
                "--sample",
                "--device",
                "cpu",
                "--output-dir",
                tmp,
            ])

            run_dir = next(Path(tmp).iterdir())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertIn("accuracy", metrics["test"])
            self.assertIn("confusion_matrix", metrics["test"])
            self.assertTrue((run_dir / "figures" / "fault_confusion_matrix.png").exists())

    def test_benchmark_sample_writes_baseline_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = run_cli([
                "benchmark",
                "--task",
                "all",
                "--baselines",
                "linear,forest",
                "--sample",
                "--output-dir",
                tmp,
            ])

            run_dir = next(Path(tmp).iterdir())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertIn("rul", metrics["results"])
            self.assertIn("fault", metrics["results"])
            self.assertGreaterEqual(len(metrics["results"]["rul"]), 1)
            self.assertTrue((run_dir / "benchmark_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
