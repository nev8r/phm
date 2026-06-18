"""
Test Streamlit GUI support helpers

this file is for verifying classroom demo run discovery, upload checks, and model reloads

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.cli import build_parser, run_cli
from USTC.SSE.BearingPrediction.gui import (
    inspect_uploaded_dataset,
    list_run_directories,
    summarize_run,
)
from USTC.SSE.BearingPrediction.workflow import evaluate_saved_training_run
from USTC.SSE.BearingPrediction.workflow import predict_feature_csv_with_run


class PhmGuiSupportTest(unittest.TestCase):
    def test_cli_exposes_gui_subcommand(self):
        help_text = build_parser().format_help()
        self.assertIn("gui", help_text)

        with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()):
            run_cli(["gui", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_run_discovery_and_summary_read_train_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260618_120000_train_rul"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"command": "train", "task": "rul", "sample": False}),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps({"test": {"mse": 0.021532, "rmse": 0.1467, "mae": 0.1050}}),
                encoding="utf-8",
            )
            (run_dir / "model_summary.json").write_text(
                json.dumps({"model": "PaperCBAMCNNLSTMRegressor", "parameter_count": 123}),
                encoding="utf-8",
            )

            runs = list_run_directories(Path(tmp), command="train", task="rul")
            summary = summarize_run(run_dir)

            self.assertEqual([item["path"] for item in runs], [run_dir])
            self.assertEqual(summary["task"], "rul")
            self.assertEqual(summary["metrics"]["MSE"], "0.021532")
            self.assertEqual(summary["metrics"]["RMSE"], "0.146700")

    def test_uploaded_dataset_detection_handles_zip_and_feature_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "feature_table.csv"
            pd.DataFrame(np.ones((10, 4)), columns=["f1", "f2", "f3", "f4"]).to_csv(csv_path, index=False)
            zip_path = root / "phm2012.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("Learning_set/Bearing1_1/acc_00001.csv", "1;2;3\n")
                archive.writestr("Full_Test_Set/Bearing1_3/acc_00001.csv", "1;2;3\n")

            csv_info = inspect_uploaded_dataset(csv_path)
            zip_info = inspect_uploaded_dataset(zip_path)

            self.assertTrue(csv_info["valid"])
            self.assertEqual(csv_info["dataset"], "single_csv")
            self.assertEqual(csv_info["numeric_columns"], 4)
            self.assertTrue(zip_info["valid"])
            self.assertEqual(zip_info["dataset"], "PHM2012")

    def test_evaluate_saved_training_run_reloads_sample_checkpoint(self):
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
            result = evaluate_saved_training_run(run_dir, device_name="cpu")

            self.assertEqual(code, 0)
            self.assertEqual(result["task"], "rul")
            self.assertIn("mse", result["metrics"])
            self.assertTrue(Path(result["figures"]["rul_prediction_curve"]).exists())
            self.assertTrue(Path(result["predictions_path"]).exists())

    def test_evaluate_saved_training_run_reloads_fault_checkpoint(self):
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
            result = evaluate_saved_training_run(run_dir, device_name="cpu")

            self.assertEqual(code, 0)
            self.assertEqual(result["task"], "fault")
            self.assertIn("accuracy", result["metrics"])
            self.assertTrue(Path(result["figures"]["fault_confusion_matrix"]).exists())

    def test_predict_feature_csv_with_run_accepts_matching_feature_table(self):
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
            summary = json.loads((run_dir / "model_summary.json").read_text(encoding="utf-8"))
            csv_path = Path(tmp) / "uploaded_features.csv"
            frame = pd.DataFrame(
                np.zeros((int(summary["sequence_length"]) + 2, int(summary["input_dim"])), dtype=np.float32)
            )
            frame.to_csv(csv_path, index=False)
            result = predict_feature_csv_with_run(run_dir, csv_path, device_name="cpu")

            self.assertEqual(code, 0)
            self.assertEqual(result["prediction_count"], 3)
            self.assertTrue(Path(result["predictions_path"]).exists())


if __name__ == "__main__":
    unittest.main()
