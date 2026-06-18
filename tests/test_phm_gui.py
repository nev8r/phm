"""
Test Streamlit GUI workbench support helpers

this file is for verifying workbench run discovery, upload checks, and model reloads

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import json
import subprocess
import sys
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
    inspect_dataset_roots,
    inspect_uploaded_dataset,
    list_run_directories,
    summarize_run,
    validate_training_run,
)
from USTC.SSE.BearingPrediction.gui_jobs import (
    read_job_log,
    start_cli_job,
    wait_for_job,
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

    def test_cli_module_is_invokable_with_python_m(self):
        completed = subprocess.run(
            [sys.executable, "-m", "USTC.SSE.BearingPrediction.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Bearing PHM", completed.stdout)

    def test_gui_source_uses_workbench_language(self):
        source = Path("src/USTC/SSE/BearingPrediction/gui.py").read_text(encoding="utf-8")

        self.assertIn("轴承 PHM 实验工作台", source)
        self.assertNotIn("轴承寿命预测与故障诊断系统演示台", source)
        self.assertNotIn("课堂展示入口", source)
        self.assertNotIn("训练 Demo", source)

    def test_gui_keeps_work_controls_in_main_canvas(self):
        source = Path("src/USTC/SSE/BearingPrediction/gui.py").read_text(encoding="utf-8")

        self.assertIn('initial_sidebar_state="collapsed"', source)
        self.assertIn("status_area, workspace_area = st.columns([0.8, 1.8]", source)
        self.assertIn("def _render_sidebar_summary", source)
        self.assertIn("def _render_main_status_area", source)
        sidebar_block = source.split("with st.sidebar:", maxsplit=1)[1].split(
            "status_area, workspace_area", maxsplit=1
        )[0]
        self.assertNotIn("_render_job_panel", sidebar_block)
        self.assertNotIn("_render_run_summary", sidebar_block)

    def test_dataset_root_inspection_reports_local_roots_and_cache_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phm_root = root / "phm2012"
            xjtu_root = root / "xjtu"
            (phm_root / "Learning_set" / "Bearing1_1").mkdir(parents=True)
            (phm_root / "Full_Test_Set" / "Bearing1_3").mkdir(parents=True)
            (phm_root / "Learning_set" / "Bearing1_1" / "acc_00001.csv").write_text("1;2;3\n", encoding="utf-8")
            (phm_root / "Full_Test_Set" / "Bearing1_3" / "acc_00001.csv").write_text("1;2;3\n", encoding="utf-8")
            (xjtu_root / "35Hz12kN" / "Bearing1_1").mkdir(parents=True)
            (xjtu_root / "37.5Hz11kN" / "Bearing2_1").mkdir(parents=True)
            (xjtu_root / "35Hz12kN" / "Bearing1_1" / "1.csv").write_text("1,2\n", encoding="utf-8")
            (xjtu_root / "37.5Hz11kN" / "Bearing2_1" / "1.csv").write_text("1,2\n", encoding="utf-8")

            status = inspect_dataset_roots(phm_root, xjtu_root, cache_dir=root / "cache" / "paper_features")

            self.assertTrue(status["PHM2012"]["valid"])
            self.assertGreaterEqual(status["PHM2012"]["bearing_count"], 2)
            self.assertGreaterEqual(status["PHM2012"]["file_count"], 2)
            self.assertFalse(status["PHM2012"]["cache"]["exists"])
            self.assertTrue(status["XJTU-SY"]["valid"])
            self.assertGreaterEqual(status["XJTU-SY"]["condition_count"], 2)

    def test_validate_training_run_reports_missing_files_and_task_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260618_120000_train_rul"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"command": "train", "task": "rul", "sample": True}),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(json.dumps({"test": {"mse": 0.1}}), encoding="utf-8")
            (run_dir / "model_summary.json").write_text(
                json.dumps({"model": "PaperCBAMCNNLSTMRegressor", "input_dim": 96, "sequence_length": 8}),
                encoding="utf-8",
            )

            validation = validate_training_run(run_dir, expected_task="fault")

            self.assertFalse(validation["valid"])
            self.assertTrue(validation["task_mismatch"])
            self.assertIn("model_state.pt", validation["missing"])
            self.assertIn("standardizer.npz", validation["missing"])

    def test_background_job_records_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp) / "jobs"
            success = start_cli_job(
                [sys.executable, "-c", "print('job-ok')"],
                kind="unit",
                jobs_root=jobs_root,
            )
            success_done = wait_for_job(success["job_dir"], timeout_seconds=10, poll_seconds=0.1)

            self.assertEqual(success_done["status"], "succeeded")
            self.assertEqual(success_done["exit_code"], 0)
            self.assertIn("job-ok", read_job_log(success["job_dir"]))

            failure = start_cli_job(
                [sys.executable, "-c", "raise SystemExit(7)"],
                kind="unit",
                jobs_root=jobs_root,
            )
            failure_done = wait_for_job(failure["job_dir"], timeout_seconds=10, poll_seconds=0.1)

            self.assertEqual(failure_done["status"], "failed")
            self.assertEqual(failure_done["exit_code"], 7)

    def test_benchmark_cli_accepts_explicit_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "explicit_benchmark_rul"
            code = run_cli([
                "benchmark",
                "--task",
                "rul",
                "--baselines",
                "linear",
                "--sample",
                "--run-dir",
                str(run_dir),
            ])

            self.assertEqual(code, 0)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "benchmark_results.csv").exists())

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
