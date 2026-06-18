"""
Test phm analysis and cli module

this file is for verifying feature analysis, labels, diagrams, and cli helpers

created by zy

copyright USTC

2026
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from USTC.SSE.BearingPrediction.analysis import (
    build_dataset_cards,
    compute_fault_label_series,
    compute_feature_analysis,
    compute_rul_labels,
    compute_tsfresh_audit,
    render_model_architecture_diagrams,
    task_relationship_summary,
)
from USTC.SSE.BearingPrediction.cli import build_parser, run_cli


class PhmAnalysisCliTest(unittest.TestCase):
    def test_dataset_cards_capture_core_dataset_facts(self):
        cards = build_dataset_cards()

        self.assertIn("PHM2012", cards)
        self.assertIn("XJTU-SY", cards)
        self.assertEqual(cards["PHM2012"]["sampling_rate_hz"], 25600)
        self.assertEqual(cards["XJTU-SY"]["sampling_rate_hz"], 25600)
        self.assertIn("RUL", cards["PHM2012"]["tasks"])
        self.assertIn("Fault", cards["XJTU-SY"]["tasks"])
        self.assertGreaterEqual(len(cards["PHM2012"]["operating_conditions"]), 3)
        self.assertGreaterEqual(len(cards["XJTU-SY"]["operating_conditions"]), 3)

    def test_label_formulas_match_rul_and_fault_rules(self):
        linear = compute_rul_labels(length=5, mode="linear")
        rectified = compute_rul_labels(length=5, mode="rectified", fpt_index=2)
        fault = compute_fault_label_series([0.10, 0.11, 0.10, 0.60, 0.61], healthy_ratio=0.6,
                                           min_bound=0.05, max_consecution=2)

        np.testing.assert_allclose(linear, np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32))
        np.testing.assert_allclose(rectified, np.array([1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(fault["labels"], np.array([0, 0, 0, 1, 1], dtype=np.int64))
        self.assertEqual(fault["fot_index"], 3)
        self.assertGreater(fault["threshold"], 0.0)

    def test_feature_analysis_reports_selection_evidence(self):
        features = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.9],
            [2.0, 0.1, 0.7],
            [3.0, 0.4, 0.2],
        ], dtype=np.float32)
        target = np.array([1.0, 0.8, 0.5, 0.1], dtype=np.float32)
        names = ["rms", "entropy", "flat"]

        summary = compute_feature_analysis(features, target, names, task="rul")

        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["feature_count"], 3)
        self.assertEqual(summary["top_correlated_features"][0]["feature"], "rms")
        self.assertIn("why_not_default_tsfresh", summary)
        self.assertIn("domain", summary["feature_strategy"])

    def test_tsfresh_audit_reports_feature_selection_evidence(self):
        features = np.array([
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.8, 0.2],
            [1.0, 0.0],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.9, 0.1],
            [1.1, 0.0],
        ], dtype=np.float32)
        target = np.array([1.0, 0.8, 0.6, 0.3, 0.0, 1.0, 0.7, 0.4, 0.1, 0.0], dtype=np.float32)
        ids = np.array(["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"])

        audit = compute_tsfresh_audit(features, target, ["rms", "entropy"], ids=ids, mode="minimal")

        self.assertEqual(audit["mode"], "minimal")
        self.assertGreater(audit["extracted_feature_count"], 0)
        self.assertIn("top_correlated_tsfresh_features", audit)
        self.assertIn("domain_overlap", audit)

    def test_model_architecture_diagrams_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_model_architecture_diagrams(Path(tmp))

            self.assertTrue(paths["rul"].exists())
            self.assertTrue(paths["fault"].exists())
            self.assertGreater(paths["rul"].stat().st_size, 1000)
            self.assertGreater(paths["fault"].stat().st_size, 1000)

    def test_cli_analyze_sample_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = run_cli([
                "analyze",
                "--task",
                "all",
                "--feature-set",
                "domain",
                "--sample",
                "--output-dir",
                tmp,
            ])

            run_dir = next(Path(tmp).iterdir())
            self.assertEqual(code, 0)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "figures" / "rul_model_architecture.png").exists())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("dataset_cards", metrics)
            self.assertIn("task_relationship", metrics)

    def test_cli_parser_exposes_required_subcommands(self):
        parser = build_parser()
        help_text = parser.format_help()

        for command in ("analyze", "train", "benchmark", "report"):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
