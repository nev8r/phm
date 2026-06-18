"""
Test training artifacts test module

this file is for verifying test training artifacts behavior

created by zy

copyright USTC

2026
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from USTC.SSE.BearingPrediction.data.paper import (
    load_training_artifacts,
    save_training_artifacts,
    training_artifact_paths,
)


class TrainingArtifactTest(unittest.TestCase):
    def test_training_artifacts_keep_model_weights_separate_from_numpy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "paper_model.pt"
            model = nn.Linear(3, 1)
            mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            std = np.array([0.5, 0.75, 1.0], dtype=np.float32)
            config = {"fft_bins": 256, "fault_types": ("OF", "IF")}

            save_training_artifacts(model, checkpoint_path, mean=mean, std=std, config=config)
            paths = training_artifact_paths(checkpoint_path)

            self.assertTrue(paths["checkpoint"].exists())
            self.assertTrue(paths["standardizer"].exists())
            self.assertTrue(paths["config"].exists())

            state_dict = torch.load(paths["checkpoint"], weights_only=True)
            self.assertIn("weight", state_dict)
            self.assertNotIn("mean", state_dict)
            self.assertNotIn("config", state_dict)

            restored_model = nn.Linear(3, 1)
            artifacts = load_training_artifacts(restored_model, checkpoint_path, map_location="cpu")

            np.testing.assert_allclose(artifacts["mean"], mean)
            np.testing.assert_allclose(artifacts["std"], std)
            self.assertEqual(artifacts["config"]["fft_bins"], 256)
            self.assertEqual(artifacts["config"]["fault_types"], ["OF", "IF"])


if __name__ == "__main__":
    unittest.main()
