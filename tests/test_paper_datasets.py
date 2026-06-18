"""
Test paper datasets test module

this file is for verifying test paper datasets behavior

created by zy

copyright USTC

2026
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from USTC.SSE.BearingPrediction.data.paper.BearingPaperDataset import (
    DEFAULT_FREQUENCY_BANDS,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_TIME_FEATURES,
    SequenceFeatureDataset,
    estimate_three_sigma_fot_index,
    extract_feature_vector,
    make_sequence_index,
    read_phm2012_acc_file,
)


class PaperDatasetTest(unittest.TestCase):
    def test_extract_feature_vector_combines_fft_and_handcrafted_features(self):
        sampling_rate = 1024
        samples = np.arange(256) / sampling_rate
        signal = np.column_stack([
            np.sin(2 * np.pi * 64 * samples),
            np.sin(2 * np.pi * 128 * samples),
        ])

        feature = extract_feature_vector(
            signal,
            sampling_rate=sampling_rate,
            fft_bins=8,
            channels=(0, 1),
            frequency_bands=((0, 100), (100, 300)),
            include_handcrafted=True,
        )

        expected_per_channel = 8 + len(DEFAULT_TIME_FEATURES) + len(DEFAULT_SPECTRAL_FEATURES) + 2
        self.assertEqual(feature.shape, (expected_per_channel * 2,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_make_sequence_index_respects_entity_ranges(self):
        windows, bearings = make_sequence_index(
            {"Bearing1_1": (0, 5), "Bearing2_1": (5, 9)},
            sequence_length=3,
            sequence_step=2,
        )

        np.testing.assert_array_equal(windows, np.array([[0, 3], [2, 5], [5, 8]]))
        self.assertEqual(bearings.tolist(), ["Bearing1_1", "Bearing1_1", "Bearing2_1"])

    def test_sequence_feature_dataset_uses_last_target(self):
        features = np.arange(20, dtype=np.float32).reshape(10, 2)
        targets = np.arange(10, dtype=np.float32).reshape(10, 1)
        windows = np.array([[0, 4], [3, 7]])
        mean = np.array([0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0], dtype=np.float32)

        dataset = SequenceFeatureDataset(features, targets, windows, mean=mean, std=std)
        x, y = dataset[1]

        self.assertEqual(tuple(x.shape), (4, 2))
        self.assertEqual(tuple(y.shape), (1,))
        self.assertEqual(float(y[0]), 6.0)

    def test_read_phm2012_acc_file_supports_semicolon_bearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acc_00001.csv"
            path.write_text("0;0;0.1;0.2\n1;1;0.3;0.4\n", encoding="utf-8")

            signal = read_phm2012_acc_file(path, "Bearing1_4")

        self.assertEqual(signal.shape, (2, 2))
        np.testing.assert_allclose(signal[1], np.array([0.3, 0.4], dtype=np.float32))

    def test_estimate_three_sigma_fot_index_detects_sustained_rms_shift(self):
        rms = np.array([0.10, 0.11, 0.10, 0.12, 0.11, 0.10, 0.55, 0.58, 0.60, 0.62], dtype=np.float32)

        fot = estimate_three_sigma_fot_index(rms, healthy_ratio=0.5, min_bound=0.05, max_consecution=2)

        self.assertEqual(fot, 6)


if __name__ == "__main__":
    unittest.main()
