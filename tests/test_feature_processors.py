"""
Test feature processors test module

this file is for verifying test feature processors behavior

created by zy

copyright USTC

2026
"""

import unittest

import numpy as np

from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor
from USTC.SSE.BearingPrediction.data.process.array.FrequencyBandEnergyProcessor import FrequencyBandEnergyProcessor
from USTC.SSE.BearingPrediction.data.process.array.SpectralFeatureProcessor import SpectralFeatureProcessor
from USTC.SSE.BearingPrediction.data.process.array.TimeDomainFeatureProcessor import TimeDomainFeatureProcessor


class FeatureProcessorTest(unittest.TestCase):
    def test_fft_magnitude_processor_detects_known_frequency(self):
        sampling_rate = 1024
        samples = np.arange(sampling_rate) / sampling_rate
        signal = np.sin(2 * np.pi * 64 * samples)

        processor = FFTMagnitudeProcessor(sampling_rate=sampling_rate, include_dc=True)
        spectrum = processor.run(signal)
        frequencies = processor.frequency_bins(len(signal))

        peak_frequency = frequencies[int(np.argmax(spectrum))]

        self.assertEqual(spectrum.shape, (sampling_rate // 2 + 1,))
        self.assertAlmostEqual(peak_frequency, 64.0)

    def test_fft_magnitude_processor_limits_bins_and_supports_log_scale(self):
        signal = np.ones(128)

        processor = FFTMagnitudeProcessor(n_bins=12, include_dc=False, log_scale=True)
        spectrum = processor.run(signal)

        self.assertEqual(spectrum.shape, (12,))
        self.assertTrue(np.all(spectrum >= 0))

    def test_frequency_band_energy_processor_returns_relative_band_energy(self):
        sampling_rate = 1024
        samples = np.arange(sampling_rate) / sampling_rate
        signal = (
            2.0 * np.sin(2 * np.pi * 50 * samples)
            + 0.5 * np.sin(2 * np.pi * 300 * samples)
        )

        processor = FrequencyBandEnergyProcessor(
            sampling_rate=sampling_rate,
            bands=[(0, 100), (100, 400)],
            relative=True,
        )
        energy = processor.run(signal)

        self.assertEqual(energy.shape, (2,))
        self.assertGreater(energy[0], energy[1])
        self.assertAlmostEqual(float(np.sum(energy)), 1.0, places=6)

    def test_spectral_feature_processor_extracts_named_features(self):
        sampling_rate = 1024
        samples = np.arange(sampling_rate) / sampling_rate
        signal = np.sin(2 * np.pi * 128 * samples)

        processor = SpectralFeatureProcessor(
            sampling_rate=sampling_rate,
            features=("centroid", "peak_frequency", "bandwidth", "entropy"),
        )
        features = processor.run(signal)

        self.assertEqual(features.shape, (4,))
        self.assertAlmostEqual(features[1], 128.0)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_time_domain_feature_processor_extracts_fault_statistics(self):
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        processor = TimeDomainFeatureProcessor(features=("rms", "ptp", "crest_factor"))
        features = processor.run(signal)

        expected_rms = np.sqrt(np.mean(np.square(signal)))
        self.assertEqual(features.shape, (3,))
        self.assertAlmostEqual(features[0], expected_rms)
        self.assertAlmostEqual(features[1], 4.0)
        self.assertAlmostEqual(features[2], 2.0 / expected_rms)


if __name__ == "__main__":
    unittest.main()
