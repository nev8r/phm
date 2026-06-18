"""
Test paper models test module

this file is for verifying test paper models behavior

created by zy

copyright USTC

2026
"""

import unittest

import torch

from USTC.SSE.BearingPrediction.model.paper.CNNLSTM import (
    CBAMCNNLSTMRegressor,
    CNNLSTMMultiLabelClassifier,
    PaperCBAMCNNLSTMRegressor,
    ResCNNLSTMClassifier,
)


class PaperModelTest(unittest.TestCase):
    def test_cbam_cnn_lstm_regressor_forward_shape(self):
        model = CBAMCNNLSTMRegressor(input_dim=32, hidden_dim=16, conv_channels=8)
        output = model(torch.randn(4, 12, 32))

        self.assertEqual(tuple(output.shape), (4, 1))

    def test_cnn_lstm_multilabel_classifier_forward_shape(self):
        model = CNNLSTMMultiLabelClassifier(input_dim=32, num_labels=4, hidden_dim=16, conv_channels=8)
        output = model(torch.randn(4, 8, 32))

        self.assertEqual(tuple(output.shape), (4, 4))

    def test_paper_cbam_cnn_lstm_regressor_forward_shape(self):
        model = PaperCBAMCNNLSTMRegressor(input_dim=256, lstm_hidden=16, lstm_layers=2)
        output = model(torch.randn(3, 5, 256))

        self.assertEqual(tuple(output.shape), (3, 1))

    def test_res_cnn_lstm_classifier_forward_shape(self):
        model = ResCNNLSTMClassifier(input_dim=32, num_classes=2, hidden_dim=16, conv_channels=8)
        output = model(torch.randn(4, 8, 32))

        self.assertEqual(tuple(output.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
