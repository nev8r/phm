"""
Test loader split-root registration behavior.

Purpose: verify test loader split-root registration behavior behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import tempfile
import unittest
from pathlib import Path

from USTC.SSE.BearingPrediction.data.loader.PHM2012Loader import PHM2012Loader
from USTC.SSE.BearingPrediction.data.loader.XJTULoader import XJTULoader


class LoaderSplitRootTest(unittest.TestCase):
    def test_phm2012_loader_registers_sparse_official_split_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bearing_dir = root / "Learning_set" / "Bearing1_1"
            bearing_dir.mkdir(parents=True)
            (root / "Learning_set" / "README.md").write_text("not a bearing directory\n")

            loader = PHM2012Loader(str(root))

            self.assertEqual(list(loader.keys()), ["Bearing1_1"])

    def test_xjtu_loader_registers_sparse_condition_split_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bearing_dir = root / "35Hz12kN" / "Bearing1_1"
            bearing_dir.mkdir(parents=True)
            (root / "35Hz12kN" / "README.md").write_text("not a bearing directory\n")

            loader = XJTULoader(str(root))

            self.assertEqual(list(loader.keys()), ["Bearing1_1"])


if __name__ == "__main__":
    unittest.main()
