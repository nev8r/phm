"""
Small fake dataset builders for Stage 1 tests.
"""

from pathlib import Path


CSV_TEXT = "Horizontal_vibration_signals,Vertical_vibration_signals\n0.1,0.2\n"


def create_fake_xjtu_root(root: Path) -> Path:
    files = [
        "35Hz12kN/Bearing1_1/1.csv",
        "35Hz12kN/Bearing1_1/2.csv",
        "35Hz12kN/Bearing1_2/1.csv",
        "35Hz12kN/Bearing1_4/1.csv",
        "35Hz12kN/Bearing1_5/1.csv",
        "37.5Hz11kN/Bearing2_1/1.csv",
        "40Hz10kN/Bearing3_1/1.csv",
    ]
    for relative_path in files:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CSV_TEXT, encoding="utf-8")
    return root


def create_fake_phm2012_root(root: Path) -> Path:
    files = [
        "Learning_set/Bearing1_1/acc_00001.csv",
        "Learning_set/Bearing1_1/acc_00002.csv",
        "Learning_set/Bearing2_1/acc_00001.csv",
        "Learning_set/Bearing2_2/acc_00001.csv",
        "Learning_set/Bearing2_2/temp_00001.csv",
        "Full_Test_Set/Bearing1_3/acc_00001.csv",
    ]
    for relative_path in files:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CSV_TEXT, encoding="utf-8")
    return root
