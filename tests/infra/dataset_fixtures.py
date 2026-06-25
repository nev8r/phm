"""
Small fake dataset builders for Stage 1 tests.

Purpose: verify small fake dataset builders for stage 1 tests behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from pathlib import Path


def _xjtu_csv_text(n=32, scale=1.0) -> str:
    lines = ["Horizontal_vibration_signals,Vertical_vibration_signals"]
    for i in range(n):
        lines.append(f"{scale * 0.1 * i},{scale * 0.2 * i}")
    return "\n".join(lines) + "\n"


def _phm_csv_text(n=32, sep=",", scale=1.0) -> str:
    lines = []
    for i in range(n):
        lines.append(f"{i}{sep}{scale * 0.1 * i}{sep}{scale * 0.2 * i}")
    return "\n".join(lines) + "\n"


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
    for idx, relative_path in enumerate(files, start=1):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_xjtu_csv_text(scale=float(idx)), encoding="utf-8")
    return root


def create_fake_phm2012_root(root: Path) -> Path:
    comma_files = [
        "Learning_set/Bearing1_1/acc_00001.csv",
        "Learning_set/Bearing1_1/acc_00002.csv",
        "Learning_set/Bearing2_1/acc_00001.csv",
        "Learning_set/Bearing2_2/acc_00001.csv",
        "Full_Test_Set/Bearing1_3/acc_00001.csv",
    ]
    for idx, relative_path in enumerate(comma_files, start=1):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_phm_csv_text(scale=float(idx)), encoding="utf-8")

    semicolon_path = root / "Learning_set/Bearing1_4/acc_00001.csv"
    semicolon_path.parent.mkdir(parents=True, exist_ok=True)
    semicolon_path.write_text(_phm_csv_text(sep=";", scale=10.0), encoding="utf-8")

    temp_path = root / "Learning_set/Bearing2_2/temp_00001.csv"
    temp_path.write_text(_phm_csv_text(), encoding="utf-8")
    return root
