#!/usr/bin/env python3
"""
Clean legacy text fragments inside course PPTX files.

PPTX files are ZIP packages containing XML slides. This script performs
bounded text replacements in those XML parts while preserving the package
structure.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]

COMMON_REPLACEMENTS = {
    "cyj": "cyy",
    "混淆矩阵": "误差分布图",
    "PHM/XJTU-SY 类 score": "PHM/RUL 非对称惩罚 score",
    "PHM/NASA 类 score": "PHM/RUL 非对称惩罚 score",
}

PROPOSAL_REPLACEMENTS = {
    "NASA Turbofan Engine Degradation Data（UCI）": "XJTU-SY 与 PHM2012/FEMTO 轴承退化数据集",
    "NASA Turbofan Engine Degradation Data (UCI)": "XJTU-SY 与 PHM2012/FEMTO 轴承退化数据集",
    "Turbofan Engine Degradation Data": "Bearing Run-to-Failure Data",
    "NASA": "XJTU-SY",
    "Turbofan": "PHM2012/FEMTO",
    "Engine": "轴承",
    "Degradation": "退化",
    "UCI": "公开数据",
}


def clean_pptx(path: Path) -> bool:
    changed = False
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with ZipFile(path, "r") as source, ZipFile(tmp_path, "w", ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename.endswith(".xml"):
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    if text is not None:
                        updated = text
                        replacements = dict(COMMON_REPLACEMENTS)
                        if "proposal" in path.parts:
                            replacements.update(PROPOSAL_REPLACEMENTS)
                        for old, new in replacements.items():
                            updated = updated.replace(old, new)
                        if updated != text:
                            data = updated.encode("utf-8")
                            changed = True
                target.writestr(item, data)
        if changed:
            shutil.move(str(tmp_path), path)
        else:
            tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return changed


def main() -> None:
    for path in sorted(ROOT.glob("docx/*/*.pptx")):
        if clean_pptx(path):
            print(f"cleaned {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
