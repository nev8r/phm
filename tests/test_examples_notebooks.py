"""
Example notebook tests

this file is for checking training notebook structure

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _read_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def _joined_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_dataset_training_notebooks_exist_and_use_distinct_algorithms() -> None:
    project_root = Path(__file__).resolve().parents[1]
    xjtu_notebook = _read_notebook(project_root / "examples" / "03_xjtu_cnn_rul_training.ipynb")
    phm_notebook = _read_notebook(project_root / "examples" / "04_phm2012_mlp_feature_training.ipynb")

    xjtu_source = _joined_source(xjtu_notebook)
    phm_source = _joined_source(phm_notebook)

    assert "XJTULoader" in xjtu_source
    assert "CNN" in xjtu_source
    assert "PHM2012Loader" in phm_source
    assert "MLP" in phm_source


def test_examples_directory_contains_only_notebooks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    non_notebook_files = [
        path.name
        for path in (project_root / "examples").iterdir()
        if path.is_file() and path.suffix != ".ipynb"
    ]

    assert non_notebook_files == []


def test_all_example_notebooks_execute_successfully(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_paths = sorted((project_root / "examples").glob("*.ipynb"))
    assert notebook_paths
    monkeypatch.setenv("BEARING_EXAMPLE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BEARING_EXAMPLE_EPOCHS", "1")
    monkeypatch.chdir(project_root)

    for notebook_path in notebook_paths:
        notebook = _read_notebook(notebook_path)
        namespace: dict[str, object] = {"__name__": "__main__"}
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.strip():
                exec(compile(source, str(notebook_path), "exec"), namespace)


def test_dataset_documentation_covers_required_practical_topics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    document_text = (project_root / "docs" / "DATASETS.md").read_text(encoding="utf-8")
    required_terms = [
        "XJTU-SY",
        "PHM2012",
        "FEMTO",
        "PRONOSTIA",
        "25.6 kHz",
        "32768",
        "2560",
        "1.28 s",
        "0.1 s",
        "1 min",
        "10 s",
        "35Hz12kN",
        "37.5Hz11kN",
        "40Hz10kN",
        "1800 rpm",
        "1650 rpm",
        "1500 rpm",
        "acc_00001.csv",
        "temp_00001.csv",
        "Horizontal Vibration",
        "Vertical Vibration",
        "Temperature",
        "elapsed_seconds",
        "rul",
        "rul_unit",
        "XJTULoader",
        "PHM2012Loader",
        "常见问题",
        "推荐实验切分",
    ]

    missing_terms = [term for term in required_terms if term not in document_text]
    assert missing_terms == []
