"""
Final defense material alignment tests

this file is for validating issue 3 final defense wording and generated decks

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_MD_DIR = PROJECT_ROOT / "docx" / "final" / "md"
FINAL_PPTX = PROJECT_ROOT / "docx" / "final" / "工业轴承设备剩余寿命预测系统的实现-结题答辩.pptx"
FINAL_WEB_PPT = PROJECT_ROOT / "docx" / "final" / "web-ppt" / "index.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pptx_text(path: Path) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as package:
        slide_names = sorted(
            name
            for name in package.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for slide_name in slide_names:
            root = ElementTree.fromstring(package.read(slide_name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    texts.append(node.text)
    return "\n".join(texts)


def test_final_outline_and_speech_have_current_issue3_boundaries() -> None:
    """
    final outline and speech should answer SOTA, tsfresh and RULSurv questions.
    """

    combined = "\n".join(
        [
            _read(FINAL_MD_DIR / "20_结题答辩提纲.md"),
            _read(FINAL_MD_DIR / "21_结题答辩演讲稿.md"),
        ]
    )

    required_phrases = [
        "外部 SOTA",
        "AutoRUL",
        "GNN",
        "Weibull",
        "尚未在本地跑出指标",
        "tsfresh",
        "EfficientFCParameters",
        "manual+tsfresh",
        "相关性整体偏弱",
        "sktime RocketRegressor",
        "RULSurv",
        "survival_probability=0.25",
        "保守解码",
        "row-level",
        "held-out",
    ]
    for phrase in required_phrases:
        assert phrase in combined

    stale_phrases = [
        "39 个全量自动化测试",
        "39 个测试",
        "39 个全量测试",
        "未做多随机种子统计",
    ]
    for phrase in stale_phrases:
        assert phrase not in combined


def test_metric_driven_taskbook_is_result_report_not_future_template() -> None:
    """
    issue 3 requires document 28 to describe completed evidence, not a future plan.
    """

    text = _read(FINAL_MD_DIR / "28_指标驱动RUL改进任务书.md")

    stale_phrases = [
        "本轮不在本地实现 tsfresh、sktime 或新模型代码",
        "不提交新的实验结果 CSV",
        "当前没有使用它",
        "下一阶段真正跑出实验后",
        "待填",
        "本轮完成不等于已经跑出 tsfresh/sktime 指标",
    ]
    for phrase in stale_phrases:
        assert phrase not in text

    required_phrases = [
        "tsfresh selected features",
        "Efficient selected",
        "manual 19 + tsfresh",
        "0.315629",
        "0.319927",
        "sktime RocketRegressor",
        "0.263706",
        "RULSurv RSF port",
        "MIGRATION_PASS",
        "survival_probability=0.25",
        "外部 SOTA",
        "source pin",
        "依赖 probe",
    ]
    for phrase in required_phrases:
        assert phrase in text


def test_generated_pptx_and_web_deck_include_updated_boundaries() -> None:
    """
    generated final decks should carry the same defense boundaries as the sources.
    """

    deck_text = _pptx_text(FINAL_PPTX)
    web_text = _read(FINAL_WEB_PPT)
    combined = deck_text + "\n" + web_text

    required_phrases = [
        "59",
        "外部 SOTA",
        "source pin",
        "依赖 probe",
        "tsfresh",
        "Efficient",
        "manual+tsfresh",
        "相关性整体偏弱",
        "RULSurv",
        "survival_probability=0.25",
        "保守解码",
    ]
    for phrase in required_phrases:
        assert phrase in combined

    assert "39 个全量测试" not in combined
    assert "39 个全量自动化测试" not in combined
    assert "未做多随机种子统计" not in combined


def test_generation_scripts_are_source_of_truth_for_issue3_materials() -> None:
    """
    source generators must contain the current boundaries so regeneration is safe.
    """

    combined = "\n".join(
        [
            _read(PROJECT_ROOT / "scripts" / "generate_final_ppt.py"),
            _read(PROJECT_ROOT / "scripts" / "generate_final_web_ppt.py"),
        ]
    )

    for phrase in [
        "59",
        "外部 SOTA",
        "tsfresh",
        "Efficient",
        "manual+tsfresh",
        "相关性整体偏弱",
        "sktime RocketRegressor",
        "RULSurv",
        "survival_probability=0.25",
        "保守解码",
    ]:
        assert phrase in combined

    for pattern in [
        r"39\s*个",
        r"未做多随机",
        r"tsfresh 仅作为可选扩展依赖",
    ]:
        assert re.search(pattern, combined) is None


def test_final_markdown_uses_current_pytest_count() -> None:
    """
    final markdown should not keep the old 39-test count after issue 2 added tests.
    """

    combined = "\n".join(path.read_text(encoding="utf-8") for path in FINAL_MD_DIR.glob("*.md"))
    assert "39 个" not in combined
    assert "39 passed" not in combined
    assert "59 passed" in combined
