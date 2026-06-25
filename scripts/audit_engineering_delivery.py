#!/usr/bin/env python3
"""Audit engineering-practice delivery artifacts.

Purpose: verify the course delivery package, process documents, demo evidence,
and Python source headers before final submission.
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docx"
MD_ROOT = DOC_ROOT / "md"
WORD_ROOT = DOC_ROOT / "word"
PDF_ROOT = DOC_ROOT / "pdf"
CLI_DEMO_ROOT = ROOT / "reports" / "cli_demo"

PLACEHOLDER_PATTERN = re.compile(r"TODO|TBD|待补|占位|未完成|XXX")
HEADER_REQUIRED = ("Author:", "Program date:", "Copyright: USTC")


def main() -> int:
    checks = [
        check_document_counts(),
        check_formal_documents(),
        check_demo_manifests(),
        check_cli_demo(),
        check_python_headers(),
    ]
    ok = all(checks)
    if ok:
        print("delivery audit passed")
        return 0
    print("delivery audit failed")
    return 1


def check_document_counts() -> bool:
    counts = {
        "md": len(list(MD_ROOT.glob("*.md"))),
        "word": len(list(WORD_ROOT.glob("*.docx"))),
        "pdf": len(list(PDF_ROOT.glob("*.pdf"))),
    }
    print(f"document counts: {counts}")
    if counts != {"md": 20, "word": 20, "pdf": 20}:
        print("ERROR: expected 20 Markdown, 20 Word, and 20 PDF files")
        return False
    return True


def check_formal_documents() -> bool:
    ok = True
    required_terms = {
        "06_": ["半月进度表", "阶段产物追踪矩阵", "配置管理记录", "风险跟踪"],
        "08_": ["源码目录结构", "CLI 模式表", "训练评估流"],
        "11_": ["Author:", "Program date:", "Copyright: USTC"],
        "12_": ["分解视图", "执行视图", "实现视图", "部署视图"],
        "17_": ["CLI 使用流程", "Dashboard", "演示视频", "常见错误"],
        "18_": ["uv sync", "bp", "测试命令", "故障排查"],
    }
    for path in sorted(MD_ROOT.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if PLACEHOLDER_PATTERN.search(text):
            print(f"ERROR: placeholder text found in {path.relative_to(ROOT)}")
            ok = False
        if len(text.splitlines()) < 50:
            print(f"ERROR: document is still too thin: {path.relative_to(ROOT)}")
            ok = False
        for prefix, terms in required_terms.items():
            if path.name.startswith(prefix):
                for term in terms:
                    if term not in text:
                        print(f"ERROR: {term!r} missing from {path.relative_to(ROOT)}")
                        ok = False
    readme = (DOC_ROOT / "README.md").read_text(encoding="utf-8")
    for term in ["正式提交文档", "docx/word", "最终压缩包", "源码", "测试"]:
        if term not in readme:
            print(f"ERROR: {term!r} missing from docx/README.md")
            ok = False
    return ok


def check_demo_manifests() -> bool:
    ok = True
    for relative in [
        Path("reports/demo_videos/MANIFEST.csv"),
        Path("reports/demo_dashboard/MANIFEST.csv"),
    ]:
        path = ROOT / relative
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        status_values = {row.get("status", "") for row in rows}
        if status_values != {"pass"}:
            print(f"ERROR: {relative} statuses are {sorted(status_values)}")
            ok = False
    return ok


def check_cli_demo() -> bool:
    ok = True
    required = [
        CLI_DEMO_ROOT / "README.md",
        CLI_DEMO_ROOT / "COMMANDS.md",
        CLI_DEMO_ROOT / "MANIFEST.csv",
        CLI_DEMO_ROOT / "RUN_OUTPUTS.md",
        CLI_DEMO_ROOT / "VIDEO_QA.md",
    ]
    for path in required:
        if not path.exists():
            print(f"ERROR: missing {path.relative_to(ROOT)}")
            ok = False
    if (CLI_DEMO_ROOT / "MANIFEST.csv").exists():
        rows = list(csv.DictReader((CLI_DEMO_ROOT / "MANIFEST.csv").open(encoding="utf-8")))
        if not rows or {row.get("status", "") for row in rows} != {"pass"}:
            print("ERROR: CLI demo manifest must contain only pass rows")
            ok = False
    return ok


def check_python_headers() -> bool:
    ok = True
    roots = [ROOT / "src", ROOT / "tests", ROOT / "recipes"]
    files = [path for root in roots for path in root.rglob("*.py")]
    missing: list[Path] = []
    for path in files:
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"ERROR: syntax error in {path.relative_to(ROOT)}: {exc}")
            ok = False
            continue
        doc = ast.get_docstring(module) or ""
        if not all(term in doc for term in HEADER_REQUIRED):
            missing.append(path)
    if missing:
        print(f"ERROR: {len(missing)} Python files missing standardized headers")
        for path in missing[:30]:
            print(f"  - {path.relative_to(ROOT)}")
        if len(missing) > 30:
            print(f"  ... {len(missing) - 30} more")
        ok = False
    else:
        print(f"python headers: {len(files)} files passed")
    return ok


if __name__ == "__main__":
    sys.exit(main())
