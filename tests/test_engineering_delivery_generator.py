"""
Engineering delivery generator tests.

Purpose: verify engineering delivery generator tests behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import ast
from pathlib import Path

from scripts import audit_engineering_delivery as audit


ROOT = Path(__file__).resolve().parents[1]


def _literal_assignment(name: str):
    source = (ROOT / "scripts/generate_engineering_delivery.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_generated_formal_document_snippets_do_not_include_placeholder_terms():
    document_sections = _literal_assignment("DOCUMENT_SECTIONS")
    formal_text = "\n".join([
        _literal_assignment("COMMON_EVIDENCE"),
        *document_sections.values(),
    ])

    assert audit.PLACEHOLDER_PATTERN.search(formal_text) is None


def test_cli_demo_generation_uses_workspace_uv_cache():
    source = (ROOT / "scripts/generate_engineering_delivery.py").read_text(encoding="utf-8")

    assert 'env["UV_CACHE_DIR"]' in source
    assert '".uv-cache"' in source
