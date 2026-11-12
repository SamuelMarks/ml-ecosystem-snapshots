"""Unit tests verifying `.pyi` default-value sanitization through public `generate_stubs`."""

from __future__ import annotations

import ast
from pathlib import Path

from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.stubs import generate_stubs


def test_sanitize_default(tmp_path: Path) -> None:
    """Verify `generate_stubs` preserves safe AST defaults and replaces unsafe ones with `...`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    snap: dict[str, JsonValue] = {
        "target": "demo_stubs",
        "version": "1.0.0",
        "categories": {
            "ops": [
                {
                    "name": "op_defaults",
                    "api_path": "demo_stubs.op_defaults",
                    "kind": "function",
                    "params": [
                        {"name": "p_unrep", "kind": "POSITIONAL_OR_KEYWORD", "default": "<unrepresentable>"},
                        {"name": "p_str", "kind": "POSITIONAL_OR_KEYWORD", "default": "'linear'"},
                        {"name": "p_int", "kind": "POSITIONAL_OR_KEYWORD", "default": "10"},
                        {"name": "p_list", "kind": "POSITIONAL_OR_KEYWORD", "default": "[1, 2, 3]"},
                        {"name": "p_enum", "kind": "POSITIONAL_OR_KEYWORD", "default": "Enum.A"},
                        {"name": "p_var", "kind": "POSITIONAL_OR_KEYWORD", "default": "my_var"},
                        {"name": "p_neg", "kind": "POSITIONAL_OR_KEYWORD", "default": "-1.0"},
                        {"name": "p_binop", "kind": "POSITIONAL_OR_KEYWORD", "default": "1 + 1"},
                        {"name": "p_cls", "kind": "POSITIONAL_OR_KEYWORD", "default": "<class 'list'>"},
                        {"name": "p_call", "kind": "POSITIONAL_OR_KEYWORD", "default": "f()"},
                    ],
                }
            ]
        },
    }
    generate_stubs(snap, str(tmp_path))
    pyi_text = (tmp_path / "demo_stubs" / "__init__.pyi").read_text(encoding="utf-8")
    ast.parse(pyi_text)
    assert "p_unrep: Incomplete = ..." in pyi_text
    assert "p_str: Incomplete = 'linear'" in pyi_text
    assert "p_int: Incomplete = 10" in pyi_text
    assert "p_list: Incomplete = [1, 2, 3]" in pyi_text
    assert "p_enum: Incomplete = Enum.A" in pyi_text
    assert "p_var: Incomplete = my_var" in pyi_text
    assert "p_neg: Incomplete = -1.0" in pyi_text
    assert "p_binop: Incomplete = 1 + 1" in pyi_text
    assert "p_cls: Incomplete = ..." in pyi_text
    assert "p_call: Incomplete = ..." in pyi_text
