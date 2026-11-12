"""Unit test suite for Chex snapshot extractor, Griffe static scanning, MCP validation, and CLI tooling."""

from __future__ import annotations

import argparse
import types
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.api import (
    extract_snapshot,
    extract_version_with_griffe,
    get_pkg_version,
)
from ml_ecosystem_snapshots.cli.commands.check import cmd_check_chex
from ml_ecosystem_snapshots.frameworks import chex
from ml_ecosystem_snapshots.frameworks._base import scan_griffe_module_members
from ml_ecosystem_snapshots.introspection.static_parsers import (
    DEFAULT_GRIFFE_CACHE,
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.mcp.registry import dispatch_tool_call
from ml_ecosystem_snapshots.mcp.tools.domain_ops import check_chex_op
from ml_ecosystem_snapshots.mcp.tools.hallucination import (
    check_code_block,
    explain_anti_pattern,
)
from ml_ecosystem_snapshots.mcp.tools.translation import translate_concept_arguments
from ml_ecosystem_snapshots.utils import get_framework_docstring_parser


def test_chex_docstring_parser_and_version() -> None:
    """Verify docstring parser selection and version resolution for chex."""
    assert get_framework_docstring_parser("chex") == "google"
    assert get_pkg_version("chex") != "unknown"


def test_get_chex_module_and_griffe_offline_and_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_get_chex_module` and `_get_chex_griffe_module` across offline and live states.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import griffe

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert chex._get_chex_module() is None
    assert chex._get_chex_griffe_module() is None

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    dummy_mod = types.ModuleType("chex")
    with patch(
        "ml_ecosystem_snapshots.frameworks.chex.importlib.import_module",
        return_value=dummy_mod,
    ):
        assert chex._get_chex_module() is dummy_mod

    with patch(
        "ml_ecosystem_snapshots.frameworks.chex.importlib.import_module",
        side_effect=ImportError("no chex"),
    ):
        assert chex._get_chex_module() is None

    real_griffe_mod = griffe.Module("chex")
    with patch(
        "ml_ecosystem_snapshots.frameworks.chex.load_griffe_module_safely",
        return_value=real_griffe_mod,
    ):
        assert chex._get_chex_griffe_module() is real_griffe_mod


def test_load_griffe_module_safely_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `load_griffe_module_safely` offline, cache hit, success, and exception branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import griffe

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert load_griffe_module_safely("chex") is None

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    cached_mod = griffe.Module("chex_cached_test")
    DEFAULT_GRIFFE_CACHE.put("chex_cached_test", cached_mod)
    try:
        assert load_griffe_module_safely("chex_cached_test") is cached_mod
    finally:
        DEFAULT_GRIFFE_CACHE.pop("chex_cached_test", None)

    with patch("griffe.load", side_effect=RuntimeError("cannot load")):
        assert load_griffe_module_safely("nonexistent_pkg_xyz_123") is None

    # Success branch loading a real standard-library module via Griffe
    DEFAULT_GRIFFE_CACHE.pop("json", None)
    loaded_json = load_griffe_module_safely("json", framework_name="chex")
    assert loaded_json is not None
    DEFAULT_GRIFFE_CACHE.pop("json", None)


def test_scan_griffe_module_members_all_branches() -> None:
    """Verify `scan_griffe_module_members` across all member filtering and error branches."""
    import griffe

    from ml_ecosystem_snapshots.introspection.runtime_boundary import StaticGriffeNode

    with patch(
        "ml_ecosystem_snapshots.frameworks._base.load_griffe_module_safely",
        return_value=None,
    ):
        assert scan_griffe_module_members("chex", "chex") == []

    griffe_mod = griffe.Module("chex")
    griffe_mod.set_member("_private_fn", griffe.Function("_private_fn"))
    griffe_mod.set_member("blocked_fn", griffe.Function("blocked_fn"))
    griffe_mod.set_member("broken_member", griffe.Alias("broken_member", target="missing_pkg.missing_sym"))
    griffe_mod.set_member("attr_only", griffe.Attribute("attr_only"))
    griffe_mod.set_member("valid_fn", griffe.Function("valid_fn"))
    griffe_mod.set_member("valid_cls", griffe.Class("valid_cls"))
    griffe_mod.set_member("inspect_error_fn", griffe.Function("inspect_error_fn"))

    def fake_inspector(
        obj: StaticGriffeNode,
        api_path: str,
        is_public: bool | None = None,
    ) -> GhostRef:
        """Return a `GhostRef` or raise on `inspect_error_fn`.

        Args:
            obj: Griffe AST member node.
            api_path: API path string.
            is_public: Public visibility flag.

        Returns:
            Constructed `GhostRef`.

        Raises:
            ValueError: When `api_path` ends with `inspect_error_fn`.
        """
        if api_path.endswith("inspect_error_fn"):
            raise ValueError("inspect failed")
        return GhostRef(
            name=api_path.split(".")[-1],
            api_path=api_path,
            kind="class" if obj.is_class else "function",
            params=[],
            environment_tags=["chex"],
        )

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks._base.load_griffe_module_safely",
            return_value=griffe_mod,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks._base.GhostInspector.inspect",
            side_effect=fake_inspector,
        ),
    ):
        refs = scan_griffe_module_members(
            "chex",
            "chex",
            include_nonpublic=False,
            block_list={"blocked_fn"},
            environment_tags=("chex", "jax"),
        )
        names = {r.name for r in refs}
        assert names == {"valid_fn", "valid_cls"}
        for r in refs:
            assert r.environment_tags is not None
            assert "jax" in r.environment_tags


def test_collect_chex_api_offline_and_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `collect_chex_api` and `collect_api` in offline, live, and Griffe modes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    assert chex.collect_api(SemanticTier.LOSS) == []

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    offline_refs = chex.collect_api(SemanticTier.ARRAY_API)
    assert len(offline_refs) == len(chex.CANONICAL_CHEX_OPS)
    offline_names = {r.name for r in offline_refs}
    assert "assert_shape" in offline_names
    assert "assert_trees_all_close" in offline_names
    assert "dataclass" in offline_names

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    fake_live = types.ModuleType("chex")

    def custom_live_assert(x: int) -> int:
        """Custom live assertion helper.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    pytest.MonkeyPatch().setattr(fake_live, "custom_live_assert", custom_live_assert, raising=False)
    pytest.MonkeyPatch().setattr(fake_live, "assert_shape", custom_live_assert, raising=False)

    griffe_extra = GhostRef(
        name="griffe_discovered_assert",
        api_path="chex.griffe_discovered_assert",
        kind="function",
        params=[],
    )
    griffe_dup = GhostRef(
        name="assert_shape",
        api_path="chex.assert_shape",
        kind="function",
        params=[],
    )

    import griffe

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_module",
            return_value=fake_live,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_griffe_module",
            return_value=griffe.Module("chex"),
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.chex.scan_griffe_module_members",
            return_value=[griffe_extra, griffe_dup],
        ),
    ):
        live_refs = chex.collect_chex_api(include_nonpublic=False)
        paths = {r.api_path for r in live_refs}
        assert "chex.custom_live_assert" in paths
        assert "chex.griffe_discovered_assert" in paths
        assert "chex.assert_shape" in paths

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_module",
            return_value=None,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_griffe_module",
            return_value=None,
        ),
    ):
        fallback_refs = chex.collect_chex_api(include_nonpublic=False)
        assert len(fallback_refs) == len(chex.CANONICAL_CHEX_OPS)


def test_check_chex_op_mcp_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `check_chex_op`, MCP dispatch, CLI `cmd_check_chex`, anti-patterns, and concept translation.

    Args:
        capsys: Pytest stdout/stderr capture fixture.
    """
    res_valid = check_chex_op("chex.assert_shape", inputs_count=2, attributes=["custom_message"])
    assert res_valid.is_valid is True

    res_bad = check_chex_op("chex.nonexistent_assert")
    assert res_bad.is_valid is False

    rpc_res = dispatch_tool_call(
        1,
        "check_chex_op",
        {"op_name": "assert_trees_all_close", "attributes": ["rtol", "atol"]},
    )
    assert rpc_res.result is not None

    args_ok = argparse.Namespace(op_name="assert_shape", inputs_count=2, attributes="custom_message")
    cmd_check_chex(args_ok)
    assert "Chex Operation 'assert_shape' is valid." in capsys.readouterr().out

    args_fail = argparse.Namespace(op_name="unknown_chex_op", inputs_count=None, attributes=None)
    with pytest.raises(SystemExit):
        cmd_check_chex(args_fail)

    anti = explain_anti_pattern("chex", "chex.assert_shape", "msg")
    assert anti.is_known_anti_pattern is True
    assert anti.canonical_argument == "custom_message"

    trans = translate_concept_arguments(
        "assert_allclose",
        "torch",
        "chex",
        {"rtol": 1e-5, "atol": 1e-6, "msg": "mismatch"},
    )
    assert trans.translated_kwargs["custom_message"] == "mismatch"

    block_res = check_code_block(
        """import chex
chex.assert_shape(x, (2, 3), msg='bad')"""
    )
    assert block_res.total_analyzed == 1


def test_extract_version_with_griffe_branches() -> None:
    """Verify `_extract_version_with_griffe` across alias, value, None, and exception branches."""
    import griffe

    col = griffe.ModulesCollection()
    alias_mod = griffe.Module("chex", modules_collection=col)
    col["chex"] = alias_mod
    sub_mod = griffe.Module("_src")
    sub_mod.set_member("VER", griffe.Attribute("VER", value="'0.1.90'"))
    alias_mod.set_member("_src", sub_mod)
    alias_mod.set_member("__version__", griffe.Alias("__version__", target="chex._src.VER"))

    with patch("griffe.load", return_value=alias_mod):
        assert extract_version_with_griffe("chex") == "0.1.90"

    attr_mod = griffe.Module("chex")
    attr_mod.set_member("__version__", griffe.Attribute("__version__", value='"0.2.0"'))
    with patch("griffe.load", return_value=attr_mod):
        assert extract_version_with_griffe("chex") == "0.2.0"

    empty_mod = griffe.Module("chex")
    with patch("griffe.load", return_value=empty_mod):
        assert extract_version_with_griffe("chex") is None

    with patch("griffe.load", return_value=griffe.Function("not_a_module")):
        assert extract_version_with_griffe("chex") is None

    with patch("griffe.load", side_effect=RuntimeError("fail")):
        assert extract_version_with_griffe("chex") is None

    with (
        patch(
            "importlib.metadata.version",
            side_effect=RuntimeError("not installed"),
        ),
        patch(
            "ml_ecosystem_snapshots.api.extract_version_with_griffe",
            return_value=None,
        ),
    ):
        assert get_pkg_version("chex") == "0.1.87"


def test_extract_snapshot_chex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end `extract_snapshot('chex')` execution in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    snap = extract_snapshot("chex")
    assert snap.get("target") == "chex"
    cats = snap.get("categories", {})
    assert isinstance(cats, dict)
    assert "array" in cats
