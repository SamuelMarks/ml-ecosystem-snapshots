"""Unit test suite for Chex snapshot extractor, Griffe static scanning, MCP validation, and CLI tooling."""

from __future__ import annotations

import argparse
import types
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pytest

from ml_ecosystem_snapshots.api import (
    _extract_version_with_griffe,
    extract_snapshot,
    get_pkg_version,
)
from ml_ecosystem_snapshots.cli.commands.check import cmd_check_chex
from ml_ecosystem_snapshots.frameworks import chex
from ml_ecosystem_snapshots.frameworks._base import scan_griffe_module_members
from ml_ecosystem_snapshots.introspection.static_parsers import (
    _GRIFFE_CACHE,
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


class _FakeGriffeMember:
    """Fake Griffe AST member implementing `GriffeMemberProtocol`."""

    def __init__(
        self,
        is_class: bool = False,
        is_function: bool = True,
        is_public: bool = True,
        raise_on_kind: bool = False,
    ) -> None:
        """Initialize fake Griffe member.

        Args:
            is_class: Whether member is a class.
            is_function: Whether member is a function.
            is_public: Whether member is public.
            raise_on_kind: Whether accessing `is_class` raises an error.
        """
        self._is_class = is_class
        self._is_function = is_function
        self.is_public = is_public
        self._raise_on_kind = raise_on_kind

    @property
    def is_class(self) -> bool:
        """Return whether node is a class."""
        if self._raise_on_kind:
            raise RuntimeError("broken alias")
        return self._is_class

    @property
    def is_function(self) -> bool:
        """Return whether node is a function."""
        return self._is_function


class _FakeGriffeModule:
    """Fake Griffe Module implementing `GriffeModuleMembersProtocol`."""

    def __init__(self, members: dict[str, _FakeGriffeMember]) -> None:
        """Initialize fake Griffe module.

        Args:
            members: Mapping of member names to `_FakeGriffeMember` instances.
        """
        self.members = members
        self.is_public = True


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

    fake_griffe = _FakeGriffeModule({})
    with patch(
        "ml_ecosystem_snapshots.frameworks.chex.load_griffe_module_safely",
        return_value=fake_griffe,
    ):
        assert chex._get_chex_griffe_module() is fake_griffe


def test_load_griffe_module_safely_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `load_griffe_module_safely` offline, cache hit, success, and exception branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert load_griffe_module_safely("chex") is None

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    fake_node = _FakeGriffeModule({})
    _GRIFFE_CACHE["chex_cached_test"] = fake_node
    try:
        assert load_griffe_module_safely("chex_cached_test") is fake_node
    finally:
        _GRIFFE_CACHE.pop("chex_cached_test", None)

    with patch("griffe.load", side_effect=RuntimeError("cannot load")):
        assert load_griffe_module_safely("nonexistent_pkg_xyz_123") is None

    # Success branch loading a real standard-library module via Griffe
    _GRIFFE_CACHE.pop("json", None)
    loaded_json = load_griffe_module_safely("json", framework_name="chex")
    assert loaded_json is not None
    _GRIFFE_CACHE.pop("json", None)


def test_scan_griffe_module_members_all_branches() -> None:
    """Verify `scan_griffe_module_members` across all member filtering and error branches."""
    with patch(
        "ml_ecosystem_snapshots.frameworks._base.load_griffe_module_safely",
        return_value=None,
    ):
        assert scan_griffe_module_members("chex", "chex") == []

    members = {
        "_private_fn": _FakeGriffeMember(is_function=True),
        "blocked_fn": _FakeGriffeMember(is_function=True),
        "broken_member": _FakeGriffeMember(raise_on_kind=True),
        "attr_only": _FakeGriffeMember(is_class=False, is_function=False),
        "valid_fn": _FakeGriffeMember(is_function=True),
        "valid_cls": _FakeGriffeMember(is_class=True, is_function=False),
        "inspect_error_fn": _FakeGriffeMember(is_function=True),
    }
    fake_mod = _FakeGriffeModule(members)

    def fake_inspector(
        obj: _FakeGriffeMember,
        api_path: str,
        is_public: bool | None = None,
    ) -> GhostRef:
        """Fake inspector returning a `GhostRef` or raising on `inspect_error_fn`.

        Args:
            obj: Fake Griffe member node.
            api_path: API path string.
            is_public: Public visibility flag.

        Returns:
            Constructed `GhostRef`.
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
            return_value=fake_mod,
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

    fake_live.__dict__["custom_live_assert"] = custom_live_assert
    fake_live.__dict__["assert_shape"] = custom_live_assert

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

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_module",
            return_value=fake_live,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.chex._get_chex_griffe_module",
            return_value=_FakeGriffeModule({}),
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
    res_valid = check_chex_op(
        "chex.assert_shape", inputs_count=2, attributes=["custom_message"]
    )
    assert res_valid["is_valid"] is True

    res_bad = check_chex_op("chex.nonexistent_assert")
    assert res_bad["is_valid"] is False

    rpc_res = dispatch_tool_call(
        1,
        "check_chex_op",
        {"op_name": "assert_trees_all_close", "attributes": ["rtol", "atol"]},
    )
    assert "result" in rpc_res

    args_ok = argparse.Namespace(
        op_name="assert_shape", inputs_count=2, attributes="custom_message"
    )
    cmd_check_chex(args_ok)
    assert "Chex Operation 'assert_shape' is valid." in capsys.readouterr().out

    args_fail = argparse.Namespace(
        op_name="unknown_chex_op", inputs_count=None, attributes=None
    )
    with pytest.raises(SystemExit):
        cmd_check_chex(args_fail)

    anti = explain_anti_pattern("chex", "chex.assert_shape", "msg")
    assert anti["is_known_anti_pattern"] is True
    assert anti["canonical_argument"] == "custom_message"

    trans = translate_concept_arguments(
        "assert_allclose",
        "torch",
        "chex",
        {"rtol": 1e-5, "atol": 1e-6, "msg": "mismatch"},
    )
    assert trans["translated_kwargs"]["custom_message"] == "mismatch"

    block_res = check_code_block(
        """import chex
chex.assert_shape(x, (2, 3), msg='bad')"""
    )
    assert block_res["total_analyzed"] == 1


def test_extract_version_with_griffe_branches() -> None:
    """Verify `_extract_version_with_griffe` across alias, value, None, and exception branches."""

    class _FakeTarget:
        """Fake Griffe alias target with a `value` attribute."""

        def __init__(self, value: str | None) -> None:
            """Initialize fake target.

            Args:
                value: Version string or None.
            """
            self.value = value

    class _FakeAlias:
        """Fake Griffe Alias member."""

        def __init__(self, target: _FakeTarget | None) -> None:
            """Initialize fake alias.

            Args:
                target: Target node or None.
            """
            self.target = target

    class _FakeAttr:
        """Fake Griffe Attribute member."""

        def __init__(self, value: str | None) -> None:
            """Initialize fake attribute.

            Args:
                value: Version string or None.
            """
            self.value = value

    class _FakeMod:
        """Fake Griffe Module with `members` dictionary."""

        def __init__(self, members: dict[str, _FakeAlias | _FakeAttr | None]) -> None:
            """Initialize fake module.

            Args:
                members: Mapping of member names to fake nodes.
            """
            self.members = members

    with patch(
        "griffe.load",
        return_value=_FakeMod({"__version__": _FakeAlias(_FakeTarget("'0.1.90'"))}),
    ):
        assert _extract_version_with_griffe("chex") == "0.1.90"

    with patch(
        "griffe.load",
        return_value=_FakeMod({"__version__": _FakeAttr('"0.2.0"')}),
    ):
        assert _extract_version_with_griffe("chex") == "0.2.0"

    with patch(
        "griffe.load",
        return_value=_FakeMod({"__version__": None}),
    ):
        assert _extract_version_with_griffe("chex") is None

    with patch("griffe.load", return_value="not_a_module"):
        assert _extract_version_with_griffe("chex") is None

    with patch("griffe.load", side_effect=RuntimeError("fail")):
        assert _extract_version_with_griffe("chex") is None

    with (
        patch(
            "importlib.metadata.version",
            side_effect=RuntimeError("not installed"),
        ),
        patch(
            "ml_ecosystem_snapshots.api._extract_version_with_griffe",
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
