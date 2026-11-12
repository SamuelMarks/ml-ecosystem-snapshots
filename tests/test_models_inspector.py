"""Unit tests and verification suite for models inspector."""

from __future__ import annotations

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import ExtendedGhostParam


def dummy_func_with_docstring(x: int) -> str:
    """Dummy function.

    Args:
        x (int): The x value.

    Returns:
        str: A string.

    Raises:
        ValueError: If x is bad.
    """
    if x < 0:
        raise ValueError()
    return "test"


def dummy_func_sphinx_docstring(y: LivePythonTarget) -> None:
    """Sphinx doc.

    :param y: The y value.
    :type y: int
    :raises TypeError: If wrong.

    Args:
        y: Input integer parameter.
    """
    pass


def test_cdd_returns_and_params() -> None:
    """Verify cdd returns and params behavior."""
    ref = GhostInspector.inspect(dummy_func_with_docstring, "tests.dummy_func_with_docstring")
    assert ref.returns_description is not None
    assert ref.params[0].name == "x"
    assert ref.params[0].description == "The x value."


def test_cdd_sphinx_raises() -> None:
    """Verify cdd sphinx raises behavior."""
    ref = GhostInspector.inspect(dummy_func_sphinx_docstring, "tests.dummy_func_sphinx_docstring")
    assert ref.raises is not None
    assert "TypeError" in ref.raises


def test_griffe_parsing() -> None:
    """Verify griffe parsing behavior."""
    import urllib.parse

    ref = GhostInspector.inspect(urllib.parse.urljoin, "urllib.parse.urljoin")
    assert ref.name == "urljoin"
    assert ref.has_arg("base")
    assert ref.has_arg("url")


def test_cdd_exception_handling(mocker: pytest_mock.MockerFixture) -> None:
    """Verify cdd exception handling behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("cdd.docstring.parse.docstring", side_effect=ValueError)
    ref = GhostInspector.inspect(dummy_func_with_docstring, "tests.dummy_func_with_docstring")
    assert ref.name == "dummy_func_with_docstring"


def test_cdd_direct_raises(mocker: pytest_mock.MockerFixture) -> None:
    """Verify cdd direct raises behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Mock cdd to return direct raises list
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"typ": "KeyError"}]},
    )
    ref = GhostInspector.inspect(dummy_func_with_docstring, "tests.dummy_func_with_docstring")
    assert ref.raises is not None
    assert "KeyError" in ref.raises


def test_inspect_annotation_fallback() -> None:
    """Verify inspect annotation fallback behavior."""

    class ForwardRefStr:
        """Custom annotation stub without a runtime class name attribute."""

        # A class that lacks __name__ but is passed as annotation
        pass

    def func_annotated(a: LivePythonTarget) -> None:
        """Accept a single parameter carrying a non-type annotation object.

        Args:
            a: First positional operand.
        """
        pass

    import pytest

    pytest.MonkeyPatch().setattr(func_annotated, "__annotations__", {"a": ForwardRefStr(), "return": None})

    ref = GhostInspector.inspect(func_annotated, "tests.func_annotated")
    assert ref.params[0].annotation is not None
    assert "ForwardRefStr" in ref.params[0].annotation


def test_griffe_self_skip() -> None:
    """Verify griffe self skip behavior."""

    # griffe self skipping
    class A:
        """Sample class defining an instance method with an explicit self parameter."""

        def m(self, x: LivePythonTarget) -> None:
            """Accept an operand value on an instance method.

            Args:
                x: Input operand value.
            """
            pass

    # we need a real class importable by griffe to test griffe skipping 'self'
    import email.message

    ref = GhostInspector.inspect(email.message.Message.set_payload, "email.message.Message.set_payload")
    assert not ref.has_arg("self")
    assert ref.has_arg("payload")


def test_griffe_varargs() -> None:
    """Verify griffe varargs behavior."""
    import subprocess

    ref = GhostInspector.inspect(subprocess.run, "subprocess.run")
    assert ref.has_varargs


def test_models_raises_no_typ(mocker: pytest_mock.MockerFixture) -> None:
    """Verify models raises no typ behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"not_typ": "KeyError"}]},
    )
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_griffe_docstring_metadata",
        return_value={"params": {}, "returns": None, "raises": []},
    )
    ref = GhostInspector.inspect(dummy_func_with_docstring, "tests.dummy_func_with_docstring")
    assert not ref.raises


def test_ghost_inspector_aten_overload_and_factory_default(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector when c_ext_params already has overloads and overload has factory default.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.c_extension_sig import CExtensionSignature

    existing_overload = CExtensionSignature(
        params=[
            ("y", "POSITIONAL_OR_KEYWORD", "<factory_default>", "float"),
            ("z", "POSITIONAL_OR_KEYWORD", "1.0", "float"),
        ]
    )
    existing_sig = CExtensionSignature(
        params=[("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[existing_overload],
    )
    aten_sig = CExtensionSignature(
        params=[("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[CExtensionSignature(params=[("z", "POSITIONAL_OR_KEYWORD", None, "str")])],
    )

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_c_extension_signature",
        return_value=existing_sig,
    )
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.extract_aten_c_extension_signature",
        return_value=aten_sig,
    )
    mocker.patch("inspect.signature", side_effect=TypeError("no python signature"))

    def dummy_torch_fn(x: int) -> None:
        """Dummy function for torch inspection test.

        Args:
            x: Integer parameter.
        """
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    p0 = ov0.params[0]
    assert isinstance(p0, ExtendedGhostParam)
    assert p0.default_factory == "<factory_default>"

    # Also test when c_ext_params has no overloads and aten_sig supplies overloads
    sig_without_overloads = CExtensionSignature(
        params=[("x", "POSITIONAL_OR_KEYWORD", None, "int")],
        overloads=[],
    )
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_c_extension_signature",
        return_value=sig_without_overloads,
    )
    ref_merged = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert ref_merged.overloads is not None
    assert len(ref_merged.overloads) == 1
    assert ref_merged.signature_completeness == "exact"

    # Also test when extract_aten_c_extension_signature raises an exception
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.extract_aten_c_extension_signature",
        side_effect=RuntimeError("aten error"),
    )
    ref_aten_err = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert ref_aten_err.signature_completeness == "heuristic"

    # Also test _build_extended_params when p_anno is None and cdd_ast_params supplies typ
    from ml_ecosystem_snapshots.introspection.inspector import (
        ExtractedParamRecord,
        build_extended_params,
    )
    from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str

    built_params = build_extended_params(
        extracted_params=[("alpha", "POSITIONAL_OR_KEYWORD", None, None)],
        cdd_params={},
        cdd_ast_params={"alpha": ExtractedParamRecord(doc="alpha doc", typ="float", default=None)},
        is_torch_target=False,
        name="sample",
        active_sanitize_type_str=sanitize_type_str,
    )
    assert built_params[0].annotation == "float"


def test_ghost_inspector_aten_none(mocker: pytest_mock.MockerFixture) -> None:
    """Test GhostInspector when extract_aten_c_extension_signature returns None.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("inspect.signature", side_effect=TypeError("no python signature"))
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.extract_aten_c_extension_signature",
        return_value=None,
    )
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_c_extension_signature",
        return_value=None,
    )

    def dummy_torch_fn() -> None:
        """Dummy function for torch inspection test."""
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert len(ref.params) == 2
    assert ref.params[0].name == "args"
    assert ref.params[1].name == "kwargs"


def test_ghost_inspector_hydrate_variations() -> None:
    """Test GhostInspector.hydrate handles missing kind and pre-hydrated parameters."""
    import typing

    from ml_switcheroo_ir.schema.ghost import GhostParam, ParameterKind

    from ml_ecosystem_snapshots.models import JsonValue

    data: dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] = {
        "name": "op_test",
        "api_path": "custom.op_test",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
            ExtendedGhostParam(name="y", kind=ParameterKind.POSITIONAL_ONLY),
        ],
    }
    hydrated = GhostInspector.hydrate(data)
    assert len(hydrated.params) == 2
    assert hydrated.params[0].name == "x"
    assert hydrated.params[0].kind == ParameterKind.POSITIONAL_OR_KEYWORD
    assert hydrated.params[1].name == "y"
    assert hydrated.params[1].kind == ParameterKind.POSITIONAL_ONLY


def test_inspector_helper_fallback_branches() -> None:
    """Verify fallback branches across internal Griffe/CDD parameter and cache helpers."""
    import types

    import pytest

    from ml_ecosystem_snapshots.introspection.inspector import (
        DocstringIrResult,
        coerce_param_record,
        get_griffe_node_overloads,
        get_griffe_node_params,
        get_griffe_param_kind_name,
        merge_griffe_return_meta,
        resolve_griffe_node_from_cache,
    )

    assert coerce_param_record(42).doc is None

    res = DocstringIrResult()
    merge_griffe_return_meta("not_a_dict_or_meta", lambda s: s, res)
    assert res.returns_type is None

    import griffe

    bare_param = griffe.Parameter(name="p", kind=None)
    assert get_griffe_param_kind_name(bare_param, default="KEYWORD_ONLY") == "KEYWORD_ONLY"

    no_load_mod = types.ModuleType("griffe_no_load")
    assert resolve_griffe_node_from_cache("pkg.fn", "google", no_load_mod, {}) is None

    root_with_leaf = griffe.Module("pkg")
    root_with_leaf.set_member("sub", griffe.Attribute("sub"))
    fake_griffe = types.ModuleType("griffe_leaf")
    pytest.MonkeyPatch().setattr(fake_griffe, "load", lambda *_a, **_k: root_with_leaf, raising=False)
    assert resolve_griffe_node_from_cache("pkg.sub.fn", "google", fake_griffe, {}) is None

    bad_alias = griffe.Alias("bad_alias", target="missing_pkg.missing_cls")
    assert get_griffe_node_params(bad_alias) is None
    assert get_griffe_node_overloads(bad_alias) is None
    assert get_griffe_node_params(None) is None
    assert get_griffe_node_overloads(None) is None

    empty_cls = griffe.Class("EmptyCls")
    empty_cls.set_member("__init__", griffe.Function("__init__", parameters=griffe.Parameters()))
    assert get_griffe_node_params(empty_cls) is None
    assert get_griffe_node_overloads(empty_cls) is None


def test_ghostref_from_obj_griffe_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `GhostInspector.inspect` succeeds using runtime/CDD extraction when `griffe` is unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys

    monkeypatch.setitem(sys.modules, "griffe", None)

    def dummy_fn(val: int = 3) -> int:
        """Return input integer unchanged.

        Args:
            val: Input integer value.

        Returns:
            Same integer value.
        """
        return val

    ref = GhostInspector.inspect(dummy_fn, "dummy_fn")
    assert ref.name == "dummy_fn"
    assert [p.name for p in ref.params] == ["val"]
    assert ref.params[0].default == "3"


def test_extended_ghost_ref_dict_with_non_string_default() -> None:
    """Verify `GhostInspector.hydrate` coerces non-string numeric defaults into canonical strings."""
    from ml_ecosystem_snapshots.models import JsonValue

    data: dict[str, JsonValue] = {
        "name": "foo",
        "api_path": "foo",
        "kind": "function",
        "params": [{"name": "a", "default": 123, "kind": "POSITIONAL_OR_KEYWORD"}],
    }
    ref = GhostInspector.hydrate(data)
    assert ref.name == "foo"
    assert len(ref.params) == 1
    assert ref.params[0].default == "123"
