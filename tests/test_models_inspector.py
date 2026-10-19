"""Unit tests and verification suite for models inspector."""

from __future__ import annotations

import pytest_mock


from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import ExtendedGhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef


def dummy_func_with_docstring(x: int) -> str:
    """Dummy function.

    Args:
        x (int): The x value.

    Returns:
        str: A string.

    Raises:
        ValueError: If x is bad.

    # noqa: DAR202, DAR402
    """
    if x < 0:
        raise ValueError()
    return "test"


def dummy_func_sphinx_docstring(y: object) -> None:
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
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert ref.returns_description is not None
    assert ref.params[0].name == "x"
    assert ref.params[0].description == "The x value."


def test_cdd_sphinx_raises() -> None:
    """Verify cdd sphinx raises behavior."""
    ref = GhostInspector.inspect(
        dummy_func_sphinx_docstring, "tests.dummy_func_sphinx_docstring"
    )
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
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
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
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
    assert ref.raises is not None
    assert "KeyError" in ref.raises


def test_inspect_annotation_fallback() -> None:
    """Verify inspect annotation fallback behavior."""

    class ForwardRefStr:
        """Mock or helper class ForwardRefStr for unit testing."""

        # A class that lacks __name__ but is passed as annotation
        pass

    def func_annotated(a: object) -> None:
        """Execute helper func annotated.

        Args:
            a: First positional operand.
        """
        pass

    func_annotated.__annotations__["a"] = ForwardRefStr()

    ref = GhostInspector.inspect(func_annotated, "tests.func_annotated")
    assert ref.params[0].annotation is not None
    assert "ForwardRefStr" in ref.params[0].annotation


def test_griffe_self_skip() -> None:
    """Verify griffe self skip behavior."""

    # griffe self skipping
    class A:
        """Mock or helper class A for unit testing."""

        def m(self, x: object) -> None:
            """Execute helper m.

            Args:
                x: Input operand value.
            """
            pass

    # we need a real class importable by griffe to test griffe skipping 'self'
    import email.message

    ref = GhostInspector.inspect(
        email.message.Message.set_payload, "email.message.Message.set_payload"
    )
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
    ref = GhostInspector.inspect(
        dummy_func_with_docstring, "tests.dummy_func_with_docstring"
    )
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
        overloads=[
            CExtensionSignature(params=[("z", "POSITIONAL_OR_KEYWORD", None, "str")])
        ],
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
    from ml_switcheroo_ir.schema.ghost import ParameterKind

    data = {
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
