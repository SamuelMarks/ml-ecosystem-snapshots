"""Unit tests and verification suite for c extension sig."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_runtime_signature,
)
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def _set_raw_c_doc(fn: LivePythonTarget, raw_c_doc: str) -> None:
    """Attach a raw C-extension or pybind11 docstring to a callable test fixture.

    Args:
        fn: Callable fixture whose runtime docstring should be replaced.
        raw_c_doc: Raw C-extension signature docstring text.
    """
    pytest.MonkeyPatch().setattr(fn, "__doc__", raw_c_doc)


def dummy_c_ext() -> None:
    """Represent a single-signature C-extension function fixture."""
    pass


_set_raw_c_doc(
    dummy_c_ext,
    "dummy_c_ext(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor",
)

original_sig = get_runtime_signature


def patched_sig(target: LivePythonTarget, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> inspect.Signature:
    """Raise ValueError for `dummy_c_ext` and delegate other targets to `inspect.signature`.

    Args:
        target: Callable object to inspect.
        *args: Positional arguments forwarded to `inspect.signature`.
        **kwargs: Keyword arguments forwarded to `inspect.signature`.

    Raises:
        ValueError: When `target` is `dummy_c_ext`.

    Returns:
        The `inspect.Signature` for `target`.
    """
    if target is dummy_c_ext:
        raise ValueError("no signature found")
    return original_sig(target)


def test_c_extension_signature_fallback() -> None:
    """Verify c extension signature fallback behavior."""
    with patch("inspect.signature", side_effect=patched_sig):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_c_ext, "dummy_c_ext")

        assert len(ref.params) == 2
        assert ref.params[0].name == "input"
        assert ref.params[0].annotation == "Tensor"
        assert ref.params[1].name == "out"
        assert ref.params[1].kind == "KEYWORD_ONLY"
        assert ref.params[1].annotation == "Tensor | None"


def dummy_posonly(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
    """Represent a C-extension function fixture with positional-only parameters.

    Args:
        *args: Variadic positional arguments.
        **kwargs: Variadic keyword arguments.
    """
    pass


_set_raw_c_doc(
    dummy_posonly,
    "dummy_posonly(a: int, /, b: str = 'x') -> None\n\nArgs:\n    *args: Variadic positional arguments.\n    **kwargs: Variadic keyword arguments.",
)


def test_c_extension_posonlyargs() -> None:
    """Test extracting positional-only parameters from C-extension docstring."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_posonly, "dummy_posonly")
        assert len(ref.params) == 2
        assert ref.params[0].name == "a"
        assert str(ref.params[0].kind).endswith("POSITIONAL_ONLY")
        assert ref.params[1].name == "b"
        assert str(ref.params[1].kind).endswith("POSITIONAL_OR_KEYWORD")


def dummy_no_doc() -> None:
    """Dummy no doc function."""
    pass


def test_c_extension_inexact_signature_fallback() -> None:
    """Test that unresolvable functions receive inexact_signature tag."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_no_doc, "dummy_no_doc")
        assert "inexact_signature" in (ref.environment_tags or [])

        # When environment_tags already contains inexact_signature
        ref2 = inspector.inspect(
            dummy_no_doc,
            "dummy_no_doc",
            environment_tags=["custom_tag", "inexact_signature"],
        )
        assert "inexact_signature" in (ref2.environment_tags or [])


def dummy_torch_add() -> None:
    """Represent a multi-overload PyTorch add C-extension function fixture."""
    pass


_set_raw_c_doc(
    dummy_torch_add,
    "add(input, other, *, out=None) -> Tensor\n"
    "add(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Adds other scaled by alpha to input.\n",
)


def dummy_torch_clamp() -> None:
    """Represent a three-overload PyTorch clamp C-extension function fixture."""
    pass


_set_raw_c_doc(
    dummy_torch_clamp,
    "clamp(input, min=None, max=None, *, out=None) -> Tensor\n"
    "clamp(input, *, min=None, max=None, out=None) -> Tensor\n"
    "clamp(input, min: Tensor, max: Tensor, *, out=None) -> Tensor\n\n"
    "Clamps all elements in input into the range [min, max].\n",
)


def dummy_pybind11_overload() -> None:
    """Represent a numbered pybind11 overloaded C-extension function fixture."""
    pass


_set_raw_c_doc(
    dummy_pybind11_overload,
    "process(*args, **kwargs)\n"
    "Overloaded function.\n\n"
    "1. process(input: Tensor, factor: float) -> Tensor\n\n"
    "2. process(input: Tensor, matrix: Tensor) -> Tensor\n",
)


def test_c_extension_multi_signature_torch_add() -> None:
    """Test multi-signature overload extraction matching torch.add patterns."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_torch_add, "add")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 3
        assert [p.name for p in ref.params] == ["input", "other", "out"]
        assert ref.params[2].kind == "KEYWORD_ONLY"

        # Check secondary overload in ref.overloads
        assert ref.overloads is not None
        assert len(ref.overloads) == 1
        ov = ref.overloads[0]
        assert isinstance(ov, ExtendedGhostRef)
        assert ov.name == ref.name
        assert ov.returns_type == "Tensor"
        assert [p.name for p in ov.params] == ["input", "other", "alpha", "out"]
        assert ov.params[2].name == "alpha"
        assert ov.params[2].kind == "KEYWORD_ONLY"
        assert ov.params[2].default == "1"


def test_c_extension_multi_signature_torch_clamp() -> None:
    """Test multi-signature overload extraction matching torch.clamp with 3 overloads."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_torch_clamp, "clamp")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 4
        assert [p.name for p in ref.params] == ["input", "min", "max", "out"]
        assert ref.params[1].kind == "POSITIONAL_OR_KEYWORD"

        # Check secondary and tertiary overloads
        assert ref.overloads is not None
        assert len(ref.overloads) == 2
        ov1 = ref.overloads[0]
        assert isinstance(ov1, ExtendedGhostRef)
        assert [p.name for p in ov1.params] == ["input", "min", "max", "out"]
        assert ov1.params[1].kind == "KEYWORD_ONLY"
        assert ov1.params[2].kind == "KEYWORD_ONLY"
        assert ov1.params[3].kind == "KEYWORD_ONLY"

        ov2 = ref.overloads[1]
        assert isinstance(ov2, ExtendedGhostRef)
        assert [p.name for p in ov2.params] == ["input", "min", "max", "out"]
        assert ov2.params[1].annotation == "Tensor"
        assert ov2.params[2].annotation == "Tensor"


def test_c_extension_multi_signature_pybind11() -> None:
    """Test PyBind11 numbered overloads promote the first numbered overload to primary."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_pybind11_overload, "process")

        assert ref.returns_type == "Tensor"
        assert len(ref.params) == 2
        assert [p.name for p in ref.params] == ["input", "factor"]
        assert ref.params[1].annotation == "float"

        assert ref.overloads is not None
        assert len(ref.overloads) == 1
        ov = ref.overloads[0]
        assert isinstance(ov, ExtendedGhostRef)
        assert [p.name for p in ov.params] == ["input", "matrix"]
        assert ov.params[1].annotation == "Tensor"


def dummy_overloaded_header() -> None:
    """Represent a C-extension fixture with an inline Overloaded function header."""
    pass


_set_raw_c_doc(
    dummy_overloaded_header,
    "foo(a: int) -> int\nOverloaded function.\nfoo(b: str) -> str\n",
)


def test_c_extension_overloaded_header_branch() -> None:
    """Test C-extension docstring containing 'overloaded function' header between signatures."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_overloaded_header, "foo")
        assert len(ref.params) == 1
        assert ref.params[0].name == "a"
        assert ref.overloads is not None
        assert len(ref.overloads) == 1
        ov0 = ref.overloads[0]
        assert isinstance(ov0, ExtendedGhostRef)
        assert ov0.params[0].name == "b"


def dummy_leading_overload_header() -> None:
    """Represent a C-extension fixture starting with Overloaded function header."""
    pass


_set_raw_c_doc(
    dummy_leading_overload_header,
    "Overloaded function.\n\n1. bar(x: int) -> int\n\n2. bar(y: str) -> str\n",
)


def dummy_overload_vararg() -> None:
    """Represent a C-extension fixture with variadic positional secondary overload."""
    pass


_set_raw_c_doc(
    dummy_overload_vararg,
    "baz(x: int) -> int\nbaz(*args) -> int\n",
)


def dummy_no_ret_type() -> None:
    """Represent a C-extension fixture signature without an explicit return arrow."""
    pass


_set_raw_c_doc(dummy_no_ret_type, "qux(x: int)")


def test_c_extension_leading_header_and_vararg_overloads() -> None:
    """Test leading 'overloaded function' header and varargs in secondary overload."""
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(dummy_leading_overload_header, "bar")
        assert len(ref.params) == 1
        assert ref.params[0].name == "x"
        assert ref.overloads is not None
        assert len(ref.overloads) == 1
        ov_bar = ref.overloads[0]
        assert isinstance(ov_bar, ExtendedGhostRef)
        assert ov_bar.params[0].name == "y"

        ref_var = inspector.inspect(dummy_overload_vararg, "baz")
        assert ref_var.overloads is not None
        assert len(ref_var.overloads) == 1
        ov_baz = ref_var.overloads[0]
        assert isinstance(ov_baz, ExtendedGhostRef)
        assert ov_baz.has_varargs is True

        ref_no_ret = inspector.inspect(dummy_no_ret_type, "qux")
        assert ref_no_ret.returns_type in (None, "NoneType")
        assert len(ref_no_ret.params) == 1


def dummy_opaque_func() -> None:
    """Opaque function without docstring signature."""
    pass


def test_opaque_c_extension_existing_environment_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test opaque fallback when environment_tags already contains tags.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(dummy_opaque_func, "__doc__", None)
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        inspector = GhostInspector()
        ref = inspector.inspect(
            dummy_opaque_func,
            "opaque_func",
            environment_tags=["inexact_signature", "opaque_c_extension"],
        )
        assert ref.signature_completeness == "opaque"
        assert ref.environment_tags == ["inexact_signature", "opaque_c_extension"]
        assert len(ref.params) == 2
        assert ref.params[0].name == "args"
        assert ref.params[1].name == "kwargs"


def test_c_extension_signature_sequence_operations() -> None:
    """Verify CExtensionSignature parameter access, iteration, equality, and repr."""
    from ml_ecosystem_snapshots.c_extension_sig import (
        CExtensionSignature,
        ParsedParamTuple,
    )

    sig = CExtensionSignature(
        params=[
            ("x", "POSITIONAL_OR_KEYWORD", None, "Tensor"),
            ParsedParamTuple(name="y", kind="POSITIONAL_OR_KEYWORD", default="1", annotation="int"),
        ],
        returns_type="Tensor",
    )
    assert len(sig) == 2
    assert sig[0].name == "x"
    assert sig[0].kind == "POSITIONAL_OR_KEYWORD"
    assert sig[0].default is None
    assert sig[0].annotation == "Tensor"
    assert sig[0].as_tuple() == ("x", "POSITIONAL_OR_KEYWORD", None, "Tensor")
    assert sig[1].name == "y"
    assert sig[1].default == "1"

    # Equality and repr
    sig_copy = CExtensionSignature(params=list(sig.iter_params()))
    assert sig == sig_copy
    assert sig == list(sig.iter_params())
    assert sig != "not_a_sig"
    assert "ParsedParamTuple" in repr(sig) or "'x'" in repr(sig)
