"""Tests for AST analysis edge cases, parser fallbacks, and exception handling in GhostInspector."""

from __future__ import annotations

import ast
import inspect
import typing
from unittest.mock import patch

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget


class BaseParentModel:
    """Base parent class."""

    def __init__(self, arg1: int = 1) -> None:
        """Initialize BaseParentModel.

        Args:
            arg1: Integer parameter.
        """
        pass


class OtherHelperModel:
    """Helper class."""

    def __init__(self) -> None:
        """Initialize helper."""
        pass


helper_instance = OtherHelperModel()


class DerivedClassModel(BaseParentModel):
    """Derived class with custom __init__ calls."""

    def __init__(self, a: int = 10) -> None:
        """Initialize DerivedClassModel.

        Args:
            a: Parameter a.
        """
        pass


class SimpleClassModel:
    """Simple class.

    Args:
        x: Input.
    """

    def __init__(self, x: int) -> None:
        """Initialize SimpleClassModel.

        Args:
            x: Input.
        """
        pass


def target_fn_ast_edges(
    self: SimpleClassModel,
    ast_arg: int = 42,
    broken_arg: str = "x",
    hint_arg: int = 1,
) -> None:
    """Target function for AST default edges.

    Args:
        self: Self parameter.
        ast_arg: AST default parameter.
        broken_arg: Broken default parameter.
        hint_arg: Parameter resolving type from hint.
    """
    pass


def target_fn_sig_failure(x: int) -> None:
    """Target function for signature failure in CDD branch.

    Args:
        x: Input tensor or scalar operand.
    """
    pass


class MroParentClass:
    """Parent class."""

    def __init__(self, p: int = 1) -> None:
        """Initialize MroParentClass.

        Args:
            p: Parameter p.
        """
        pass


class MroChildClass(MroParentClass):
    """Child class calling super with **kwargs."""

    def __init__(self, **kwargs: int) -> None:
        """Initialize MroChildClass.

        Args:
            **kwargs: Keyword args passed to super.
        """
        pass


def test_extract_accepted_kwargs_subscript_branches() -> None:
    """Test extract_accepted_kwargs_from_ast across all Subscript AST branches.

    Returns:
        None.
    """
    tree = ast.parse(
        "def sample_fn(**kwargs):\n    kwargs[123]\n    kwargs['alpha']\n    kwargs[var]\n    other_dict['ignored']\n"
    )

    with patch("inspect.getsource", return_value="def sample_fn(**kwargs): pass"):
        with patch(
            "ml_ecosystem_snapshots.introspection.ast_kwargs.ast.parse",
            return_value=tree,
        ):
            res = extract_accepted_kwargs_from_ast(lambda **kwargs: None)
            assert res == ["alpha"]


def test_ghost_inspector_unwrap_max_depth() -> None:
    """Test recursive unwrap loop when max depth 10 is reached without break.

    Returns:
        None.
    """

    def make_wrapper(
        inner: typing.Callable[[int], int],
    ) -> typing.Callable[[int], int]:
        """Create function wrapper with __wrapped__ attribute.

        Args:
            inner: Inner callable.

        Returns:
            Wrapped callable.
        """

        def wrapped(x: int) -> int:
            """Wrapped function.

            Args:
                x: Input value.

            Returns:
                Call result.
            """
            return inner(x)

        pytest.MonkeyPatch().setattr(wrapped, "__wrapped__", inner, raising=False)
        return wrapped

    def base_func(x: int) -> int:
        """Base function.

        Args:
            x: Input value.

        Returns:
            Result value.
        """
        return x

    curr: typing.Callable[[int], int] = base_func
    for _ in range(12):
        curr = make_wrapper(curr)

    ref = GhostInspector.inspect(curr, "test_deep_wrapper")
    assert ref.has_arg("x")


def test_ghost_inspector_cdd_docstring_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector handles exception in cdd_docstring_parsers.docstring.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.docstring.parse as cdd_docstring_parsers

    mocker.patch.object(
        cdd_docstring_parsers,
        "docstring",
        side_effect=RuntimeError("simulated cdd failure"),
    )

    def dummy(x: int) -> None:
        """Dummy docstring.

        Args:
            x: Value.
        """
        pass

    ref = GhostInspector.inspect(dummy, "dummy_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_griffe_docstring_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector handles exception in extract_griffe_docstring_metadata.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_griffe_docstring_metadata",
        side_effect=RuntimeError("simulated griffe doc error"),
    )

    def dummy(x: int) -> None:
        """Dummy docstring.

        Args:
            x: Value.
        """
        pass

    ref = GhostInspector.inspect(dummy, "dummy_griffe_doc_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_super_init_branches(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector AST inspection branches for super().__init__ calls.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.get_runtime_source",
        return_value=(
            "class DerivedClassModel(BaseParentModel):\n    def __init__(self, a: int = 10, **kwargs: int) -> None:\n        OtherHelperModel.__init__(helper_instance)\n        super().__init__(arg1=a)\n"
        ),
    )
    ref = GhostInspector.inspect(DerivedClassModel, "DerivedClassModel")
    assert ref.has_arg("a")


def test_ghost_inspector_cdd_parsed_ir_none(mocker: pytest_mock.MockerFixture) -> None:
    """Test GhostInspector when cdd_parsed_ir has no params attribute.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.class_.parse

    mocker.patch.object(cdd.class_.parse, "class_", return_value={})

    ref = GhostInspector.inspect(SimpleClassModel, "SimpleClassModel")
    assert ref.has_arg("x")


def test_ghost_inspector_cdd_ast_params_edges(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector cdd_ast_params handling for self, AST defaults, and hints.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.function.parse

    ast_default = ast.Constant(value=42)

    class BrokenAST(ast.AST):
        """Broken AST node."""

        pass

    broken_default = BrokenAST()

    fake_cdd_ir = {
        "params": {
            "self": {"typ": "Any"},
            "ast_arg": {"default": ast_default, "typ": "int"},
            "broken_arg": {"default": broken_default, "typ": "str"},
            "hint_arg": {"default": None},
        }
    }

    orig_unparse = ast.unparse

    def mock_unparse(node: ast.AST) -> str:
        """Mock unparse raising on BrokenAST.

        Args:
            node: AST node.

        Raises:
            ValueError: Simulated unparse error on BrokenAST.

        Returns:
            Unparsed string.
        """
        if isinstance(node, BrokenAST):
            raise ValueError("unparse failed")
        return orig_unparse(node)

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.ast.unparse",
        side_effect=mock_unparse,
    )
    mocker.patch.object(cdd.function.parse, "function", return_value=fake_cdd_ir)

    ref = GhostInspector.inspect(target_fn_ast_edges, "target_fn_ast_edges")
    assert not ref.has_arg("self")
    assert ref.has_arg("ast_arg")
    assert ref.has_arg("broken_arg")
    assert ref.has_arg("hint_arg")
    ast_param = next(p for p in ref.params if p.name == "ast_arg")
    assert ast_param.default == "42"
    hint_param = next(p for p in ref.params if p.name == "hint_arg")
    assert hint_param.annotation == "int"


def test_ghost_inspector_standard_fallback_exception_in_cdd_branch(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector handles exception in inspect.signature within CDD branch.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    import cdd.function.parse

    fake_cdd_ir = {
        "params": {
            "x": {"typ": "int"},
        }
    }
    mocker.patch.object(cdd.function.parse, "function", return_value=fake_cdd_ir)
    mocker.patch("inspect.signature", side_effect=TypeError("broken signature"))

    ref = GhostInspector.inspect(target_fn_sig_failure, "target_fn_sig_failure")
    assert ref.has_arg("x")


def test_ghost_inspector_parent_mro_signature_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector handles exception when parent signature inspection fails during super kwargs resolution.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    from ml_ecosystem_snapshots.introspection.runtime_boundary import (
        get_member_dict,
        get_runtime_signature,
    )

    def mock_signature(
        obj: LivePythonTarget,
        *args: LivePythonTarget,
        **kwargs: LivePythonTarget,
    ) -> inspect.Signature:
        """Mock signature raising on MroParentClass.__init__.

        Args:
            obj: Target object.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: Simulated signature failure on parent __init__.

        Returns:
            Signature object.
        """
        if obj == get_member_dict(MroParentClass).get("__init__"):
            raise TypeError("parent init signature failure")
        return get_runtime_signature(obj)

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.get_runtime_source",
        return_value=(
            "class MroChildClass(MroParentClass):\n    def __init__(self, **kwargs: int) -> None:\n        super().__init__(**kwargs)\n"
        ),
    )
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.get_runtime_signature",
        side_effect=mock_signature,
    )

    ref = GhostInspector.inspect(MroChildClass, "MroChildClass")
    assert ref.api_path == "MroChildClass"


def test_ghost_inspector_torch_target_infer_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector handles infer_torch_dtype_and_rank exceptions.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(
        torch_fw,
        "infer_torch_dtype_and_rank",
        side_effect=ValueError("simulated infer error"),
    )

    def dummy_torch_fn(x: int) -> None:
        """Dummy function.

        Args:
            x: Input.
        """
        pass

    ref = GhostInspector.inspect(dummy_torch_fn, "torch.dummy_torch_fn")
    assert ref.has_arg("x")


def test_ghost_inspector_overload_without_parameters() -> None:
    """Test GhostInspector overload branch when overload object has no parameters attribute.

    Returns:
        None.
    """
    import griffe

    bare_ov = griffe.Function("bare_overload_fn", parameters=griffe.Parameters(), returns="int")
    fn_node = griffe.Function("bare_overload_fn", parameters=griffe.Parameters())
    fn_node.overloads = [bare_ov]

    ref = GhostInspector.inspect(fn_node, "bare_overload_fn")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    assert ov0.returns_type == "int"


def test_ghost_inspector_c_ext_overload_infer_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test GhostInspector c_ext overloads when infer_torch_dtype_and_rank raises.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    from ml_ecosystem_snapshots.c_extension_sig import (
        CExtensionSignature,
        ParsedParamTuple,
    )
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(
        torch_fw,
        "infer_torch_dtype_and_rank",
        side_effect=RuntimeError("simulated c_ext infer error"),
    )

    param = ParsedParamTuple(
        name="x",
        kind="POSITIONAL_OR_KEYWORD",
        default=None,
        annotation="Tensor",
    )
    fake_sig = CExtensionSignature(
        params=[param],
        returns_type="Tensor",
        overloads=[
            CExtensionSignature(params=[param], returns_type="Tensor"),
        ],
    )

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_c_extension_signature",
        return_value=fake_sig,
    )
    mocker.patch("inspect.signature", side_effect=TypeError("no signature"))

    def c_ext_dummy(x: LivePythonTarget) -> None:
        """Dummy C extension function.

        Args:
            x: Tensor.
        """
        pass

    ref = GhostInspector.inspect(c_ext_dummy, "torch.c_ext_dummy")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    assert ov0.params[0].name == "x"
