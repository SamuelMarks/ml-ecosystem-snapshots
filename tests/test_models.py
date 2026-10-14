"""Unit tests and verification suite for models."""

from typing import Any, Dict, List


from ml_ecosystem_snapshots.models import (
    ExtendedGhostParam,
    GhostInspector,
    GhostIsaRef,
)
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind


def test_ghost_param() -> None:
    """Verify ghost param behavior."""
    param = GhostParam(
        name="x",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        default="1",
        annotation="int",
    )
    assert param.name == "x"
    assert param.kind == ParameterKind.POSITIONAL_OR_KEYWORD
    assert param.default == "1"
    assert param.annotation == "int"


def test_ghost_ref_has_arg() -> None:
    """Verify ghost ref has arg behavior."""
    params = [GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)]
    ref = GhostRef(
        name="foo",
        api_path="pkg.foo",
        kind="function",
        params=params,
        has_varargs=False,
    )
    assert ref.has_arg("x") is True
    assert ref.has_arg("y") is False


def dummy_func(a: int, b: int = 2, *args: Any, **kwargs: Any) -> None:
    """Docstring for dummy_func. a: int, b: int, args: tuple, kwargs: dict.

    # noqa: DAR101
    """
    pass


class DummyClass:
    """Mock or helper class DummyClass for unit testing."""

    def __init__(self, c: str = "test") -> None:
        """Execute helper init  . c.

        # noqa: DAR101

        Args:
            c: Third parameter value.
        """
        pass


def dummy_c_extension(*args: Any, **kwargs: Any) -> None:
    """Execute helper dummy c extension without docstring args for C-ext fallback.

    *args: Variadic positional arguments.
    **kwargs: Variadic keyword arguments.
    """
    # simulate something that inspect.signature fails on
    pass


def test_ghost_inspector_function() -> None:
    """Verify ghost inspector function behavior."""
    ref = GhostInspector.inspect(dummy_func, "tests.dummy_func")
    assert ref.name == "dummy_func"
    assert ref.api_path == "tests.dummy_func"
    assert ref.kind == "function"
    assert ref.docstring is not None
    assert "Docstring for dummy_func." in ref.docstring
    assert ref.has_varargs is True

    assert (
        len(ref.params) == 4
    )  # a, b (args/kwargs aren't counted explicitly as params here? Wait, *args is VAR_POSITIONAL)
    assert ref.has_arg("a")
    assert ref.has_arg("b")


def test_ghost_inspector_class() -> None:
    """Verify ghost inspector class behavior."""
    ref = GhostInspector.inspect(DummyClass, "tests.DummyClass")
    assert ref.name == "DummyClass"
    assert ref.kind == "class"
    assert ref.has_arg("c")
    assert ref.params[0].default == "'test'"


def test_ghost_inspector_c_extension(mocker: Any) -> None:
    """Verify ghost inspector c extension behavior.

    Args:
        mocker: Input parameter value.
    """
    # Mock inspect.signature to raise ValueError
    mocker.patch("inspect.signature", side_effect=ValueError)
    ref = GhostInspector.inspect(dummy_c_extension, "tests.dummy_c_extension")
    assert ref.name == "dummy_c_extension"
    assert ref.kind == "function"
    assert ref.has_varargs is True
    assert ref.has_arg("args")
    assert ref.has_arg("kwargs")


def test_ghost_inspector_hydrate() -> None:
    """Verify ghost inspector hydrate behavior."""
    data: dict[str, Any] = {
        "name": "foo",
        "api_path": "foo",
        "kind": "function",
        "params": [],
        "docstring": None,
        "has_varargs": False,
    }
    ref = GhostInspector.hydrate(data)
    assert ref.name == "foo"

    # Test hydrate without explicit kind to verify default function kind assignment
    data_no_kind: dict[str, Any] = {
        "name": "bar",
        "api_path": "bar",
        "params": [],
    }
    ref_no_kind = GhostInspector.hydrate(data_no_kind)
    assert ref_no_kind.name == "bar"
    assert ref_no_kind.kind == "function"


def test_unrepresentable_default() -> None:
    """Verify unrepresentable default behavior."""

    class Unrepresentable:
        """Mock or helper class Unrepresentable for unit testing."""

        def __repr__(self) -> Any:
            """Execute helper repr  .

            Raises:
                Exception: Exception.
            """
            raise Exception("no")

    def f(a: Any = Unrepresentable()) -> None:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_memory_address_default() -> None:
    """Verify memory address default behavior."""

    class MemoryDefault:
        """Mock or helper class MemoryDefault for unit testing."""

        pass

    def f(a: Any = MemoryDefault()) -> None:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_callable_default() -> None:
    """Verify callable default behavior."""

    def my_default() -> Any:
        """Execute helper my default."""
        pass

    def f(a: Any = my_default) -> None:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_ghost_inspector_str_throws() -> None:
    """Verify ghost inspector str throws behavior."""

    class StrThrows:
        """Mock or helper class StrThrows for unit testing."""

        def __repr__(self) -> Any:
            """Execute helper repr  .

            Returns:
                Computed result.
            """
            return "fine"

        def __str__(self) -> Any:
            """Execute helper str  .

            Raises:
                Exception: Exception.
            """
            raise Exception("no")

    def f(a: Any = StrThrows()) -> None:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_ghost_inspector_c_extension_class(mocker: Any) -> None:
    """Verify ghost inspector c extension class behavior.

    Args:
        mocker: Input parameter value.
    """
    mocker.patch("inspect.signature", side_effect=ValueError)

    class DummyCClass:
        """Mock or helper class DummyCClass for unit testing."""

        pass

    ref = GhostInspector.inspect(DummyCClass, "tests.DummyCClass")
    assert ref.name == "DummyCClass"
    assert ref.kind == "class"
    assert not ref.has_varargs
    assert len(ref.params) == 0


def test_ghost_inspector_str_has_address() -> None:
    """Verify ghost inspector str has address behavior."""

    class StrAddr:
        """Mock or helper class StrAddr for unit testing."""

        def __repr__(self) -> Any:
            """Execute helper repr  .

            Returns:
                Computed result.
            """
            return "fine"

        def __str__(self) -> Any:
            """Execute helper str  .

            Returns:
                Computed result.
            """
            return "something at 0x1234"

    def f(a: Any = StrAddr()) -> None:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_ghost_inspector_cdd_raises_no_typ(mocker: Any) -> None:
    """Verify ghost inspector cdd raises no typ behavior.

    Args:
        mocker: Input parameter value.
    """
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"doc": "some error but no typ"}]},
    )

    def f() -> Any:
        """Doc."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.raises == []


def test_ghost_inspector_is_public(mocker: Any) -> None:
    """Verify ghost inspector is public behavior.

    Args:
        mocker: Input parameter value.
    """
    # test explicit override
    ref = GhostInspector.inspect(dummy_func, "tests.dummy_func", is_public=False)
    assert ref.is_public is False

    # test griffe is_public
    class FakeGriffeNode:
        """Mock or helper class FakeGriffeNode for unit testing."""

        is_public = False

    mocker.patch("griffe.load", return_value=FakeGriffeNode())
    ref2 = GhostInspector.inspect(dummy_func, "tests.dummy_func")
    assert ref2.is_public is False

    # test griffe is_public missing attribute
    class FakeGriffeNodeNoPub:
        """Mock or helper class FakeGriffeNodeNoPub for unit testing."""

        pass

    mocker.patch("griffe.load", return_value=FakeGriffeNodeNoPub())

    def _private_dummy() -> Any:
        """Execute helper private dummy."""
        pass

    ref3 = GhostInspector.inspect(_private_dummy, "tests._private_dummy")
    assert ref3.is_public is False


def test_ghost_inspector_annotation_str() -> None:
    """Verify ghost inspector annotation str behavior."""

    def f(a: str) -> Any:
        """Execute helper f. a.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_fallback() -> None:
    """Verify ghost inspector cdd fallback behavior."""

    # Test when p_anno is missing but cdd has it
    def dummy_func_cdd(x: Any) -> None:
        """Dummy func.

        # noqa: DAR101

        Args:
        x (int): The value.

            x: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_cdd, "dummy_func_cdd")
    assert ref.params[0].annotation == "int"


def test_ghost_inspector_annotation_name() -> None:
    """Verify ghost inspector annotation name behavior."""

    class MyType:
        """Mock or helper class MyType for unit testing."""

        pass

    def dummy_func_anno(x: MyType) -> Any:
        """Execute helper dummy func anno. x.

        Args:
            x: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    assert ref.params[0].annotation is not None
    assert "MyType" in ref.params[0].annotation


def test_ghost_inspector_annotation_name_mock(mocker: Any) -> None:
    """Verify ghost inspector annotation name mock behavior.

    Args:
        mocker: Input parameter value.
    """
    # Mock get_type_hints to raise an Exception so it falls back to raw annotations
    mocker.patch("typing.get_type_hints", side_effect=Exception("Failed"))

    class MyType:
        """Mock or helper class MyType for unit testing."""

        pass

    def dummy_func_anno(x: MyType) -> Any:
        """Execute helper dummy func anno. x.

        Args:
            x: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    assert ref.params[0].annotation == "MyType"


def test_ghost_inspector_annotation_str_mock(mocker: Any) -> None:
    """Verify ghost inspector annotation str mock behavior.

    Args:
        mocker: Input parameter value.
    """
    mocker.patch("typing.get_type_hints", side_effect=Exception("Failed"))

    def dummy_func_anno_str(x: "str") -> Any:
        """Execute helper dummy func anno str. x.

        Args:
            x: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_anno_str, "dummy_func_anno_str")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_anno_fallback() -> None:
    """Verify ghost inspector cdd anno fallback behavior."""

    # cdd_params gives 'typ' but inspect does not find it
    def dummy_func_no_anno(x: Any) -> None:
        """Dummy func.

        # noqa: DAR101

        Args:
        x (int): The value.

            x: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_no_anno, "dummy_func_no_anno")
    assert ref.params[0].annotation == "int"
    assert ref.params[0].description == "The value."


def test_ghost_inspector_return_type_hints() -> None:
    """Verify ghost inspector return type hints behavior."""

    def dummy_func_ret() -> int:
        """Execute helper dummy func ret.

        Returns:
            Computed result.
        """
        return 0

    ref = GhostInspector.inspect(dummy_func_ret, "dummy_func_ret")
    assert ref.returns_type == "int"


def test_ghost_inspector_griffe_params(mocker: Any) -> None:
    """Verify ghost inspector griffe params behavior.

    Args:
        mocker: Input parameter value.
    """

    class FakeKind:
        """Mock or helper class FakeKind for unit testing."""

        def __init__(self, name: Any) -> None:
            """Execute helper init  . name.

            Args:
                name: Input parameter value.
            """
            self.name = name

    class FakeParam:
        """Mock or helper class FakeParam for unit testing."""

        def __init__(
            self, name: Any, kind_name: Any, default: Any, annotation: Any
        ) -> None:
            """Execute helper init  . name kind_name default annotation.

            Args:
                annotation: Input parameter value.
                default: Input parameter value.
                kind_name: Input parameter value.
                name: Input parameter value.
            """
            self.name = name
            self.kind = FakeKind(kind_name) if kind_name else None
            self.default = default
            self.annotation = annotation

    class FakeGriffeNode:
        """Mock or helper class FakeGriffeNode for unit testing."""

        is_public = True
        parameters = [
            FakeParam("self", None, None, None),
            FakeParam("x", "VAR_POSITIONAL", None, "int"),
            FakeParam("y", "POSITIONAL_OR_KEYWORD", "10", None),
        ]

    mocker.patch("griffe.load", return_value=FakeGriffeNode())

    def dummy_func_griffe() -> Any:
        """Execute helper dummy func griffe."""
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert ref.has_varargs is True
    assert ref.params[0].name == "x"
    assert ref.params[0].annotation == "int"
    assert ref.params[1].name == "y"
    assert ref.params[1].default == "10"


def test_ghost_inspector_overloads_cdd() -> None:
    """Verify GhostInspector handles overload inspection when griffe is unavailable."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"griffe": None}):

        def dummy_func_no_griffe() -> None:
            """Execute dummy function with no Griffe backing."""
            pass

        ref = GhostInspector.inspect(dummy_func_no_griffe, "dummy_func_no_griffe")
        assert ref.overloads is None or len(ref.overloads) == 0


def test_ghost_inspector_griffe_overloads(mocker: Any) -> None:
    """Verify ghost inspector griffe overloads behavior.

    Args:
        mocker: Input parameter value.
    """

    class FakeParam:
        """Mock or helper class FakeParam for unit testing."""

        def __init__(
            self, name: Any, kind_name: Any, default: Any, annotation: Any
        ) -> None:
            """Execute helper init  . name kind_name default annotation.

            Args:
                annotation: Input parameter value.
                default: Input parameter value.
                kind_name: Input parameter value.
                name: Input parameter value.
            """
            self.name = name

            class FakeKind:
                """Mock or helper class FakeKind for unit testing."""

                pass

            self.kind = FakeKind()
            setattr(
                self.kind,
                "name",
                kind_name if kind_name else "POSITIONAL_OR_KEYWORD",
            )
            self.default = default
            self.annotation = annotation

    class FakeOverload:
        """Mock or helper class FakeOverload for unit testing."""

        def __init__(self, parameters: Any, returns: Any) -> None:
            """Execute helper init  . parameters returns.

            Args:
                parameters: Input parameter value.
                returns: Input parameter value.
            """
            self.parameters = parameters
            self.returns = returns

    class FakeGriffeNode:
        """Mock or helper class FakeGriffeNode for unit testing."""

        is_public = True
        parameters: List[Any] = []
        overloads = [
            FakeOverload(
                [
                    FakeParam("self", None, None, None),
                    FakeParam("a", "VAR_POSITIONAL", "1", "int"),
                ],
                "int",
            )
        ]

    mocker.patch("griffe.load", return_value=FakeGriffeNode())

    def dummy_func_griffe() -> Any:
        """Execute helper dummy func griffe."""
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    assert ref.overloads[0].has_varargs is True
    assert ref.overloads[0].params[0].name == "a"
    assert ref.overloads[0].params[0].annotation == "int"
    assert ref.overloads[0].params[0].default == "1"
    assert ref.overloads[0].returns_type == "int"


def test_models_branches() -> None:
    """Verify models branches behavior."""
    from ml_ecosystem_snapshots.models import sanitize_type_str

    assert sanitize_type_str("typing.List") == "list"
    assert sanitize_type_str("builtins.str") == "builtins.str"


def test_preload_griffe_cache() -> None:
    """Test preloading griffe cache for specified and default frameworks."""
    from ml_ecosystem_snapshots.models import preload_griffe_cache, _GRIFFE_CACHE

    preload_griffe_cache(["ml_ecosystem_snapshots"])
    assert "ml_ecosystem_snapshots" in _GRIFFE_CACHE

    # Call again to hit the already cached branch
    preload_griffe_cache(["ml_ecosystem_snapshots"])

    preload_griffe_cache(["non_existent_framework_xyz"])


def test_inspect_griffe_node_direct() -> None:
    """Test inspecting a Griffe node directly without live object."""
    import griffe
    from ml_ecosystem_snapshots.models import GhostInspector

    mod = griffe.load("ml_ecosystem_snapshots.models")
    ref = GhostInspector.inspect(
        mod["GhostInspector"], "ml_ecosystem_snapshots.models.GhostInspector"
    )
    assert ref.name == "GhostInspector"
    assert ref.kind == "class"


def test_sanitize_type_str_empty_after_strip() -> None:
    """Test sanitize_type_str when strip_sphinx_roles produces an empty string."""
    from ml_ecosystem_snapshots.models import sanitize_type_str

    assert sanitize_type_str(":class:``") == ""


def test_ghost_inspector_griffe_class_init(mocker: Any) -> None:
    """Test griffe class inspection resolving parameters from __init__.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_param = mocker.MagicMock()
    mock_param.name = "val"
    mock_param.kind.name = "POSITIONAL_OR_KEYWORD"
    mock_param.default = None
    mock_param.annotation = "int"
    mock_param.description = "The value."

    mock_init = mocker.MagicMock()
    mock_init.parameters = [mock_param]

    mock_node = mocker.MagicMock(
        spec=[
            "name",
            "is_class",
            "is_function",
            "parameters",
            "members",
            "docstring",
            "is_public",
            "overloads",
            "returns",
        ]
    )
    mock_node.name = "DummyClass"
    mock_node.docstring = None
    mock_node.parameters = None
    mock_node.is_class = True
    mock_node.is_function = False
    mock_node.members = {"__init__": mock_init}
    mock_node.is_public = True
    mock_node.overloads = None
    mock_node.returns = "int"

    ref = GhostInspector.inspect(mock_node, "DummyClass")
    assert any(p.name == "val" for p in ref.params)
    assert ref.returns_type == "int"

    # Cover empty griffe_params (branch 531->533)
    mock_node_empty = mocker.MagicMock(
        spec=[
            "name",
            "is_class",
            "is_function",
            "parameters",
            "members",
            "docstring",
            "is_public",
            "overloads",
            "returns",
        ]
    )
    mock_node_empty.name = "EmptyClass"
    mock_node_empty.docstring = None
    mock_node_empty.parameters = []
    mock_node_empty.is_class = True
    mock_node_empty.is_function = False
    mock_node_empty.members = {}
    mock_node_empty.is_public = True
    mock_node_empty.overloads = None
    mock_node_empty.returns = None

    ref_empty = GhostInspector.inspect(mock_node_empty, "EmptyClass")
    assert ref_empty.name == "EmptyClass"


def test_extended_ghost_param_allowed_values() -> None:
    """Test ExtendedGhostParam instantiation with allowed_values."""
    param = ExtendedGhostParam(
        name="reduction",
        kind=ParameterKind.KEYWORD_ONLY,
        default="mean",
        allowed_values=["none", "mean", "sum"],
    )
    assert param.allowed_values == ["none", "mean", "sum"]


def test_ghost_isa_ref_hydrate_overloads() -> None:
    """Test GhostInspector.hydrate creates overloads for multi-signature instructions."""
    raw_data = {
        "mnemonic": "AL2P",
        "operands": [["R", "R", "I"], ["R", "I"]],
        "architecture": ["sm_80", "sm_90"],
    }
    hydrated = GhostInspector.hydrate(raw_data)
    assert isinstance(hydrated, GhostIsaRef)
    assert hydrated.overloads is not None
    assert len(hydrated.overloads) == 2
    assert len(hydrated.overloads[0].params) == 3
    assert len(hydrated.overloads[1].params) == 2


def test_ghost_inspector_overloads_with_literal_and_standard_enum(
    mocker: Any,
) -> None:
    """Test extracting allowed_values from Literal and standard enums in overloads.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_param_lit = mocker.MagicMock()
    mock_param_lit.name = "mode"
    mock_param_lit.kind.name = "KEYWORD_ONLY"
    mock_param_lit.default = None
    mock_param_lit.annotation = 'Literal["fast", "slow"]'

    mock_param_enum = mocker.MagicMock()
    mock_param_enum.name = "reduction"
    mock_param_enum.kind.name = "KEYWORD_ONLY"
    mock_param_enum.default = None
    mock_param_enum.annotation = "str"

    mock_overload = mocker.MagicMock(spec=["parameters", "returns"])
    mock_overload.parameters = [mock_param_lit, mock_param_enum]
    mock_overload.returns = "int"

    mock_node = mocker.MagicMock(
        spec=[
            "name",
            "is_class",
            "is_function",
            "parameters",
            "docstring",
            "is_public",
            "overloads",
            "returns",
        ]
    )
    mock_node.name = "func_with_ov"
    mock_node.is_class = False
    mock_node.is_function = True
    mock_node.parameters = []
    mock_node.docstring = None
    mock_node.is_public = True
    mock_node.overloads = [mock_overload]
    mock_node.returns = "int"

    ref = GhostInspector.inspect(mock_node, "tests.func_with_ov")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov_params = ref.overloads[0].params
    assert ov_params[0].allowed_values == ["fast", "slow"]
    assert ov_params[1].allowed_values == ["none", "mean", "sum"]


def test_ghost_inspector_c_ext_overloads_with_literal_and_standard_enum(
    mocker: Any,
) -> None:
    """Test extracting allowed_values in C-extension overloads.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("inspect.signature", side_effect=ValueError)

    class MockCExtSig(list[Any]):
        """Mock C-extension signature list with overloads."""

        overloads: Any = None
        returns_type: Any = "int"

    class MockOverloadList(list[Any]):
        """Mock overload list with returns_type."""

        returns_type: Any = "int"

    mock_c_sig = MockCExtSig([("x", "POSITIONAL_OR_KEYWORD", None, "int")])
    mock_ov = MockOverloadList(
        [
            ("mode", "KEYWORD_ONLY", None, 'Literal["fast", "slow"]'),
            ("reduction", "KEYWORD_ONLY", None, "str"),
        ]
    )
    mock_c_sig.overloads = [mock_ov]

    mocker.patch(
        "ml_ecosystem_snapshots.models.extract_c_extension_signature",
        return_value=mock_c_sig,
    )

    def dummy_fn(*args: Any, **kwargs: Any) -> Any:
        """Dummy C function.

        Args:
            args: Variadic positional arguments or CLI namespace.
            kwargs: Variadic keyword arguments.
        """
        return None

    ref = GhostInspector.inspect(dummy_fn, "pkg.dummy_fn")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov_params = ref.overloads[0].params
    assert ov_params[0].allowed_values == ["fast", "slow"]
    assert ov_params[1].allowed_values == ["none", "mean", "sum"]


def test_ghost_inspector_literal_param() -> None:
    """Test GhostInspector extracting allowed_values from Literal type annotation."""
    from typing import Literal

    def dummy_lit(mode: Literal["train", "eval"] = "train") -> None:
        """Dummy function with Literal parameter.

        Args:
            mode: Operating mode.
        """
        pass

    ref = GhostInspector.inspect(dummy_lit, "tests.dummy_lit")
    assert len(ref.params) == 1
    assert isinstance(ref.params[0], ExtendedGhostParam)
    assert ref.params[0].allowed_values == ["train", "eval"]


def test_extract_accepted_kwargs_from_ast() -> None:
    """Test extracting accepted kwargs via static AST analysis."""
    from ml_ecosystem_snapshots.models import extract_accepted_kwargs_from_ast

    def sample_kwargs_fn(**kwargs: Any) -> None:
        """Sample function querying kwargs in various ways.

        Args:
            kwargs: Variadic keyword arguments.
        """
        var_key = "test"
        _ = kwargs.get("alpha", 1.0)
        _ = kwargs.get(var_key)
        _ = kwargs.pop("beta", None)
        _ = kwargs["gamma"]
        other_dict: Dict[str, Any] = {}
        other_val = None
        if "delta" in kwargs:
            pass
        if "delta" not in kwargs:
            pass
        if other_val == kwargs:
            pass
        if "delta" in other_dict:
            pass
        if var_key in kwargs:
            pass

    extracted = extract_accepted_kwargs_from_ast(sample_kwargs_fn)
    assert extracted == ["alpha", "beta", "delta", "gamma"]

    # Test function without kwargs
    def no_kwargs_fn(a: int, b: int) -> None:
        """Function with positional arguments and no variable kwargs.

        Args:
            a: First argument.
            b: Second argument.
        """
        pass

    assert extract_accepted_kwargs_from_ast(no_kwargs_fn) is None

    # Test inspection populating accepted_kwargs
    ref = GhostInspector.inspect(sample_kwargs_fn, "tests.sample_kwargs_fn")
    assert getattr(ref, "accepted_kwargs", None) == ["alpha", "beta", "delta", "gamma"]


def test_ghost_instruction_and_operation_refs() -> None:
    """Test first-class GhostInstructionRef and GhostOperationRef schemas and hydration."""
    from ml_ecosystem_snapshots.models import (
        GhostInstructionRef,
        GhostOperationRef,
        GhostResult,
        ExtendedGhostParam,
        IRParameterRole,
    )

    # 1. GhostInstructionRef
    inst = GhostInstructionRef(
        name="FADD",
        api_path="FADD",
        kind="instruction",
        condition_codes=["CC.EQ", "CC.LT"],
        supported_architectures=["sm_80", "sm_90"],
    )
    assert inst.domain_type == "isa"
    assert inst.condition_codes == ["CC.EQ", "CC.LT"]
    assert inst.supported_architectures == ["sm_80", "sm_90"]

    # 2. GhostOperationRef
    op = GhostOperationRef(
        name="AddFOp",
        api_path="arith.addf",
        kind="operation",
        traits=["SameOperandsAndResultType", "Commutative"],
        operands=[
            ExtendedGhostParam(
                name="lhs",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="AnyFloat",
                role=IRParameterRole.OPERAND,
            ),
            ExtendedGhostParam(
                name="rhs",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="AnyFloat",
                role=IRParameterRole.OPERAND,
            ),
        ],
        returns=[GhostResult(name="result", type="AnyFloat")],
    )
    assert op.domain_type == "mlir"
    assert op.traits == ["SameOperandsAndResultType", "Commutative"]
    assert len(op.operands or []) == 2
    assert len(op.returns or []) == 1

    # 3. Hydrate with domain_type="instruction" and "operation"
    hydrated_inst = GhostInspector.hydrate(
        {
            "name": "v_add_f32",
            "api_path": "v_add_f32",
            "kind": "instruction",
            "domain_type": "instruction",
            "supported_architectures": ["GFX10+"],
        }
    )
    assert isinstance(hydrated_inst, GhostInstructionRef)

    hydrated_op = GhostInspector.hydrate(
        {
            "name": "DotGeneralOp",
            "api_path": "stablehlo.dot_general",
            "kind": "operation",
            "domain_type": "operation",
            "traits": [],
        }
    )
    assert isinstance(hydrated_op, GhostOperationRef)


def test_ghost_inspector_griffe_param_not_in_cdd(mocker: Any) -> None:
    """Test GhostInspector merging a Griffe parameter not found in cdd_params (line 765).

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """

    def sample_func(x: int, extra_param: int = 0) -> None:
        """Sample docstring for testing docstring parameter merge.

        Args:
            x: First input operand or value.
            extra_param: Input extra_param parameter.
        """
        pass

    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"params": {"x": {"doc": "x doc"}}},
    )
    mocker.patch(
        "ml_ecosystem_snapshots.models.extract_griffe_docstring_metadata",
        return_value={
            "params": {"extra_param": {"doc": "extra param doc", "typ": "int"}}
        },
    )

    ref = GhostInspector.inspect(sample_func, "sample_func")
    p = next((p for p in ref.params if p.name == "extra_param"), None)
    assert p is not None
    assert p.description == "extra param doc"
