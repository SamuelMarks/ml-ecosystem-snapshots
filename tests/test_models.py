"""Unit tests and verification suite for models."""

from __future__ import annotations

import typing

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import (
    ExtendedGhostParam,
    GhostIsaRef,
)


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


def dummy_func(a: int, b: int = 2, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
    """Docstring for dummy_func.

    Args:
        a: Primary integer input.
        b: Optional integer input with default value.
        *args: Variadic positional arguments.
        **kwargs: Variadic keyword arguments.
    """
    pass


class DummyClass:
    """Sample class fixture with a default constructor argument."""

    def __init__(self, c: str = "test") -> None:
        """Initialize DummyClass with an optional string value.

        Args:
            c: Constructor string parameter.
        """
        pass


def dummy_c_extension(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
    """Simulate a C-extension callable whose Python signature lookup fails.

    Args:
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


def test_ghost_inspector_c_extension(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector c extension behavior.

    Args:
        mocker: Pytest mocker fixture.
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
    import pytest

    from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
    from ml_ecosystem_snapshots.models import JsonValue

    data: dict[str, JsonValue] = {
        "name": "foo",
        "api_path": "foo",
        "kind": "function",
        "params": [],
        "docstring": None,
        "has_varargs": False,
    }
    ref = GhostInspector.hydrate(data)
    assert ref.name == "foo"
    assert ref.schema_version == "2.0.0"

    # Test hydrate rejects legacy schema_version and legacy param fields
    with pytest.raises(SnapshotSchemaError, match="Unsupported schema_version"):
        GhostInspector.hydrate(
            {
                "name": "bar",
                "api_path": "bar",
                "kind": "function",
                "schema_version": "1.2",
            }
        )

    with pytest.raises(SnapshotSchemaError, match="Legacy v1.x"):
        GhostInspector.hydrate(
            {
                "name": "bar",
                "api_path": "bar",
                "kind": "function",
                "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "dtypes": ["f32"]}],
            }
        )


def test_unrepresentable_default() -> None:
    """Verify unrepresentable default behavior."""

    class Unrepresentable:
        """Sentinel default value whose repr raises a generic Exception."""

        def __repr__(self) -> str:
            """Fail intentionally when formatting representation.

            Raises:
                RuntimeError: Always raised to simulate an unrepresentable object.

            Returns:
                Never returns.
            """
            raise RuntimeError("no")

    default_unrep = Unrepresentable()

    def f(a: Unrepresentable = default_unrep) -> None:
        """Accept an unrepresentable default value.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_memory_address_default() -> None:
    """Verify memory address default behavior."""

    class MemoryDefault:
        """Object whose default repr embeds a runtime memory address."""

        pass

    default_mem = MemoryDefault()

    def f(a: MemoryDefault = default_mem) -> None:
        """Accept an object default carrying a memory address repr.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_callable_default() -> None:
    """Verify callable default behavior."""

    def my_default() -> None:
        """Act as a no-op callable default factory."""
        pass

    def f(a: typing.Callable[[], None] = my_default) -> None:
        """Accept a callable default parameter.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_ghost_inspector_str_throws() -> None:
    """Verify ghost inspector str throws behavior."""

    class StrThrows:
        """Default value stub with valid repr but raising str conversion."""

        def __repr__(self) -> str:
            """Return a benign representation string.

            Returns:
                Constant representation string `'fine'`.
            """
            return "fine"

        def __str__(self) -> str:
            """Raise an error when converted to str.

            Raises:
                RuntimeError: Always raised during string conversion.

            Returns:
                Never returns.
            """
            raise RuntimeError("no")

    default_str_throws = StrThrows()

    def f(a: StrThrows = default_str_throws) -> None:
        """Accept a default object whose str conversion fails.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<unrepresentable>"


def test_ghost_inspector_c_extension_class(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector c extension class behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("inspect.signature", side_effect=ValueError)

    class DummyCClass:
        """Class stub simulating an uninspectable C-extension type."""

        pass

    ref = GhostInspector.inspect(DummyCClass, "tests.DummyCClass")
    assert ref.name == "DummyCClass"
    assert ref.kind == "class"
    assert not ref.has_varargs
    assert len(ref.params) == 0


def test_ghost_inspector_str_has_address() -> None:
    """Verify ghost inspector str has address behavior."""

    class StrAddr:
        """Default value stub whose str contains a hex memory address."""

        def __repr__(self) -> str:
            """Return a clean representation without a memory address.

            Returns:
                Constant string `'fine'`.
            """
            return "fine"

        def __str__(self) -> str:
            """Return a string containing a hexadecimal memory address.

            Returns:
                String containing `'0x1234'`.
            """
            return "something at 0x1234"

    default_str_addr = StrAddr()

    def f(a: StrAddr = default_str_addr) -> None:
        """Accept a default object whose str embeds a memory address.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].default == "<factory_default>"


def test_ghost_inspector_cdd_raises_no_typ(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector cdd raises no typ behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch(
        "cdd.docstring.parse.docstring",
        return_value={"raises": [{"doc": "some error but no typ"}]},
    )

    def f() -> None:
        """Doc."""
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.raises == []


def test_ghost_inspector_is_public(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector is public behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import griffe

    # test explicit override
    ref = GhostInspector.inspect(dummy_func, "tests.dummy_func", is_public=False)
    assert ref.is_public is False

    # test griffe is_public
    priv_mod = griffe.Module("tests")
    priv_fn = griffe.Function("dummy_func")
    priv_fn.public = False
    priv_mod.set_member("dummy_func", priv_fn)
    mocker.patch("griffe.load", return_value=priv_mod)
    ref2 = GhostInspector.inspect(dummy_func, "tests.dummy_func")
    assert ref2.is_public is False

    # test griffe node missing
    empty_mod = griffe.Module("tests")
    mocker.patch("griffe.load", return_value=empty_mod)

    def _private_dummy() -> None:
        """Represent a underscore-prefixed private helper function."""
        pass

    ref3 = GhostInspector.inspect(_private_dummy, "tests._private_dummy")
    assert ref3.is_public is False


def test_ghost_inspector_annotation_str() -> None:
    """Verify ghost inspector annotation str behavior."""

    def f(a: str) -> None:
        """Accept a string-annotated parameter.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(f, "f")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_fallback() -> None:
    """Verify ghost inspector cdd fallback behavior."""

    # Test when p_anno is missing but cdd has it
    def dummy_func_cdd(x: LivePythonTarget) -> None:
        """Dummy func.

        Args:
            x (int): The value.
        """
        pass

    pytest.MonkeyPatch().setattr(dummy_func_cdd, "__annotations__", {"return": None})
    ref = GhostInspector.inspect(dummy_func_cdd, "dummy_func_cdd")
    assert ref.params[0].annotation == "int"


def test_ghost_inspector_annotation_name() -> None:
    """Verify ghost inspector annotation name behavior."""

    class MyType:
        """Custom local parameter annotation type."""

        pass

    def dummy_func_anno(x: MyType) -> None:
        """Accept a parameter annotated with local class MyType.

        Args:
            x: Input tensor or scalar operand.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    assert ref.params[0].annotation is not None
    assert "MyType" in ref.params[0].annotation


def test_ghost_inspector_annotation_name_mock(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Verify ghost inspector annotation name mock behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Mock get_type_hints to raise an Exception so it falls back to raw annotations
    mocker.patch("typing.get_type_hints", side_effect=RuntimeError("Failed"))

    class MyType:
        """Custom local type for raw annotation fallback inspection."""

        pass

    def dummy_func_anno(x: MyType) -> None:
        """Accept a parameter resolved via raw runtime annotations.

        Args:
            x: Input tensor or scalar operand.
        """
        pass

    pytest.MonkeyPatch().setattr(dummy_func_anno, "__annotations__", {"x": MyType, "return": None})
    ref = GhostInspector.inspect(dummy_func_anno, "dummy_func_anno")
    assert ref.params[0].annotation == "MyType"


def test_ghost_inspector_annotation_str_mock(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector annotation str mock behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("typing.get_type_hints", side_effect=RuntimeError("Failed"))

    def dummy_func_anno_str(x: str) -> None:
        """Accept a parameter carrying a string raw annotation.

        Args:
            x: Input tensor or scalar operand.
        """
        pass

    pytest.MonkeyPatch().setattr(dummy_func_anno_str, "__annotations__", {"x": "str", "return": None})
    ref = GhostInspector.inspect(dummy_func_anno_str, "dummy_func_anno_str")
    assert ref.params[0].annotation == "str"


def test_ghost_inspector_cdd_anno_fallback() -> None:
    """Verify ghost inspector cdd anno fallback behavior."""

    # cdd_params gives 'typ' but inspect does not find it
    def dummy_func_no_anno(x: LivePythonTarget) -> None:
        """Dummy func.

        Args:
            x (int): The value.
        """
        pass

    pytest.MonkeyPatch().setattr(dummy_func_no_anno, "__annotations__", {"return": None})
    ref = GhostInspector.inspect(dummy_func_no_anno, "dummy_func_no_anno")
    assert ref.params[0].annotation == "int"
    assert ref.params[0].description == "The value."


def test_ghost_inspector_return_type_hints() -> None:
    """Verify ghost inspector return type hints behavior."""

    def dummy_func_ret() -> int:
        """Return zero with an explicit integer return annotation.

        Returns:
            Zero integer constant.
        """
        return 0

    ref = GhostInspector.inspect(dummy_func_ret, "dummy_func_ret")
    assert ref.returns_type == "int"


def test_ghost_inspector_griffe_params(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector griffe params behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import griffe

    real_fn_node = griffe.Function(
        "dummy_func_griffe",
        parameters=griffe.Parameters(
            griffe.Parameter("self", kind=None),
            griffe.Parameter("x", kind=griffe.ParameterKind.var_positional, annotation="int"),
            griffe.Parameter("y", kind=griffe.ParameterKind.positional_or_keyword, default="10"),
            griffe.Parameter("z", kind=griffe.ParameterKind.positional_or_keyword, annotation="str"),
        ),
    )

    mocker.patch("griffe.load", return_value=real_fn_node)

    def dummy_func_griffe(z: int) -> None:
        """Provide a live Python callable merged with Griffe parameter metadata.

        Args:
            z: Input parameter operand.
        """
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert ref.has_varargs is True
    assert ref.params[0].name == "x"
    assert ref.params[0].annotation == "int"
    assert ref.params[1].name == "y"
    assert ref.params[1].default == "10"
    assert ref.params[2].name == "z"
    assert ref.params[2].annotation == "int"


def test_ghost_inspector_overloads_cdd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GhostInspector handles overload inspection when griffe is unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys

    monkeypatch.setitem(sys.modules, "griffe", None)

    def dummy_func_no_griffe() -> None:
        """Execute dummy function with no Griffe backing."""
        pass

    ref = GhostInspector.inspect(dummy_func_no_griffe, "dummy_func_no_griffe")
    assert ref.overloads is None or len(ref.overloads) == 0


def test_ghost_inspector_griffe_overloads(mocker: pytest_mock.MockerFixture) -> None:
    """Verify ghost inspector griffe overloads behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import griffe

    ov_fn = griffe.Function(
        "dummy_func_griffe",
        parameters=griffe.Parameters(
            griffe.Parameter("self", kind=None),
            griffe.Parameter("a", kind=griffe.ParameterKind.var_positional, default="1", annotation="int"),
        ),
        returns="int",
    )
    base_fn = griffe.Function("dummy_func_griffe", parameters=griffe.Parameters())
    base_fn.overloads = [ov_fn]

    mocker.patch("griffe.load", return_value=base_fn)

    def dummy_func_griffe() -> None:
        """Provide a base callable backed by synthetic Griffe overloads."""
        pass

    ref = GhostInspector.inspect(dummy_func_griffe, "dummy_func_griffe")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    assert ov0.has_varargs is True
    assert ov0.params[0].name == "a"
    assert ov0.params[0].annotation == "int"
    assert ov0.params[0].default == "1"
    assert ov0.returns_type == "int"


def test_models_branches() -> None:
    """Verify models branches behavior."""
    from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str

    assert sanitize_type_str("typing.List") == "list"
    assert sanitize_type_str("builtins.str") == "builtins.str"


def test_preload_griffe_cache() -> None:
    """Test preloading griffe cache for specified and default frameworks."""
    from ml_ecosystem_snapshots.introspection.static_parsers import (
        DEFAULT_GRIFFE_CACHE,
        preload_griffe_cache,
    )

    preload_griffe_cache(["ml_ecosystem_snapshots"])
    assert DEFAULT_GRIFFE_CACHE.contains("ml_ecosystem_snapshots")

    # Call again to hit the already cached branch
    preload_griffe_cache(["ml_ecosystem_snapshots"])

    preload_griffe_cache(["non_existent_framework_xyz"])


def test_inspect_griffe_node_direct() -> None:
    """Test inspecting a Griffe node directly without live object."""
    import griffe

    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    mod = griffe.load("ml_ecosystem_snapshots.introspection.inspector")
    ref = GhostInspector.inspect(
        mod["GhostInspector"],
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector",
    )
    assert ref.name == "GhostInspector"
    assert ref.kind == "class"


def test_sanitize_type_str_empty_after_strip() -> None:
    """Test sanitize_type_str when strip_sphinx_roles produces an empty string."""
    from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str

    assert sanitize_type_str(":class:``") == ""


def test_ghost_inspector_griffe_class_init(mocker: pytest_mock.MockerFixture) -> None:
    """Test griffe class inspection resolving parameters from __init__.

    Args:
        mocker: Pytest mocker fixture.
    """
    import griffe

    del mocker
    cls_node = griffe.Class("DummyClass")
    init_fn = griffe.Function(
        "__init__",
        parameters=griffe.Parameters(
            griffe.Parameter("val", kind=griffe.ParameterKind.positional_or_keyword, annotation="int"),
        ),
        returns="int",
    )
    cls_node.set_member("__init__", init_fn)
    fn_with_ret = griffe.Function(
        "DummyFn",
        parameters=griffe.Parameters(
            griffe.Parameter("val", kind=griffe.ParameterKind.positional_or_keyword, annotation="int"),
        ),
        returns="int",
    )

    ref = GhostInspector.inspect(cls_node, "DummyClass")
    assert any(p.name == "val" for p in ref.params)
    ref_fn = GhostInspector.inspect(fn_with_ret, "DummyFn")
    assert ref_fn.returns_type == "int"

    cls_empty = griffe.Class("EmptyClass")
    ref_empty = GhostInspector.inspect(cls_empty, "EmptyClass")
    assert ref_empty.name == "EmptyClass"

    raising_alias = griffe.Alias("RaisingNode", target="nonexistent_pkg.NonExistentClass")
    ref_raising = GhostInspector.inspect(raising_alias, "RaisingNode")
    assert ref_raising.name == "RaisingNode"


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
    from ml_ecosystem_snapshots.models import JsonValue

    raw_data: dict[str, JsonValue] = {
        "mnemonic": "AL2P",
        "operands": [["R", "R", "I"], ["R", "I"]],
        "architecture": ["sm_80", "sm_90"],
    }
    hydrated = GhostInspector.hydrate(raw_data)
    assert isinstance(hydrated, GhostIsaRef)
    assert hydrated.overloads is not None
    assert len(hydrated.overloads) == 2
    ov0 = hydrated.overloads[0]
    ov1 = hydrated.overloads[1]
    assert isinstance(ov0, GhostRef)
    assert isinstance(ov1, GhostRef)
    assert len(ov0.params) == 3
    assert len(ov1.params) == 2


def test_ghost_inspector_overloads_with_literal_and_standard_enum(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test extracting allowed_values from Literal and standard enums in overloads.

    Args:
        mocker: Pytest mocker fixture.
    """
    import griffe

    del mocker
    overload_fn = griffe.Function(
        "func_with_ov",
        parameters=griffe.Parameters(
            griffe.Parameter("mode", kind=griffe.ParameterKind.keyword_only, annotation='Literal["fast", "slow"]'),
            griffe.Parameter("reduction", kind=griffe.ParameterKind.keyword_only, annotation="str"),
        ),
        returns="int",
    )
    fn_node = griffe.Function("func_with_ov", parameters=griffe.Parameters(), returns="int")
    fn_node.overloads = [overload_fn]

    ref = GhostInspector.inspect(fn_node, "tests.func_with_ov")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    ov_params = ov0.params
    p0 = ov_params[0]
    p1 = ov_params[1]
    assert isinstance(p0, ExtendedGhostParam)
    assert isinstance(p1, ExtendedGhostParam)
    assert p0.allowed_values == ["fast", "slow"]
    assert p1.allowed_values == ["none", "mean", "sum"]


def test_ghost_inspector_c_ext_overloads_with_literal_and_standard_enum(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test extracting allowed_values in C-extension overloads.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch("inspect.signature", side_effect=ValueError)

    from ml_ecosystem_snapshots.c_extension_sig import CExtensionSignature, ParsedParamTuple

    mock_ov = CExtensionSignature(
        params=[
            ParsedParamTuple(name="mode", kind="KEYWORD_ONLY", default=None, annotation='Literal["fast", "slow"]'),
            ParsedParamTuple(name="reduction", kind="KEYWORD_ONLY", default=None, annotation="str"),
        ],
        returns_type="int",
        overloads=[],
    )
    mock_c_sig = CExtensionSignature(
        params=[ParsedParamTuple(name="x", kind="POSITIONAL_OR_KEYWORD", default=None, annotation="int")],
        returns_type="int",
        overloads=[mock_ov],
    )

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.extract_c_extension_signature",
        return_value=mock_c_sig,
    )

    def dummy_fn(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
        """Dummy C function.

        Args:
            *args: Variadic positional arguments or CLI namespace.
            **kwargs: Variadic keyword arguments.
        """
        return None

    ref = GhostInspector.inspect(dummy_fn, "pkg.dummy_fn")
    assert ref.overloads is not None
    assert len(ref.overloads) == 1
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    ov_params = ov0.params
    p0 = ov_params[0]
    p1 = ov_params[1]
    assert isinstance(p0, ExtendedGhostParam)
    assert isinstance(p1, ExtendedGhostParam)
    assert p0.allowed_values == ["fast", "slow"]
    assert p1.allowed_values == ["none", "mean", "sum"]


def test_ghost_inspector_literal_param() -> None:
    """Test GhostInspector extracting allowed_values from Literal type annotation."""

    def dummy_lit(mode: typing.Literal["train", "eval"] = "train") -> None:
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
    from ml_ecosystem_snapshots.introspection.ast_kwargs import (
        extract_accepted_kwargs_from_ast,
    )

    def sample_kwargs_fn(**kwargs: LivePythonTarget) -> None:
        """Sample function querying kwargs in various ways.

        Args:
            **kwargs: Variadic keyword arguments.
        """
        var_key = "test"
        _ = kwargs.get("alpha", 1.0)
        _ = kwargs.get(var_key)
        _ = kwargs.pop("beta", None)
        _ = kwargs["gamma"]
        other_dict: dict[str, str] = {}
        other_val: dict[str, str] | None = None
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
    assert ref.accepted_kwargs == ["alpha", "beta", "delta", "gamma"]


def test_ghost_instruction_and_operation_refs() -> None:
    """Test first-class GhostInstructionRef and GhostOperationRef schemas and hydration."""
    from ml_ecosystem_snapshots.models import (
        ExtendedGhostParam,
        GhostInstructionRef,
        GhostOperationRef,
        GhostResult,
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


def test_ghost_inspector_griffe_param_not_in_cdd(
    mocker: pytest_mock.MockerFixture,
) -> None:
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
        "ml_ecosystem_snapshots.introspection.inspector.extract_griffe_docstring_metadata",
        return_value={"params": {"extra_param": {"doc": "extra param doc", "typ": "int"}}},
    )

    ref = GhostInspector.inspect(sample_func, "sample_func")
    p = next((p for p in ref.params if p.name == "extra_param"), None)
    assert p is not None
    assert p.description == "extra param doc"


def test_ensure_json_dict_and_list() -> None:
    """Verify `ensure_json_dict`, `ensure_json_list`, and `ensure_present` validate inputs and raise `SnapshotSchemaError`."""
    from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
    from ml_ecosystem_snapshots.models import ensure_json_dict, ensure_json_list, ensure_present

    assert ensure_json_dict({"k": 1}) == {"k": 1}
    assert ensure_json_list([1, "a"]) == [1, "a"]
    assert ensure_present("ok") == "ok"

    with pytest.raises(SnapshotSchemaError, match="dict"):
        ensure_json_dict("not_a_dict")

    with pytest.raises(SnapshotSchemaError, match="list"):
        ensure_json_list({"not": "a_list"})

    with pytest.raises(SnapshotSchemaError, match="non-None"):
        ensure_present(None)
