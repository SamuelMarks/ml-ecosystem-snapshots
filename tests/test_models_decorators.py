"""Unit tests and verification suite for models decorators."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
)
from ml_ecosystem_snapshots.models import JsonValue


def dummy_decorator(
    func: InspectableTarget,
) -> InspectableTarget:
    """Execute helper dummy decorator.

    Args:
        func: Wrapped callable function.


    Returns:
        Computed result.
    """

    def wrapper(*args: JsonValue, **kwargs: JsonValue) -> JsonValue:
        """Execute helper wrapper.

        Args:
            args: Variadic positional arguments.
            kwargs: Variadic keyword arguments.


        Returns:
            Computed result.
        """
        return None

    wrapper.__dict__["__wrapped__"] = func
    return wrapper


def tf_function_mock(func: InspectableTarget) -> InspectableTarget:
    """Execute helper tf function mock.

    Args:
        func: Wrapped callable function.


    Returns:
        Computed result.
    """

    class TFFunction:
        """Mock or helper class TFFunction for unit testing."""

        def __init__(self, f: InspectableTarget) -> None:
            """Execute helper init  .

            Args:
                f: Target callable to decorate.
            """
            self._python_function = f

    return TFFunction(func)


def test_unwrap_standard_decorator() -> None:
    """Verify unwrap standard decorator behavior."""

    @dummy_decorator
    def my_func(a: int, b: str = "test") -> None:
        """My doc.

        Args:
            a: First positional operand.
            b: Second positional operand.
        """
        pass

    ref = GhostInspector.inspect(my_func, "my_func")
    assert ref.has_arg("a")
    assert ref.has_arg("b")
    assert ref.params[0].annotation == "int"


def test_unwrap_tf_decorator() -> None:
    """Verify unwrap tf decorator behavior."""

    @tf_function_mock
    def my_tf_func(tensor: JsonValue, training: bool = False) -> None:
        """Execute helper my tf func.

        Args:
            tensor: Input activation tensor.
            training: Whether layer is in training mode.
        """
        pass

    ref = GhostInspector.inspect(my_tf_func, "my_tf_func")
    assert ref.has_arg("tensor")
    assert ref.has_arg("training")


def test_unwrap_generic_decorator() -> None:
    """Verify unwrap generic decorator behavior."""

    def generic_dec(func: InspectableTarget) -> InspectableTarget:
        """Execute helper generic dec.

        Args:
            func: Wrapped callable function.


        Returns:
            Computed result.
        """

        class Wrapper:
            """Mock or helper class Wrapper for unit testing."""

            pass

        w = Wrapper()
        w.__dict__["__original_fn"] = func
        return w

    @generic_dec
    def fn1(a: JsonValue) -> None:
        """Execute helper fn1.

        Args:
            a: First positional operand.
        """
        pass

    ref = GhostInspector.inspect(fn1, "fn1")
    assert ref.has_arg("a")


def test_unwrap_variant_decorator() -> None:
    """Verify unwrap variant decorator behavior."""

    def variant_dec(func: InspectableTarget) -> InspectableTarget:
        """Execute helper variant dec.

        Args:
            func: Wrapped callable function.


        Returns:
            Computed result.
        """

        class Wrapper:
            """Mock or helper class Wrapper for unit testing."""

            pass

        w = Wrapper()
        w.__dict__["_original_fn"] = func
        return w

    @variant_dec
    def fn2(b: JsonValue) -> None:
        """Execute helper fn2.

        Args:
            b: Second positional operand.
        """
        pass

    ref = GhostInspector.inspect(fn2, "fn2")
    assert ref.has_arg("b")
