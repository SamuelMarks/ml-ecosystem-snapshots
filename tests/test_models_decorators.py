"""Unit tests and verification suite for models decorators."""

from typing import Any


from ml_ecosystem_snapshots.models import GhostInspector


def dummy_decorator(func: Any) -> Any:
    """Execute helper dummy decorator.

    Args:
        func: Input parameter value.


    Returns:
        Computed result.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Execute helper wrapper.

        Args:
            args: Input parameter value.
            kwargs: Input parameter value.


        Returns:
            Computed result.
        """
        return func(*args, **kwargs)

    setattr(wrapper, "__wrapped__", func)
    return wrapper


def tf_function_mock(func: Any) -> Any:
    """Execute helper tf function mock.

    Args:
        func: Input parameter value.


    Returns:
        Computed result.
    """

    class TFFunction:
        """Mock or helper class TFFunction for unit testing."""

        def __init__(self, f: Any) -> None:
            """Execute helper init  .

            Args:
                f: Input parameter value.
            """
            self._python_function = f

    return TFFunction(func)


def test_unwrap_standard_decorator() -> None:
    """Verify unwrap standard decorator behavior."""

    @dummy_decorator
    def my_func(a: int, b: str = "test") -> Any:
        """My doc.

        Args:
            a: Input parameter value.
            b: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(my_func, "my_func")
    assert ref.has_arg("a")
    assert ref.has_arg("b")
    assert ref.params[0].annotation == "int"


def test_unwrap_tf_decorator() -> None:
    """Verify unwrap tf decorator behavior."""

    @tf_function_mock
    def my_tf_func(tensor: Any, training: bool = False) -> None:
        """Execute helper my tf func.

        Args:
            tensor: Input parameter value.
            training: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(my_tf_func, "my_tf_func")
    assert ref.has_arg("tensor")
    assert ref.has_arg("training")


def test_unwrap_generic_decorator() -> None:
    """Verify unwrap generic decorator behavior."""

    def generic_dec(func: Any) -> Any:
        """Execute helper generic dec.

        Args:
            func: Input parameter value.


        Returns:
            Computed result.
        """

        class Wrapper:
            """Mock or helper class Wrapper for unit testing."""

            pass

        w = Wrapper()
        setattr(w, "__original_fn", func)
        return w

    @generic_dec
    def fn1(a: Any) -> Any:
        """Execute helper fn1.

        Args:
            a: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(fn1, "fn1")
    assert ref.has_arg("a")


def test_unwrap_variant_decorator() -> None:
    """Verify unwrap variant decorator behavior."""

    def variant_dec(func: Any) -> Any:
        """Execute helper variant dec.

        Args:
            func: Input parameter value.


        Returns:
            Computed result.
        """

        class Wrapper:
            """Mock or helper class Wrapper for unit testing."""

            pass

        w = Wrapper()
        setattr(w, "_original_fn", func)
        return w

    @variant_dec
    def fn2(b: Any) -> Any:
        """Execute helper fn2.

        Args:
            b: Input parameter value.
        """
        pass

    ref = GhostInspector.inspect(fn2, "fn2")
    assert ref.has_arg("b")
