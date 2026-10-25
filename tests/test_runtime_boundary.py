"""Unit tests for the runtime reflection boundary (`introspection/runtime_boundary.py`)."""

from __future__ import annotations

import inspect

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    DescriptorProtocol,
    InspectableCallableProtocol,
    WrappedTargetProtocol,
    get_runtime_signature,
    get_runtime_source,
    has_runtime_attr,
    read_runtime_attr,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


class _SampleCallable:
    """Sample callable implementing `InspectableCallableProtocol`."""

    __doc__ = "Sample callable docstring."

    def __call__(self, *args: JsonValue, **kwargs: JsonValue) -> JsonValue:
        """Return first positional argument or None.

        Args:
            args: Variadic positional arguments.
            kwargs: Variadic keyword arguments.

        Returns:
            First argument or None.
        """
        return args[0] if args else None


class _SampleDescriptor:
    """Sample descriptor implementing `DescriptorProtocol`."""

    def __get__(
        self, instance: ScalarValue, owner: type | None = None
    ) -> InspectableCallableProtocol | ScalarValue:
        """Return a scalar value when accessed on class or instance.

        Args:
            instance: Target instance or None.
            owner: Owner class or None.

        Returns:
            Integer descriptor value.
        """
        return 42


class _RaisingAttrTarget:
    """Target whose property raises RuntimeError upon attribute access."""

    __doc__ = "Raising target."

    @property
    def boom(self) -> str:
        """Raise RuntimeError unconditionally.

        Raises:
            RuntimeError: Always raised to test safe boundary error handling.
        """
        raise RuntimeError("boom")


def _sample_fn(alpha: int, beta: str = "ok") -> str:
    """Sample function for runtime source and signature tests.

    Args:
        alpha: Integer input parameter.
        beta: String input parameter.

    Returns:
        Formatted string.
    """
    return f"{alpha}:{beta}"


def test_runtime_boundary_protocols_and_helpers() -> None:
    """Verify runtime checkable protocols and safe attribute/source/signature helpers."""
    callable_obj = _SampleCallable()
    assert isinstance(callable_obj, InspectableCallableProtocol)
    assert isinstance(callable_obj, WrappedTargetProtocol)
    assert callable_obj("hello") == "hello"

    desc_obj = _SampleDescriptor()
    assert isinstance(desc_obj, DescriptorProtocol)
    assert desc_obj.__get__(None, _SampleDescriptor) == 42

    assert has_runtime_attr(_sample_fn, "__name__") is True
    assert has_runtime_attr(_sample_fn, "nonexistent_attr_xyz") is False
    assert read_runtime_attr(_sample_fn, "__name__") == "_sample_fn"
    assert (
        read_runtime_attr(_sample_fn, "nonexistent_attr_xyz", "fallback") == "fallback"
    )

    raising_target = _RaisingAttrTarget()
    assert has_runtime_attr(raising_target, "boom") is False
    assert read_runtime_attr(raising_target, "boom", "safe_default") == "safe_default"

    src = get_runtime_source(_sample_fn)
    assert "def _sample_fn" in src

    sig = get_runtime_signature(_sample_fn)
    assert isinstance(sig, inspect.Signature)
    assert list(sig.parameters.keys()) == ["alpha", "beta"]
