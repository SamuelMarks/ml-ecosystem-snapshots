"""Unit tests for the runtime reflection boundary (`introspection/runtime_boundary.py`)."""

from __future__ import annotations

import inspect

import pytest

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    DescriptorProtocol,
    InspectableCallableProtocol,
    WrappedTargetProtocol,
    get_member_dict,
    get_runtime_signature,
    get_runtime_source,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


class _SampleCallable:
    """Sample callable implementing `InspectableCallableProtocol`."""

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

    def __get__(self, instance: ScalarValue, owner: type | None = None) -> InspectableCallableProtocol | ScalarValue:
        """Return a scalar value when accessed on class or instance.

        Args:
            instance: Target instance or None.
            owner: Owner class or None.

        Returns:
            Integer descriptor value.
        """
        return 42


class _RaisingAttrTarget:
    """Target whose `__dir__` raises RuntimeError upon member enumeration."""

    def __dir__(self) -> list[str]:
        """Raise RuntimeError unconditionally.

        Raises:
            RuntimeError: Always raised to test safe boundary error handling.

        Returns:
            Never returns.
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
    """Verify runtime checkable protocols and safe member/source/signature helpers."""
    callable_obj = _SampleCallable()
    assert isinstance(callable_obj, InspectableCallableProtocol)
    assert isinstance(callable_obj, WrappedTargetProtocol)
    assert callable_obj("hello") == "hello"

    desc_obj = _SampleDescriptor()
    assert isinstance(desc_obj, DescriptorProtocol)

    class _DescriptorHost:
        """Host class for exercising `_SampleDescriptor`."""

        value = desc_obj

    assert _DescriptorHost.value == 42

    fn_members = get_member_dict(_sample_fn)
    assert fn_members.get("__name__") == "_sample_fn"
    assert "nonexistent_attr_xyz" not in fn_members
    assert get_member_dict(None) == {}

    raising_target = _RaisingAttrTarget()
    assert get_member_dict(raising_target) == {}

    src = get_runtime_source(_sample_fn)
    assert "def _sample_fn" in src
    with pytest.raises(TypeError):
        get_runtime_source(None)

    sig = get_runtime_signature(_sample_fn)
    assert isinstance(sig, inspect.Signature)
    assert list(sig.parameters.keys()) == ["alpha", "beta"]
    with pytest.raises(TypeError):
        get_runtime_signature(None)
