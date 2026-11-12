"""Tests for Bohrium heterogeneous runtime snapshot extractor."""

from __future__ import annotations

import sys
import types

import pytest
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks.bohrium import (
    CANONICAL_BOHRIUM_OPS,
    collect_api,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def test_collect_api_bohrium_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_BOHRIUM_OPS fallback when bohrium is absent.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.bohrium._get_bohrium", lambda: None)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_BOHRIUM_OPS)
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "empty" in names
    assert "flush" in names


def test_collect_api_bohrium_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects members when bohrium is present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeBohrium:
        """Mock Bohrium module."""

        @staticmethod
        def array(obj: LivePythonTarget) -> LivePythonTarget:
            """Mock array creation.

            Args:
                obj: Target object under inspection.

            Returns:
                The input target object.
            """
            return obj

        @staticmethod
        def flush() -> None:
            """Mock flush call."""
            pass

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.bohrium._get_bohrium", lambda: FakeBohrium())
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) >= 2
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "flush" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) >= 2

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_get_bohrium_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` resolves live `bohrium` module from `sys.modules` when present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("bohrium")
    monkeypatch.setitem(sys.modules, "bohrium", fake_mod)
    assert collect_api(SemanticTier.ARRAY_API) == []


def test_get_bohrium_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` falls back to canonical ops when `bohrium` import raises `ImportError`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "bohrium", None)
    assert len(collect_api(SemanticTier.ARRAY_API)) == len(CANONICAL_BOHRIUM_OPS)


def test_collect_api_bohrium_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakyBohrium:
        """Mock Bohrium namespace with broken attribute."""

        array = "broken"

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.bohrium._get_bohrium", lambda: FlakyBohrium())

    def mock_inspect(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A mocked GhostRef.

        Raises:
            RuntimeError: Always raised to simulate inspection failure.
        """
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.bohrium.GhostInspector.inspect", mock_inspect)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_bohrium_roundtrip_serialization() -> None:
    """Verify roundtrip serialization through GhostPythonRef and Pydantic models."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    for ref in refs:
        assert isinstance(ref, GhostPythonRef)
        serialized = ref.model_dump()
        reconstructed = GhostPythonRef.model_validate(serialized)
        assert reconstructed.name == ref.name
        assert reconstructed.api_path == ref.api_path
        assert len(reconstructed.params) == len(ref.params)
