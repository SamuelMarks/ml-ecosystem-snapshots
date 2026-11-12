"""Tests for Intel Data Parallel Extension for NumPy (DPNP) snapshot extractor."""

from __future__ import annotations

import types

import pytest
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks.dpnp import (
    CANONICAL_DPNP_OPS,
    collect_api,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def test_collect_api_dpnp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_DPNP_OPS fallback when dpnp is not installed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: None)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_DPNP_OPS)
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "empty" in names
    assert "matmul" in names


def test_collect_api_dpnp_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects members when dpnp is present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeDPNP:
        """Mock DPNP module."""

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
        def empty(shape: LivePythonTarget) -> LivePythonTarget:
            """Mock empty creation.

            Args:
                shape: Input shape parameter.

            Returns:
                The input shape object.
            """
            return shape

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: FakeDPNP())
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) >= 2
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "empty" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) >= 2

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_get_dpnp_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` resolves live `dpnp` module from `sys.modules` when present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys

    fake_mod = types.ModuleType("dpnp")
    monkeypatch.setitem(sys.modules, "dpnp", fake_mod)
    assert collect_api(SemanticTier.ARRAY_API) == []


def test_get_dpnp_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` falls back to canonical ops when `dpnp` import raises `ImportError`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys

    monkeypatch.setitem(sys.modules, "dpnp", None)
    assert len(collect_api(SemanticTier.ARRAY_API)) == len(CANONICAL_DPNP_OPS)


def test_collect_api_dpnp_inspection_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakyDPNP:
        """Mock DPNP namespace with broken attribute."""

        array = "broken"

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: FlakyDPNP())

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

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.dpnp.GhostInspector.inspect", mock_inspect)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_dpnp_roundtrip_serialization() -> None:
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
