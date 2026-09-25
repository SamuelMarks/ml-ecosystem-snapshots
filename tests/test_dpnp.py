"""Tests for Intel Data Parallel Extension for NumPy (DPNP) snapshot extractor."""

from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.dpnp import (
    CANONICAL_DPNP_OPS,
    _get_dpnp,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_dpnp_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_DPNP_OPS fallback when dpnp is not installed."""
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_DPNP_OPS)
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "empty" in names
    assert "matmul" in names


def test_collect_api_dpnp_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects members when dpnp is present."""

    class FakeDPNP:
        """Mock DPNP module."""

        @staticmethod
        def array(obj: Any) -> Any:
            """Mock array creation."""
            return obj

        @staticmethod
        def empty(shape: Any) -> Any:
            """Mock empty creation."""
            return shape

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: FakeDPNP()
    )
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
    """Verify _get_dpnp returns module when import succeeds."""
    import sys
    import types

    fake_mod = types.ModuleType("dpnp")
    monkeypatch.setitem(sys.modules, "dpnp", fake_mod)
    assert _get_dpnp() is fake_mod


def test_get_dpnp_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_dpnp handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "dpnp" in name:
            raise ImportError("Simulated dpnp absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_dpnp() is None


def test_collect_api_dpnp_inspection_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakyDPNP:
        array = "broken"

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.dpnp._get_dpnp", lambda: FlakyDPNP()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.dpnp.GhostInspector.inspect", mock_inspect
    )
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
