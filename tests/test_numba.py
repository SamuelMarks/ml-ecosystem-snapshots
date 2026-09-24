"""Tests for Numba JIT and array primitives snapshot extractor."""

from typing import Any
import pytest

from ml_framework_snapshots.frameworks.numba import (
    CANONICAL_NUMBA_OPS,
    _get_numba,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_numba_live() -> None:
    """Verify collect_api introspects live Numba functions when numba is available."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "njit" in names or "jit" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_numba_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api uses CANONICAL_NUMBA_OPS fallback when numba is not installed."""
    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.numba._get_numba", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_NUMBA_OPS)
    names = {ref.name for ref in refs}
    assert "njit" in names
    assert "vectorize" in names
    assert "prange" in names


def test_get_numba_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_numba handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "numba" in name:
            raise ImportError("Simulated numba absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_numba() is None


def test_collect_api_numba_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakyNumba:
        njit = "broken"

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.numba._get_numba", lambda: FlakyNumba()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.numba.GhostInspector.inspect", mock_inspect
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_numba_roundtrip_serialization() -> None:
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
