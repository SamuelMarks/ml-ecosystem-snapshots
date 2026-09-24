"""Tests for PyData Sparse format snapshot extractor."""

from typing import Any
import pytest

from ml_framework_snapshots.frameworks.sparse import (
    CANONICAL_SPARSE_OPS,
    _get_sparse,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_sparse_live() -> None:
    """Verify collect_api introspects live PyData Sparse classes and functions."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "COO" in names
    assert "dot" in names or "matmul" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_sparse_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api uses CANONICAL_SPARSE_OPS fallback when sparse is not installed."""
    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.sparse._get_sparse", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_SPARSE_OPS)
    names = {ref.name for ref in refs}
    assert "COO" in names
    assert "GCXS" in names
    assert "DOK" in names


def test_get_sparse_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_sparse handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "sparse" in name:
            raise ImportError("Simulated sparse absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_sparse() is None


def test_collect_api_sparse_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakySparse:
        COO = "broken"

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.sparse._get_sparse", lambda: FlakySparse()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.sparse.GhostInspector.inspect", mock_inspect
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_sparse_roundtrip_serialization() -> None:
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
