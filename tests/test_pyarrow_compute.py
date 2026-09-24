"""Tests for Apache Arrow Compute kernel snapshot extractor."""

from typing import Any
import pytest

from ml_framework_snapshots.frameworks.pyarrow_compute import (
    CANONICAL_PYARROW_COMPUTE_OPS,
    _get_pyarrow_compute,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_pyarrow_compute_live() -> None:
    """Verify collect_api introspects live pyarrow.compute functions."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "add" in names
    assert "sum" in names or "mean" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_pyarrow_compute_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_PYARROW_COMPUTE_OPS fallback when pyarrow is absent."""
    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.pyarrow_compute._get_pyarrow_compute",
        lambda: None,
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_PYARROW_COMPUTE_OPS)
    names = {ref.name for ref in refs}
    assert "add" in names
    assert "subtract" in names
    assert "filter" in names


def test_get_pyarrow_compute_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_pyarrow_compute handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "pyarrow" in name:
            raise ImportError("Simulated pyarrow absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_pyarrow_compute() is None


def test_collect_api_pyarrow_compute_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakyPyArrow:
        add = "broken"

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.pyarrow_compute._get_pyarrow_compute",
        lambda: FlakyPyArrow(),
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.pyarrow_compute.GhostInspector.inspect",
        mock_inspect,
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_pyarrow_compute_roundtrip_serialization() -> None:
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
