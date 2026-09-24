"""Tests for Awkward Array ragged and nested layout snapshot extractor."""

from typing import Any
import pytest

from ml_framework_snapshots.frameworks.awkward import (
    CANONICAL_AWKWARD_OPS,
    _get_awkward,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_awkward_live() -> None:
    """Verify collect_api introspects live Awkward Array constructors and functions."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "Array" in names
    assert "Record" in names
    assert "flatten" in names or "to_numpy" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_awkward_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_AWKWARD_OPS fallback when awkward is not installed."""
    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.awkward._get_awkward", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_AWKWARD_OPS)
    names = {ref.name for ref in refs}
    assert "Array" in names
    assert "ArrayBuilder" in names
    assert "flatten" in names


def test_get_awkward_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_awkward handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "awkward" in name:
            raise ImportError("Simulated awkward absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_awkward() is None


def test_collect_api_awkward_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakyAwkward:
        Array = "broken"

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.awkward._get_awkward", lambda: FlakyAwkward()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.awkward.GhostInspector.inspect", mock_inspect
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert refs == []


def test_awkward_roundtrip_serialization() -> None:
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
