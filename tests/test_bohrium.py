"""Tests for Bohrium heterogeneous runtime snapshot extractor."""

import sys
import types
from typing import Any
import pytest

from ml_framework_snapshots.frameworks.bohrium import (
    CANONICAL_BOHRIUM_OPS,
    _get_bohrium,
    collect_api,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_bohrium_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_BOHRIUM_OPS fallback when bohrium is absent."""
    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.bohrium._get_bohrium", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_BOHRIUM_OPS)
    names = {ref.name for ref in refs}
    assert "array" in names
    assert "empty" in names
    assert "flush" in names


def test_collect_api_bohrium_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects members when bohrium is present."""

    class FakeBohrium:
        """Mock Bohrium module."""

        @staticmethod
        def array(obj: Any) -> Any:
            """Mock array creation."""
            return obj

        @staticmethod
        def flush() -> None:
            """Mock flush call."""
            pass

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.bohrium._get_bohrium", lambda: FakeBohrium()
    )
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
    """Verify _get_bohrium returns module when import succeeds."""
    fake_mod = types.ModuleType("bohrium")
    monkeypatch.setitem(sys.modules, "bohrium", fake_mod)
    assert _get_bohrium() is fake_mod


def test_get_bohrium_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_bohrium handles ImportError gracefully."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "bohrium" in name:
            raise ImportError("Simulated bohrium absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_bohrium() is None


def test_collect_api_bohrium_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully."""

    class FlakyBohrium:
        array = "broken"

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.bohrium._get_bohrium", lambda: FlakyBohrium()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_framework_snapshots.frameworks.bohrium.GhostInspector.inspect", mock_inspect
    )
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
