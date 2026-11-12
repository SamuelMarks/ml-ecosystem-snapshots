"""Tests for Awkward Array ragged and nested layout snapshot extractor."""

from __future__ import annotations

import sys
import types

import pytest
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks.awkward import (
    CANONICAL_AWKWARD_OPS,
    collect_api,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def test_collect_api_awkward_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live Awkward Array constructors and functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeAwkward:
        """Mock Awkward Array module."""

        @staticmethod
        def Array(data: LivePythonTarget) -> LivePythonTarget:
            """Mock Array constructor.

            Args:
                data: Input data.

            Returns:
                Input data.
            """
            return data

        @staticmethod
        def Record(data: LivePythonTarget) -> LivePythonTarget:
            """Mock Record constructor.

            Args:
                data: Input data.

            Returns:
                Input data.
            """
            return data

        @staticmethod
        def flatten(data: LivePythonTarget) -> LivePythonTarget:
            """Mock flatten method.

            Args:
                data: Input data.

            Returns:
                Flattened data.
            """
            return data

        @staticmethod
        def mean(data: LivePythonTarget) -> LivePythonTarget:
            """Mock mean method.

            Args:
                data: Input data.

            Returns:
                Mean result.
            """
            return data

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: FakeAwkward())

    real_inspect = GhostInspector.inspect

    def mock_inspect(obj: LivePythonTarget, name: str, is_public: bool = True) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            obj: Target Awkward Array symbol to inspect.
            name: Fully-qualified symbol path.
            is_public: Whether the symbol is publicly exported.

        Returns:
            A mocked GhostRef.

        Raises:
            RuntimeError: When inspecting `mean` to simulate failure.
        """
        if "mean" in name:
            raise RuntimeError("Simulated inspection failure")
        return real_inspect(obj, name, is_public=is_public)

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.awkward.GhostInspector.inspect", mock_inspect)

    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "Array" in names
    assert "Record" in names
    assert "flatten" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_awkward_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api returns CANONICAL_AWKWARD_OPS fallback when awkward is not installed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: None)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_AWKWARD_OPS)
    names = {ref.name for ref in refs}
    assert "Array" in names
    assert "ArrayBuilder" in names
    assert "flatten" in names


def test_get_awkward_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` resolves live `awkward` module from `sys.modules` when present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("awkward")
    monkeypatch.setitem(sys.modules, "awkward", fake_mod)
    assert collect_api(SemanticTier.ARRAY_API) == []


def test_get_awkward_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` falls back to canonical ops when `awkward` import raises `ImportError`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "awkward", None)
    assert len(collect_api(SemanticTier.ARRAY_API)) == len(CANONICAL_AWKWARD_OPS)


def test_collect_api_awkward_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakyAwkward:
        """Mock Awkward module with broken Array attribute."""

        Array = "broken"

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: FlakyAwkward())

    def mock_inspect(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> ExtendedGhostRef:
        """Mock inspect raising RuntimeError.

        Args:
            *args: Positional inspect arguments.
            **kwargs: Keyword inspect arguments.

        Raises:
            RuntimeError: Always raised.

        Returns:
            Never returns.
        """
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.awkward.GhostInspector.inspect", mock_inspect)
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
