"""Tests for Awkward Array ragged and nested layout snapshot extractor."""

import sys
import types
from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.awkward import (
    CANONICAL_AWKWARD_OPS,
    _get_awkward,
    collect_api,
)
from ml_ecosystem_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_awkward_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live Awkward Array constructors and functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeAwkward:
        """Mock Awkward Array module."""

        @staticmethod
        def Array(data: Any) -> Any:
            """Mock Array constructor.

            Args:
                data: Input data.

            Returns:
                Input data.
            """
            return data

        @staticmethod
        def Record(data: Any) -> Any:
            """Mock Record constructor.

            Args:
                data: Input data.

            Returns:
                Input data.
            """
            return data

        @staticmethod
        def flatten(data: Any) -> Any:
            """Mock flatten method.

            Args:
                data: Input data.

            Returns:
                Flattened data.
            """
            return data

        @staticmethod
        def mean(data: Any) -> Any:
            """Mock mean method.

            Args:
                data: Input data.

            Returns:
                Mean result.
            """
            return data

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: FakeAwkward()
    )

    real_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect raising error on mean.

        Args:
            obj: Target object.
            name: Fully qualified target name.
            is_public: Visibility flag.

        Returns:
            GhostRef object.

        Raises:
            RuntimeError: If target is mean.
        """
        if "mean" in name:
            raise RuntimeError("Simulated inspection failure")
        return real_inspect(obj, name, is_public=is_public)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.awkward.GhostInspector.inspect", mock_inspect
    )

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
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_AWKWARD_OPS)
    names = {ref.name for ref in refs}
    assert "Array" in names
    assert "ArrayBuilder" in names
    assert "flatten" in names


def test_get_awkward_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_awkward returns module when import succeeds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("awkward")
    monkeypatch.setitem(sys.modules, "awkward", fake_mod)
    assert _get_awkward() is fake_mod


def test_get_awkward_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_awkward handles ImportError gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on awkward.

        Args:
            name: Module name.
            *args: Positional import arguments.
            **kwargs: Keyword import arguments.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing awkward.
        """
        if "awkward" in name:
            raise ImportError("Simulated awkward absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_awkward() is None


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

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.awkward._get_awkward", lambda: FlakyAwkward()
    )

    def mock_inspect(*args: Any, **kwargs: Any) -> Any:
        """Mock inspect raising RuntimeError.

        Args:
            *args: Positional inspect arguments.
            **kwargs: Keyword inspect arguments.

        Raises:
            RuntimeError: Always raised.
        """
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.awkward.GhostInspector.inspect", mock_inspect
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
