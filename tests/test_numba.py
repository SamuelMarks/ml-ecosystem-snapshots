"""Tests for Numba JIT and array primitives snapshot extractor."""

import sys
import types
from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.numba import (
    CANONICAL_NUMBA_OPS,
    _get_numba,
    collect_api,
)
from ml_ecosystem_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_numba_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live Numba functions when numba is available.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeNumba:
        """Mock Numba module."""

        @staticmethod
        def njit(func: Any) -> Any:
            """Mock njit decorator.

            Args:
                func: Target function.

            Returns:
                Target function.
            """
            return func

        @staticmethod
        def vectorize(func: Any) -> Any:
            """Mock vectorize decorator.

            Args:
                func: Target function.

            Returns:
                Target function.
            """
            return func

        @staticmethod
        def prange(n: int) -> range:
            """Mock prange function.

            Args:
                n: Upper bound.

            Returns:
                Range object.
            """
            return range(n)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.numba._get_numba", lambda: FakeNumba()
    )

    real_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect raising error on prange.

        Args:
            obj: Target object.
            name: Fully qualified target name.
            is_public: Visibility flag.

        Returns:
            GhostRef object.

        Raises:
            RuntimeError: If target is prange.
        """
        if "prange" in name:
            raise RuntimeError("Simulated inspection failure")
        return real_inspect(obj, name, is_public=is_public)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.numba.GhostInspector.inspect", mock_inspect
    )

    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "njit" in names
    assert "vectorize" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_numba_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api uses CANONICAL_NUMBA_OPS fallback when numba is not installed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.numba._get_numba", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_NUMBA_OPS)
    names = {ref.name for ref in refs}
    assert "njit" in names
    assert "vectorize" in names
    assert "prange" in names


def test_get_numba_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_numba returns module when import succeeds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("numba")
    monkeypatch.setitem(sys.modules, "numba", fake_mod)
    assert _get_numba() is fake_mod


def test_get_numba_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_numba handles ImportError gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on numba.

        Args:
            name: Module name.
            *args: Positional import arguments.
            **kwargs: Keyword import arguments.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing numba.
        """
        if "numba" in name:
            raise ImportError("Simulated numba absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_numba() is None


def test_collect_api_numba_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakyNumba:
        """Mock Numba module with broken njit."""

        njit = "broken"

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.numba._get_numba", lambda: FlakyNumba()
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
        "ml_ecosystem_snapshots.frameworks.numba.GhostInspector.inspect", mock_inspect
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
