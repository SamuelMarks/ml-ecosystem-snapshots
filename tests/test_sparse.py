"""Tests for PyData Sparse format snapshot extractor."""

from __future__ import annotations

import sys
import types

import pytest
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks.sparse import (
    CANONICAL_SPARSE_OPS,
    collect_api,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def test_collect_api_sparse_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live PyData Sparse classes and functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeSparse:
        """Mock Sparse module."""

        class COO:
            """Mock COO format array."""

            def __init__(self, data: LivePythonTarget) -> None:
                """Initialize mock COO.

                Args:
                    data: Input coordinates or data.
                """
                self.data = data

        @staticmethod
        def dot(a: LivePythonTarget, b: LivePythonTarget) -> LivePythonTarget:
            """Mock dot operator.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                Matrix dot product.
            """
            return a

        @staticmethod
        def matmul(a: LivePythonTarget, b: LivePythonTarget) -> LivePythonTarget:
            """Mock matmul operator.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                Matrix multiplication result.
            """
            return a

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.sparse._get_sparse",
        lambda: FakeSparse(),
    )

    real_inspect = GhostInspector.inspect

    def mock_inspect(obj: LivePythonTarget, name: str, is_public: bool = True) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            obj: Target PyData Sparse symbol to inspect.
            name: Fully-qualified symbol path.
            is_public: Whether the symbol is publicly exported.

        Returns:
            A mocked GhostRef.

        Raises:
            RuntimeError: When `name` contains `matmul`.
        """
        if "matmul" in name:
            raise RuntimeError("Simulated inspection failure")
        return real_inspect(obj, name, is_public=is_public)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.sparse.GhostInspector.inspect",
        mock_inspect,
    )

    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "COO" in names
    assert "dot" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_sparse_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api uses CANONICAL_SPARSE_OPS fallback when sparse is not installed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.sparse._get_sparse", lambda: None)
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_SPARSE_OPS)
    names = {ref.name for ref in refs}
    assert "COO" in names
    assert "GCXS" in names
    assert "DOK" in names


def test_get_sparse_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` resolves live `sparse` module from `sys.modules` when present.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("sparse")
    monkeypatch.setitem(sys.modules, "sparse", fake_mod)
    assert collect_api(SemanticTier.ARRAY_API) == []


def test_get_sparse_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_api` falls back to canonical ops when `sparse` import raises `ImportError`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "sparse", None)
    assert len(collect_api(SemanticTier.ARRAY_API)) == len(CANONICAL_SPARSE_OPS)


def test_collect_api_sparse_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakySparse:
        """Mock Sparse module with broken COO."""

        COO = "broken"

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.sparse._get_sparse", lambda: FlakySparse())

    def mock_inspect(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
        """Mock inspect raising RuntimeError.

        Args:
            *args: Positional inspect arguments.
            **kwargs: Keyword inspect arguments.

        Raises:
            RuntimeError: Always raised.
        """
        raise RuntimeError("Inspection failure")

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.sparse.GhostInspector.inspect", mock_inspect)
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
