"""Tests for PyData Sparse format snapshot extractor."""

import sys
import types
from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.sparse import (
    CANONICAL_SPARSE_OPS,
    _get_sparse,
    collect_api,
)
from ml_ecosystem_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_sparse_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live PyData Sparse classes and functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeSparse:
        """Mock Sparse module."""

        class COO:
            """Mock COO format array."""

            def __init__(self, data: Any) -> None:
                """Initialize mock COO.

                Args:
                    data: Input coordinates or data.
                """
                self.data = data

        @staticmethod
        def dot(a: Any, b: Any) -> Any:
            """Mock dot operator.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                Matrix dot product.
            """
            return a

        @staticmethod
        def matmul(a: Any, b: Any) -> Any:
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

    def mock_inspect(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect raising error on matmul.

        Args:
            obj: Target object.
            name: Fully qualified target name.
            is_public: Visibility flag.

        Returns:
            GhostRef object.

        Raises:
            RuntimeError: If target is matmul.
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
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.sparse._get_sparse", lambda: None
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_SPARSE_OPS)
    names = {ref.name for ref in refs}
    assert "COO" in names
    assert "GCXS" in names
    assert "DOK" in names


def test_get_sparse_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_sparse returns module when import succeeds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_mod = types.ModuleType("sparse")
    monkeypatch.setitem(sys.modules, "sparse", fake_mod)
    assert _get_sparse() is fake_mod


def test_get_sparse_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_sparse handles ImportError gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on sparse.

        Args:
            name: Module name.
            *args: Positional import arguments.
            **kwargs: Keyword import arguments.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing sparse.
        """
        if "sparse" in name:
            raise ImportError("Simulated sparse absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_sparse() is None


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

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.sparse._get_sparse", lambda: FlakySparse()
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
        "ml_ecosystem_snapshots.frameworks.sparse.GhostInspector.inspect", mock_inspect
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
