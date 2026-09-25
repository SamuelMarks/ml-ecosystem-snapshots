"""Tests for Apache Arrow Compute kernel snapshot extractor."""

import sys
import types
from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.pyarrow_compute import (
    CANONICAL_PYARROW_COMPUTE_OPS,
    _get_pyarrow_compute,
    collect_api,
)
from ml_ecosystem_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef,
    SemanticTier,
)


def test_collect_api_pyarrow_compute_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api introspects live pyarrow.compute functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakePyArrowCompute:
        """Mock pyarrow.compute module."""

        @staticmethod
        def add(a: Any, b: Any) -> Any:
            """Mock add kernel.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                Sum value.
            """
            return a + b

        @staticmethod
        def sum(arr: Any) -> Any:
            """Mock sum kernel.

            Args:
                arr: Input array.

            Returns:
                Sum reduction.
            """
            return arr

        @staticmethod
        def subtract(a: Any, b: Any) -> Any:
            """Mock subtract kernel.

            Args:
                a: Left operand.
                b: Right operand.

            Returns:
                Difference value.
            """
            return a - b

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.pyarrow_compute._get_pyarrow_compute",
        lambda: FakePyArrowCompute(),
    )

    real_inspect = GhostInspector.inspect

    def mock_inspect(obj: Any, name: str, is_public: bool = True) -> Any:
        """Mock inspect raising error on subtract.

        Args:
            obj: Target object.
            name: Fully qualified target name.
            is_public: Visibility flag.

        Returns:
            GhostRef object.

        Raises:
            RuntimeError: If target is subtract.
        """
        if "subtract" in name:
            raise RuntimeError("Simulated inspection failure")
        return real_inspect(obj, name, is_public=is_public)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.pyarrow_compute.GhostInspector.inspect",
        mock_inspect,
    )

    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    names = {ref.name for ref in refs}
    assert "add" in names
    assert "sum" in names

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_unsupported = collect_api(SemanticTier.ACTIVATION)
    assert refs_unsupported == []


def test_collect_api_pyarrow_compute_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api returns CANONICAL_PYARROW_COMPUTE_OPS fallback when pyarrow is absent.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.pyarrow_compute._get_pyarrow_compute",
        lambda: None,
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == len(CANONICAL_PYARROW_COMPUTE_OPS)
    names = {ref.name for ref in refs}
    assert "add" in names
    assert "subtract" in names
    assert "filter" in names


def test_get_pyarrow_compute_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_pyarrow_compute returns module when import succeeds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_pa = types.ModuleType("pyarrow")
    fake_pc = types.ModuleType("pyarrow.compute")
    setattr(fake_pa, "compute", fake_pc)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.compute", fake_pc)
    assert _get_pyarrow_compute() is fake_pc


def test_get_pyarrow_compute_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_pyarrow_compute handles ImportError gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on pyarrow.

        Args:
            name: Module name.
            *args: Positional import arguments.
            **kwargs: Keyword import arguments.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing pyarrow.
        """
        if "pyarrow" in name:
            raise ImportError("Simulated pyarrow absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_pyarrow_compute() is None


def test_collect_api_pyarrow_compute_inspection_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collect_api handles individual object inspection errors gracefully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FlakyPyArrow:
        """Mock PyArrow module with broken add."""

        add = "broken"

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.pyarrow_compute._get_pyarrow_compute",
        lambda: FlakyPyArrow(),
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
        "ml_ecosystem_snapshots.frameworks.pyarrow_compute.GhostInspector.inspect",
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
