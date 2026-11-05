"""Unit tests and verification suite for high perf."""

from __future__ import annotations

import importlib
import types

import pytest
from ml_ecosystem_snapshots.frameworks.deepspeed import collect_api as ds_collect
from ml_ecosystem_snapshots.frameworks.onnxruntime import collect_api as onnx_collect
from ml_ecosystem_snapshots.frameworks.triton import collect_api as triton_collect
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier


def test_triton_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify triton collect behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_triton = types.ModuleType("triton")
    mock_tl = types.ModuleType("triton.language")

    def constexpr() -> None:
        """Execute helper constexpr."""

    def tensor() -> None:
        """Execute helper tensor."""

    def _priv() -> None:
        """Execute helper _priv."""

    monkeypatch.setattr(mock_tl, "constexpr", constexpr, raising=False)
    monkeypatch.setattr(mock_tl, "tensor", tensor, raising=False)
    monkeypatch.setattr(mock_tl, "not_callable", 123, raising=False)
    monkeypatch.setattr(mock_tl, "_priv", _priv, raising=False)

    mock_math = types.ModuleType("triton.language.math")
    monkeypatch.setattr(mock_math, "a_bad", 123, raising=False)
    monkeypatch.setattr(mock_math, "sin", lambda: None, raising=False)
    monkeypatch.setattr(mock_math, "_priv", lambda: None, raising=False)
    monkeypatch.setattr(mock_math, "bad", 123, raising=False)
    monkeypatch.setattr(mock_tl, "math", mock_math, raising=False)

    def fake_inspect(obj: InspectableTarget, path: str, **kw: InspectableTarget) -> GhostRef:
        """Return a stub GhostRef for the given path.

        Args:
            obj: Target callable.
            path: Dotted API path.
            kw: Extra keyword arguments.

        Returns:
            Stub `GhostRef`.
        """
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

    def fake_import_module(name: str, package: str | None = None) -> types.ModuleType:
        """Return fake triton modules for import_module.

        Args:
            name: Module name.
            package: Optional package anchor.

        Returns:
            Fake module object.

        Raises:
            ImportError: If an unknown module is requested.
        """
        if name == "triton":
            return mock_triton
        if name == "triton.language":
            return mock_tl
        raise ImportError(name)

    monkeypatch.setattr(GhostInspector, "inspect", staticmethod(fake_inspect))
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    res = triton_collect(SemanticTier.UTIL)
    assert len(res) == 3
    assert {r.api_path for r in res} == {
        "triton.language.constexpr",
        "triton.language.tensor",
        "triton.language.math.sin",
    }

    res_priv = triton_collect(SemanticTier.UTIL, include_nonpublic=True)
    assert len(res_priv) == 5


def test_onnx_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify onnx collect behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_onnx = types.ModuleType("onnxruntime")

    class InferenceSession:
        """Mock or helper class InferenceSession for unit testing."""

    monkeypatch.setattr(mock_onnx, "InferenceSession", InferenceSession, raising=False)
    monkeypatch.setattr(mock_onnx, "utils", 123, raising=False)

    def fake_inspect(self: GhostInspector, obj: InspectableTarget, path: str) -> GhostRef:
        """Return a stub GhostRef for ONNX Runtime symbols.

        Args:
            self: GhostInspector instance.
            obj: Target symbol.
            path: Dotted API path.

        Returns:
            Stub `GhostRef`.
        """
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

    monkeypatch.setattr(importlib, "import_module", lambda _name: mock_onnx)
    monkeypatch.setattr(GhostInspector, "inspect", fake_inspect)

    res = onnx_collect(SemanticTier.MODEL)
    assert len(res) == 1
    assert res[0].name == "InferenceSession"
    assert any(p.name == "providers" for p in res[0].params)


def test_ds_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ds collect behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_ds = types.ModuleType("deepspeed")

    def initialize() -> None:
        """Execute helper initialize."""

    monkeypatch.setattr(mock_ds, "initialize", initialize, raising=False)
    monkeypatch.setattr(mock_ds, "utils", 123, raising=False)

    def fake_inspect(self: GhostInspector, obj: InspectableTarget, path: str) -> GhostRef:
        """Return a stub GhostRef for DeepSpeed symbols.

        Args:
            self: GhostInspector instance.
            obj: Target symbol.
            path: Dotted API path.

        Returns:
            Stub `GhostRef`.
        """
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

    monkeypatch.setattr(importlib, "import_module", lambda _name: mock_ds)
    monkeypatch.setattr(GhostInspector, "inspect", fake_inspect)

    res = ds_collect(SemanticTier.MODEL)
    assert len(res) == 1
    assert res[0].name == "initialize"
    assert any(p.name == "config_params" for p in res[0].params)
