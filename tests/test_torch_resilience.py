"""Tests for fault tolerance, fallback handling, and inspection resilience in the PyTorch framework collector."""

from __future__ import annotations

import importlib
import sys
import types
import warnings

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_ecosystem_snapshots.frameworks import torch as torch_fw
from ml_ecosystem_snapshots.models import GhostPythonRef


def test_torch_module_import_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test torch module reload handling when importing torch.nn/optim/data fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from torch_mock import build_mock_torch_modules

    for mod_name, mod_obj in build_mock_torch_modules().items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    def _get_submods() -> tuple[types.ModuleType | None, types.ModuleType | None, types.ModuleType | None]:
        """Return current `(nn, optim, data)` references from `torch_fw`.

        Returns:
            Tuple of `(nn, optim, data)` module references.
        """
        return (torch_fw.nn, torch_fw.optim, torch_fw.data)

    importlib.reload(torch_fw)
    assert all(m is not None for m in _get_submods())

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "torch.nn", None)
        mp.setitem(sys.modules, "torch.optim", None)
        mp.setitem(sys.modules, "torch.utils.data", None)
        importlib.reload(torch_fw)
        assert _get_submods() == (None, None, None)

    importlib.reload(torch_fw)
    assert _get_submods()[0] is not None


def test_torch_scan_metrics_inspect_exception(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test `collect_api(SemanticTier.METRIC)` catches exceptions during `GhostInspector.inspect`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """

    class DummyMetric:
        """Dummy metric class."""

        pass

    fake_metrics = types.ModuleType("torchmetrics")
    monkeypatch.setattr(fake_metrics, "DummyMetric", DummyMetric, raising=False)
    monkeypatch.setitem(sys.modules, "torchmetrics", fake_metrics)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.GhostInspector.inspect",
        side_effect=RuntimeError("simulated inspect failure"),
    )
    refs = torch_fw.collect_api(SemanticTier.METRIC, include_nonpublic=False)
    assert refs == []


def test_torch_scan_array_api_inspect_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test `collect_api(SemanticTier.ARRAY_API)` catches exceptions across all submodules and ops.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    fake_torch = types.ModuleType("torch")
    monkeypatch.setattr(fake_torch, "dummy_op", lambda x: x, raising=False)

    linalg_mod = types.ModuleType("torch.linalg")
    monkeypatch.setattr(linalg_mod, "norm", lambda x: x, raising=False)
    monkeypatch.setattr(fake_torch, "linalg", linalg_mod, raising=False)

    special_mod = types.ModuleType("torch.special")
    monkeypatch.setattr(special_mod, "erf", lambda x: x, raising=False)
    monkeypatch.setattr(fake_torch, "special", special_mod, raising=False)

    fft_mod = types.ModuleType("torch.fft")
    monkeypatch.setattr(fft_mod, "fft", lambda x: x, raising=False)
    monkeypatch.setattr(fake_torch, "fft", fft_mod, raising=False)

    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    monkeypatch.setattr(functional_mod, "relu", lambda x: x, raising=False)
    monkeypatch.setattr(nn_mod, "functional", functional_mod, raising=False)
    monkeypatch.setattr(fake_torch, "nn", nn_mod, raising=False)

    autograd_mod = types.ModuleType("torch.autograd")
    monkeypatch.setattr(autograd_mod, "grad", lambda x: x, raising=False)
    monkeypatch.setattr(fake_torch, "autograd", autograd_mod, raising=False)

    dist_mod = types.ModuleType("torch.distributed")
    monkeypatch.setattr(dist_mod, "all_reduce", lambda x: x, raising=False)
    monkeypatch.setattr(fake_torch, "distributed", dist_mod, raising=False)

    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: MockTensor) -> MockTensor:
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    monkeypatch.setattr(fake_torch, "Tensor", MockTensor, raising=False)

    class MockOps:
        """Mock ops container."""

        class MockAten:
            """Mock aten namespace."""

            add = staticmethod(lambda x, y: x)
            dummy_noncallable = 42

        aten = MockAten()

    monkeypatch.setattr(fake_torch, "ops", MockOps(), raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.GhostInspector.inspect",
        side_effect=ValueError("simulated inspection failure"),
    )
    refs = torch_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert refs == []


def test_torch_scan_array_api_tensor_inplace_tags_none(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test `collect_api(SemanticTier.ARRAY_API)` populates `environment_tags` when initial tags are None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    fake_torch = types.ModuleType("torch")

    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: MockTensor) -> MockTensor:
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    monkeypatch.setattr(fake_torch, "Tensor", MockTensor, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mock_ref = GhostPythonRef(
        name="add_",
        api_path="torch.Tensor.add_",
        kind="method",
        domain_metadata=None,
        environment_tags=None,
    )
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.torch.GhostInspector.inspect",
        return_value=mock_ref,
    )
    refs = torch_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert len(refs) == 1
    assert isinstance(refs[0], GhostPythonRef)
    assert refs[0].domain_metadata == {"is_in_place": True}
    assert refs[0].environment_tags == ["in_place_mutation"]


def test_get_jit_schemas_for_op_import_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `get_jit_schemas_for_op` handles exception during torch import.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    res = torch_fw.get_jit_schemas_for_op("add")
    assert res == []


def test_get_aten_op_schema_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `get_aten_op_schema` handles `ImportError` on torch import.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    res = torch_fw.get_aten_op_schema("add")
    assert res is None


def test_torch_numpy_warning_ignored() -> None:
    """Test that Failed to initialize NumPy UserWarning is filtered out."""
    importlib.reload(torch_fw)
    matching = [
        f
        for f in warnings.filters
        if f[0] == "ignore" and f[2] is UserWarning and "Failed to initialize NumPy" in str(f[1])
    ]
    assert len(matching) > 0
