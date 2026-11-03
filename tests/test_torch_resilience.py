"""Tests for fault tolerance, fallback handling, and inspection resilience in the PyTorch framework collector."""

from __future__ import annotations

import operator
import pytest

import importlib
import pytest_mock
import types
from unittest.mock import patch

from ml_ecosystem_snapshots.frameworks.torch import (
    _scan_array_api,
    _scan_metrics,
    get_aten_op_schema,
    get_jit_schemas_for_op,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_ecosystem_snapshots.models import GhostPythonRef


def test_torch_module_import_reload() -> None:
    """Test torch module reload handling when importing torch.nn/optim/data fails.

    Returns:
        None.
    """
    from torch_mock import ensure_torch

    ensure_torch()
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    importlib.reload(torch_fw)
    assert operator.attrgetter("nn")(torch_fw) is not None
    assert operator.attrgetter("optim")(torch_fw) is not None
    assert operator.attrgetter("data")(torch_fw) is not None

    import builtins
    import typing

    real_import = builtins.__import__

    def mock_import(
        name: str,
        globals: typing.Mapping[str, InspectableTarget] | None = None,
        locals: typing.Mapping[str, InspectableTarget] | None = None,
        fromlist: typing.Sequence[str] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Mock import raising ImportError on torch submodules.

        Args:
            name: Module name.
            globals: Globals dictionary.
            locals: Locals dictionary.
            fromlist: From-import list.
            level: Relative import level.

        Raises:
            ImportError: Simulated import error for torch submodules.

        Returns:
            Imported module.
        """
        if name in ("torch.nn", "torch.optim", "torch.utils.data"):
            raise ImportError(f"simulated {name} import failure")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(torch_fw)
        assert operator.attrgetter("nn")(torch_fw) is None
        assert operator.attrgetter("optim")(torch_fw) is None
        assert operator.attrgetter("data")(torch_fw) is None

    # Restore clean reload
    importlib.reload(torch_fw)
    assert operator.attrgetter("nn")(torch_fw) is not None


def test_torch_scan_metrics_inspect_exception(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test _scan_metrics catches exceptions during GhostInspector.inspect.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """

    class DummyMetric:
        """Dummy metric class."""

        pass

    fake_metrics = types.ModuleType("torchmetrics")
    pytest.MonkeyPatch().setattr(
        fake_metrics, "DummyMetric", DummyMetric, raising=False
    )

    with patch.dict("sys.modules", {"torchmetrics": fake_metrics}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            side_effect=RuntimeError("simulated inspect failure"),
        )
        refs = _scan_metrics(include_nonpublic=False)
        assert refs == []


def test_torch_scan_array_api_inspect_exceptions(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test _scan_array_api catches exceptions across all submodules and ops.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_torch = types.ModuleType("torch")

    # 1. Top-level callable
    pytest.MonkeyPatch().setattr(fake_torch, "dummy_op", lambda x: x, raising=False)

    # 2. linalg callable
    linalg_mod = types.ModuleType("torch.linalg")
    pytest.MonkeyPatch().setattr(linalg_mod, "norm", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "linalg", linalg_mod, raising=False)

    # 3. special callable
    special_mod = types.ModuleType("torch.special")
    pytest.MonkeyPatch().setattr(special_mod, "erf", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "special", special_mod, raising=False)

    # 4. fft callable
    fft_mod = types.ModuleType("torch.fft")
    pytest.MonkeyPatch().setattr(fft_mod, "fft", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "fft", fft_mod, raising=False)

    # 5. nn.functional callable
    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    pytest.MonkeyPatch().setattr(functional_mod, "relu", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(nn_mod, "functional", functional_mod, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "nn", nn_mod, raising=False)

    # 6. autograd callable
    autograd_mod = types.ModuleType("torch.autograd")
    pytest.MonkeyPatch().setattr(autograd_mod, "grad", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "autograd", autograd_mod, raising=False)

    # 7. distributed callable
    dist_mod = types.ModuleType("torch.distributed")
    pytest.MonkeyPatch().setattr(dist_mod, "all_reduce", lambda x: x, raising=False)
    pytest.MonkeyPatch().setattr(fake_torch, "distributed", dist_mod, raising=False)

    # 8. Tensor instance method
    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: "MockTensor") -> "MockTensor":
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    pytest.MonkeyPatch().setattr(fake_torch, "Tensor", MockTensor, raising=False)

    # 9. torch.ops.aten callable
    class MockOps:
        """Mock ops container."""

        class MockAten:
            """Mock aten namespace."""

            add = staticmethod(lambda x, y: x)
            dummy_noncallable = 42

        aten = MockAten()

    pytest.MonkeyPatch().setattr(fake_torch, "ops", MockOps(), raising=False)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            side_effect=ValueError("simulated inspection failure"),
        )
        refs = _scan_array_api(include_nonpublic=True)
        assert refs == []


def test_torch_scan_array_api_tensor_inplace_tags_none(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test _scan_array_api populates environment_tags when initial tags are None.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_torch = types.ModuleType("torch")

    class MockTensor:
        """Mock Tensor class."""

        def add_(self, other: "MockTensor") -> "MockTensor":
            """In-place add.

            Args:
                other: Other tensor.

            Returns:
                Self.
            """
            return self

    pytest.MonkeyPatch().setattr(fake_torch, "Tensor", MockTensor, raising=False)

    mock_ref = GhostPythonRef(
        name="add_",
        api_path="torch.Tensor.add_",
        kind="method",
        domain_metadata=None,
        environment_tags=None,
    )

    with patch.dict("sys.modules", {"torch": fake_torch}):
        mocker.patch.object(
            GhostInspector,
            "inspect",
            return_value=mock_ref,
        )
        refs = _scan_array_api(include_nonpublic=False)
        assert len(refs) == 1
        assert isinstance(refs[0], GhostPythonRef)
        assert refs[0].domain_metadata == {"is_in_place": True}
        assert refs[0].environment_tags == ["in_place_mutation"]


def test_get_jit_schemas_for_op_import_exception() -> None:
    """Test get_jit_schemas_for_op handles exception during torch import.

    Returns:
        None.
    """
    with patch.dict("sys.modules", {"torch": None}):
        res = get_jit_schemas_for_op("add")
        assert res == []


def test_get_aten_op_schema_import_error() -> None:
    """Test get_aten_op_schema handles ImportError on torch import.

    Returns:
        None.
    """
    with patch.dict("sys.modules", {"torch": None}):
        res = get_aten_op_schema("add")
        assert res is None


def test_torch_numpy_warning_ignored() -> None:
    """Test that Failed to initialize NumPy UserWarning is filtered out.

    Returns:
        None.
    """
    import importlib
    import warnings
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    importlib.reload(torch_fw)
    matching = [
        f
        for f in warnings.filters
        if f[0] == "ignore"
        and f[2] is UserWarning
        and f[1] is not None
        and "Failed to initialize NumPy"
        in (operator.attrgetter("pattern")(f[1]) if "pattern" in dir(f[1]) else "")
    ]
    assert len(matching) > 0
