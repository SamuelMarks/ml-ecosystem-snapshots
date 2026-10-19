"""Tests for cupy and dask framework collectors."""

from __future__ import annotations

import sys
import pytest_mock
import types

from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import ExtendedGhostRef
import ml_ecosystem_snapshots.frameworks.cupy as cupy_fw
import ml_ecosystem_snapshots.frameworks.dask as dask_fw


def create_module(name: str, attrs: dict[str, object]) -> types.ModuleType:
    """Create a mock module with given attributes.

    Args:
        name: Synthetic module name.
        attrs: Dictionary of module attributes to populate.

    Returns:
        The created mock module.
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_cupy_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Test cupy_fw.collect_api function.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Test when cp is None
    mocker.patch("ml_ecosystem_snapshots.frameworks.cupy.cp", None)
    assert cupy_fw.collect_api(SemanticTier.ACTIVATION) == []

    # Mock cp
    def fake_tanh() -> None:
        """Fake tanh function."""
        pass

    def fake_exp() -> None:
        """Fake exp function."""
        pass

    def fake_private() -> None:
        """Fake private function."""
        pass

    fake_cp = create_module(
        "cupy",
        {
            "tanh": fake_tanh,
            "exp": fake_exp,
            "maximum": 42,
            "_private": fake_private,
            "not_callable": 42,
            "error_func": lambda: (
                None
            ),  # We'll mock GhostInspector.inspect to raise exception for this
        },
    )
    mocker.patch("ml_ecosystem_snapshots.frameworks.cupy.cp", fake_cp)

    # Test ACTIVATION
    orig_inspect = GhostInspector.inspect

    def mock_inspect(obj: object, fqn: str, is_public: bool) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            obj: Target live object to inspect.
            fqn: Fully-qualified API path.
            is_public: Whether the symbol is publicly exported.

        Returns:
            A mocked GhostRef.
        """
        if fqn == "cupy.exp":
            raise ValueError("mock error")
        return orig_inspect(obj, fqn, is_public=is_public)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.cupy.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = cupy_fw.collect_api(SemanticTier.ACTIVATION)
    assert len(res) == 1  # tanh only

    # Test ARRAY_API
    res = cupy_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert any(r.api_path == "cupy.tanh" for r in res)
    assert not any(r.api_path == "cupy._private" for r in res)

    res = cupy_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert any(r.api_path == "cupy._private" for r in res)

    # Test other tier
    assert cupy_fw.collect_api(SemanticTier.OPTIMIZER) == []


def test_dask_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Test dask_fw.collect_api function.

    Args:
        mocker: Pytest mocker fixture.
    """
    # Test when da is None
    mocker.patch("ml_ecosystem_snapshots.frameworks.dask.da", None)
    assert dask_fw.collect_api(SemanticTier.ARRAY_API) == []

    def fake_func() -> None:
        """Fake function."""
        pass

    def fake_private() -> None:
        """Fake private function."""
        pass

    fake_da = create_module(
        "dask.array",
        {
            "func": fake_func,
            "_private": fake_private,
            "not_callable": 42,
        },
    )
    mocker.patch("ml_ecosystem_snapshots.frameworks.dask.da", fake_da)

    orig_inspect = GhostInspector.inspect

    def mock_inspect(obj: object, fqn: str, is_public: bool) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            obj: Target live object to inspect.
            fqn: Fully-qualified API path.
            is_public: Whether the symbol is publicly exported.

        Returns:
            A mocked GhostRef.
        """
        if fqn == "dask.array._private":
            raise ValueError("mock error")
        return orig_inspect(obj, fqn, is_public=is_public)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.dask.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = dask_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert any(r.api_path == "dask.array.func" for r in res)

    res = dask_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    # _private raises error, so it won't be collected
    assert not any(r.api_path == "dask.array._private" for r in res)

    # Test other tier
    assert dask_fw.collect_api(SemanticTier.OPTIMIZER) == []

    # Test UTIL tier with mocked top-level dask module
    fake_top_dask = create_module(
        "dask",
        {
            "compute": fake_func,
            "visualize": fake_func,
            "_private_util": fake_private,
            "not_callable": 123,
        },
    )
    mocker.patch.dict(sys.modules, {"dask": fake_top_dask})

    res_util = dask_fw.collect_api(SemanticTier.UTIL, include_nonpublic=False)
    assert len(res_util) > 0
    res_util_nonpublic = dask_fw.collect_api(SemanticTier.UTIL, include_nonpublic=True)
    assert len(res_util_nonpublic) >= len(res_util)

    # Test UTIL tier when dask import raises
    def mock_import_raise(*args: object, **kwargs: object) -> object:
        """Mock import raising error when importing dask.

        Args:
            *args: Positional import arguments.
            **kwargs: Keyword import arguments.

        Raises:
            RuntimeError: Always raised to simulate import failure.
        """
        raise RuntimeError("Simulated dask error")

    mocker.patch("builtins.__import__", side_effect=mock_import_raise)
    assert dask_fw.collect_api(SemanticTier.UTIL) == []

    # Test UTIL tier when inspect raises
    mocker.stopall()
    mocker.patch.dict(sys.modules, {"dask": fake_top_dask})
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.dask.GhostInspector.inspect",
        side_effect=ValueError("simulated util error"),
    )
    assert dask_fw.collect_api(SemanticTier.UTIL) == []


def test_cupy_import_success(mocker: pytest_mock.MockerFixture) -> None:
    """Test cupy import logic when module is available.

    Args:
        mocker: Pytest mocker fixture.
    """
    import sys

    # Mock cupy as a valid module
    fake_cupy = types.ModuleType("cupy")
    mocker.patch.dict(sys.modules, {"cupy": fake_cupy})

    # Reload framework to trigger the try block successfully
    if "ml_ecosystem_snapshots.frameworks.cupy" in sys.modules:
        del sys.modules["ml_ecosystem_snapshots.frameworks.cupy"]

    import ml_ecosystem_snapshots.frameworks.cupy as c_fw

    assert c_fw.cp is fake_cupy


def test_cupy_import_error(mocker: pytest_mock.MockerFixture) -> None:
    """Test cupy import logic when module is not available.

    Args:
        mocker: Pytest mocker fixture.
    """
    import sys

    mocker.patch.dict(sys.modules, {"cupy": None})
    if "ml_ecosystem_snapshots.frameworks.cupy" in sys.modules:
        del sys.modules["ml_ecosystem_snapshots.frameworks.cupy"]

    import ml_ecosystem_snapshots.frameworks.cupy as c_fw

    assert c_fw.cp is None


def test_dask_import_error(mocker: pytest_mock.MockerFixture) -> None:
    """Test dask import logic when module is available and unavailable.

    Args:
        mocker: Pytest mocker fixture.
    """
    import importlib
    import sys
    import types
    import ml_ecosystem_snapshots.frameworks.dask as d_fw

    # 1. Available
    fake_dask = types.ModuleType("dask")
    fake_da = types.ModuleType("dask.array")
    mocker.patch.dict(sys.modules, {"dask": fake_dask, "dask.array": fake_da})
    importlib.reload(d_fw)
    assert getattr(d_fw, "da") is fake_da

    # 2. Unavailable
    mocker.patch.dict(sys.modules, {"dask": None, "dask.array": None})
    importlib.reload(d_fw)
    assert getattr(d_fw, "da") is None
    assert d_fw.collect_api(SemanticTier.UTIL) == []

    # Restore
    mocker.stopall()
    importlib.reload(d_fw)
