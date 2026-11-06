"""Tests for multi-framework collectors: IR, ONNX Runtime, NumPy, MLX, HuggingFace, TensorFlow, and JAX."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
import typing
from unittest.mock import patch

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks import (
    huggingface as hf_mod,
)
from ml_ecosystem_snapshots.frameworks import (
    ir as ir_mod,
)
from ml_ecosystem_snapshots.frameworks import (
    jax as jax_fw,
)
from ml_ecosystem_snapshots.frameworks import (
    mlx as mlx_mod,
)
from ml_ecosystem_snapshots.frameworks import (
    numpy as numpy_mod,
)
from ml_ecosystem_snapshots.frameworks import (
    onnxruntime as ort_mod,
)
from ml_ecosystem_snapshots.frameworks import (
    tensorflow as tf_fw,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget


def test_ir_collect_api_all_categories() -> None:
    """Test ir_mod.collect_api across supported and unsupported categories."""
    # Supported categories
    refs_array = ir_mod.collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) == 59
    names = {r.name for r in refs_array}
    assert "LogicalGraph" in names
    assert "LogicalNode" in names
    assert "TensorSpec" in names
    assert "LogicalEdge" in names
    assert "topological_sort" in names

    refs_neural = ir_mod.collect_api(SemanticTier.NEURAL, include_nonpublic=True)
    assert len(refs_neural) == 59

    refs_util = ir_mod.collect_api(SemanticTier.UTIL)
    assert len(refs_util) == 59

    # Unsupported category triggers empty return
    refs_loss = ir_mod.collect_api(SemanticTier.LOSS)
    assert refs_loss == []


class _OrtInferenceSession:
    """Stub ONNX Runtime InferenceSession class."""


class _OrtOtherSession:
    """Stub secondary ONNX Runtime session class."""


def _ort_dummy_util() -> None:
    """Stub public utility function for ONNX Runtime."""


def _ort_private_func() -> None:
    """Stub private utility function for ONNX Runtime."""


def _build_fake_ort_module() -> types.ModuleType:
    """Construct a populated fake `onnxruntime` module for collector testing.

    Returns:
        Populated fake `onnxruntime` module instance.
    """
    fake_mod = types.ModuleType("onnxruntime")
    mp = pytest.MonkeyPatch()
    mp.setattr(fake_mod, "InferenceSession", _OrtInferenceSession, raising=False)
    mp.setattr(fake_mod, "OtherSession", _OrtOtherSession, raising=False)
    mp.setattr(fake_mod, "dummy_util", _ort_dummy_util, raising=False)
    mp.setattr(fake_mod, "_private_func", _ort_private_func, raising=False)
    mp.setattr(fake_mod, "none_obj", None, raising=False)
    mp.setattr(fake_mod, "raise_exc", lambda: None, raising=False)
    mp.setattr(fake_mod, "return_none_ref", lambda: None, raising=False)
    return fake_mod


def _make_ort_inspect_side_effect(
    providers_present: list[bool],
) -> typing.Callable[[InspectableTarget, str], GhostRef | None]:
    """Create a side-effect callback for `GhostInspector.inspect` on ONNX Runtime targets.

    Args:
        providers_present: Single-element boolean list controlling whether `providers` is pre-populated.

    Returns:
        Callable side-effect function returning a `GhostRef` or `None`.
    """

    def _mock_inspect(obj: InspectableTarget, path: str) -> GhostRef | None:
        """Simulate `GhostInspector.inspect` for ONNX Runtime test targets.

        Args:
            obj: Live target being inspected.
            path: Dotted API path of the target.

        Raises:
            RuntimeError: When `raise_exc` is encountered in `path`.

        Returns:
            Constructed `GhostRef` or `None`.
        """
        if "raise_exc" in path:
            raise RuntimeError("simulated inspect error")
        if "return_none_ref" in path:
            return None
        if "InferenceSession" in path and providers_present[0]:
            return GhostRef(
                name="InferenceSession",
                api_path=path,
                kind="class",
                params=[GhostParam(name="providers", kind=ParameterKind.KEYWORD_ONLY)],
            )
        return GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="class" if inspect.isclass(obj) else "function",
            params=[],
        )

    return _mock_inspect


def test_onnxruntime_collect_api_edge_branches(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Exercise `onnxruntime.collect_api` across all category and error branches.

    Args:
        mocker: Pytest mocker fixture.
    """
    assert ort_mod.collect_api(SemanticTier.LOSS) == []

    real_import = importlib.import_module

    def mock_import_err(name: str, package: str | None = None) -> types.ModuleType:
        """Raise `ImportError` when importing `onnxruntime`.

        Args:
            name: Target module name.
            package: Optional package anchor.

        Raises:
            ImportError: When `name` equals `'onnxruntime'`.

        Returns:
            Imported module for non-onnxruntime names.
        """
        if name == "onnxruntime":
            raise ImportError("simulated onnxruntime error")
        return real_import(name, package)

    with patch("importlib.import_module", side_effect=mock_import_err):
        assert ort_mod.collect_api(SemanticTier.MODEL) == []

    fake_mod = _build_fake_ort_module()

    def mock_import_fake(name: str, package: str | None = None) -> types.ModuleType:
        """Return `fake_mod` when importing `onnxruntime`.

        Args:
            name: Target module name.
            package: Optional package anchor.

        Returns:
            Fake `onnxruntime` module or real imported module.
        """
        if name == "onnxruntime":
            return fake_mod
        return real_import(name, package)

    providers_present = [False]
    mocker.patch.object(GhostInspector, "inspect", side_effect=_make_ort_inspect_side_effect(providers_present))

    with patch("importlib.import_module", side_effect=mock_import_fake):
        providers_present[0] = False
        res_model = ort_mod.collect_api(SemanticTier.MODEL, include_nonpublic=False)
        names_model = {r.name for r in res_model}
        assert "InferenceSession" in names_model
        assert "OtherSession" in names_model

        inf_ref = next(r for r in res_model if r.name == "InferenceSession")
        assert any(p.name == "providers" for p in inf_ref.params)

        providers_present[0] = True
        res_model2 = ort_mod.collect_api(SemanticTier.MODEL, include_nonpublic=False)
        inf_ref2 = next(r for r in res_model2 if r.name == "InferenceSession")
        assert len([p for p in inf_ref2.params if p.name == "providers"]) == 1

        res_util = ort_mod.collect_api(SemanticTier.UTIL, include_nonpublic=True)
        names_util = {r.name for r in res_util}
        assert "dummy_util" in names_util
        assert "_private_func" in names_util


class _NumpyLinAlgClass:
    """Stub class inside `numpy.linalg`."""


class _NumpyFFTClass:
    """Stub class inside `numpy.fft`."""


def _np_noop() -> None:
    """No-op stub callable for fake NumPy functions."""


def _make_named_np_stubs() -> dict[str, typing.Callable[[], None]]:
    """Create stub functions with exact NumPy symbol names for `GhostInspector.inspect`.

    Returns:
        Dictionary mapping NumPy symbol names to matching stub callables.
    """

    def tanh() -> None:
        """Stub `numpy.tanh` function."""

    def exp() -> None:
        """Stub `numpy.exp` function."""

    def abs() -> None:
        """Stub `numpy.abs` function."""

    def add() -> None:
        """Stub `numpy.add` function."""

    def det() -> None:
        """Stub `numpy.linalg.det` function."""

    def inv() -> None:
        """Stub `numpy.linalg.inv` function."""

    def fft() -> None:
        """Stub `numpy.fft.fft` function."""

    def ifft() -> None:
        """Stub `numpy.fft.ifft` function."""

    return {
        "tanh": tanh,
        "exp": exp,
        "abs": abs,
        "add": add,
        "det": det,
        "inv": inv,
        "fft": fft,
        "ifft": ifft,
    }


def _build_fake_numpy_array_module(
    stubs: dict[str, typing.Callable[[], None]],
) -> types.ModuleType:
    """Build a fake `numpy` module with `linalg` and `fft` submodules.

    Args:
        stubs: Mapping of NumPy operation names to stub functions.

    Returns:
        Populated fake `numpy` module instance.
    """
    mp = pytest.MonkeyPatch()
    fake_np_array = types.ModuleType("numpy")
    mp.setattr(fake_np_array, "abs", stubs["abs"], raising=False)
    mp.setattr(fake_np_array, "add", stubs["add"], raising=False)
    mp.setattr(fake_np_array, "all", "not_callable", raising=False)

    fake_linalg = types.ModuleType("numpy.linalg")
    mp.setattr(fake_linalg, "_private", _np_noop, raising=False)
    mp.setattr(fake_linalg, "LinAlgClass", _NumpyLinAlgClass, raising=False)
    mp.setattr(fake_linalg, "non_callable", 123, raising=False)
    mp.setattr(fake_linalg, "det", stubs["det"], raising=False)
    mp.setattr(fake_linalg, "inv", stubs["inv"], raising=False)
    mp.setattr(fake_np_array, "linalg", fake_linalg, raising=False)

    fake_fft_mod = types.ModuleType("numpy.fft")
    mp.setattr(fake_fft_mod, "_private", _np_noop, raising=False)
    mp.setattr(fake_fft_mod, "FFTClass", _NumpyFFTClass, raising=False)
    mp.setattr(fake_fft_mod, "non_callable", 456, raising=False)
    mp.setattr(fake_fft_mod, "fft", stubs["fft"], raising=False)
    mp.setattr(fake_fft_mod, "ifft", stubs["ifft"], raising=False)
    mp.setattr(fake_np_array, "fft", fake_fft_mod, raising=False)
    return fake_np_array


def test_numpy_collect_api_all_branches(mocker: pytest_mock.MockerFixture) -> None:
    """Exercise `numpy.collect_api` across missing module, activation, and array branches.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch.object(numpy_mod, "np", None)
    assert numpy_mod.collect_api(SemanticTier.ACTIVATION) == []
    assert numpy_mod.collect_api(SemanticTier.ARRAY_API) == []

    fake_np = types.ModuleType("numpy")
    mocker.patch.object(numpy_mod, "np", fake_np)
    assert numpy_mod.collect_api(SemanticTier.MODEL) == []

    stubs = _make_named_np_stubs()
    mp = pytest.MonkeyPatch()
    mp.setattr(fake_np, "tanh", stubs["tanh"], raising=False)
    mp.setattr(fake_np, "exp", stubs["exp"], raising=False)
    mp.setattr(fake_np, "maximum", 42, raising=False)

    original_inspect = GhostInspector.inspect

    def mock_inspect_act(obj: InspectableTarget, name: str, is_public: bool = True) -> GhostRef:
        """Simulate `GhostInspector.inspect` for NumPy activation symbols.

        Args:
            obj: Live target object to inspect.
            name: Dotted API path string.
            is_public: Public visibility flag.

        Raises:
            RuntimeError: When inspecting `exp`.

        Returns:
            Inspected `GhostRef` result.
        """
        if "exp" in name:
            raise RuntimeError("simulated inspect error")
        return original_inspect(obj, name, is_public=is_public)

    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect_act)
    res_act = numpy_mod.collect_api(SemanticTier.ACTIVATION)
    assert len(res_act) == 1
    assert res_act[0].name == "tanh"

    fake_np_array = _build_fake_numpy_array_module(stubs)
    failing_tokens = ("add", "inv", "ifft")

    def mock_inspect_arr(obj: InspectableTarget, name: str, is_public: bool = True) -> GhostRef:
        """Simulate `GhostInspector.inspect` for NumPy array API symbols.

        Args:
            obj: Live target object to inspect.
            name: Dotted API path string.
            is_public: Public visibility flag.

        Raises:
            RuntimeError: When `name` contains any token in `failing_tokens`.

        Returns:
            Inspected `GhostRef` result.
        """
        if any(tok in name for tok in failing_tokens):
            raise RuntimeError("simulated inspect error")
        return original_inspect(obj, name, is_public=is_public)

    mocker.patch.object(numpy_mod, "np", fake_np_array)
    mocker.patch.object(GhostInspector, "inspect", side_effect=mock_inspect_arr)

    res_arr = numpy_mod.collect_api(SemanticTier.ARRAY_API)
    names_arr = {r.name for r in res_arr}
    assert {"abs", "det", "fft"} <= names_arr

    fake_np_bare = types.ModuleType("numpy")
    mp.setattr(fake_np_bare, "abs", stubs["abs"], raising=False)
    mocker.patch.object(numpy_mod, "np", fake_np_bare)
    res_bare = numpy_mod.collect_api(SemanticTier.ARRAY_API)
    assert len(res_bare) == 1


def test_numpy_import_error_handling() -> None:
    """Test numpy module import failure fallback branch."""
    saved_np = numpy_mod.np
    saved_sys_np = sys.modules.get("numpy")

    with patch.dict(sys.modules, {"numpy": None}):
        importlib.reload(numpy_mod)
        assert numpy_mod.np is None

    # Restore clean numpy import
    if saved_sys_np is not None:
        sys.modules["numpy"] = saved_sys_np
    numpy_mod.np = saved_np
    assert numpy_mod.np is not None


class _MlxUpperLayer:
    """Stub MLX layer class with uppercase leading character."""


class _mlx_lower_layer:
    """Stub MLX layer class with lowercase leading character."""


class _MlxPrivateLayer:
    """Stub private MLX layer class."""


class _MlxCustomLoss:
    """Stub MLX loss class."""


class _MlxAdam:
    """Stub MLX optimizer class with uppercase leading character."""


class _mlx_adam_lower:
    """Stub MLX optimizer class with lowercase leading character."""


class _MlxPrivateOptimizer:
    """Stub private MLX optimizer class."""


class _MlxCoreClass:
    """Stub class in `mlx.core`."""


def _mlx_noop() -> None:
    """No-op callable stub for fake MLX functions."""


def _build_fake_mlx_modules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Build fake `mlx`, `mlx.nn`, and `mlx.nn.losses` modules for collector tests.

    Returns:
        Three-element tuple of `(fake_mlx, fake_nn, fake_losses)`.
    """
    mp = pytest.MonkeyPatch()
    fake_mlx = types.ModuleType("mlx")
    fake_nn = types.ModuleType("mlx.nn")
    fake_losses = types.ModuleType("mlx.nn.losses")
    fake_optimizers = types.ModuleType("mlx.optimizers")
    fake_core = types.ModuleType("mlx.core")
    fake_fft = types.ModuleType("mlx.core.fft")
    fake_linalg = types.ModuleType("mlx.core.linalg")

    mp.setattr(fake_mlx, "nn", fake_nn, raising=False)
    mp.setattr(fake_mlx, "optimizers", fake_optimizers, raising=False)
    mp.setattr(fake_mlx, "core", fake_core, raising=False)

    mp.setattr(fake_nn, "UpperLayer", _MlxUpperLayer, raising=False)
    mp.setattr(fake_nn, "lower_layer", _mlx_lower_layer, raising=False)
    mp.setattr(fake_nn, "_PrivateLayer", _MlxPrivateLayer, raising=False)
    mp.setattr(fake_nn, "not_a_class", _mlx_noop, raising=False)
    mp.setattr(fake_nn, "relu", _mlx_noop, raising=False)
    mp.setattr(fake_nn, "_relu", _mlx_noop, raising=False)
    mp.setattr(fake_nn, "unrelated_act", _mlx_noop, raising=False)

    mp.setattr(fake_losses, "cross_entropy_loss", _mlx_noop, raising=False)
    mp.setattr(fake_losses, "CustomLoss", _MlxCustomLoss, raising=False)
    mp.setattr(fake_losses, "_private_loss", _mlx_noop, raising=False)
    mp.setattr(fake_losses, "something_else", _mlx_noop, raising=False)
    mp.setattr(fake_losses, "number_loss", 999, raising=False)
    mp.setattr(fake_nn, "losses", fake_losses, raising=False)

    mp.setattr(fake_optimizers, "Adam", _MlxAdam, raising=False)
    mp.setattr(fake_optimizers, "adam_lower", _mlx_adam_lower, raising=False)
    mp.setattr(fake_optimizers, "_PrivateOptimizer", _MlxPrivateOptimizer, raising=False)
    mp.setattr(fake_optimizers, "optim_func", _mlx_noop, raising=False)

    mp.setattr(fake_core, "abs", _mlx_noop, raising=False)
    mp.setattr(fake_core, "fake_core_exc", _mlx_noop, raising=False)
    mp.setattr(fake_core, "CoreClass", _MlxCoreClass, raising=False)
    mp.setattr(fake_core, "_private_core", _mlx_noop, raising=False)

    mp.setattr(fake_fft, "fft", _mlx_noop, raising=False)
    mp.setattr(fake_fft, "fake_fft_exc", _mlx_noop, raising=False)
    mp.setattr(fake_fft, "ClassFFT", _MlxCoreClass, raising=False)
    mp.setattr(fake_fft, "_private_fft", _mlx_noop, raising=False)
    mp.setattr(fake_core, "fft", fake_fft, raising=False)

    mp.setattr(fake_linalg, "norm", _mlx_noop, raising=False)
    mp.setattr(fake_linalg, "fake_linalg_exc", _mlx_noop, raising=False)
    mp.setattr(fake_linalg, "ClassLinalg", _MlxCoreClass, raising=False)
    mp.setattr(fake_linalg, "_private_linalg", _mlx_noop, raising=False)
    mp.setattr(fake_core, "linalg", fake_linalg, raising=False)
    return fake_mlx, fake_nn, fake_losses


def _mock_mlx_inspect(obj: InspectableTarget, path: str) -> GhostRef:
    """Simulate `GhostInspector.inspect` for MLX collector tests.

    Args:
        obj: Target live object to inspect.
        path: Dotted canonical API path.

    Raises:
        RuntimeError: When `path` references a simulated failing function.

    Returns:
        Constructed `GhostRef` instance.
    """
    if "_exc" in path:
        raise RuntimeError("simulated inspect error")
    return GhostRef(
        name=path.split(".")[-1],
        api_path=path,
        kind="class" if inspect.isclass(obj) else "function",
        params=[],
    )


def test_mlx_collect_api_all_branches(mocker: pytest_mock.MockerFixture) -> None:
    """Exercise `mlx.collect_api` across all categories, visibility flags, and fallbacks.

    Args:
        mocker: Pytest mocker fixture.
    """
    mocker.patch.object(mlx_mod, "mlx", None)
    assert mlx_mod.collect_api(SemanticTier.LAYER) == []

    fake_mlx, fake_nn, fake_losses = _build_fake_mlx_modules()
    mocker.patch.object(mlx_mod, "mlx", fake_mlx)
    mocker.patch.object(GhostInspector, "inspect", side_effect=_mock_mlx_inspect)

    assert mlx_mod.collect_api(SemanticTier.MODEL) == []

    layers_pub = mlx_mod.collect_api(SemanticTier.LAYER, include_nonpublic=False)
    assert {r.name for r in layers_pub} == {"UpperLayer"}
    layers_all = mlx_mod.collect_api(SemanticTier.LAYER, include_nonpublic=True)
    assert {r.name for r in layers_all} == {"UpperLayer"}

    act_pub = mlx_mod.collect_api(SemanticTier.ACTIVATION, include_nonpublic=False)
    assert {r.name for r in act_pub} == {"relu"}
    act_all = mlx_mod.collect_api(SemanticTier.ACTIVATION, include_nonpublic=True)
    assert {r.name for r in act_all} == {"relu"}

    losses_pub = mlx_mod.collect_api(SemanticTier.LOSS, include_nonpublic=False)
    assert {r.name for r in losses_pub} == {"cross_entropy_loss", "CustomLoss"}
    losses_all = mlx_mod.collect_api(SemanticTier.LOSS, include_nonpublic=True)
    assert "_private_loss" in {r.name for r in losses_all}

    del fake_nn.losses
    assert mlx_mod.collect_api(SemanticTier.LOSS) == []
    pytest.MonkeyPatch().setattr(fake_nn, "losses", fake_losses, raising=False)

    opt_pub = mlx_mod.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=False)
    assert {r.name for r in opt_pub} == {"Adam"}
    opt_all = mlx_mod.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=True)
    assert {r.name for r in opt_all} == {"Adam"}

    arr_pub = mlx_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    names_arr = {r.name for r in arr_pub}
    assert {"abs", "fft", "norm"} <= names_arr
    assert "fake_core_exc" not in names_arr

    arr_all = mlx_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    names_arr_all = {r.name for r in arr_all}
    assert {"_private_core", "_private_fft", "_private_linalg"} <= names_arr_all

    mocker.patch.object(mlx_mod, "mlx", types.ModuleType("mlx"))
    assert mlx_mod.collect_api(SemanticTier.ARRAY_API) == []

    fake_mlx_no_sub = types.ModuleType("mlx")
    pytest.MonkeyPatch().setattr(fake_mlx_no_sub, "core", types.ModuleType("mlx.core"), raising=False)
    mocker.patch.object(mlx_mod, "mlx", fake_mlx_no_sub)
    assert mlx_mod.collect_api(SemanticTier.ARRAY_API) == []

    mocker.patch.object(mlx_mod, "mlx", fake_mlx)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.mlx.get_all_members",
        side_effect=RuntimeError("simulated get_all_members crash"),
    )
    assert mlx_mod.collect_api(SemanticTier.LAYER) == []


def test_mlx_import_error_handling() -> None:
    """Test mlx module import success and failure fallback branches."""
    # 1. Test simulated import success of mlx
    mock_mlx_root = types.ModuleType("mlx")
    mock_mlx_core = types.ModuleType("mlx.core")
    mock_mlx_nn = types.ModuleType("mlx.nn")
    mock_mlx_opt = types.ModuleType("mlx.optimizers")

    def _get_mlx() -> types.ModuleType | None:
        """Read current `mlx_mod.mlx` reference after module reload.

        Returns:
            Current `mlx_mod.mlx` value.
        """
        return mlx_mod.mlx

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx_root,
            "mlx.core": mock_mlx_core,
            "mlx.nn": mock_mlx_nn,
            "mlx.optimizers": mock_mlx_opt,
        },
    ):
        importlib.reload(mlx_mod)
        assert _get_mlx() is mock_mlx_root

    # 2. Test simulated import failure of mlx
    with patch.dict(
        sys.modules,
        {
            "mlx": None,
            "mlx.core": None,
            "mlx.nn": None,
            "mlx.optimizers": None,
        },
    ):
        importlib.reload(mlx_mod)
        assert _get_mlx() is None

    # Restore clean reload
    importlib.reload(mlx_mod)


class _HfDummyGenWithException:
    """Stub class whose `generate` method signature inspection fails."""

    def generate(self) -> None:
        """Run stub generation."""


class _HfDummyGenFull:
    """Stub class with a multi-parameter `generate` signature."""

    @typing.no_type_check
    def generate(
        self,
        existing_param: str,
        default_val: int = 10,
        no_default: float = 0.0,
        no_anno=None,
        *args: InspectableTarget,
        **kwargs: InspectableTarget,
    ) -> None:
        """Run stub generation with diverse parameter kinds.

        Args:
            existing_param: Pre-existing parameter name.
            default_val: Parameter with integer default.
            no_default: Parameter with float annotation.
            no_anno: Untyped parameter.
            *args: Variadic positional arguments.
            **kwargs: Variadic keyword arguments.
        """


class _HfNoAnnotations:
    """Stub class without class-level annotations."""


class _HfAnnotatedConfig:
    """Stub configuration class with class-level annotations."""

    __annotations__ = {
        "existing": int,
        "str_anno": "str",
        "type_anno": float,
    }


class _HfSampleGenModel:
    """Stub HuggingFace model exposing a `generate` method."""

    def generate(self, input_ids: str = "") -> None:
        """Run stub generation with `input_ids`.

        Args:
            input_ids: Prompt token identifier string.
        """


class _HfLargeMod:
    """Large fake `transformers` module with >100 attributes."""

    def __dir__(self) -> list[str]:
        """Return attribute names exceeding the 100-item threshold.

        Returns:
            List of attribute names on the fake module.
        """
        names = [f"ignored_symbol_{i}" for i in range(105)]
        names.extend(
            [
                "_private_skip",
                "AutoConfig",
                "AutoModelWithGen",
                "PreTrainedModelValid",
                "CustomPipeline",
                "CustomTokenizer",
                "CustomTokenizerFast",
                "TFAutoModelValid",
                "FlaxAutoModelValid",
                "AutoScheduler",
                "pipeline_fn",
                "AutoRaiseGetattr",
                "AutoNone",
                "AutoReturnNoneRef",
                "AutoInspectExc",
            ]
        )
        return names

    def __getattr__(self, name: str) -> InspectableTarget:
        """Resolve dynamic attributes on the fake `transformers` module.

        Args:
            name: Requested attribute name.

        Raises:
            AttributeError: When `name` is a dunder or `'AutoRaiseGetattr'`.

        Returns:
            Simulated target object or `None`.
        """
        if name.startswith("__") or name == "AutoRaiseGetattr":
            raise AttributeError(name)
        if name == "AutoNone":
            return None
        if name == "AutoConfig":
            return _HfAnnotatedConfig
        if name == "AutoModelWithGen":
            return _HfSampleGenModel()
        return _mlx_noop


def _check_hf_helper_extractors(mocker: pytest_mock.MockerFixture) -> None:
    """Exercise `_extract_generation_kwargs`, `_parse_pretrained_config`, and `_handle_automodel_factory`.

    Args:
        mocker: Pytest mocker fixture.
    """
    ref = GhostRef(name="DummyGen", api_path="DummyGen", kind="class", params=[])
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.huggingface.inspect.signature",
        side_effect=TypeError("cannot inspect signature"),
    )
    hf_mod._extract_generation_kwargs(_HfDummyGenWithException(), ref)
    assert len(ref.params) == 0

    ref_gen = GhostRef(
        name="DummyGenFull",
        api_path="DummyGenFull",
        kind="class",
        params=[GhostParam(name="existing_param", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    mocker.stopall()
    hf_mod._extract_generation_kwargs(_HfDummyGenFull(), ref_gen)
    gen_param_names = {p.name for p in ref_gen.params}
    assert {"existing_param", "default_val", "no_default", "no_anno"} <= gen_param_names
    assert len([p for p in ref_gen.params if p.name == "existing_param"]) == 1

    ref2 = GhostRef(name="NoAnno", api_path="NoAnno", kind="class", params=[])
    hf_mod._parse_pretrained_config(_HfNoAnnotations(), ref2)
    assert len(ref2.params) == 0

    ref3 = GhostRef(
        name="AnnotatedConfig",
        api_path="AnnotatedConfig",
        kind="class",
        params=[GhostParam(name="existing", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    hf_mod._parse_pretrained_config(_HfAnnotatedConfig, ref3)
    param_names = {p.name for p in ref3.params}
    assert {"existing", "str_anno", "type_anno"} <= param_names
    assert len([p for p in ref3.params if p.name == "existing"]) == 1

    ref4 = GhostRef(
        name="AutoModel",
        api_path="AutoModel",
        kind="class",
        params=[GhostParam(name="config", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    hf_mod._handle_automodel_factory(None, "AutoModel", ref4)
    assert len([p for p in ref4.params if p.name == "config"]) == 1


def _mock_hf_inspect_large(obj: InspectableTarget, path: str) -> GhostRef | None:
    """Simulate `GhostInspector.inspect` for the fake `transformers` module.

    Args:
        obj: Inspected live object.
        path: Dotted API path string.

    Raises:
        RuntimeError: When `path` contains `'AutoInspectExc'`.

    Returns:
        Constructed `GhostRef` or `None`.
    """
    if "AutoInspectExc" in path:
        raise RuntimeError("simulated inspect error")
    if "AutoReturnNoneRef" in path:
        return None
    return GhostRef(
        name=path.split(".")[-1],
        api_path=path,
        kind="function",
        params=[],
    )


def test_huggingface_collect_api_all_branches(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Exercise `huggingface.collect_huggingface` and library entrypoints across all branches.

    Args:
        mocker: Pytest mocker fixture.
    """
    mapping = {SemanticTier.MODEL: ["Model"]}
    assert hf_mod.collect_huggingface("transformers", mapping, SemanticTier.OPTIMIZER) == []

    real_import = importlib.import_module

    def mock_import_err(name: str, package: str | None = None) -> types.ModuleType:
        """Raise `ImportError` when importing `transformers`.

        Args:
            name: Module name.
            package: Optional package anchor.

        Raises:
            ImportError: When `name` equals `'transformers'`.

        Returns:
            Imported module for other package names.
        """
        if name == "transformers":
            raise ImportError("simulated transformers error")
        return real_import(name, package)

    with patch("importlib.import_module", side_effect=mock_import_err):
        assert hf_mod.collect_huggingface("transformers", mapping, SemanticTier.MODEL) == []

    _check_hf_helper_extractors(mocker)

    def mock_import_large(name: str, package: str | None = None) -> types.ModuleType | _HfLargeMod:
        """Return `_HfLargeMod` when importing `transformers`.

        Args:
            name: Module name.
            package: Optional package anchor.

        Returns:
            Fake large `transformers` module or real module.
        """
        if name == "transformers":
            return _HfLargeMod()
        return real_import(name, package)

    mocker.patch.object(GhostInspector, "inspect", side_effect=_mock_hf_inspect_large)

    mapping_transformers = {
        SemanticTier.MODEL: [
            "Config",
            "Model",
            "Pipeline",
            "Tokenizer",
            "TFAuto",
            "FlaxAuto",
            "pipeline",
            "PreTrained",
        ],
        SemanticTier.OPTIMIZER: ["Scheduler"],
        SemanticTier.UTIL: ["other"],
    }
    with patch("importlib.import_module", side_effect=mock_import_large):
        res_large_model = hf_mod.collect_huggingface(
            "transformers",
            mapping_transformers,
            SemanticTier.MODEL,
            include_nonpublic=False,
        )
        names_large_model = {r.name for r in res_large_model}
        expected_models = {
            "AutoConfig",
            "AutoModelWithGen",
            "PreTrainedModelValid",
            "CustomPipeline",
            "CustomTokenizer",
            "CustomTokenizerFast",
            "TFAutoModelValid",
            "FlaxAutoModelValid",
        }
        assert expected_models <= names_large_model
        assert "ignored_symbol_0" not in names_large_model

        res_large_opt = hf_mod.collect_huggingface("transformers", mapping_transformers, SemanticTier.OPTIMIZER)
        assert "AutoScheduler" in {r.name for r in res_large_opt}

        res_large_util = hf_mod.collect_huggingface(
            "transformers",
            mapping_transformers,
            SemanticTier.UTIL,
            include_nonpublic=True,
        )
        assert "pipeline_fn" in {r.name for r in res_large_util}
        assert len(hf_mod.collect_transformers(SemanticTier.MODEL)) > 0

    fake_diff_lib = types.ModuleType("diffusers")
    pytest.MonkeyPatch().setattr(fake_diff_lib, "DiffusionPipeline", _mlx_noop, raising=False)
    fake_tok_lib = types.ModuleType("tokenizers")
    pytest.MonkeyPatch().setattr(fake_tok_lib, "TokenizerModel", _mlx_noop, raising=False)
    fake_libs = {"diffusers": fake_diff_lib, "tokenizers": fake_tok_lib}

    def mock_import_helpers(name: str, package: str | None = None) -> types.ModuleType:
        """Return fake `diffusers` or `tokenizers` modules on import.

        Args:
            name: Module name.
            package: Optional package anchor.

        Returns:
            Fake module if matched, otherwise real imported module.
        """
        if name in fake_libs:
            return fake_libs[name]
        return real_import(name, package)

    with patch("importlib.import_module", side_effect=mock_import_helpers):
        assert len(hf_mod.collect_diffusers(SemanticTier.MODEL)) >= 1
        assert len(hf_mod.collect_tokenizers(SemanticTier.MODEL)) >= 1


def test_tensorflow_collect_api_inspection_failures(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test tensorflow collect_api when GhostInspector.inspect raises exceptions across all branches.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_math = types.ModuleType("tf.math")
    pytest.MonkeyPatch().setattr(fake_math, "abs", lambda: None, raising=False)

    fake_linalg = types.ModuleType("tf.linalg")
    pytest.MonkeyPatch().setattr(fake_linalg, "matmul", lambda: None, raising=False)

    fake_raw_ops = types.ModuleType("tf.raw_ops")
    pytest.MonkeyPatch().setattr(fake_raw_ops, "raw_add", lambda: None, raising=False)

    class FakeTensor:
        """Fake Tensor class."""

        def get_shape(self) -> None:
            """Method docstring."""
            pass

    class FakeVariable:
        """Fake Variable class."""

        def assign(self) -> None:
            """Method docstring."""
            pass

    fake_tf = types.ModuleType("tf")
    pytest.MonkeyPatch().setattr(fake_tf, "math", fake_math, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "linalg", fake_linalg, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "raw_ops", fake_raw_ops, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "concat", lambda: None, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "custom_tensor_op", lambda: None, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "Tensor", FakeTensor, raising=False)
    pytest.MonkeyPatch().setattr(fake_tf, "Variable", FakeVariable, raising=False)

    mocker.patch.object(tf_fw, "tf", fake_tf)
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect",
        side_effect=Exception("inspection failure"),
    )

    res = tf_fw.collect_api(SemanticTier.ARRAY_API)
    assert res == []


def test_jax_import_reload_branches() -> None:
    """Test jax module reload when jax is successfully imported and when it fails.

    Returns:
        None.
    """
    fake_jax_mod = types.ModuleType("jax")
    with patch.dict("sys.modules", {"jax": fake_jax_mod}):
        importlib.reload(jax_fw)
        assert jax_fw.jax is fake_jax_mod

    with patch.dict("sys.modules", {"jax": None}):
        importlib.reload(jax_fw)
        assert jax_fw.jax is None


def test_jax_sharding_and_pallas_collection(mocker: pytest_mock.MockerFixture) -> None:
    """Test collection of jax.sharding and jax.experimental.pallas symbols.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """

    class Mesh:
        """Mock Mesh class."""

        pass

    def shard_map() -> None:
        """Mock shard_map function."""
        pass

    def _priv_sharding() -> None:
        """Private sharding function."""
        pass

    fake_sharding = types.ModuleType("jax.sharding")
    pytest.MonkeyPatch().setattr(fake_sharding, "Mesh", Mesh, raising=False)
    pytest.MonkeyPatch().setattr(fake_sharding, "shard_map", shard_map, raising=False)
    pytest.MonkeyPatch().setattr(fake_sharding, "_priv_sharding", _priv_sharding, raising=False)
    pytest.MonkeyPatch().setattr(fake_sharding, "not_callable", 42, raising=False)

    def pallas_call() -> None:
        """Mock pallas_call kernel."""
        pass

    def _priv_pallas() -> None:
        """Private pallas kernel."""
        pass

    fake_pallas = types.ModuleType("jax.experimental.pallas")
    pytest.MonkeyPatch().setattr(fake_pallas, "pallas_call", pallas_call, raising=False)
    pytest.MonkeyPatch().setattr(fake_pallas, "_priv_pallas", _priv_pallas, raising=False)
    pytest.MonkeyPatch().setattr(fake_pallas, "not_callable", 99, raising=False)

    fake_experimental = types.ModuleType("jax.experimental")
    pytest.MonkeyPatch().setattr(fake_experimental, "pallas", fake_pallas, raising=False)

    fake_jax = types.ModuleType("jax")
    pytest.MonkeyPatch().setattr(fake_jax, "sharding", fake_sharding, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "experimental", fake_experimental, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "lax", types.ModuleType("jax.lax"), raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "random", types.ModuleType("jax.random"), raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "Array", type("Array", (), {}), raising=False)

    fake_jnp = types.ModuleType("jax.numpy")
    fake_numpy = types.ModuleType("numpy")
    pytest.MonkeyPatch().setattr(fake_numpy, "float32", float, raising=False)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.sharding": fake_sharding,
            "jax.experimental": fake_experimental,
            "jax.experimental.pallas": fake_pallas,
            "jax.lax": fake_jax.lax,
            "jax.random": fake_jax.random,
            "jax.numpy": fake_jnp,
            "numpy": fake_numpy,
        },
    )

    # 1. Public symbols only
    refs = jax_fw._scan_array_api(include_nonpublic=False)
    paths = {r.api_path for r in refs}
    assert "jax.sharding.shard_map" in paths
    assert "jax.experimental.pallas.pallas_call" in paths
    assert "jax.sharding._priv_sharding" not in paths
    assert "jax.experimental.pallas._priv_pallas" not in paths

    # 2. Including non-public symbols
    refs_all = jax_fw._scan_array_api(include_nonpublic=True)
    paths_all = {r.api_path for r in refs_all}
    assert "jax.sharding._priv_sharding" in paths_all
    assert "jax.experimental.pallas._priv_pallas" in paths_all


def test_jax_array_ops_inspect_exceptions(mocker: pytest_mock.MockerFixture) -> None:
    """Test jax array ops collection when GhostInspector.inspect raises exceptions across all branches.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_jnp = types.ModuleType("jax.numpy")
    pytest.MonkeyPatch().setattr(fake_jnp, "abs", lambda: None, raising=False)

    fake_lax = types.ModuleType("jax.lax")
    pytest.MonkeyPatch().setattr(fake_lax, "add", lambda: None, raising=False)

    fake_random = types.ModuleType("jax.random")
    pytest.MonkeyPatch().setattr(fake_random, "uniform", lambda: None, raising=False)

    class FakeArray:
        """Fake Array class."""

        def reshape(self) -> None:
            """Method docstring."""
            pass

    fake_sharding = types.ModuleType("jax.sharding")
    pytest.MonkeyPatch().setattr(fake_sharding, "shard_map", lambda: None, raising=False)

    fake_pallas = types.ModuleType("jax.experimental.pallas")
    pytest.MonkeyPatch().setattr(fake_pallas, "pallas_call", lambda: None, raising=False)

    fake_experimental = types.ModuleType("jax.experimental")
    pytest.MonkeyPatch().setattr(fake_experimental, "pallas", fake_pallas, raising=False)

    fake_jax = types.ModuleType("jax")
    pytest.MonkeyPatch().setattr(fake_jax, "lax", fake_lax, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "random", fake_random, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "sharding", fake_sharding, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "experimental", fake_experimental, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "jit", lambda fn: fn, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "grad", lambda fn: fn, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "vmap", lambda fn: fn, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "pmap", lambda fn: fn, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "checkpoint", lambda fn: fn, raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "Array", FakeArray, raising=False)

    fake_numpy = types.ModuleType("numpy")
    pytest.MonkeyPatch().setattr(fake_numpy, "float32", float, raising=False)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            "jax.lax": fake_lax,
            "jax.random": fake_random,
            "jax.sharding": fake_sharding,
            "jax.experimental": fake_experimental,
            "jax.experimental.pallas": fake_pallas,
            "numpy": fake_numpy,
        },
    )

    def mock_inspect(obj: InspectableTarget, name: str, **kwargs: InspectableTarget) -> GhostRef:
        """Mock inspect function.

        Args:
            obj: Target live object to inspect.
            name: Dotted canonical API path.
            **kwargs: Variable arguments.

        Returns:
            A mocked GhostRef.
        """
        if name == "jax.numpy.float32":
            return GhostRef(name="float32", api_path="jax.numpy.float32", kind="class")
        raise RuntimeError("inspection failure")

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    refs = jax_fw._scan_array_api(include_nonpublic=False)
    assert len(refs) == 1
    assert refs[0].api_path == "jax.numpy.float32"


def test_jax_pallas_import_failure(mocker: pytest_mock.MockerFixture) -> None:
    """Test jax array ops collection when importing jax.experimental.pallas raises an exception.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    fake_jax = types.ModuleType("jax")
    pytest.MonkeyPatch().setattr(fake_jax, "lax", types.ModuleType("jax.lax"), raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "random", types.ModuleType("jax.random"), raising=False)
    pytest.MonkeyPatch().setattr(fake_jax, "Array", type("Array", (), {}), raising=False)
    fake_jnp = types.ModuleType("jax.numpy")
    fake_numpy = types.ModuleType("numpy")
    pytest.MonkeyPatch().setattr(fake_numpy, "float32", float, raising=False)

    mocker.patch.dict(
        "sys.modules",
        {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            "jax.lax": fake_jax.lax,
            "jax.random": fake_jax.random,
            "jax.experimental": None,
            "jax.experimental.pallas": None,
            "numpy": fake_numpy,
        },
    )

    refs = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    assert any(ref.api_path == "jax.numpy.float32" for ref in refs)


def test_jax_array_ops_import_error(mocker: pytest_mock.MockerFixture) -> None:
    """Test jax array ops collection when import jax raises ImportError.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        None.
    """
    mocker.patch.dict("sys.modules", {"jax": None})
    assert jax_fw.collect_api(SemanticTier.ARRAY_API) == []
