"""Tests for missing framework dependencies."""

import sys
import types
from typing import Any
from unittest.mock import patch, MagicMock
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier


def test_deepspeed_missing() -> None:
    """Verify DeepSpeed collector fallback and parameter extraction when DeepSpeed is absent or mocked."""
    from ml_ecosystem_snapshots.frameworks.deepspeed import collect_api

    with patch(
        "ml_ecosystem_snapshots.frameworks.deepspeed.importlib.import_module",
        side_effect=ImportError,
    ):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class DeepspeedMock:
        """Mock DeepSpeed module exposing initialize and _hidden attributes."""

        def __dir__(self) -> Any:
            """Return mock directory listing.

            Returns:
                List of mock member names.
            """
            return ["_hidden", "initialize", "nothing"]

        def _hidden(self, x: int = 1) -> int:
            """Mock private helper.

            Args:
                x: Integer parameter.

            Returns:
                Unchanged integer.
            """
            return x

        def initialize(self, model: Any = None) -> Any:
            """Mock initialize method without config_params so collector injects it.

            Args:
                model: Model object.
            """
            pass

        nothing = None

    with patch(
        "ml_ecosystem_snapshots.frameworks.deepspeed.importlib.import_module",
        return_value=DeepspeedMock(),
    ):
        res_pub = collect_api(SemanticTier.MODEL)
        assert len(res_pub) == 1
        assert res_pub[0].api_path == "deepspeed.initialize"
        assert any(p.name == "model" for p in res_pub[0].params)
        assert any(p.name == "config_params" for p in res_pub[0].params)
        res_nonpub = collect_api(SemanticTier.MODEL, include_nonpublic=True)
        assert len(res_nonpub) == 1


def test_onnxruntime_missing() -> None:
    """Verify ONNX Runtime collector fallback and InferenceSession extraction when mocked."""
    from ml_ecosystem_snapshots.frameworks.onnxruntime import collect_api
    from ml_ecosystem_snapshots.models import GhostInspector

    with patch(
        "ml_ecosystem_snapshots.frameworks.onnxruntime.importlib.import_module",
        side_effect=ImportError,
    ):
        assert collect_api(SemanticTier.MODEL) == []
    assert collect_api(SemanticTier.LOSS) == []

    class OnnxMock:
        """Mock ONNX Runtime module exposing InferenceSession."""

        def __dir__(self) -> Any:
            """Return mock directory listing.

            Returns:
                List of mock member names.
            """
            return [
                "_hidden",
                "InferenceSession",
                "SessionWithProviders",
                "get_available_providers",
                "nothing",
            ]

        def InferenceSession(self, path_or_bytes: str) -> None:
            """Mock InferenceSession factory without providers parameter.

            Args:
                path_or_bytes: Path or bytes to ONNX model.
            """
            return None

        def get_available_providers(self) -> Any:
            """Mock UTIL function in onnxruntime."""
            return []

        class SessionWithProviders:
            """Mock Session class already containing providers parameter."""

            def __init__(self, path_or_bytes: str, providers: Any = None) -> None:
                """Initialize session with providers parameter.

                Args:
                    path_or_bytes: Path or bytes to ONNX model.
                    providers: Execution providers sequence.
                """
                self.path = path_or_bytes
                self.providers = providers

        nothing = None

    with (
        patch(
            "griffe.load",
            side_effect=ImportError("no griffe"),
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.onnxruntime.importlib.import_module",
            return_value=OnnxMock(),
        ),
    ):
        res_pub = collect_api(SemanticTier.MODEL)
        assert len(res_pub) == 2
        inf_ref = next(r for r in res_pub if r.name == "InferenceSession")
        assert any(p.name == "path_or_bytes" for p in inf_ref.params)
        assert any(p.name == "providers" for p in inf_ref.params)
        res_nonpub = collect_api(SemanticTier.MODEL, include_nonpublic=True)
        assert len(res_nonpub) == 2
        res_util = collect_api(SemanticTier.UTIL)
        assert len(res_util) >= 1

    # Cover branch where InferenceSession already defines providers and when inspect returns None or raises
    class OnnxAlreadyHasProviders:
        """Mock ONNX Runtime module where InferenceSession already defines providers."""

        def __dir__(self) -> Any:
            """Return mock directory listing.

            Returns:
                List of mock member names.
            """
            return ["InferenceSession", "BadSession"]

        class InferenceSession:
            """Mock InferenceSession class with providers parameter."""

            def __init__(self, path_or_bytes: str, providers: Any = None) -> None:
                """Initialize session with providers parameter.

                Args:
                    path_or_bytes: Model path.
                    providers: Execution providers.
                """
                self.path = path_or_bytes
                self.providers = providers

        def BadSession(self) -> Any:
            """Mock session raising TypeError during inspection."""
            pass

    orig_inspect = GhostInspector.inspect

    def selective_inspect(obj: Any, name: str) -> Any:
        """Delegate to GhostInspector.inspect except for BadSession.

        Args:
            obj: Object to inspect.
            name: API path string.

        Returns:
            Extracted GhostRef.

        Raises:
            TypeError: When inspecting BadSession.
        """
        if "BadSession" in name:
            raise TypeError("broken session")
        return orig_inspect(obj, name)

    with patch.object(GhostInspector, "inspect", side_effect=selective_inspect):
        with patch(
            "ml_ecosystem_snapshots.frameworks.onnxruntime.importlib.import_module",
            return_value=OnnxAlreadyHasProviders(),
        ):
            res_prov = collect_api(SemanticTier.MODEL)
            assert len(res_prov) == 1
            assert sum(1 for p in res_prov[0].params if p.name == "providers") == 1

    with patch.object(GhostInspector, "inspect", return_value=None):
        with patch(
            "ml_ecosystem_snapshots.frameworks.onnxruntime.importlib.import_module",
            return_value=OnnxAlreadyHasProviders(),
        ):
            assert collect_api(SemanticTier.MODEL) == []


def test_triton_missing() -> None:
    """Verify Triton collector fallback and constexpr kernel annotation extraction."""
    from ml_ecosystem_snapshots.frameworks.triton import (
        _extract_triton_kernel,
        collect_api,
    )
    from ml_ecosystem_snapshots.models import GhostInspector

    with patch(
        "ml_ecosystem_snapshots.frameworks.triton.importlib.import_module",
        side_effect=ImportError,
    ):
        assert collect_api(SemanticTier.UTIL) == []
    assert collect_api(SemanticTier.LOSS) == []

    fake_tl = types.ModuleType("triton.language")
    fake_tl_math = types.ModuleType("triton.language.math")

    def fake_fn(a: Any, b: Any, c: Any) -> Any:
        """Mock Triton JIT function with constexpr annotation.

        Args:
            a: First operand with constexpr annotation.
            b: Second float operand.
            c: Unannotated third operand.
        """
        pass

    fake_fn.__annotations__ = {"a": "int constexpr", "b": "float"}

    class Wrap:
        """Callable wrapper class holding a Triton JIT function."""

        fn = staticmethod(fake_fn)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Invoke wrapped kernel function.

            Args:
                args: Positional arguments.
                kwargs: Keyword arguments.

            Returns:
                Result of `fake_fn`.
            """
            return fake_fn(*args, **kwargs)

    setattr(fake_tl, "kernel_op", Wrap())
    setattr(fake_tl, "_private_op", fake_fn)
    setattr(fake_tl, "non_callable", 123)
    setattr(fake_tl_math, "sin", fake_fn)
    setattr(fake_tl_math, "_priv_math", fake_fn)
    setattr(fake_tl_math, "non_callable", 456)
    setattr(fake_tl, "math", fake_tl_math)

    with patch(
        "ml_ecosystem_snapshots.frameworks.triton.importlib.import_module",
        return_value=fake_tl,
    ):
        res_triton = collect_api(SemanticTier.UTIL)
        assert len(res_triton) == 2
        kernel_ref = next(
            r for r in res_triton if r.api_path == "triton.language.kernel_op"
        )
        param_a = next(p for p in kernel_ref.params if p.name == "a")
        assert param_a.annotation == "tl.constexpr"

        res_nonpub = collect_api(SemanticTier.UTIL, include_nonpublic=True)
        assert len(res_nonpub) >= 4

    # Cover branch where tl.math is not a module
    fake_tl_no_math = types.ModuleType("triton.language")
    setattr(fake_tl_no_math, "kernel_op", Wrap())
    with patch(
        "ml_ecosystem_snapshots.frameworks.triton.importlib.import_module",
        return_value=fake_tl_no_math,
    ):
        assert len(collect_api(SemanticTier.UTIL)) == 1

    # Cover branches where _extract_triton_kernel returns None or triton.language raises ImportError
    def import_triton_only(name: str) -> Any:
        """Return fake_tl for triton and raise ImportError for triton.language.

        Args:
            name: Module name.

        Returns:
            Fake triton module.

        Raises:
            ImportError: When importing triton.language.
        """
        if name == "triton":
            return fake_tl
        raise ImportError("no triton.language")

    with patch(
        "ml_ecosystem_snapshots.frameworks.triton.importlib.import_module",
        side_effect=import_triton_only,
    ):
        assert collect_api(SemanticTier.UTIL) == []

    with patch.object(GhostInspector, "inspect", return_value=None):
        with patch(
            "ml_ecosystem_snapshots.frameworks.triton.importlib.import_module",
            return_value=fake_tl,
        ):
            assert collect_api(SemanticTier.UTIL) == []

    class NoAnno:
        """Kernel callable without annotations."""

        def __call__(self, x: Any) -> Any:
            """Execute kernel callable.

            Args:
                x: Input kernel argument.

            Returns:
                The unchanged input argument.
            """
            return x

    no_anno = NoAnno()
    ref_no_anno = _extract_triton_kernel(no_anno, "no_anno", "mod", GhostInspector())
    assert ref_no_anno is not None
    assert ref_no_anno.name == "no_anno"


def test_sklearn_missing() -> None:
    """Verify scikit-learn collector returns empty list when member discovery raises an exception."""
    from ml_ecosystem_snapshots.frameworks.sklearn import collect_api

    with patch(
        "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
        side_effect=Exception,
    ):
        res = collect_api(SemanticTier.LAYER)
        assert res == []


def test_huggingface_missing() -> None:
    """Verify HuggingFace Transformers collector handles missing packages, generation kwargs, and PretrainedConfig."""
    from ml_ecosystem_snapshots.frameworks.huggingface import (
        _extract_generation_kwargs,
        _parse_pretrained_config,
        collect_transformers,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    class M:
        """Mock generation-capable model class."""

        def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock generate method.

            Args:
                input: Input token ids.
                args: Variadic positional arguments.
                kwargs: Variadic keyword arguments.
            """
            pass

    r = GhostRef(name="A", api_path="A", kind="class", params=[])
    _extract_generation_kwargs(M(), r)
    assert any(p.name == "input" for p in r.params)

    class M2:
        """Mock PretrainedConfig class with type annotations."""

        __annotations__ = {"a": int, "b": "str"}

    r2 = GhostRef(name="A", api_path="A", kind="class", params=[])
    _parse_pretrained_config(M2(), r2)
    assert len(r2.params) == 2

    with patch("importlib.import_module", side_effect=ImportError):
        assert collect_transformers(SemanticTier.MODEL) == []

    assert collect_transformers(SemanticTier.LOSS) == []

    class BadProp:
        """Mock class with a property that raises ValueError on access."""

        @property
        def bad(self) -> Any:
            """Raise ValueError when accessed.

            Raises:
                ValueError: Always raised to test property access resilience.
            """
            raise ValueError()

    mock_mod = MagicMock()
    setattr(
        mock_mod,
        "__dir__",
        lambda self: [
            "_hidden",
            "DummyConfig",
            "AutoModel",
            "Other",
            "bad",
        ],
    )
    mock_mod.DummyConfig = type("DummyConfig", (), {})()  # No __annotations__
    mock_mod.AutoModel = MagicMock()

    # Mock getattr correctly so it raises for 'bad'
    # we need to simulate the module behavior
    class FakeMod:
        """Fake transformers module exposing config and AutoModel members."""

        def __dir__(self) -> Any:
            """Return list of fake module attributes.

            Returns:
                List of attribute names.
            """
            return [
                "_hidden",
                "DummyConfig",
                "AutoModelA",
                "AutoModelB",
                "Other",
                "bad",
                "ThrowConfig",
                "ReturnNone",
                "GenModel",
                "RefNone",
                "EmptyConfig",
                "EmptyConfigB",
            ]

        @property
        def bad(self) -> Any:
            """Raise ValueError on attribute access.

            Raises:
                ValueError: Simulated broken property.
            """
            raise ValueError()

        @property
        def DummyConfig(self) -> Any:
            """Return a dummy config instance with empty annotations.

            Returns:
                DummyConfig instance.
            """
            return type("DummyConfig", (), {"__annotations__": {}})()

        @property
        def EmptyConfig(self) -> Any:
            """Return an EmptyConfig instance with 'a' annotation.

            Returns:
                EmptyConfig instance.
            """
            return type("EmptyConfig", (), {"__annotations__": {"a": int}})()

        @property
        def EmptyConfigB(self) -> Any:
            """Return an EmptyConfigB instance with 'b' annotation.

            Returns:
                EmptyConfigB instance.
            """
            return type("EmptyConfigB", (), {"__annotations__": {"b": int}})()

        @property
        def AutoModelA(self) -> Any:
            """Return a mock AutoModelA object.

            Returns:
                MagicMock instance.
            """
            return MagicMock()

        @property
        def AutoModelB(self) -> Any:
            """Return a mock AutoModelB object.

            Returns:
                MagicMock instance.
            """
            return MagicMock()

        @property
        def Other(self) -> Any:
            """Return a mock Other object.

            Returns:
                MagicMock instance.
            """
            return MagicMock()

        @property
        def ThrowConfig(self) -> Any:
            """Return a mock ThrowConfig object.

            Returns:
                MagicMock instance.
            """
            return MagicMock()

        @property
        def ReturnNone(self) -> Any:
            """Return None for attribute lookup.

            Returns:
                None.
            """
            return None

        @property
        def GenModel(self) -> Any:
            """Return a model instance with a generate method.

            Returns:
                Mock model instance.
            """

            class M:
                """Mock inner model class with generate method."""

                def generate(self, input: Any, *args: Any, **kwargs: Any) -> Any:
                    """Mock generate method.

                    Args:
                        input: Input tensor.
                        args: Variadic positional arguments.
                        kwargs: Variadic keyword arguments.
                    """
                    pass

            return M()

        @property
        def RefNone(self) -> Any:
            """Return a mock RefNone object.

            Returns:
                MagicMock instance.
            """
            return MagicMock()

    def mock_inspect(obj: Any, name: Any) -> Any:
        """Return a mocked GhostRef or raise for ThrowConfig.

        Args:
            obj: The target object being inspected.
            name: The API path name.

        Raises:
            Exception: When inspecting ThrowConfig.

        Returns:
            A mocked GhostRef or None.
        """
        if "ThrowConfig" in name:
            raise RuntimeError("test")
        if "RefNone" in name:
            return None
        if "AutoModelA" in name:
            return MagicMock(params=[])
        if "AutoModelB" in name:
            return MagicMock(params=[MagicMock(**{"name": "config"})])
        if "GenModel" in name:
            return GhostRef(
                name="GenModel",
                api_path="GenModel",
                kind="class",
                params=[
                    GhostParam(
                        name="input",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation="Any",
                    )
                ],
            )
        if "EmptyConfig" in name:
            return MagicMock(params=[MagicMock(**{"name": "a"})])
        return MagicMock(params=[MagicMock(**{"name": "config"})])

    with patch(
        "ml_ecosystem_snapshots.models.GhostInspector.inspect",
        side_effect=mock_inspect,
    ):
        with patch("importlib.import_module", return_value=FakeMod()):
            models_res = collect_transformers(SemanticTier.MODEL)
            assert len(models_res) >= 1
            utils_res = collect_transformers(SemanticTier.UTIL)
            assert len(utils_res) >= 1


def test_hardware_exhaustive_fallbacks() -> None:
    """Test offline canonical fallbacks for all hardware/compiler targets when exhaustive files are missing."""
    import os
    from unittest.mock import patch
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_ecosystem_snapshots.frameworks import (
        amd_rdna,
        mlir,
        nvidia_ptx,
        nvidia_sass,
        stablehlo,
    )

    orig_exists = os.path.exists

    def fake_exists(p: str) -> bool:
        """Simulate missing exhaustive JSON dataset files.

        Args:
            p: Path to check.

        Returns:
            False if target is an exhaustive JSON file, otherwise delegates to real os.path.exists.
        """
        if any(p.endswith(s) for s in ["_exhaustive.json", "_exhaustive.json.gz"]):
            return False
        return orig_exists(p)

    with patch("os.path.exists", side_effect=fake_exists):
        rdna_refs = amd_rdna.collect_api(SemanticTier.UTIL)
        assert len(rdna_refs) > 0

        mlir_refs = mlir.collect_api(SemanticTier.UTIL)
        assert len(mlir_refs) > 0

        ptx_refs = nvidia_ptx.collect_api(SemanticTier.UTIL)
        assert len(ptx_refs) > 0

        shlo_refs = stablehlo.collect_api(SemanticTier.UTIL)
        assert len(shlo_refs) > 0

        sass_refs = nvidia_sass.collect_api(SemanticTier.UTIL)
        assert len(sass_refs) > 0


def test_orbax_checkpoint_inspect_error(mocker: Any) -> None:
    """Test orbax checkpoint exception handling during member inspection and member retrieval.

    Args:
        mocker: Pytest mocker fixture.
    """
    import ml_ecosystem_snapshots.frameworks.orbax_checkpoint as orbax_mod
    from ml_ecosystem_snapshots.models import GhostInspector
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    def bad_fn() -> None:
        """Dummy function for raising inspection errors."""
        pass

    mock_ocp = mocker.MagicMock()
    mock_ocp.bad_fn = bad_fn
    mocker.patch.object(orbax_mod, "ocp", mock_ocp)

    # 1. Inner exception handler (lines 39-40)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.get_all_members",
        return_value=[("bad_fn", bad_fn)],
    )
    mocker.patch.object(
        GhostInspector,
        "inspect",
        side_effect=Exception("inspection failed"),
    )
    res = orbax_mod.collect_api(SemanticTier.ARRAY_API)
    assert res == []

    # 2. Outer exception handler (lines 41-42)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.get_all_members",
        side_effect=RuntimeError("member enumeration error"),
    )
    res_outer = orbax_mod.collect_api(SemanticTier.ARRAY_API)
    assert res_outer == []


def test_pax_import_fallback(monkeypatch: Any) -> None:
    """Test pax import logic when praxis is available and unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import importlib
    import sys
    import types
    import ml_ecosystem_snapshots.frameworks.pax as pax_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # Available
    fake_paxml: Any = types.ModuleType("paxml")
    fake_praxis: Any = types.ModuleType("praxis")
    fake_layers: Any = types.ModuleType("praxis.layers")
    fake_praxis.layers = fake_layers
    monkeypatch.setitem(sys.modules, "paxml", fake_paxml)
    monkeypatch.setitem(sys.modules, "praxis", fake_praxis)
    monkeypatch.setitem(sys.modules, "praxis.layers", fake_layers)
    importlib.reload(pax_mod)
    assert getattr(pax_mod, "praxis") is fake_praxis

    # Unavailable
    monkeypatch.setitem(sys.modules, "paxml", None)
    importlib.reload(pax_mod)
    assert getattr(pax_mod, "praxis") is None
    assert pax_mod.collect_api(SemanticTier.MODEL) == []

    # Restore
    monkeypatch.undo()
    importlib.reload(pax_mod)


def test_sklearn_import_fallback(monkeypatch: Any) -> None:
    """Test sklearn import logic when sklearn is unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import importlib
    import sys
    import ml_ecosystem_snapshots.frameworks.sklearn as sklearn_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    monkeypatch.setitem(sys.modules, "sklearn", None)
    importlib.reload(sklearn_mod)
    assert getattr(sklearn_mod, "sklearn") is None
    assert sklearn_mod.collect_api(SemanticTier.MODEL) == []

    # Restore
    monkeypatch.undo()
    importlib.reload(sklearn_mod)


def test_flax_nnx_import_fallback(monkeypatch: Any, mocker: Any) -> None:
    """Test flax.nnx import fallback and inspection exception handling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import importlib
    import sys
    import types
    import ml_ecosystem_snapshots.frameworks.flax_nnx as nnx_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    # 1. Unavailable import
    monkeypatch.setitem(sys.modules, "flax", None)
    monkeypatch.setitem(sys.modules, "flax.nnx", None)
    importlib.reload(nnx_mod)
    assert nnx_mod.nnx is None
    assert nnx_mod.collect_api(SemanticTier.LAYER) == []

    # 2. Available with issubclass TypeError
    fake_nnx: Any = types.ModuleType("flax.nnx")

    class BadClass:
        """Dummy uninheritable class to trigger TypeError in issubclass."""

        pass

    fake_nnx.BadClass = BadClass
    fake_nnx.Module = (
        123  # Non-class causes issubclass(BadClass, 123) to raise TypeError
    )
    monkeypatch.setitem(sys.modules, "flax", types.ModuleType("flax"))
    monkeypatch.setitem(sys.modules, "flax.nnx", fake_nnx)
    importlib.reload(nnx_mod)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.flax_nnx.get_all_members",
        return_value=[("BadClass", BadClass)],
    )
    res = nnx_mod.collect_api(SemanticTier.LAYER)
    assert res == []

    # Restore
    monkeypatch.undo()
    importlib.reload(nnx_mod)


def test_deepspeed_fallback_edges(mocker: Any) -> None:
    """Test deepspeed inspect exception handling and existing config_params.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types
    import ml_ecosystem_snapshots.frameworks.deepspeed as ds_mod
    from ml_ecosystem_snapshots.models import GhostInspector
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, SemanticTier

    fake_ds: Any = types.ModuleType("deepspeed")
    fake_ds.initialize = lambda: None
    fake_ds.none_ref = lambda: None
    fake_ds.error_op = lambda: None
    fake_ds.z_extra_util = lambda: None
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.deepspeed.importlib.import_module",
        return_value=fake_ds,
    )

    def mock_inspect(obj: Any, name: str) -> Any:
        """Mock inspect function raising ValueError for error_op.

        Args:
            obj: Target object being inspected.
            name: Fully qualified API name.

        Returns:
            A mocked GhostRef.

        Raises:
            ValueError: When inspecting `error_op`.
        """
        if "error_op" in name:
            raise ValueError("simulated error")
        if "none_ref" in name:
            return None
        if "initialize" in name:
            return GhostRef(
                name="initialize",
                api_path="deepspeed.initialize",
                kind="function",
                params=[
                    GhostParam(name="config_params", kind=ParameterKind.KEYWORD_ONLY)
                ],
            )
        return GhostRef(
            name="z_extra_util",
            api_path="deepspeed.z_extra_util",
            kind="function",
            params=[],
        )

    mocker.patch.object(
        GhostInspector,
        "inspect",
        side_effect=mock_inspect,
    )
    # MODEL category checks initialize with config_params and non-matching UTIL attrs
    res_model = ds_mod.collect_api(SemanticTier.MODEL)
    assert len(res_model) == 1

    # UTIL category triggers none_ref (branch 53 -> 36), inspect exception on error_op, and appends z_extra_util
    res_util = ds_mod.collect_api(SemanticTier.UTIL)
    assert len(res_util) == 1
    assert res_util[0].name == "z_extra_util"


def test_optax_shim_import_reload() -> None:
    """Test optax_shim module reload handling when optax is missing and present.

    Returns:
        None.
    """
    import importlib
    from ml_ecosystem_snapshots.frameworks import optax_shim

    with patch.dict("sys.modules", {"optax": None}):
        importlib.reload(optax_shim)
        assert getattr(optax_shim, "optax") is None

    # Restore clean reload
    importlib.reload(optax_shim)
    import ml_ecosystem_snapshots.frameworks.jax as jax_fw

    importlib.reload(jax_fw)


def test_orbax_checkpoint_import_reload() -> None:
    """Test orbax_checkpoint module reload handling when orbax.checkpoint is missing and present.

    Returns:
        None.
    """
    import importlib
    from ml_ecosystem_snapshots.frameworks import orbax_checkpoint

    with patch.dict("sys.modules", {"orbax.checkpoint": None, "orbax": None}):
        importlib.reload(orbax_checkpoint)
        assert getattr(orbax_checkpoint, "ocp") is None

    # Restore clean reload
    importlib.reload(orbax_checkpoint)


def test_tensorflow_import_reload() -> None:
    """Test tensorflow module reload handling when tensorflow is missing and present.

    Returns:
        None.
    """
    import builtins
    import importlib
    import types
    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    with patch.dict("sys.modules", {"tensorflow": None}):
        importlib.reload(tf_fw)
        assert getattr(tf_fw, "tf") is None

    # Test _get_tf exception handler
    tf_fw.tf = None
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import function.

        Args:
            name: The module name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The imported module or raises.
        """
        """Mock import raising ImportError on tensorflow.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing tensorflow.
        """
        if name == "tensorflow":
            raise ImportError("simulated tf error")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules):
        sys.modules.pop("tensorflow", None)
        with patch("builtins.__import__", side_effect=mock_import):
            assert tf_fw._get_tf() is None
            assert tf_fw.collect_api(SemanticTier.ACTIVATION) == []

    # Test tf without nn attribute
    fake_tf_no_nn = types.ModuleType("tensorflow")
    with patch.object(tf_fw, "tf", fake_tf_no_nn):
        assert (
            tf_fw._collect_live(SemanticTier.ACTIVATION, include_nonpublic=False) == []
        )

    # Restore clean reload
    importlib.reload(tf_fw)


def test_tensorflow_collect_live_image_lookup(monkeypatch: Any) -> None:
    """Verify _collect_live inspects tf.image and tf.lookup members.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import types
    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw
    from ml_switcheroo_ir.schema.ghost import SemanticTier
    from ml_ecosystem_snapshots.models import GhostInspector

    class MockLookupTable:
        """Mock lookup table class."""

        pass

    def mock_resize(x: Any) -> Any:
        """Mock resize function.

        Args:
            x: Input parameter.

        Returns:
            Input value.
        """
        return x

    def mock_private(x: Any) -> Any:
        """Mock private function.

        Args:
            x: Input parameter.

        Returns:
            Input value.
        """
        return x

    fake_image = types.SimpleNamespace(
        resize=mock_resize,
        _private=mock_private,
        not_callable=42,
    )
    fake_lookup = types.SimpleNamespace(
        StaticHashTable=MockLookupTable,
        _private_table=MockLookupTable,
        not_callable=99,
    )
    fake_nn = types.SimpleNamespace(
        relu=mock_resize,
    )
    fake_tf = types.SimpleNamespace(
        nn=fake_nn,
        image=fake_image,
        lookup=fake_lookup,
        math=types.SimpleNamespace(),
        linalg=types.SimpleNamespace(),
    )

    monkeypatch.setattr(tf_fw, "tf", fake_tf)

    # 1. include_nonpublic=False
    refs_public = tf_fw._collect_live(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert any("tf.image.resize" in r.api_path for r in refs_public)
    assert any("tf.lookup.StaticHashTable" in r.api_path for r in refs_public)
    assert any("tf.nn.relu" in r.api_path for r in refs_public)
    assert not any("tf.image._private" in r.api_path for r in refs_public)
    assert not any("tf.lookup._private_table" in r.api_path for r in refs_public)

    # 2. include_nonpublic=True
    refs_nonpublic = tf_fw._collect_live(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert any("tf.image._private" in r.api_path for r in refs_nonpublic)
    assert any("tf.lookup._private_table" in r.api_path for r in refs_nonpublic)

    # 3. Exception in inspect
    def mock_broken_inspect(obj: Any, name: str, **kwargs: Any) -> Any:
        """Mock broken inspect raising on image, lookup, and nn.

        Args:
            obj: Inspected object.
            name: Fully qualified target name.
            **kwargs: Extra inspect kwargs.

        Returns:
            GhostRef object.

        Raises:
            RuntimeError: If target belongs to image, lookup, or nn.
        """
        if "image" in name or "lookup" in name or "nn" in name:
            raise RuntimeError("Inspect failure")
        return GhostInspector.inspect(obj, name, **kwargs)

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.tensorflow.GhostInspector.inspect",
        mock_broken_inspect,
    )
    refs_err = tf_fw._collect_live(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert not any("tf.image.resize" in r.api_path for r in refs_err)
    assert not any("tf.nn.relu" in r.api_path for r in refs_err)


def test_tensorflow_get_tf_exception(monkeypatch: Any) -> None:
    """Verify _get_tf returns None when importing tensorflow raises Exception.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins
    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    monkeypatch.setattr(tf_fw, "tf", None)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    real_import = builtins.__import__

    def failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Failing import for tensorflow.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Module or raises RuntimeError.
        """
        if name == "tensorflow":
            raise RuntimeError("TF boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert tf_fw._get_tf() is None

    # Test _get_tf when import succeeds
    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setattr(tf_fw, "tf", None)
    fake_tf_success = types.ModuleType("tensorflow")
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf_success)
    assert tf_fw._get_tf() is fake_tf_success
