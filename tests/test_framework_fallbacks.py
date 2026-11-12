"""Tests for missing framework dependencies and live/static collector fallbacks."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from conftest import FakePackageBuilder
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind, SemanticTier

from ml_ecosystem_snapshots.frameworks import (
    amd_rdna,
    deepspeed,
    flax_nnx,
    huggingface,
    mlir,
    nvidia_ptx,
    nvidia_sass,
    onnxruntime,
    orbax_checkpoint,
    stablehlo,
    triton,
)
from ml_ecosystem_snapshots.frameworks import jax as jax_fw
from ml_ecosystem_snapshots.frameworks import optax as optax_fw
from ml_ecosystem_snapshots.frameworks import pax as pax_mod
from ml_ecosystem_snapshots.frameworks import sklearn as sklearn_mod
from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget, get_member_dict
from ml_ecosystem_snapshots.models import ExtendedGhostRef


def test_deepspeed_missing_and_fallback_edges(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify DeepSpeed collector handles missing package, parameter injection, and inspect errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.block_imports("deepspeed")
    assert deepspeed.collect_api(SemanticTier.MODEL) == []
    assert deepspeed.collect_api(SemanticTier.LOSS) == []

    def _initialize(model: LivePythonTarget = None) -> None:
        """Simulate `deepspeed.initialize` without `config_params`.

        Args:
            model: Model instance.
        """
        _ = model

    def _hidden(x: int = 1) -> int:
        """Simulate private helper function.

        Args:
            x: Integer input.

        Returns:
            Unchanged integer.
        """
        return x

    fake_package_builder.build_module(
        "deepspeed",
        members={"initialize": _initialize, "_hidden": _hidden, "nothing": None},
    )
    res_pub = deepspeed.collect_api(SemanticTier.MODEL)
    assert len(res_pub) == 1
    assert res_pub[0].api_path == "deepspeed.initialize"
    assert {p.name for p in res_pub[0].params} >= {"model", "config_params"}
    assert len(deepspeed.collect_api(SemanticTier.MODEL, include_nonpublic=True)) == 1

    # Cover existing config_params, inspect -> None, and inspect raising ValueError
    fake_package_builder.build_module(
        "deepspeed",
        members={
            "initialize": lambda: None,
            "none_ref": lambda: None,
            "error_op": lambda: None,
            "z_extra_util": lambda: None,
        },
    )

    def _mock_ds_inspect(self_insp: GhostInspector, _obj: LivePythonTarget, name: str) -> GhostRef | None:
        """Simulate `GhostInspector.inspect` edge cases for DeepSpeed members.

        Args:
            self_insp: Active `GhostInspector` instance.
            _obj: Target object.
            name: Fully qualified API path.

        Returns:
            Simulated `GhostRef` or `None`.

        Raises:
            ValueError: When `error_op` is inspected.
        """
        _ = self_insp
        if "error_op" in name:
            raise ValueError("simulated error")
        if "none_ref" in name:
            return None
        if "initialize" in name:
            return GhostRef(
                name="initialize",
                api_path="deepspeed.initialize",
                kind="function",
                params=[GhostParam(name="config_params", kind=ParameterKind.KEYWORD_ONLY)],
            )
        return GhostRef(name="z_extra_util", api_path="deepspeed.z_extra_util", kind="function", params=[])

    monkeypatch.setattr(GhostInspector, "inspect", _mock_ds_inspect)
    assert len(deepspeed.collect_api(SemanticTier.MODEL)) == 1
    res_util = deepspeed.collect_api(SemanticTier.UTIL)
    assert [r.name for r in res_util] == ["z_extra_util"]


def test_onnxruntime_missing_and_providers_injection(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify ONNX Runtime collector fallback and `InferenceSession` `providers` parameter handling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.block_imports("onnxruntime")
    assert onnxruntime.collect_api(SemanticTier.MODEL) == []
    assert onnxruntime.collect_api(SemanticTier.LOSS) == []

    def InferenceSession(path_or_bytes: str) -> None:
        """Simulate `InferenceSession` factory without `providers` parameter.

        Args:
            path_or_bytes: Path or bytes to ONNX model.
        """
        _ = path_or_bytes

    def get_available_providers() -> list[str]:
        """Simulate `onnxruntime.get_available_providers`.

        Returns:
            Empty list of providers.
        """
        return []

    class SessionWithProviders:
        """Simulated session class already defining `providers`."""

        def __init__(self, path_or_bytes: str, providers: list[str] | None = None) -> None:
            """Initialize session with `providers`.

            Args:
                path_or_bytes: Model path.
                providers: Optional execution providers.
            """
            self.path = path_or_bytes
            self.providers = providers

    fake_package_builder.build_module(
        "onnxruntime",
        members={
            "_hidden": lambda: None,
            "InferenceSession": InferenceSession,
            "SessionWithProviders": SessionWithProviders,
            "get_available_providers": get_available_providers,
            "nothing": None,
        },
    )
    res_pub = onnxruntime.collect_api(SemanticTier.MODEL)
    assert len(res_pub) == 2
    inf_ref = next(r for r in res_pub if r.name == "InferenceSession")
    assert {p.name for p in inf_ref.params} >= {"path_or_bytes", "providers"}
    assert len(onnxruntime.collect_api(SemanticTier.MODEL, include_nonpublic=True)) == 2
    assert len(onnxruntime.collect_api(SemanticTier.UTIL)) >= 1

    # Cover InferenceSession already defining `providers`, inspect raising TypeError, and inspect returning None
    fake_package_builder.build_module(
        "onnxruntime",
        members={"InferenceSession": SessionWithProviders, "BadSession": lambda: None},
    )
    orig_inspect = GhostInspector.inspect

    def _selective_inspect(self_insp: GhostInspector, obj: LivePythonTarget, name: str) -> ExtendedGhostRef:
        """Raise `TypeError` on `BadSession` while delegating valid classes to `GhostInspector.inspect`.

        Args:
            self_insp: Active `GhostInspector` instance.
            obj: Target object.
            name: API path string.

        Returns:
            Inspected `ExtendedGhostRef`.

        Raises:
            TypeError: When `BadSession` is inspected.
        """
        _ = self_insp
        if "BadSession" in name:
            raise TypeError("broken session")
        return orig_inspect(obj, name)

    with monkeypatch.context() as mp:
        mp.setattr(GhostInspector, "inspect", _selective_inspect)
        res_prov = onnxruntime.collect_api(SemanticTier.MODEL)
        assert len(res_prov) == 1
        assert sum(1 for p in res_prov[0].params if p.name == "providers") == 1

    with monkeypatch.context() as mp:
        mp.setattr(GhostInspector, "inspect", lambda _self, _obj, _name: None)
        assert onnxruntime.collect_api(SemanticTier.MODEL) == []


def test_triton_missing_and_constexpr_kernels(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify Triton collector fallback and `tl.constexpr` kernel annotation normalization.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.block_imports("triton", "triton.language")
    assert triton.collect_api(SemanticTier.UTIL) == []
    assert triton.collect_api(SemanticTier.LOSS) == []

    def _fake_fn(a: LivePythonTarget, b: LivePythonTarget, c: LivePythonTarget) -> None:
        """Simulate Triton JIT kernel with constexpr parameter annotation.

        Args:
            a: Constexpr operand.
            b: Float operand.
            c: Unannotated operand.
        """
        _ = (a, b, c)

    monkeypatch.setattr(_fake_fn, "__annotations__", {"a": "int constexpr", "b": "float"})

    class KernelWrapper:
        """Callable wrapper holding `.fn` like `triton.runtime.JITFunction`."""

        fn = staticmethod(_fake_fn)

        def __call__(self, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> None:
            """Forward invocation to `_fake_fn`.

            Args:
                *args: Positional kernel arguments.
                **kwargs: Keyword kernel arguments.
            """
            _fake_fn(*args, **kwargs)

    class NoAnnoKernel:
        """Callable kernel whose `__annotations__` attribute is `None`."""

        def __call__(self, x: LivePythonTarget) -> LivePythonTarget:
            """Return `x` unchanged.

            Args:
                x: Input value.

            Returns:
                Unchanged input value.
            """
            return x

    no_anno = NoAnnoKernel()
    monkeypatch.setattr(no_anno, "__annotations__", None, raising=False)

    tl_math = fake_package_builder.build_module(
        "triton.language.math",
        members={"sin": _fake_fn, "_priv_math": _fake_fn, "non_callable": 456},
    )
    tl_mod = fake_package_builder.build_module(
        "triton.language",
        members={
            "kernel_op": KernelWrapper(),
            "no_anno_op": no_anno,
            "_private_op": _fake_fn,
            "non_callable": 123,
        },
        submodules={"math": tl_math},
    )
    fake_package_builder.build_module("triton", submodules={"language": tl_mod})

    res_triton = triton.collect_api(SemanticTier.UTIL)
    kernel_ref = next(r for r in res_triton if r.api_path == "triton.language.kernel_op")
    assert next(p for p in kernel_ref.params if p.name == "a").annotation == "tl.constexpr"
    assert any(r.name == "no_anno_op" for r in res_triton)
    assert len(triton.collect_api(SemanticTier.UTIL, include_nonpublic=True)) >= 5

    # `triton.language` without `math` submodule
    tl_no_math = fake_package_builder.build_module(
        "triton.language",
        members={"kernel_op": KernelWrapper()},
    )
    fake_package_builder.build_module("triton", submodules={"language": tl_no_math})
    assert len(triton.collect_api(SemanticTier.UTIL)) == 1

    # `triton` present while `triton.language` blocked
    fake_package_builder.build_module("triton")
    fake_package_builder.block_imports("triton.language")
    assert triton.collect_api(SemanticTier.UTIL) == []


def test_huggingface_transformers_missing_and_augmentation(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify HuggingFace `collect_transformers` handles missing package, `PretrainedConfig`, `AutoModel`, and `generate`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.block_imports("transformers")
    assert huggingface.collect_transformers(SemanticTier.MODEL) == []
    assert huggingface.collect_transformers(SemanticTier.LOSS) == []

    class SampleConfig:
        """Simulated `PreTrainedConfig` subclass with typed class annotations."""

        vocab_size: int
        model_type: str

        def __init__(self, vocab_size: int = 32000) -> None:
            """Initialize sample configuration.

            Args:
                vocab_size: Vocabulary size.
            """
            self.vocab_size = vocab_size

    class NoDictAnnotationsConfig:
        """Simulated config class with non-dict `__annotations__`."""

        pass

    monkeypatch.setattr(NoDictAnnotationsConfig, "__annotations__", None, raising=False)

    class AutoModelSample:
        """Simulated `AutoModel` factory class without explicit `config` parameter."""

        def __init__(self) -> None:
            """Initialize `AutoModelSample`."""

    class CausalGenModel:
        """Simulated causal LM model exposing `generate`."""

        def __init__(self, hidden_size: int = 768) -> None:
            """Initialize `CausalGenModel`.

            Args:
                hidden_size: Hidden dimension.
            """
            self.hidden_size = hidden_size

        def generate(
            self,
            input_ids: LivePythonTarget,
            max_new_tokens: int = 32,
            *args: LivePythonTarget,
            **kwargs: LivePythonTarget,
        ) -> None:
            """Generate token sequence.

            Args:
                input_ids: Prompt token IDs.
                max_new_tokens: Maximum tokens to generate.
                *args: Variadic positional arguments.
                **kwargs: Variadic keyword arguments.
            """
            _ = (input_ids, max_new_tokens, args, kwargs)

    class BrokenGenModel:
        """Model class whose `generate` attribute is not inspectable."""

        generate = 123

    def helper_util(x: int) -> int:
        """Utility function classified under `SemanticTier.UTIL`.

        Args:
            x: Integer input.

        Returns:
            Unchanged integer.
        """
        return x

    fake_package_builder.build_module(
        "transformers",
        members={
            "_hidden": helper_util,
            "NoneAttr": None,
            "SampleConfig": SampleConfig,
            "NoDictAnnotationsConfig": NoDictAnnotationsConfig,
            "AutoModelSample": AutoModelSample,
            "CausalGenModel": CausalGenModel,
            "BrokenGenModel": BrokenGenModel,
            "ThrowModel": lambda: None,
            "NoneRefModel": lambda: None,
            "helper_util": helper_util,
        },
    )

    orig_inspect = GhostInspector.inspect

    def _hf_inspect(self_insp: GhostInspector, obj: LivePythonTarget, name: str) -> GhostRef | None:
        """Simulate `ThrowModel` exception and `NoneRefModel` returning `None` while inspecting real classes.

        Args:
            self_insp: Active `GhostInspector` instance.
            obj: Target object.
            name: Qualified API path.

        Returns:
            Inspected `GhostRef` or `None`.

        Raises:
            RuntimeError: When inspecting `ThrowModel`.
        """
        _ = self_insp
        if "ThrowModel" in name:
            raise RuntimeError("simulated hf error")
        if "NoneRefModel" in name:
            return None
        return orig_inspect(obj, name)

    monkeypatch.setattr(GhostInspector, "inspect", _hf_inspect)
    models_res = huggingface.collect_transformers(SemanticTier.MODEL)
    by_name = {r.name: r for r in models_res}
    assert {"SampleConfig", "AutoModelSample", "CausalGenModel"} <= set(by_name)
    assert "model_type" in {p.name for p in by_name["SampleConfig"].params}
    assert "config" in {p.name for p in by_name["AutoModelSample"].params}
    assert "max_new_tokens" in {p.name for p in by_name["CausalGenModel"].params}
    assert any(r.name == "helper_util" for r in huggingface.collect_transformers(SemanticTier.UTIL))


def test_hardware_exhaustive_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify canonical fallbacks for all hardware/compiler targets when exhaustive JSON files are absent.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    orig_exists = os.path.exists
    _ = tmp_path

    def _hide_exhaustive(path_str: str) -> bool:
        """Simulate missing `*_exhaustive.json` files while delegating other paths.

        Args:
            path_str: Path to check.

        Returns:
            False for exhaustive JSON paths, otherwise `os.path.exists(path_str)`.
        """
        if path_str.endswith(("_exhaustive.json", "_exhaustive.json.gz")):
            return False
        return orig_exists(path_str)

    monkeypatch.setattr("os.path.exists", _hide_exhaustive)
    for mod in (amd_rdna, mlir, nvidia_ptx, nvidia_sass, stablehlo):
        assert len(mod.collect_api(SemanticTier.UTIL)) > 0


def test_sklearn_pax_and_flax_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `sklearn`, `pax`, and `flax_nnx` reload and member-enumeration fallbacks.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    with monkeypatch.context() as mp:
        mp.setattr(
            "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
            lambda _m: (_ for _ in ()).throw(RuntimeError("err")),
        )
        assert sklearn_mod.collect_api(SemanticTier.LAYER) == []

    with monkeypatch.context() as mp:
        mp.setitem(os.environ, "ML_SNAPSHOTS_OFFLINE", "0")
        fb = FakePackageBuilder(mp)
        fb.block_imports("sklearn")
        importlib.reload(sklearn_mod)
        assert sklearn_mod.sklearn is None
        assert sklearn_mod.collect_api(SemanticTier.MODEL) == []
    importlib.reload(sklearn_mod)

    with monkeypatch.context() as mp:
        fb = FakePackageBuilder(mp)
        fake_layers = fb.build_module("praxis.layers")
        fake_praxis = fb.build_module("praxis", submodules={"layers": fake_layers})
        fb.build_module("paxml")
        importlib.reload(pax_mod)
        assert get_member_dict(pax_mod).get("praxis") is fake_praxis

        fb.block_imports("paxml")
        importlib.reload(pax_mod)
        assert get_member_dict(pax_mod).get("praxis") is None
        assert pax_mod.collect_api(SemanticTier.MODEL) == []
    importlib.reload(pax_mod)

    with monkeypatch.context() as mp:
        fb = FakePackageBuilder(mp)
        fb.block_imports("flax", "flax.nnx")
        importlib.reload(flax_nnx)
        assert flax_nnx.nnx is None
        assert flax_nnx.collect_api(SemanticTier.LAYER) == []

        class BadClass:
            """Candidate class tested against invalid `nnx.Module = 123`."""

        fake_nnx = fb.build_module("flax.nnx", members={"BadClass": BadClass, "Module": 123})
        fb.build_module("flax", submodules={"nnx": fake_nnx})
        importlib.reload(flax_nnx)
        mp.setattr("ml_ecosystem_snapshots.frameworks.flax_nnx.get_all_members", lambda _m: [("BadClass", BadClass)])
        assert flax_nnx.collect_api(SemanticTier.LAYER) == []
    importlib.reload(flax_nnx)


def test_orbax_and_optax_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `orbax_checkpoint` and `optax` reload and member-inspection fallbacks.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_ocp = fake_package_builder.build_module("orbax.checkpoint", members={"bad_fn": lambda: None})
    fake_package_builder.build_module("orbax", submodules={"checkpoint": fake_ocp})
    with monkeypatch.context() as mp:
        mp.setattr(orbax_checkpoint, "ocp", fake_ocp)
        mp.setattr(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.get_all_members",
            lambda _m: [("bad_fn", lambda: None)],
        )
        mp.setattr(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.GhostInspector.inspect",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("inspection failed")),
        )
        assert len(orbax_checkpoint.collect_api(SemanticTier.ARRAY_API)) == len(orbax_checkpoint.CANONICAL_ORBAX_OPS)

        mp.setattr(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.get_all_members",
            lambda _m: (_ for _ in ()).throw(RuntimeError("member enumeration error")),
        )
        assert len(orbax_checkpoint.collect_api(SemanticTier.ARRAY_API)) == len(orbax_checkpoint.CANONICAL_ORBAX_OPS)

    with monkeypatch.context() as mp:
        fb = FakePackageBuilder(mp)
        fb.block_imports("orbax", "orbax.checkpoint")
        importlib.reload(orbax_checkpoint)
        assert get_member_dict(orbax_checkpoint).get("ocp") is None

        reloaded_ocp = fb.build_module("orbax.checkpoint")
        fb.build_module("orbax", submodules={"checkpoint": reloaded_ocp})
        importlib.reload(orbax_checkpoint)
        assert get_member_dict(orbax_checkpoint).get("ocp") is reloaded_ocp
    importlib.reload(orbax_checkpoint)

    with monkeypatch.context() as mp:
        fb = FakePackageBuilder(mp)
        fb.block_imports("optax")
        importlib.reload(optax_fw)
        assert get_member_dict(optax_fw).get("optax") is None

        mock_optax = fb.build_module("optax")
        importlib.reload(optax_fw)
        assert get_member_dict(optax_fw).get("optax") is mock_optax
    importlib.reload(optax_fw)
    importlib.reload(jax_fw)


def test_tensorflow_live_and_import_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify TensorFlow collector handles missing module, missing submodules, `image`/`lookup` tiers, and errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    with monkeypatch.context() as mp:
        fb = FakePackageBuilder(mp)
        fb.block_imports("tensorflow")
        importlib.reload(tf_fw)
        assert get_member_dict(tf_fw).get("tf") is None
        assert tf_fw.collect_api(SemanticTier.ACTIVATION) == []
    importlib.reload(tf_fw)

    # Lazy import success via public `collect_api` when `tf_fw.tf` starts as None
    empty_tf = fake_package_builder.build_module("tensorflow")
    with monkeypatch.context() as mp:
        mp.setattr(tf_fw, "tf", None)
        assert tf_fw.collect_api(SemanticTier.ACTIVATION) == []
        assert get_member_dict(tf_fw).get("tf") is empty_tf

    class MockLookupTable:
        """Simulated `tf.lookup.StaticHashTable` class."""

    def _mock_op(x: LivePythonTarget) -> LivePythonTarget:
        """Simulated public TensorFlow callable.

        Args:
            x: Input tensor.

        Returns:
            Unchanged input tensor.
        """
        return x

    def _mock_private(x: LivePythonTarget) -> LivePythonTarget:
        """Simulated private TensorFlow callable.

        Args:
            x: Input tensor.

        Returns:
            Unchanged input tensor.
        """
        return x

    tf_image = fake_package_builder.build_module(
        "tensorflow.image",
        members={"resize": _mock_op, "_private": _mock_private, "not_callable": 42},
    )
    tf_lookup = fake_package_builder.build_module(
        "tensorflow.lookup",
        members={"StaticHashTable": MockLookupTable, "_private_table": MockLookupTable, "not_callable": 99},
    )
    tf_nn = fake_package_builder.build_module("tensorflow.nn", members={"relu": _mock_op})
    tf_math = fake_package_builder.build_module("tensorflow.math")
    tf_linalg = fake_package_builder.build_module("tensorflow.linalg")
    full_tf = fake_package_builder.build_module(
        "tensorflow",
        submodules={"nn": tf_nn, "image": tf_image, "lookup": tf_lookup, "math": tf_math, "linalg": tf_linalg},
    )
    monkeypatch.setattr(tf_fw, "tf", full_tf)

    refs_pub = tf_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    pub_paths = {r.api_path for r in refs_pub}
    assert {"tf.image.resize", "tf.lookup.StaticHashTable", "tf.nn.relu"} <= pub_paths
    assert "tf.image._private" not in pub_paths
    assert "tf.lookup._private_table" not in pub_paths

    refs_nonpub = tf_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    nonpub_paths = {r.api_path for r in refs_nonpub}
    assert {"tf.image._private", "tf.lookup._private_table"} <= nonpub_paths

    orig_inspect = GhostInspector.inspect

    def _broken_tf_inspect(obj: LivePythonTarget, name: str, **kwargs: LivePythonTarget) -> ExtendedGhostRef:
        """Raise `RuntimeError` when inspecting `image`, `lookup`, or `nn` members.

        Args:
            obj: Target object.
            name: Qualified API path.
            **kwargs: Additional inspect keyword arguments.

        Returns:
            Inspected `ExtendedGhostRef`.

        Raises:
            RuntimeError: For `image`, `lookup`, or `nn` targets.
        """
        _ = kwargs
        if any(seg in name for seg in ("image", "lookup", "nn")):
            raise RuntimeError("Inspect failure")
        return orig_inspect(obj, name)

    monkeypatch.setattr("ml_ecosystem_snapshots.frameworks.tensorflow.GhostInspector.inspect", _broken_tf_inspect)
    refs_err = tf_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert not any("tf.image.resize" in r.api_path for r in refs_err)
    assert not any("tf.nn.relu" in r.api_path for r in refs_err)
