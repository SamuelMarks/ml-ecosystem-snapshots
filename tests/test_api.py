"""Unit and integration tests for the `ml_ecosystem_snapshots.api` SDK and version resolution."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types
import typing
from pathlib import Path

import griffe
import pytest
from conftest import FakePackageBuilder
from hatchling.metadata.core import ProjectMetadata
from hatchling.plugin.manager import PluginManager
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

import ml_ecosystem_snapshots
from ml_ecosystem_snapshots.api import (
    FRAMEWORK_COLLECTORS,
    consolidate_aliases,
    extract_all_snapshots,
    extract_snapshot,
    extract_snapshot_isolated,
    get_available_frameworks,
    get_pkg_version,
    make_lazy_collector,
    validate_snapshot_envelope,
    write_snapshot,
)
from ml_ecosystem_snapshots.exceptions import SnapshotError, SnapshotSchemaError
from ml_ecosystem_snapshots.frameworks._base import CollectorDomain, DomainCollectorAdapter
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget, get_member_dict
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list
from ml_ecosystem_snapshots.network_env import ZSTD_MAGIC_HEADER, load_json_data, stream_snapshot_items
from ml_ecosystem_snapshots.registry import DOMAIN_COLLECTORS, ProvenanceMetadata, make_lazy_domain_collector
from ml_ecosystem_snapshots.utils import get_all_members


def test_get_all_members_lazy_namespace() -> None:
    """Verify `get_all_members` resolves lazy `__dir__`/`__getattr__` attributes and skips broken ones."""

    class LazyModule(types.SimpleNamespace):
        """Fake lazy-loading module namespace for testing member discovery."""

        def __init__(self) -> None:
            """Initialize visible and cached lazy attributes."""
            self.VisibleClass = int
            self._cache = {"hidden_func": lambda: 42}

        def __dir__(self) -> list[str]:
            """Return simulated module attribute names including broken entries.

            Returns:
                List of attribute names exposed by `dir()`.
            """
            return ["VisibleClass", "hidden_func", "broken_all", "broken_dir"]

        def __getattr__(self, name: str) -> LivePythonTarget:
            """Resolve lazy attributes or raise `AttributeError` for broken entries.

            Args:
                name: Attribute name to look up on the lazy module.

            Returns:
                Resolved target attribute from the internal cache.

            Raises:
                AttributeError: When `name` is simulated as broken or missing.
            """
            if name in ("broken_all", "broken_dir"):
                raise AttributeError("simulated error")
            if name in self._cache:
                return self._cache[name]
            raise AttributeError(name)

    members = dict(get_all_members(LazyModule()))
    assert "VisibleClass" in members
    assert "hidden_func" in members
    assert "broken_all" not in members
    assert "broken_dir" not in members
    assert members["VisibleClass"] is int
    hidden_fn = members["hidden_func"]
    assert callable(hidden_fn)
    assert hidden_fn() == 42


def test_extract_snapshot_and_extract_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `extract_snapshot` and `extract_all_snapshots` handle missing, partial, and empty collectors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    assert extract_snapshot("nonexistent") == {}

    monkeypatch.setattr("ml_ecosystem_snapshots.api.get_pkg_version", lambda _fw: "unknown")
    assert extract_snapshot("torch") == {}

    monkeypatch.setattr("ml_ecosystem_snapshots.api.get_pkg_version", lambda _fw: "1.0.0")
    sample_ref = GhostRef(name="MSELoss", api_path="torch.nn.MSELoss", kind="class")

    def fake_collect(cat: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
        """Simulate framework tier collection with partial failure.

        Args:
            cat: Semantic tier category to collect.
            include_nonpublic: Whether non-public symbols are included.

        Returns:
            Collected `GhostRef` items for the requested tier.

        Raises:
            RuntimeError: When collecting the optimizer tier to test resilience.
        """
        _ = include_nonpublic
        if cat == SemanticTier.LOSS:
            return [sample_ref]
        if cat == SemanticTier.OPTIMIZER:
            raise RuntimeError("simulate exception")
        return []

    monkeypatch.setitem(FRAMEWORK_COLLECTORS, "torch", fake_collect)
    res = extract_snapshot("torch")
    assert res["version"] == "1.0.0"
    categories = ensure_json_dict(res["categories"])
    loss_items = ensure_json_list(categories["loss"])
    assert len(loss_items) == 1
    assert ensure_json_dict(loss_items[0])["name"] == "MSELoss"

    monkeypatch.setitem(FRAMEWORK_COLLECTORS, "torch", lambda _cat, _inc=False: [])
    assert extract_snapshot("torch") == {}

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.api.extract_snapshot",
        lambda fw, include_nonpublic=False: {"version": "1"} if fw == "torch" else {},
    )
    res_all = extract_all_snapshots()
    assert "torch" in res_all
    assert "jax" not in res_all

    monkeypatch.setattr("ml_ecosystem_snapshots.api.extract_snapshot", lambda _fw, _inc=False: {})
    assert extract_all_snapshots() == {}


def test_get_available_frameworks_with_fake_package_builder(
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `get_available_frameworks` discovers `collect_*` callables, skips `_` modules, and handles import errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.build_module(
        "ml_ecosystem_snapshots.frameworks.test_fw",
        members={
            "collect_api": lambda _cat, _inc=False: [],
            "collect_custom": lambda _cat, _inc=False: [],
            "other_func": lambda: None,
        },
    )
    fake_package_builder.build_module(
        "ml_ecosystem_snapshots.frameworks.other_mod",
        members={"collect_other": lambda _cat, _inc=False: []},
    )
    fake_package_builder.build_module(
        "ml_ecosystem_snapshots.frameworks.bar_mod",
        members={"collect_": lambda _cat, _inc=False: []},
    )
    fake_package_builder.build_module(
        "ml_ecosystem_snapshots.frameworks.no_collect_mod",
        members={"some_var": 123},
    )
    fake_package_builder.block_imports("ml_ecosystem_snapshots.frameworks.broken_mod")

    monkeypatch.setattr(
        "pkgutil.iter_modules",
        lambda _paths: [
            (None, "_private_mod", False),
            (None, "test_fw", False),
            (None, "other_mod", False),
            (None, "bar_mod", False),
            (None, "no_collect_mod", False),
            (None, "broken_mod", False),
            (None, "sklearn", False),
            (None, "tensorflow", False),
        ],
    )
    discovered = get_available_frameworks()
    assert "_private_mod" not in discovered
    assert "test_fw" in discovered
    assert "test_fw_custom" in discovered
    assert "other_mod_other" in discovered
    assert "bar_mod" not in discovered
    assert "no_collect_mod" in discovered
    assert "broken_mod" in discovered
    assert "sklearn" in discovered
    assert "tensorflow" in discovered


def test_consolidate_aliases_variants() -> None:
    """Verify `consolidate_aliases` merges duplicate signatures onto shortest canonical `api_path`."""
    r_long = GhostRef(name="func", api_path="long.path.func", kind="function", aliases=["a"])
    r_short = GhostRef(name="func", api_path="short.func", kind="function", aliases=["b"])
    merged_short = consolidate_aliases([r_long, r_short])
    assert len(merged_short) == 1
    assert merged_short[0].api_path == "short.func"
    assert set(merged_short[0].aliases or []) == {"a", "b", "long.path.func"}

    r_solo = GhostRef(name="alone", api_path="alone.func", kind="function", aliases=None)
    assert consolidate_aliases([r_solo])[0].aliases == []

    r_none_1 = GhostRef(name="f", api_path="longer.f", kind="function", aliases=None)
    r_none_2 = GhostRef(name="f", api_path="f", kind="function", aliases=None)
    merged_none = consolidate_aliases([r_none_1, r_none_2])
    assert len(merged_none) == 1
    assert merged_none[0].aliases == ["longer.f"]

    r_eq_a = GhostRef(name="func", api_path="a.func", kind="function", aliases=["x"])
    r_eq_b = GhostRef(name="func", api_path="b.func", kind="function", aliases=["y"])
    merged_eq = consolidate_aliases([r_eq_a, r_eq_b])
    assert len(merged_eq) == 1
    assert merged_eq[0].api_path == "a.func"
    assert "b.func" in (merged_eq[0].aliases or [])


def test_get_pkg_version_metadata_and_cupy_tensorflow(
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `get_pkg_version` resolves aliases, static specs, and `cupy`/`tensorflow` fallback chains.

    Args:
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.set_metadata_versions(
        {
            "torch": "2.4.0",
            "flax": "0.8.5",
            "scikit-learn": "1.5.1",
            "paxml": "1.4.0",
            "orbax-checkpoint": "0.6.4",
            "optax": "0.2.3",
            "transformers": "4.42.0",
            "mlir": "19.1.2",
        }
    )
    assert get_pkg_version("torch") == "2.4.0"
    assert get_pkg_version("pytorch") == "2.4.0"
    assert get_pkg_version("flax_nnx") == "0.8.5"
    assert get_pkg_version("sklearn") == "1.5.1"
    assert get_pkg_version("pax") == "1.4.0"
    assert get_pkg_version("orbax") == "0.6.4"
    assert get_pkg_version("orbax_checkpoint") == "0.6.4"
    assert get_pkg_version("optax") == "0.2.3"
    assert get_pkg_version("huggingface") == "4.42.0"
    assert get_pkg_version("mlir") == "19.1.2"
    assert get_pkg_version("missing_pkg") == "unknown"
    assert get_pkg_version("my_custom_pkg") == "unknown"

    for dsl_name in ("html_dsl", "latex_dsl", "tikz"):
        assert get_pkg_version(dsl_name) == "0.0.2"
    assert get_pkg_version("wgsl") == "draft-2024"
    assert get_pkg_version("metal") == "3.2"
    assert get_pkg_version("wasm_simd") == "2.0"
    assert get_pkg_version("wasm") == "2.0"
    assert get_pkg_version("webgl") == "2.0"
    assert get_pkg_version("cpp_runtime") == "17"
    assert get_pkg_version("cpp") == "17"
    assert get_pkg_version("array_api") == "2024.12"
    assert get_pkg_version("nccl") == "2.21"
    assert get_pkg_version("rccl") == "2.21"

    # CuPy & TensorFlow live vs metadata fallback chains
    fake_package_builder.build_module("cupy", version="1.0.0")
    assert get_pkg_version("cupy") == "1.0.0"

    fake_package_builder.block_imports("cupy")
    fake_package_builder.set_metadata_versions({"cupy-cuda12x": "2.0.0"})
    assert get_pkg_version("cupy") == "2.0.0"

    fake_package_builder.build_module("tensorflow", version="3.0.0")
    assert get_pkg_version("tensorflow") == "3.0.0"

    fake_package_builder.block_imports("tensorflow")
    fake_package_builder.set_metadata_versions({"tensorflow-macos": "4.0.0"})
    assert get_pkg_version("tensorflow") == "4.0.0"

    fake_package_builder.set_metadata_versions({"tensorflow-cpu": "5.0.0"})
    assert get_pkg_version("tensorflow") == "5.0.0"


def test_get_pkg_version_keras_griffe_and_ir_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `get_pkg_version` resolves `keras` via real Griffe packages in `tmp_path` and `ir` aliases.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    fake_package_builder.set_metadata_versions({"keras": "3.1.0"})
    assert get_pkg_version("keras") == "3.1.0"

    fake_package_builder.set_metadata_versions({})
    keras_pkg = tmp_path / "keras"
    keras_pkg.mkdir()
    (keras_pkg / "__init__.py").write_text("__version__ = '2.3.4'", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert get_pkg_version("keras") == "2.3.4"

    # Alias pointing to Attribute in submodule
    (keras_pkg / "_ver.py").write_text('VERSION = "3.4.5"', encoding="utf-8")
    (keras_pkg / "__init__.py").write_text("from keras._ver import VERSION as __version__", encoding="utf-8")
    assert get_pkg_version("keras") == "3.4.5"

    # Alias pointing to Function (non-Attribute) -> unknown
    (keras_pkg / "_ver.py").write_text("def VERSION() -> str: return '0'", encoding="utf-8")
    assert get_pkg_version("keras") == "unknown"

    # Module without __version__ -> unknown
    (keras_pkg / "__init__.py").write_text("X = 1", encoding="utf-8")
    assert get_pkg_version("keras") == "unknown"

    # Non-Module object returned by griffe.load and RuntimeError fallback
    with monkeypatch.context() as mp:
        mp.setattr("griffe.load", lambda _name: griffe.Function("not_a_module"))
        assert get_pkg_version("keras") == "unknown"
        mp.setattr("griffe.load", lambda _name: (_ for _ in ()).throw(RuntimeError("griffe error")))
        assert get_pkg_version("keras") == "unknown"

    # `ir` / `ml_switcheroo_ir` live vs metadata branches
    fake_package_builder.build_module("ml_switcheroo_ir", version="0.0.9")
    assert get_pkg_version("ir") == "0.0.9"
    fake_package_builder.block_imports("ml_switcheroo_ir")
    fake_package_builder.set_metadata_versions({"ml_switcheroo_ir": "0.0.6"})
    assert get_pkg_version("ml_switcheroo_ir") == "0.0.6"


def test_get_pkg_version_live_modules_and_exhaustive_json_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `get_pkg_version` across live modules, metadata fallbacks, and real exhaustive JSON files in `tmp_path`.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    pkgs = [
        "scipy",
        "torchvision",
        "torchaudio",
        "safetensors",
        "torch",
        "numba",
        "sparse",
        "dpnp",
        "awkward",
        "pyarrow",
        "bohrium",
        "onnx",
    ]
    for mod_name in pkgs:
        fake_package_builder.build_module(mod_name, version="7.7.7")

    for target_key in (
        "scipy",
        "torchvision",
        "torchaudio",
        "safetensors",
        "aten",
        "numba",
        "sparse",
        "dpnp",
        "awkward",
        "pyarrow_compute",
        "bohrium",
        "onnx",
        "onnx_spec",
    ):
        assert get_pkg_version(target_key) == "7.7.7"

    fake_package_builder.block_imports(*pkgs, "flash_attn")
    fake_package_builder.set_metadata_versions(
        {
            "scipy": "9.9.9",
            "torchvision": "9.9.9",
            "torchaudio": "9.9.9",
            "safetensors": "9.9.9",
            "torch": "9.9.9",
            "numba": "9.9.9",
            "sparse": "9.9.9",
            "onnx": "9.9.9",
        }
    )
    assert get_pkg_version("scipy") == "9.9.9"
    assert get_pkg_version("aten") == "9.9.9"
    assert get_pkg_version("numba") == "9.9.9"
    assert get_pkg_version("sparse") == "9.9.9"
    assert get_pkg_version("onnx") == "9.9.9"
    assert get_pkg_version("onnx_spec") == "9.9.9"
    assert get_pkg_version("dpnp") == "0.15.0"
    assert get_pkg_version("awkward") == "2.6.8"
    assert get_pkg_version("pyarrow_compute") == "17.0.0"
    assert get_pkg_version("bohrium") == "0.13.0"
    assert get_pkg_version("flash_attention") == "2.6.3"
    assert get_pkg_version("flash_attn") == "2.6.3"

    # Real exhaustive JSON files in tmp_path/frameworks via api.__file__ anchor
    fw_dir = tmp_path / "frameworks"
    fw_dir.mkdir()
    monkeypatch.setattr("ml_ecosystem_snapshots.api.__file__", str(tmp_path / "api.py"))

    assert get_pkg_version("mlir") == "19.1.0"
    assert get_pkg_version("nvidia_sass") == "12.6.0"
    assert get_pkg_version("amd_rdna") == "19.1.0"
    assert get_pkg_version("nvidia_ptx") == "8.5"
    assert get_pkg_version("stablehlo") == "1.9.0"

    (fw_dir / "mlir_exhaustive.json").write_text('{"version": "llvm-20"}', encoding="utf-8")
    (fw_dir / "nvidia_sass_exhaustive.json").write_text('{"version": "13.0.0"}', encoding="utf-8")
    assert get_pkg_version("mlir") == "llvm-20"
    assert get_pkg_version("nvidia_sass") == "13.0.0"

    (fw_dir / "mlir_exhaustive.json").write_text("{corrupt_json", encoding="utf-8")
    (fw_dir / "nvidia_sass_exhaustive.json").write_text('{"version": null}', encoding="utf-8")
    assert get_pkg_version("mlir") == "19.1.0"
    assert get_pkg_version("nvidia_sass") == "12.6.0"


def test_validate_snapshot_envelope_and_provenance_metadata() -> None:
    """Verify `validate_snapshot_envelope`, hardware/DSL provenance metadata, and domain collector fallbacks."""
    with pytest.raises(ValueError, match="Snapshot must be a dictionary"):
        validate_snapshot_envelope("not a dict")

    env_default = validate_snapshot_envelope({"target": "torch", "version": "2.2.0", "categories": {}})
    assert env_default.target == "torch"
    assert env_default.schema_version == "2.0.0"
    assert env_default.generated_at is not None

    assert validate_snapshot_envelope({"target": "torch", "schema_version": "1.0.0"}).schema_version == "2.0.0"
    assert validate_snapshot_envelope({"target": "torch", "schema_version": "2.0.0"}).schema_version == "2.0.0"
    with pytest.raises(SnapshotSchemaError, match="Unsupported envelope schema_version"):
        validate_snapshot_envelope({"target": "torch", "schema_version": "9.9.9"})

    custom_env = validate_snapshot_envelope({"target": "jax", "generated_at": "2026-01-01T00:00:00Z"})
    assert custom_env.generated_at == "2026-01-01T00:00:00Z"
    assert validate_snapshot_envelope({}).target == "unknown"
    assert validate_snapshot_envelope({"target": "", "generated_at": ""}).generated_at != ""

    for hw_target, expected_arch in (("nvidia_sass", "sm_80"), ("amd_rdna", "GFX11/RDNA3"), ("nvidia_ptx", "sm_70")):
        hw_snap = extract_snapshot(hw_target)
        assert hw_snap["target"] == hw_target
        assert expected_arch in ensure_json_list(hw_snap["supported_microarchitectures"])

    shlo_snap = extract_snapshot("stablehlo")
    assert shlo_snap["version"] == "1.9.0"
    assert len(ensure_json_dict(shlo_snap["categories"])) > 0

    mlir_snap = extract_snapshot("mlir")
    assert mlir_snap["source_type"] == "tablegen"

    for dsl in ("html_dsl", "latex_dsl", "tikz"):
        dsl_snap = extract_snapshot(dsl)
        assert dsl_snap["source_type"] == "python_dsl"

    assert ProvenanceMetadata(upstream_version="1.0", emit_in_envelope=True).items() == [
        ("source_type", "python_package"),
        ("upstream_version", "1.0"),
    ]
    assert len(make_lazy_collector("metal", "collect_api")(SemanticTier.ARRAY_API)) > 0
    assert (
        len(make_lazy_collector("ml_ecosystem_snapshots.frameworks.metal", "collect_api")(SemanticTier.ARRAY_API)) > 0
    )
    assert len(make_lazy_domain_collector("metal", CollectorDomain.HARDWARE_ISA, "instructions").collect_domain()) > 0

    orig_dom = DOMAIN_COLLECTORS["html_dsl"]
    try:
        DOMAIN_COLLECTORS["html_dsl"] = DomainCollectorAdapter(
            CollectorDomain.DOCUMENT_DSL, "elements", lambda _inc: []
        )
        assert extract_snapshot("html_dsl") == {}

        def _raise_dom(_inc: bool) -> typing.NoReturn:
            """Raise RuntimeError during domain collection.

            Args:
                _inc: Include non-public flag.

            Raises:
                RuntimeError: Always raised.
            """
            raise RuntimeError("domain failure")

        DOMAIN_COLLECTORS["html_dsl"] = DomainCollectorAdapter(CollectorDomain.DOCUMENT_DSL, "elements", _raise_dom)
        assert extract_snapshot("html_dsl") == {}
    finally:
        DOMAIN_COLLECTORS["html_dsl"] = orig_dom


def test_extract_snapshot_isolated_and_package_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `extract_snapshot_isolated`, `PYTHONPATH` forwarding, and `ml_ecosystem_snapshots` `__init__` fallback.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    snap = extract_snapshot_isolated("html_dsl", timeout=30)
    assert snap.get("target") == "html_dsl"
    assert extract_snapshot_isolated("nonexistent_framework_xyz", timeout=10) == {}

    with monkeypatch.context() as mp:
        mp.setattr(
            "subprocess.run",
            lambda *_a, **_kw: subprocess.CompletedProcess(args=["python"], returncode=1, stdout=""),
        )
        assert extract_snapshot_isolated("html_dsl") == {}
        mp.setattr(
            "subprocess.run",
            lambda *_a, **_kw: subprocess.CompletedProcess(args=["python"], returncode=0, stdout="   "),
        )
        assert extract_snapshot_isolated("html_dsl") == {}
        mp.setattr("subprocess.run", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("Process failed")))
        assert extract_snapshot_isolated("html_dsl") == {}

    captured_envs: list[dict[str, str]] = []

    def _capture_subproc(
        *_args: list[str] | str,
        **kwargs: str | float | bool | dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Record subprocess environment dictionary and return minimal JSON stdout.

        Args:
            *_args: Ignored positional subprocess arguments.
            **kwargs: Subprocess keyword arguments including `env`.

        Returns:
            CompletedProcess with minimal JSON stdout.
        """
        env_arg = kwargs.get("env")
        if isinstance(env_arg, dict):
            captured_envs.append(env_arg)
        return subprocess.CompletedProcess(args=["python"], returncode=0, stdout='{"target": "html_dsl"}')

    with monkeypatch.context() as mp:
        mp.setenv("PYTHONPATH", "/custom/test/path")
        mp.setattr("subprocess.run", _capture_subproc)
        assert extract_snapshot_isolated("html_dsl").get("target") == "html_dsl"
        assert "/custom/test/path" in captured_envs[0]["PYTHONPATH"]

    with monkeypatch.context() as mp:
        mp.setattr(
            "ml_ecosystem_snapshots.api.FRAMEWORK_COLLECTORS",
            {"html_dsl": FRAMEWORK_COLLECTORS["html_dsl"]},
        )
        mp.setattr(
            "ml_ecosystem_snapshots.api.extract_snapshot_isolated",
            lambda _fw, _inc=False: {"target": "html_dsl"},
        )
        assert "html_dsl" in extract_all_snapshots(isolated=True)

    # Package __version__ and __init__ ImportError fallback
    pkg_ver = str(get_member_dict(ml_ecosystem_snapshots).get("__version__", ""))
    assert pkg_ver == ProjectMetadata(".", PluginManager()).version

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "ml_ecosystem_snapshots.api", None)
        importlib.reload(ml_ecosystem_snapshots)
        assert get_member_dict(ml_ecosystem_snapshots).get("__all__") == ["__version__"]
    importlib.reload(ml_ecosystem_snapshots)
    restored_all = get_member_dict(ml_ecosystem_snapshots).get("__all__")
    assert isinstance(restored_all, list) and "extract_snapshot" in restored_all


def test_write_snapshot_compression_and_streaming(
    tmp_path: Path,
    fake_package_builder: FakePackageBuilder,
) -> None:
    """Verify `write_snapshot` plain/gzip/zstd formats and `stream_snapshot_items` edge payloads.

    Args:
        tmp_path: Pytest temporary directory fixture.
        fake_package_builder: Typed fake-package builder fixture.
    """
    special_path = write_snapshot(
        "special_fw",
        {"schema_version": "1.0.0", "target": "special_fw", "version": "1.0.0+cu120 debug", "categories": {}},
        str(tmp_path / "special"),
    )
    assert special_path.endswith("special_fw_v1.0.0_cu120_debug.json")
    assert os.path.exists(special_path)

    snap_data: dict[str, JsonValue] = {
        "target": "compress_test",
        "version": "1.0.0",
        "categories": {
            "math": [
                {"name": "op1", "api_path": "test.op1", "kind": "function"},
                {"name": "op2", "api_path": "test.op2", "kind": "function"},
            ]
        },
        "items": [{"name": "op3", "api_path": "test.op3", "kind": "function"}],
    }

    p_plain = write_snapshot("compress_test", snap_data, str(tmp_path))
    assert ensure_json_dict(load_json_data(p_plain))["version"] == "1.0.0"

    p_gz = write_snapshot("compress_test", snap_data, str(tmp_path), compress="gzip")
    assert p_gz.endswith(".json.gz")
    assert ensure_json_dict(load_json_data(p_gz))["version"] == "1.0.0"

    class FakeCompressor:
        """Minimal Zstandard compressor prefixing `ZSTD_MAGIC_HEADER`."""

        def compress(self, data: bytes) -> bytes:
            """Prefix Zstandard magic header onto raw bytes.

            Args:
                data: Uncompressed payload bytes.

            Returns:
                Framed payload bytes.
            """
            return ZSTD_MAGIC_HEADER + data

    class FakeDecompressor:
        """Minimal Zstandard decompressor stripping `ZSTD_MAGIC_HEADER`."""

        def decompress(self, data: bytes) -> bytes:
            """Strip Zstandard magic header from framed bytes.

            Args:
                data: Framed payload bytes.

            Returns:
                Original uncompressed payload bytes.
            """
            return data[len(ZSTD_MAGIC_HEADER) :]

    fake_package_builder.build_module(
        "zstandard",
        members={"ZstdCompressor": FakeCompressor, "ZstdDecompressor": FakeDecompressor},
    )
    p_zst = write_snapshot("compress_test", snap_data, str(tmp_path), compress="zstd")
    assert p_zst.endswith(".json.zst")
    assert ensure_json_dict(load_json_data(p_zst))["version"] == "1.0.0"

    fake_package_builder.build_module("zstandard", members={"not_a_compressor": 1})
    with pytest.raises(SnapshotError, match="ZstdModuleProtocol"):
        write_snapshot("compress_bad_proto", snap_data, str(tmp_path), compress="zstd")

    fake_package_builder.block_imports("zstandard")
    with pytest.raises(SnapshotError, match="zstandard"):
        write_snapshot("compress_fallback", snap_data, str(tmp_path), compress="zstd")
    with pytest.raises(SnapshotError, match="zstandard"):
        load_json_data(p_zst)

    p_utf8_zst = tmp_path / "plain_utf8.json.zst"
    p_utf8_zst.write_bytes(b'{"key": "fallback"}')
    assert load_json_data(p_utf8_zst) == {"key": "fallback"}

    p_non_utf8_zst = tmp_path / "corrupt.json.zst"
    p_non_utf8_zst.write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(SnapshotError, match="zstandard"):
        load_json_data(p_non_utf8_zst)

    assert {it["name"] for it in stream_snapshot_items(p_plain)} == {"op1", "op2", "op3"}

    # Edge payloads for stream_snapshot_items
    for fname, payload, expected_names in (
        ("raw_list.json", [{"name": "list_op"}, "not_a_dict"], ["list_op"]),
        ("int_payload.json", 12345, []),
        ("bad_cat.json", {"categories": {"ops": ["not_a_dict"]}, "items": ["not_a_dict"]}, []),
        ("nondict_cat.json", {"categories": "not_a_dict"}, []),
        (
            "bad_cat_entry.json",
            {"categories": {"bad_cat": "not_a_list", "good_cat": [{"name": "valid_cat_op"}]}},
            ["valid_cat_op"],
        ),
        ("mixed_cat.json", {"categories": {"ops": ["not_a_dict", {"name": "valid_op"}]}}, ["valid_op"]),
        ("nonlist_items.json", {"items": "not_a_list"}, []),
    ):
        fp = tmp_path / fname
        fp.write_text(json.dumps(payload), encoding="utf-8")
        assert [str(it["name"]) for it in stream_snapshot_items(fp)] == expected_names
