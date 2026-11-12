"""Framework collector registry, domain collector specs, and unified provenance/version metadata."""

from __future__ import annotations

import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.frameworks._base import (
    CollectorDomain,
    DomainCollectorAdapter,
    DomainSnapshotCollector,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)


class CollectorFn(typing.Protocol):
    """Protocol for Python framework snapshot collector callables."""

    def __call__(self, category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
        """Collect API symbols for a given semantic tier.

        Args:
            category: Semantic category tier to collect.
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of collected `GhostRef` items.
        """
        ...


class ProvenanceMetadata(pydantic.BaseModel):
    """Unified provenance and version-resolution metadata for a framework or specification target."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    source_type: str = "python_package"
    upstream_commit: str | None = None
    upstream_version: str = ""
    supported_microarchitectures: list[str] | None = None
    resolution_mode: typing.Literal["static_spec", "live_module", "exhaustive_json", "pkg_alias"] = "static_spec"
    dist_name: str | None = None
    live_module: str | None = None
    griffe_fallback_module: str | None = None
    emit_in_envelope: bool = False

    def items(self) -> list[tuple[str, str | list[str]]]:
        """Return envelope-emitted provenance metadata key-value pairs.

        Returns:
            List of `(key, value)` pairs for `SnapshotEnvelope` headers.
        """
        if not self.emit_in_envelope:
            return []
        res: list[tuple[str, str | list[str]]] = [
            ("source_type", self.source_type),
            ("upstream_version", self.upstream_version),
        ]
        if self.upstream_commit is not None:
            res.append(("upstream_commit", self.upstream_commit))
        if self.supported_microarchitectures is not None:
            res.append(
                (
                    "supported_microarchitectures",
                    list(self.supported_microarchitectures),
                )
            )
        return res


def _is_collector_callable(val: LivePythonTarget) -> typing_extensions.TypeGuard[CollectorFn]:
    """Narrow an inspectable runtime value to a callable `CollectorFn`.

    Args:
        val: Runtime member value to check.

    Returns:
        True if `val` is callable, False otherwise.
    """
    return callable(val)


def _get_callable_collectors(mod: LivePythonTarget) -> dict[str, CollectorFn]:
    """Extract callable collector members from an imported module.

    Args:
        mod: Imported module or namespace target to inspect.

    Returns:
        Mapping from member name to callable collector function.
    """
    return {k: v for k, v in get_member_dict(mod).items() if _is_collector_callable(v)}


def make_lazy_collector(module_name: str, func_name: str = "collect_api") -> CollectorFn:
    """Create a lazy loader for a framework collection function to defer heavy C runtime imports.

    Args:
        module_name: Submodule name under ml_ecosystem_snapshots.frameworks or absolute module path.
        func_name: Target collection function name.

    Returns:
        Callable function forwarding arguments to the dynamically loaded collector.
    """

    def _collector(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
        """Dynamically load module and execute target collection function.

        Args:
            category: Semantic tier category to collect.
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            Output from target collection function.
        """
        import importlib

        if "." not in module_name:
            full_mod = f"ml_ecosystem_snapshots.frameworks.{module_name}"
        else:
            full_mod = module_name
        mod: LivePythonTarget = importlib.import_module(full_mod)
        members: dict[str, CollectorFn] = _get_callable_collectors(mod)
        func = members[func_name]
        return func(category, include_nonpublic)

    return _collector


_make_lazy_collector = make_lazy_collector


class DomainCollectFn(typing.Protocol):
    """Protocol for domain-native `collect_domain(include_nonpublic)` callables."""

    def __call__(self, include_nonpublic: bool = False) -> list[GhostRef]:
        """Collect domain-native `GhostRef` items without `SemanticTier`.

        Args:
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of collected `GhostRef` items.
        """
        ...


def _is_domain_collect_callable(val: LivePythonTarget) -> typing_extensions.TypeGuard[DomainCollectFn]:
    """Narrow a runtime member value to `DomainCollectFn`.

    Args:
        val: Runtime member value to inspect.

    Returns:
        True if `val` is callable.
    """
    return callable(val)


def make_lazy_domain_collector(
    module_name: str,
    collector_domain: CollectorDomain,
    category_key: str,
) -> DomainSnapshotCollector:
    """Create a lazy `DomainSnapshotCollector` delegating to `<module>.collect_domain` or `collect_api`.

    Args:
        module_name: Submodule name under `ml_ecosystem_snapshots.frameworks`.
        collector_domain: Target `CollectorDomain` classification.
        category_key: Snapshot category key emitted for this domain.

    Returns:
        Configured `DomainSnapshotCollector` instance.
    """

    def _collect(include_nonpublic: bool = False) -> list[GhostRef]:
        """Import target domain module and execute domain-native symbol collection.

        Args:
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of domain `GhostRef` items.
        """
        import importlib

        mod: LivePythonTarget = importlib.import_module(f"ml_ecosystem_snapshots.frameworks.{module_name}")
        members = get_member_dict(mod)
        dom_fn = members.get("collect_domain")
        if dom_fn is not None and _is_domain_collect_callable(dom_fn):
            return dom_fn(include_nonpublic)
        collect_fns = _get_callable_collectors(mod)
        return collect_fns["collect_api"](SemanticTier.UTIL, include_nonpublic)

    return DomainCollectorAdapter(collector_domain, category_key, _collect)


_make_lazy_domain_collector = make_lazy_domain_collector


_CANONICAL_COLLECTOR_SPECS: dict[str, tuple[str, str]] = {
    "torch": ("torch", "collect_api"),
    "jax": ("jax", "collect_api"),
    "keras": ("keras", "collect_api"),
    "tensorflow": ("tensorflow", "collect_api"),
    "mlx": ("mlx", "collect_api"),
    "cupy": ("cupy", "collect_api"),
    "dask": ("dask", "collect_api"),
    "flax_nnx": ("flax_nnx", "collect_api"),
    "sklearn": ("sklearn", "collect_api"),
    "transformers": ("huggingface", "collect_transformers"),
    "diffusers": ("huggingface", "collect_diffusers"),
    "tokenizers": ("huggingface", "collect_tokenizers"),
    "huggingface": ("huggingface", "collect_transformers"),
    "triton": ("triton", "collect_api"),
    "onnxruntime": ("onnxruntime", "collect_api"),
    "deepspeed": ("deepspeed", "collect_api"),
    "stablehlo": ("stablehlo", "collect_api"),
    "onnx": ("onnx_spec", "collect_api"),
    "metal": ("metal", "collect_api"),
    "numba": ("numba", "collect_api"),
    "sparse": ("sparse", "collect_api"),
    "dpnp": ("dpnp", "collect_api"),
    "awkward": ("awkward", "collect_api"),
    "pyarrow_compute": ("pyarrow_compute", "collect_api"),
    "bohrium": ("bohrium", "collect_api"),
    "wasm_simd": ("wasm_simd", "collect_api"),
    "wasm": ("wasm_simd", "collect_api"),
    "webgl": ("webgl", "collect_api"),
    "cpp_runtime": ("cpp_runtime", "collect_api"),
    "cpp": ("cpp_runtime", "collect_api"),
    "chex": ("chex", "collect_api"),
    "grain": ("grain", "collect_api"),
    "pygrain": ("grain", "collect_api"),
    "optax": ("optax", "collect_api"),
    "orbax": ("orbax_checkpoint", "collect_api"),
    "orbax_checkpoint": ("orbax_checkpoint", "collect_api"),
    "array_api": ("array_api", "collect_api"),
    "scipy": ("scipy", "collect_api"),
    "torchvision": ("torchvision", "collect_api"),
    "torchaudio": ("torchaudio", "collect_api"),
    "safetensors": ("safetensors", "collect_api"),
    "aten": ("aten", "collect_api"),
    "nccl": ("nccl", "collect_api"),
    "rccl": ("nccl", "collect_api"),
    "flash_attention": ("flash_attention", "collect_api"),
    "flash_attn": ("flash_attention", "collect_api"),
}

_DOMAIN_COLLECTOR_SPECS: dict[str, tuple[str, CollectorDomain, str]] = {
    "mlir": ("mlir", CollectorDomain.COMPILER_IR, "operations"),
    "stablehlo": ("stablehlo", CollectorDomain.COMPILER_IR, "operations"),
    "onnx_spec": ("onnx_spec", CollectorDomain.COMPILER_IR, "operations"),
    "onnx": ("onnx_spec", CollectorDomain.COMPILER_IR, "operations"),
    "wgsl": ("wgsl", CollectorDomain.COMPILER_IR, "operations"),
    "webgl": ("webgl", CollectorDomain.COMPILER_IR, "operations"),
    "nvidia_sass": ("nvidia_sass", CollectorDomain.HARDWARE_ISA, "instructions"),
    "nvidia_ptx": ("nvidia_ptx", CollectorDomain.HARDWARE_ISA, "instructions"),
    "amd_rdna": ("amd_rdna", CollectorDomain.HARDWARE_ISA, "instructions"),
    "metal": ("metal", CollectorDomain.HARDWARE_ISA, "instructions"),
    "wasm_simd": ("wasm_simd", CollectorDomain.HARDWARE_ISA, "instructions"),
    "wasm": ("wasm_simd", CollectorDomain.HARDWARE_ISA, "instructions"),
    "cpp_runtime": ("cpp_runtime", CollectorDomain.CPP_RUNTIME, "operations"),
    "cpp": ("cpp_runtime", CollectorDomain.CPP_RUNTIME, "operations"),
    "nccl": ("nccl", CollectorDomain.CPP_RUNTIME, "collectives"),
    "rccl": ("nccl", CollectorDomain.CPP_RUNTIME, "collectives"),
    "html_dsl": ("html_dsl", CollectorDomain.DOCUMENT_DSL, "elements"),
    "latex_dsl": ("latex_dsl", CollectorDomain.DOCUMENT_DSL, "elements"),
    "tikz": ("tikz", CollectorDomain.DOCUMENT_DSL, "elements"),
}

DOMAIN_COLLECTORS: dict[str, DomainSnapshotCollector] = {
    target: _make_lazy_domain_collector(mod_name, dom, cat_key)
    for target, (mod_name, dom, cat_key) in _DOMAIN_COLLECTOR_SPECS.items()
}


def get_available_frameworks() -> dict[str, CollectorFn]:
    """Get a dictionary of available framework collectors.

    Returns:
        A dictionary mapping framework identifiers to their collection functions.
    """
    import importlib.util
    import pkgutil

    collectors: dict[str, CollectorFn] = {}

    pkg_spec = importlib.util.find_spec("ml_ecosystem_snapshots.frameworks")
    search_paths = (
        list(pkg_spec.submodule_search_locations)
        if pkg_spec is not None and pkg_spec.submodule_search_locations is not None
        else []
    )
    for _, module_name, _ in pkgutil.iter_modules(search_paths):
        if module_name.startswith("_"):
            continue
        try:
            import importlib

            mod = importlib.import_module(f"ml_ecosystem_snapshots.frameworks.{module_name}")
            found_any = False
            members: dict[str, CollectorFn] = _get_callable_collectors(mod)
            for name, obj in members.items():
                if name.startswith("collect_"):
                    found_any = True
                    if name == "collect_api":
                        identifier = module_name
                    else:
                        identifier = f"{module_name}_{name[8:]}"
                    collectors[identifier] = obj
            if not found_any:
                collectors[module_name] = _make_lazy_collector(module_name, "collect_api")
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
            collectors[module_name] = _make_lazy_collector(module_name, "collect_api")

    for alias_name, (mod_name, fn_name) in _CANONICAL_COLLECTOR_SPECS.items():
        collectors[alias_name] = _make_lazy_collector(mod_name, fn_name)

    return collectors


FRAMEWORK_COLLECTORS: dict[str, CollectorFn] = get_available_frameworks()
COLLECTOR_REGISTRY: dict[str, CollectorFn] = FRAMEWORK_COLLECTORS

FRAMEWORK_PROVENANCE_METADATA: dict[str, ProvenanceMetadata] = {
    # Exhaustive JSON-backed hardware ISA and compiler IR specifications
    "nvidia_sass": ProvenanceMetadata(
        supported_microarchitectures=[
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
        ],
        source_type="binary_disassembly",
        upstream_commit="cuda-12.6-toolkit",
        upstream_version="12.6.0",
        resolution_mode="exhaustive_json",
        emit_in_envelope=True,
    ),
    "nvidia_ptx": ProvenanceMetadata(
        supported_microarchitectures=[
            "sm_50",
            "sm_60",
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
        ],
        source_type="tablegen",
        upstream_commit="cuda-12.6",
        upstream_version="8.5",
        resolution_mode="exhaustive_json",
        emit_in_envelope=True,
    ),
    "amd_rdna": ProvenanceMetadata(
        supported_microarchitectures=[
            "GFX9/CDNA",
            "GFX10/RDNA1",
            "GFX10.3/RDNA2",
            "GFX11/RDNA3",
            "GFX11.5",
            "GFX12/RDNA4",
        ],
        source_type="tablegen",
        upstream_commit="llvm-project-19.1.0",
        upstream_version="19.1.0",
        resolution_mode="exhaustive_json",
        emit_in_envelope=True,
    ),
    "stablehlo": ProvenanceMetadata(
        source_type="tablegen",
        upstream_commit="v1.9.0",
        upstream_version="1.9.0",
        resolution_mode="exhaustive_json",
        emit_in_envelope=True,
    ),
    "mlir": ProvenanceMetadata(
        source_type="tablegen",
        upstream_commit="llvmorg-19.1.0",
        upstream_version="19.1.0",
        resolution_mode="pkg_alias",
        dist_name="mlir",
        emit_in_envelope=True,
    ),
    # Document DSL specifications
    "html_dsl": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
        resolution_mode="static_spec",
        emit_in_envelope=True,
    ),
    "latex_dsl": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
        resolution_mode="static_spec",
        emit_in_envelope=True,
    ),
    "tikz": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
        resolution_mode="static_spec",
        emit_in_envelope=True,
    ),
    # Static specification versions
    "wgsl": ProvenanceMetadata(source_type="wgsl_spec", upstream_version="draft-2024", resolution_mode="static_spec"),
    "metal": ProvenanceMetadata(source_type="metal_spec", upstream_version="3.2", resolution_mode="static_spec"),
    "wasm_simd": ProvenanceMetadata(source_type="wasm_spec", upstream_version="2.0", resolution_mode="static_spec"),
    "wasm": ProvenanceMetadata(source_type="wasm_spec", upstream_version="2.0", resolution_mode="static_spec"),
    "webgl": ProvenanceMetadata(source_type="webgl_spec", upstream_version="2.0", resolution_mode="static_spec"),
    "cpp_runtime": ProvenanceMetadata(source_type="cpp_spec", upstream_version="17", resolution_mode="static_spec"),
    "cpp": ProvenanceMetadata(source_type="cpp_spec", upstream_version="17", resolution_mode="static_spec"),
    "array_api": ProvenanceMetadata(
        source_type="array_api_spec", upstream_version="2024.12", resolution_mode="static_spec"
    ),
    "nccl": ProvenanceMetadata(source_type="nccl_spec", upstream_version="2.21", resolution_mode="static_spec"),
    "rccl": ProvenanceMetadata(source_type="nccl_spec", upstream_version="2.21", resolution_mode="static_spec"),
    "flash_attention": ProvenanceMetadata(
        source_type="python_package", upstream_version="2.6.3", resolution_mode="static_spec"
    ),
    "flash_attn": ProvenanceMetadata(
        source_type="python_package", upstream_version="2.6.3", resolution_mode="static_spec"
    ),
    # Live module version fallbacks
    "ir": ProvenanceMetadata(
        upstream_version="0.0.6",
        resolution_mode="live_module",
        live_module="ml_switcheroo_ir",
        dist_name="ml-switcheroo-ir",
    ),
    "ml_switcheroo_ir": ProvenanceMetadata(
        upstream_version="0.0.6",
        resolution_mode="live_module",
        live_module="ml_switcheroo_ir",
        dist_name="ml-switcheroo-ir",
    ),
    "scipy": ProvenanceMetadata(
        upstream_version="1.13.1", resolution_mode="live_module", live_module="scipy", dist_name="scipy"
    ),
    "torchvision": ProvenanceMetadata(
        upstream_version="0.23.0", resolution_mode="live_module", live_module="torchvision", dist_name="torchvision"
    ),
    "torchaudio": ProvenanceMetadata(
        upstream_version="2.8.0", resolution_mode="live_module", live_module="torchaudio", dist_name="torchaudio"
    ),
    "safetensors": ProvenanceMetadata(
        upstream_version="0.7.0", resolution_mode="live_module", live_module="safetensors", dist_name="safetensors"
    ),
    "aten": ProvenanceMetadata(
        upstream_version="2.8.0", resolution_mode="live_module", live_module="torch", dist_name="torch"
    ),
    "numba": ProvenanceMetadata(
        upstream_version="0.60.0", resolution_mode="live_module", live_module="numba", dist_name="numba"
    ),
    "sparse": ProvenanceMetadata(
        upstream_version="0.15.4", resolution_mode="live_module", live_module="sparse", dist_name="sparse"
    ),
    "dpnp": ProvenanceMetadata(upstream_version="0.15.0", resolution_mode="live_module", live_module="dpnp"),
    "awkward": ProvenanceMetadata(upstream_version="2.6.8", resolution_mode="live_module", live_module="awkward"),
    "pyarrow_compute": ProvenanceMetadata(
        upstream_version="17.0.0", resolution_mode="live_module", live_module="pyarrow"
    ),
    "bohrium": ProvenanceMetadata(upstream_version="0.13.0", resolution_mode="live_module", live_module="bohrium"),
    "onnx": ProvenanceMetadata(
        upstream_version="1.17.0", resolution_mode="live_module", live_module="onnx", dist_name="onnx"
    ),
    "onnx_spec": ProvenanceMetadata(
        upstream_version="1.17.0", resolution_mode="live_module", live_module="onnx", dist_name="onnx"
    ),
    # Package distribution aliases and Griffe static fallbacks
    "flax_nnx": ProvenanceMetadata(resolution_mode="pkg_alias", dist_name="flax"),
    "sklearn": ProvenanceMetadata(resolution_mode="pkg_alias", dist_name="scikit-learn"),
    "pytorch": ProvenanceMetadata(resolution_mode="pkg_alias", dist_name="torch"),
    "pax": ProvenanceMetadata(resolution_mode="pkg_alias", dist_name="paxml"),
    "huggingface": ProvenanceMetadata(resolution_mode="pkg_alias", dist_name="transformers"),
    "chex": ProvenanceMetadata(
        upstream_version="0.1.87",
        resolution_mode="pkg_alias",
        dist_name="chex",
        griffe_fallback_module="chex",
    ),
    "grain": ProvenanceMetadata(
        upstream_version="0.2.2",
        resolution_mode="pkg_alias",
        dist_name="grain",
        griffe_fallback_module="grain",
    ),
    "pygrain": ProvenanceMetadata(
        upstream_version="0.2.2",
        resolution_mode="pkg_alias",
        dist_name="grain",
        griffe_fallback_module="grain",
    ),
    "optax": ProvenanceMetadata(
        upstream_version="0.2.3",
        resolution_mode="pkg_alias",
        dist_name="optax",
        griffe_fallback_module="optax",
    ),
    "orbax": ProvenanceMetadata(
        upstream_version="0.6.4",
        resolution_mode="pkg_alias",
        dist_name="orbax-checkpoint",
        griffe_fallback_module="orbax.checkpoint",
    ),
    "orbax_checkpoint": ProvenanceMetadata(
        upstream_version="0.6.4",
        resolution_mode="pkg_alias",
        dist_name="orbax-checkpoint",
        griffe_fallback_module="orbax.checkpoint",
    ),
    "orbax-checkpoint": ProvenanceMetadata(
        upstream_version="0.6.4",
        resolution_mode="pkg_alias",
        dist_name="orbax-checkpoint",
        griffe_fallback_module="orbax.checkpoint",
    ),
}

__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFn",
    "DOMAIN_COLLECTORS",
    "FRAMEWORK_COLLECTORS",
    "FRAMEWORK_PROVENANCE_METADATA",
    "ProvenanceMetadata",
    "get_available_frameworks",
    "make_lazy_collector",
    "make_lazy_domain_collector",
]
