"""Framework collector registry and provenance metadata specifications."""

from __future__ import annotations

import inspect
import typing

import pydantic
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget


class CollectorFn(typing.Protocol):
    """Protocol for framework snapshot collector callables."""

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
    """Typed provenance metadata for a framework snapshot."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    source_type: str
    upstream_commit: str
    upstream_version: str
    supported_microarchitectures: list[str] | None = None

    def items(self) -> list[tuple[str, str | list[str]]]:
        """Return non-None provenance metadata key-value pairs.

        Returns:
            List of `(key, value)` pairs.
        """
        res: list[tuple[str, str | list[str]]] = [
            ("source_type", self.source_type),
            ("upstream_commit", self.upstream_commit),
            ("upstream_version", self.upstream_version),
        ]
        if self.supported_microarchitectures is not None:
            res.append(
                (
                    "supported_microarchitectures",
                    list(self.supported_microarchitectures),
                )
            )
        return res


def _make_lazy_collector(module_name: str, func_name: str = "collect_api") -> CollectorFn:
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
        mod: InspectableTarget = importlib.import_module(full_mod)
        members: dict[str, CollectorFn] = dict(inspect.getmembers(mod, callable))
        func = members[func_name]
        return func(category, include_nonpublic)

    return _collector


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


def get_available_frameworks() -> dict[str, CollectorFn]:
    """Get a dictionary of available framework collectors.

    Returns:
        A dictionary mapping framework identifiers to their collection functions.
    """
    import importlib.util
    import pkgutil

    collectors: dict[str, CollectorFn] = {}

    pkg_spec = importlib.util.find_spec("ml_ecosystem_snapshots.frameworks")
    search_paths = list(pkg_spec.submodule_search_locations) if pkg_spec is not None and pkg_spec.submodule_search_locations is not None else []
    for _, module_name, _ in pkgutil.iter_modules(search_paths):
        if module_name.startswith("_"):
            continue
        try:
            import importlib

            mod = importlib.import_module(f"ml_ecosystem_snapshots.frameworks.{module_name}")
            found_any = False
            members: list[tuple[str, CollectorFn]] = inspect.getmembers(mod, callable)
            for name, obj in members:
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
    ),
    "stablehlo": ProvenanceMetadata(
        source_type="tablegen",
        upstream_commit="v1.9.0",
        upstream_version="1.9.0",
    ),
    "mlir": ProvenanceMetadata(
        source_type="tablegen",
        upstream_commit="llvmorg-19.1.0",
        upstream_version="19.1.0",
    ),
    "html_dsl": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
    ),
    "latex_dsl": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
    ),
    "tikz": ProvenanceMetadata(
        source_type="python_dsl",
        upstream_commit="ml-switcheroo-v0.0.2",
        upstream_version="0.0.2",
    ),
}

__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFn",
    "FRAMEWORK_COLLECTORS",
    "FRAMEWORK_PROVENANCE_METADATA",
    "ProvenanceMetadata",
    "get_available_frameworks",
]
