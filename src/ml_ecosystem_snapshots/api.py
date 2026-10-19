"""ML Framework Snapshots SDK.

Provides a programmatic interface for capturing and saving API signatures
from various machine learning frameworks.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import sys
import typing
from typing_extensions import TypeAlias

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pydantic

from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
from ml_ecosystem_snapshots.models import (
    JsonValue,
    SnapshotEnvelope,
    SnapshotEnvelopeDict,
    SnapshotSymbolDict,
)


class CollectorFn(typing.Protocol):
    """Protocol for framework snapshot collector callables."""

    def __call__(
        self, category: SemanticTier, include_nonpublic: bool = False
    ) -> list[GhostRef]:
        """Collect API symbols for a given semantic tier.

        Args:
            category: Semantic category tier to collect.
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of collected `GhostRef` items.
        """
        ...


if typing.TYPE_CHECKING:
    HashableKey: TypeAlias = str | int | float | bool | None | tuple["HashableKey", ...]
else:
    HashableKey: TypeAlias = typing.Union[
        str, int, float, bool, None, tuple[object, ...]
    ]


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


def _make_lazy_collector(
    module_name: str, func_name: str = "collect_api"
) -> CollectorFn:
    """Create a lazy loader for a framework collection function to defer heavy C runtime imports.

    Args:
        module_name: Submodule name under ml_ecosystem_snapshots.frameworks or absolute module path.
        func_name: Target collection function name.

    Returns:
        Callable function forwarding arguments to the dynamically loaded collector.
    """

    def _collector(
        category: SemanticTier, include_nonpublic: bool = False
    ) -> list[GhostRef]:
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
        mod = importlib.import_module(full_mod)
        func: CollectorFn = getattr(mod, func_name)
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
    "orbax": ("orbax_checkpoint", "collect_api"),
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
    import pkgutil
    import ml_ecosystem_snapshots.frameworks as frameworks_pkg

    collectors: dict[str, CollectorFn] = {}

    package = frameworks_pkg
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_"):
            continue
        try:
            import importlib

            mod = importlib.import_module(
                f"ml_ecosystem_snapshots.frameworks.{module_name}"
            )
            found_any = False
            for name, obj in vars(mod).items():
                if callable(obj) and name.startswith("collect_"):
                    found_any = True
                    if name == "collect_api":
                        identifier = module_name
                    else:
                        identifier = f"{module_name}_{name[8:]}"
                    collectors[identifier] = obj
            if not found_any:
                collectors[module_name] = _make_lazy_collector(
                    module_name, "collect_api"
                )
        except Exception:
            collectors[module_name] = _make_lazy_collector(module_name, "collect_api")

    for alias_name, (mod_name, fn_name) in _CANONICAL_COLLECTOR_SPECS.items():
        collectors[alias_name] = _make_lazy_collector(mod_name, fn_name)

    return collectors


FRAMEWORK_COLLECTORS: dict[str, CollectorFn] = get_available_frameworks()


_PKG_NAME_ALIASES: dict[str, str] = {
    "flax_nnx": "flax",
    "sklearn": "scikit-learn",
    "pytorch": "torch",
    "pax": "paxml",
    "optax": "optax",
    "huggingface": "transformers",
    "orbax_checkpoint": "orbax-checkpoint",
    "orbax": "orbax-checkpoint",
}

_STATIC_SPEC_VERSIONS: dict[str, str] = {
    "wgsl": "draft-2024",
    "metal": "3.2",
    "wasm_simd": "2.0",
    "wasm": "2.0",
    "webgl": "2.0",
    "cpp_runtime": "17",
    "cpp": "17",
    "array_api": "2024.12",
    "nccl": "2.21",
    "rccl": "2.21",
    "flash_attention": "2.6.3",
    "flash_attn": "2.6.3",
    "html_dsl": "0.0.2",
    "latex_dsl": "0.0.2",
    "tikz": "0.0.2",
}

_LIVE_MODULE_VERSION_FALLBACKS: dict[str, tuple[str, str, str | None]] = {
    "ir": ("ml_switcheroo_ir", "0.0.4", "ml-switcheroo-ir"),
    "ml_switcheroo_ir": ("ml_switcheroo_ir", "0.0.4", "ml-switcheroo-ir"),
    "scipy": ("scipy", "1.13.1", "scipy"),
    "torchvision": ("torchvision", "0.23.0", "torchvision"),
    "torchaudio": ("torchaudio", "2.8.0", "torchaudio"),
    "safetensors": ("safetensors", "0.7.0", "safetensors"),
    "aten": ("torch", "2.8.0", "torch"),
    "numba": ("numba", "0.60.0", "numba"),
    "sparse": ("sparse", "0.15.4", "sparse"),
    "dpnp": ("dpnp", "0.15.0", None),
    "awkward": ("awkward", "2.6.8", None),
    "pyarrow_compute": ("pyarrow", "17.0.0", None),
    "bohrium": ("bohrium", "0.13.0", None),
    "onnx": ("onnx", "1.17.0", "onnx"),
    "onnx_spec": ("onnx", "1.17.0", "onnx"),
}

_EXHAUSTIVE_JSON_VERSIONS: dict[str, str] = {
    "nvidia_sass": "12.6.0",
    "amd_rdna": "19.1.0",
    "nvidia_ptx": "8.5",
    "stablehlo": "1.9.0",
}


def get_pkg_version(package_name: str) -> str:
    """Safely retrieves the installed version of a python package.

    Args:
        package_name: The name of the pip package.

    Returns:
        The version string, or "unknown" if not installed.
    """
    import builtins
    import importlib.metadata

    try:
        if package_name in _STATIC_SPEC_VERSIONS:
            return _STATIC_SPEC_VERSIONS[package_name]

        if package_name in _LIVE_MODULE_VERSION_FALLBACKS:
            mod_name, default_ver, dist_fallback = _LIVE_MODULE_VERSION_FALLBACKS[
                package_name
            ]
            try:
                mod = builtins.__import__(mod_name)
                return str(getattr(mod, "__version__", default_ver))
            except Exception:
                if dist_fallback is None:
                    return default_ver
                try:
                    return importlib.metadata.version(dist_fallback)
                except Exception:
                    return default_ver
        elif package_name in _PKG_NAME_ALIASES:
            package_name = _PKG_NAME_ALIASES[package_name]
        elif package_name == "keras":
            try:
                return importlib.metadata.version("keras")
            except Exception:
                try:
                    import griffe

                    k = griffe.load("keras")
                    ver = k.members.get("__version__")
                    if ver:
                        if hasattr(ver, "target"):
                            val = getattr(getattr(ver, "target", None), "value", None)
                        else:
                            val = getattr(ver, "value", None)
                        if val is not None:
                            return str(val).strip("'\"")
                except Exception:
                    pass
        elif package_name == "cupy":
            try:
                import cupy

                return str(cupy.__version__)
            except ImportError:
                return importlib.metadata.version("cupy-cuda12x")
        elif package_name == "tensorflow":
            try:
                import tensorflow as tf

                return str(tf.__version__)
            except ImportError:
                try:
                    return importlib.metadata.version("tensorflow-macos")
                except Exception:
                    return importlib.metadata.version("tensorflow-cpu")
        elif package_name == "mlir":
            for mlir_pkg in ("mlir", "mlir-native"):
                try:
                    return importlib.metadata.version(mlir_pkg)
                except Exception:
                    pass
            json_file = os.path.join(
                os.path.dirname(__file__), "frameworks", "mlir_exhaustive.json"
            )
            if os.path.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        h = json.load(f)
                        if isinstance(h, dict) and h.get("version"):
                            return str(h["version"])
                except Exception:
                    pass
            return "19.1.0"
        elif package_name in _EXHAUSTIVE_JSON_VERSIONS:
            json_file = os.path.join(
                os.path.dirname(__file__),
                "frameworks",
                f"{package_name}_exhaustive.json",
            )
            if os.path.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        h = json.load(f)
                        if isinstance(h, dict) and h.get("version"):
                            return str(h["version"])
                except Exception:
                    pass
            return _EXHAUSTIVE_JSON_VERSIONS[package_name]

        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def _domain_meta_to_key(meta: object) -> HashableKey:
    """Convert domain metadata into a deterministic, hashable key.

    Args:
        meta: Domain metadata value (dict, list, primitive, etc.).

    Returns:
        A hashable representation of the metadata.
    """
    if meta is None:
        return None
    if isinstance(meta, dict):
        return tuple(sorted((str(k), _domain_meta_to_key(v)) for k, v in meta.items()))
    if isinstance(meta, (list, tuple, set)):
        return tuple(_domain_meta_to_key(x) for x in meta)
    return str(meta)


def _consolidate_aliases(refs: list[GhostRef]) -> list[GhostRef]:
    """Consolidate identical GhostRefs into a single reference with aliases.

    Args:
        refs: A list of GhostRef objects.

    Returns:
        A consolidated list of GhostRef objects.
    """
    consolidated: dict[HashableKey, GhostRef] = {}
    for ref in refs:
        param_sigs = tuple(
            (
                p.name,
                str(p.kind),
                p.default,
                p.annotation,
                str(getattr(p, "direction", None)),
                str(getattr(p, "role", None)),
                tuple(getattr(p, "allowed_dtypes", None) or ()),
                str(getattr(p, "rank_constraint", None)),
            )
            for p in ref.params
        )
        returns = getattr(ref, "returns", None)
        returns_sig = (
            tuple(
                (str(getattr(r, "name", None)), str(getattr(r, "type", None)))
                for r in returns
            )
            if returns
            else None
        )
        domain_type = str(getattr(ref, "domain_type", None))
        domain_metadata_key = _domain_meta_to_key(getattr(ref, "domain_metadata", None))

        key: HashableKey = (
            ref.name,
            ref.kind,
            domain_type,
            param_sigs,
            returns_sig,
            domain_metadata_key,
            ref.docstring,
        )

        if key not in consolidated:
            consolidated[key] = ref
        else:
            existing = consolidated[key]
            if ref.aliases is None:
                ref.aliases = []
            if existing.aliases is None:
                existing.aliases = []
            if len(ref.api_path) < len(existing.api_path):
                ref.aliases.append(existing.api_path)
                ref.aliases.extend(existing.aliases)
                ref.is_public = ref.is_public or existing.is_public
                consolidated[key] = ref
            else:
                existing.aliases.append(ref.api_path)
                existing.aliases.extend(ref.aliases)
                existing.is_public = existing.is_public or ref.is_public

    for ref in consolidated.values():
        if ref.aliases is not None:
            ref.aliases = sorted(list(set(ref.aliases)))
        else:
            ref.aliases = []

    return list(consolidated.values())


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


def extract_snapshot(
    framework_name: str, include_nonpublic: bool = False
) -> SnapshotEnvelopeDict:
    """Extract a snapshot dictionary for a specific framework.

    Args:
        framework_name: The framework identifier (e.g., "torch", "jax").
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A dictionary containing the version and category signatures.
        Returns an empty dict if the framework is unknown or not installed.
    """
    if framework_name not in FRAMEWORK_COLLECTORS:
        return SnapshotEnvelopeDict()

    version = get_pkg_version(framework_name)
    if version == "unknown":
        return SnapshotEnvelopeDict()

    collect_func = FRAMEWORK_COLLECTORS[framework_name]
    categories_map: dict[str, JsonValue] = {}
    snapshot_data = SnapshotEnvelopeDict(
        {
            "schema_version": "1.0.0",
            "target": framework_name,
            "version": version,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "categories": categories_map,
        }
    )
    if framework_name in FRAMEWORK_PROVENANCE_METADATA:
        for k, v in FRAMEWORK_PROVENANCE_METADATA[framework_name].items():
            if isinstance(v, list):
                snapshot_data[k] = list(v)
            else:
                snapshot_data[k] = v
    found_any = False

    def _process_category(
        cat: SemanticTier,
    ) -> tuple[str, list[SnapshotSymbolDict]]:
        """Collect, consolidate, and serialize all API symbols for a semantic tier.

        Args:
            cat: The `SemanticTier` enum category to extract.

        Returns:
            A tuple of `(category_name, serialized_ghost_ref_dicts)`.
        """
        try:
            refs = collect_func(cat, include_nonpublic)
            if refs:
                refs = _consolidate_aliases(refs)
                refs.sort(key=lambda x: x.name)
                dumped_refs: list[SnapshotSymbolDict] = []
                for r in refs:
                    d: dict[str, JsonValue] = r.model_dump(exclude_unset=True)
                    kwargs_list: list[JsonValue] = [
                        p.name
                        for p in r.params
                        if p.kind
                        in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD", "VAR_KEYWORD")
                    ]
                    d["kwargs"] = kwargs_list
                    dumped_refs.append(SnapshotSymbolDict(d))
                return cat.value, dumped_refs
        except Exception:
            pass
        return cat.value, []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_cat = {
            executor.submit(_process_category, cat): cat for cat in SemanticTier
        }
        for future in concurrent.futures.as_completed(future_to_cat):
            cat_val, dumped_refs = future.result()
            if dumped_refs:
                found_any = True
                cat_items: list[JsonValue] = [dict(r) for r in dumped_refs]
                categories_map[cat_val] = cat_items

    if not found_any:
        return SnapshotEnvelopeDict()
    return snapshot_data


def extract_snapshot_isolated(
    framework: str, include_nonpublic: bool = False, timeout: int = 300
) -> SnapshotEnvelopeDict:
    """Extract a framework snapshot within an isolated child subprocess.

    Prevents CUDA/Metal memory pre-allocation locks, framework-level C-extension
    conflicts, and memory leaks from polluting the parent process.

    Args:
        framework: Name of the framework to extract.
        include_nonpublic: Whether to include non-public APIs.
        timeout: Subprocess timeout in seconds.

    Returns:
        The extracted snapshot dictionary, or empty dict on failure.
    """
    import logging
    import subprocess

    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    src_paths = [p for p in sys.path if p]
    if existing_pypath:
        src_paths.append(existing_pypath)
    env["PYTHONPATH"] = os.pathsep.join(src_paths)

    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from ml_ecosystem_snapshots.api import extract_snapshot; "
            f"snap = extract_snapshot({repr(framework)}, include_nonpublic={include_nonpublic}); "
            "sys.stdout.write(json.dumps(snap))"
        ),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parsed_snap: object = json.loads(proc.stdout)
            assert isinstance(parsed_snap, dict)
            return SnapshotEnvelopeDict(parsed_snap)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Subprocess extraction failed for %s: %s", framework, e
        )

    return SnapshotEnvelopeDict()


def extract_all_snapshots(
    include_nonpublic: bool = False, isolated: bool = False
) -> dict[str, SnapshotEnvelopeDict]:
    """Extract snapshots for all supported and installed frameworks.

    Args:
        include_nonpublic: Whether to include non-public APIs.
        isolated: Whether to execute each framework extraction in an isolated child subprocess.

    Returns:
        A dictionary mapping framework identifiers to their snapshot data.
    """
    results: dict[str, SnapshotEnvelopeDict] = {}
    for fw in FRAMEWORK_COLLECTORS:
        if isolated:
            data = extract_snapshot_isolated(fw, include_nonpublic)
        else:
            data = extract_snapshot(fw, include_nonpublic)
        if data:
            results[fw] = data
    return results


def write_snapshot(
    framework_name: str,
    snapshot_data: typing.Mapping[str, object],
    output_dir: str,
    compress: str | None = None,
) -> str:
    """Write a snapshot dictionary to a JSON or compressed file on disk.

    Args:
        framework_name: The framework identifier.
        snapshot_data: The snapshot dictionary.
        output_dir: The directory path where the file should be written.
        compress: Optional compression algorithm ('gzip', 'gz', 'zstd').

    Returns:
        The path to the generated snapshot file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_version = snapshot_data.get("version")
    version = str(raw_version) if raw_version is not None else "unknown"
    safe_ver = version.replace("+", "_").replace(" ", "_")

    if compress in ("gzip", "gz"):
        file_path = Path(
            os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.gz")
        )
        import gzip

        with gzip.open(file_path, "wt", encoding="utf-8", newline="\n") as f:
            json.dump(dict(snapshot_data), f, indent=2, sort_keys=True)
            f.write("\n")
    elif compress == "zstd":
        file_path = Path(
            os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.zst")
        )
        raw_bytes = (
            json.dumps(dict(snapshot_data), indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        try:
            import importlib

            zstd = importlib.import_module("zstandard")
            cctx = getattr(zstd, "ZstdCompressor")()
            compressed: bytes = cctx.compress(raw_bytes)
            with open(file_path, "wb") as f:
                f.write(compressed)
        except Exception:
            import gzip

            file_path = Path(
                os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.gz")
            )
            with gzip.open(file_path, "wt", encoding="utf-8", newline="\n") as f:
                json.dump(dict(snapshot_data), f, indent=2, sort_keys=True)
                f.write("\n")
    else:
        file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json"))
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(snapshot_data), f, indent=2, sort_keys=True)
            f.write("\n")

    return str(file_path)


def validate_snapshot_envelope(
    snapshot_dict: typing.Mapping[str, JsonValue] | object,
) -> SnapshotEnvelope:
    """Validate and normalize a snapshot dictionary into a SnapshotEnvelope.

    Args:
        snapshot_dict: Raw snapshot dictionary.

    Returns:
        Validated SnapshotEnvelope instance.

    Raises:
        SnapshotSchemaError: If required fields are missing or invalid.
    """
    if not isinstance(snapshot_dict, dict):
        raise SnapshotSchemaError("Snapshot must be a dictionary.")
    target = snapshot_dict.get("target") or "unknown"
    envelope_data = dict(snapshot_dict)
    envelope_data.setdefault("target", target)
    envelope_data.setdefault("schema_version", "1.0.0")
    if "generated_at" not in envelope_data or not envelope_data["generated_at"]:
        envelope_data["generated_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    try:
        return SnapshotEnvelope.model_validate(envelope_data)
    except (ValueError, TypeError) as exc:
        raise SnapshotSchemaError(f"Invalid snapshot envelope schema: {exc}") from exc
