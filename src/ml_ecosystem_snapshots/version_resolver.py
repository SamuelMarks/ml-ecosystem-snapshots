"""Declarative package and specification version resolver for ML frameworks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import typing

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    get_member_dict,
)


@typing.runtime_checkable
class _GriffeValueProtocol(typing.Protocol):
    """Protocol for Griffe AST objects exposing a static `.value` attribute."""

    value: str | None


@typing.runtime_checkable
class _GriffeTargetProtocol(typing.Protocol):
    """Protocol for Griffe alias objects exposing a `.target` attribute."""

    target: _GriffeValueProtocol | str | None


@typing.runtime_checkable
class _GriffeMembersMappingProtocol(typing.Protocol):
    """Protocol for a Griffe members mapping supporting `.get(key)`."""

    def get(self, key: str) -> _GriffeTargetProtocol | _GriffeValueProtocol | None:
        """Retrieve a member by name.

        Args:
            key: Member attribute name.

        Returns:
            Matching Griffe member object or `None`.
        """


@typing.runtime_checkable
class _GriffeModuleProtocol(typing.Protocol):
    """Protocol for a loaded Griffe module exposing a `.members` dictionary."""

    members: _GriffeMembersMappingProtocol


_PKG_NAME_ALIASES: dict[str, str] = {
    "flax_nnx": "flax",
    "sklearn": "scikit-learn",
    "pytorch": "torch",
    "pax": "paxml",
    "chex": "chex",
    "grain": "grain",
    "pygrain": "grain",
    "optax": "optax",
    "huggingface": "transformers",
    "orbax_checkpoint": "orbax-checkpoint",
    "orbax": "orbax-checkpoint",
}

_FALLBACK_PKG_VERSIONS: dict[str, tuple[str, str]] = {
    "chex": ("chex", "0.1.87"),
    "grain": ("grain", "0.2.2"),
    "optax": ("optax", "0.2.3"),
    "orbax-checkpoint": ("orbax.checkpoint", "0.6.4"),
}


def _extract_version_with_griffe(module_name: str) -> str | None:
    """Extract `__version__` statically from a module using the custom Griffe fork.

    Args:
        module_name: Dotted module name to inspect (e.g. 'keras', 'chex', 'optax').

    Returns:
        Extracted version string with quotes stripped, or `None` if unavailable.
    """
    try:
        import griffe as _griffe_mod

        loaded: _GriffeModuleProtocol | InspectableTarget = _griffe_mod.load(module_name)
        if not isinstance(loaded, str) and isinstance(loaded, _GriffeModuleProtocol):
            ver = loaded.members.get("__version__")
            if isinstance(ver, _GriffeTargetProtocol):
                target = ver.target
                val = target.value if isinstance(target, _GriffeValueProtocol) else None
            elif isinstance(ver, _GriffeValueProtocol):
                val = ver.value
            else:
                val = None
            if val is not None:
                return str(val).strip("'\"")
    except Exception:
        pass
    return None


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


def _read_module_dunder_version(mod: InspectableTarget) -> str:
    """Read `__version__` from a loaded module via `get_member_dict`.

    Args:
        mod: Loaded Python module object.

    Returns:
        Version string.

    Raises:
        AttributeError: If `__version__` is absent on `mod`.
    """
    ver = get_member_dict(mod).get("__version__")
    if ver is None:
        raise AttributeError("__version__")
    return str(ver)


def _resolve_live_module_version(mod_name: str, default_ver: str, dist_fallback: str | None) -> str:
    """Resolve version from a live imported module, falling back to dist metadata or default.

    Args:
        mod_name: Importable module name.
        default_ver: Canonical fallback version string.
        dist_fallback: Optional pip distribution name for `importlib.metadata.version`.

    Returns:
        Resolved version string.
    """
    try:
        mod: InspectableTarget = __import__(mod_name)
        return _read_module_dunder_version(mod)
    except Exception:
        if dist_fallback is None:
            return default_ver
        try:
            return importlib.metadata.version(dist_fallback)
        except Exception:
            return default_ver


def _read_exhaustive_json_version(anchor_dir: str, pkg_prefix: str, default_ver: str) -> str:
    """Read the `version` header from a bundled `<pkg_prefix>_exhaustive.json` file.

    Args:
        anchor_dir: Directory containing `frameworks/`.
        pkg_prefix: Framework prefix (e.g. 'mlir', 'nvidia_sass').
        default_ver: Default version if the file is absent or unreadable.

    Returns:
        Resolved version string.
    """
    json_file = os.path.join(anchor_dir, "frameworks", f"{pkg_prefix}_exhaustive.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, encoding="utf-8") as f:
                h = json.load(f)
                if isinstance(h, dict) and h.get("version"):
                    return str(h["version"])
        except Exception:
            pass
    return default_ver


def _resolve_tensorflow_or_cupy_version(package_name: str) -> str | None:
    """Resolve version for `cupy` or `tensorflow` with platform distribution fallbacks.

    Args:
        package_name: Package name ('cupy' or 'tensorflow').

    Returns:
        Resolved version string or None.
    """
    if package_name == "cupy":
        try:
            import cupy

            return _read_module_dunder_version(cupy)
        except ImportError:
            return importlib.metadata.version("cupy-cuda12x")
    try:
        import tensorflow as tf

        return _read_module_dunder_version(tf)
    except ImportError:
        try:
            return importlib.metadata.version("tensorflow-macos")
        except Exception:
            return importlib.metadata.version("tensorflow-cpu")


def _resolve_special_pkg_version(
    package_name: str,
    anchor_dir: str,
    griffe_extractor: typing.Callable[[str], str | None],
) -> str | None:
    """Resolve version for special packages (`keras`, `cupy`, `tensorflow`, `mlir`).

    Args:
        package_name: Normalized package name.
        anchor_dir: Package anchor directory.
        griffe_extractor: Griffe static version extractor callback.

    Returns:
        Resolved version string, or None to fall through to `importlib.metadata.version`.
    """
    if package_name == "keras":
        try:
            return importlib.metadata.version("keras")
        except Exception:
            return griffe_extractor("keras")
    if package_name in ("cupy", "tensorflow"):
        return _resolve_tensorflow_or_cupy_version(package_name)
    if package_name == "mlir":
        for mlir_pkg in ("mlir", "mlir-native"):
            try:
                return importlib.metadata.version(mlir_pkg)
            except Exception:
                pass
        return _read_exhaustive_json_version(anchor_dir, "mlir", "19.1.0")
    return None


def get_pkg_version(
    package_name: str,
    base_file: str | None = None,
    extract_griffe_fn: typing.Callable[[str], str | None] | None = None,
) -> str:
    """Safely retrieve the installed or canonical version of a framework package.

    Args:
        package_name: The name of the framework or pip package.
        base_file: Optional anchor file path used to locate bundled exhaustive JSON files.
        extract_griffe_fn: Optional Griffe version extractor callback.

    Returns:
        The version string, or "unknown" if not installed.
    """
    anchor_dir = os.path.dirname(base_file or __file__)
    griffe_extractor = extract_griffe_fn or _extract_version_with_griffe
    try:
        resolved: str | None = _STATIC_SPEC_VERSIONS.get(package_name)
        if resolved is None and package_name in _LIVE_MODULE_VERSION_FALLBACKS:
            mod_name, default_ver, dist_fallback = _LIVE_MODULE_VERSION_FALLBACKS[package_name]
            resolved = _resolve_live_module_version(mod_name, default_ver, dist_fallback)
        elif resolved is None and package_name in _PKG_NAME_ALIASES:
            package_name = _PKG_NAME_ALIASES[package_name]
        elif resolved is None and package_name in _EXHAUSTIVE_JSON_VERSIONS:
            resolved = _read_exhaustive_json_version(anchor_dir, package_name, _EXHAUSTIVE_JSON_VERSIONS[package_name])
        elif resolved is None:
            resolved = _resolve_special_pkg_version(package_name, anchor_dir, griffe_extractor)

        return resolved if resolved is not None else importlib.metadata.version(package_name)
    except Exception:
        if package_name in _FALLBACK_PKG_VERSIONS:
            griffe_mod, fallback_ver = _FALLBACK_PKG_VERSIONS[package_name]
            extracted = griffe_extractor(griffe_mod)
            return extracted if extracted is not None else fallback_ver
        return "unknown"


__all__ = [
    "_EXHAUSTIVE_JSON_VERSIONS",
    "_FALLBACK_PKG_VERSIONS",
    "_LIVE_MODULE_VERSION_FALLBACKS",
    "_PKG_NAME_ALIASES",
    "_STATIC_SPEC_VERSIONS",
    "_extract_version_with_griffe",
    "get_pkg_version",
]
