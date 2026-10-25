"""Declarative package and specification version resolver for ML frameworks."""

from __future__ import annotations

import builtins
import importlib.metadata
import json
import os
import typing

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    has_runtime_attr,
    read_runtime_attr,
)


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

        loaded: InspectableTarget = _griffe_mod.load(module_name)
        if not isinstance(loaded, str) and has_runtime_attr(loaded, "members"):
            members_obj = read_runtime_attr(loaded, "members", None)
            get_fn: typing.Callable[[str], InspectableTarget] = getattr(
                members_obj, "get"
            )
            ver = get_fn("__version__")
            if ver is not None and has_runtime_attr(ver, "target"):
                target = read_runtime_attr(ver, "target", None)
                val = (
                    read_runtime_attr(target, "value", None)
                    if target is not None and has_runtime_attr(target, "value")
                    else None
                )
            elif ver is not None and has_runtime_attr(ver, "value"):
                val = read_runtime_attr(ver, "value", None)
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
                griffe_ver = griffe_extractor("keras")
                if griffe_ver is not None:
                    return griffe_ver
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
            json_file = os.path.join(anchor_dir, "frameworks", "mlir_exhaustive.json")
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
                anchor_dir,
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
