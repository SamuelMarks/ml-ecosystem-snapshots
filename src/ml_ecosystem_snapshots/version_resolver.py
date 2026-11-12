"""Declarative package and specification version resolver backed by `FRAMEWORK_PROVENANCE_METADATA`."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import threading
import typing

import griffe

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)
from ml_ecosystem_snapshots.registry import FRAMEWORK_PROVENANCE_METADATA

_LOGGER = logging.getLogger(__name__)
_VERSION_LOCK = threading.RLock()


def _extract_version_with_griffe(module_name: str) -> str | None:
    """Extract `__version__` statically from a module using Griffe.

    Args:
        module_name: Dotted module name to inspect (e.g. 'keras', 'chex', 'optax').

    Returns:
        Extracted version string with quotes stripped, or `None` if unavailable.
    """
    try:
        loaded = griffe.load(module_name)
        if isinstance(loaded, griffe.Module):
            ver = loaded.members.get("__version__")
            if isinstance(ver, griffe.Alias):
                target = ver.final_target
                val = target.value if isinstance(target, griffe.Attribute) else None
            elif isinstance(ver, griffe.Attribute):
                val = ver.value
            else:
                val = None
            if val is not None:
                return str(val).strip("'").strip('"')
    except (
        ImportError,
        OSError,
        SyntaxError,
        ValueError,
        TypeError,
        AttributeError,
        LookupError,
        RuntimeError,
        griffe.GriffeError,
    ) as exc:
        _LOGGER.debug("Static Griffe version lookup failed for %s: %s", module_name, exc)
    return None


def _read_module_dunder_version(mod: LivePythonTarget) -> str:
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
        mod: LivePythonTarget = importlib.import_module(mod_name)
        return _read_module_dunder_version(mod)
    except (ImportError, AttributeError, TypeError, ValueError, OSError, RuntimeError) as exc:
        _LOGGER.debug("Live module version lookup failed for %s: %s", mod_name, exc)
        if dist_fallback is None:
            return default_ver
        try:
            return importlib.metadata.version(dist_fallback)
        except (
            importlib.metadata.PackageNotFoundError,
            ImportError,
            OSError,
            ValueError,
            RuntimeError,
        ) as meta_exc:
            _LOGGER.debug("Distribution metadata lookup failed for %s: %s", dist_fallback, meta_exc)
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
    with _VERSION_LOCK:
        json_file = os.path.join(anchor_dir, "frameworks", f"{pkg_prefix}_exhaustive.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, encoding="utf-8") as f:
                    h = json.load(f)
                    if isinstance(h, dict) and h.get("version"):
                        return str(h["version"])
            except (OSError, ValueError, TypeError) as exc:
                _LOGGER.debug("Failed reading exhaustive JSON version from %s: %s", json_file, exc)
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
        except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError) as exc:
            _LOGGER.debug("tensorflow-macos metadata not found, trying tensorflow-cpu: %s", exc)
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
        except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError) as exc:
            _LOGGER.debug("keras metadata lookup failed, falling back to Griffe: %s", exc)
            return griffe_extractor("keras")
    if package_name in ("cupy", "tensorflow"):
        return _resolve_tensorflow_or_cupy_version(package_name)
    if package_name == "mlir":
        for mlir_pkg in ("mlir", "mlir-native"):
            try:
                return importlib.metadata.version(mlir_pkg)
            except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError) as exc:
                _LOGGER.debug("MLIR metadata lookup failed for %s: %s", mlir_pkg, exc)
        return _read_exhaustive_json_version(anchor_dir, "mlir", FRAMEWORK_PROVENANCE_METADATA["mlir"].upstream_version)
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
    meta = FRAMEWORK_PROVENANCE_METADATA.get(package_name)
    lookup_name = package_name
    try:
        resolved: str | None = None
        if meta is not None and meta.resolution_mode == "static_spec":
            resolved = meta.upstream_version
        elif meta is not None and meta.resolution_mode == "live_module" and meta.live_module is not None:
            resolved = _resolve_live_module_version(meta.live_module, meta.upstream_version, meta.dist_name)
        elif meta is not None and meta.resolution_mode == "exhaustive_json":
            resolved = _read_exhaustive_json_version(anchor_dir, package_name, meta.upstream_version)
        else:
            if meta is not None and meta.resolution_mode == "pkg_alias" and meta.dist_name is not None:
                lookup_name = meta.dist_name
            resolved = _resolve_special_pkg_version(lookup_name, anchor_dir, griffe_extractor)

        return resolved if resolved is not None else importlib.metadata.version(lookup_name)
    except (
        importlib.metadata.PackageNotFoundError,
        ImportError,
        AttributeError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:
        _LOGGER.debug("Package version lookup failed for %s: %s", lookup_name, exc)
        fallback_meta = FRAMEWORK_PROVENANCE_METADATA.get(lookup_name) or meta
        if fallback_meta is not None and fallback_meta.griffe_fallback_module is not None:
            extracted = griffe_extractor(fallback_meta.griffe_fallback_module)
            return extracted if extracted is not None else fallback_meta.upstream_version
        return "unknown"


extract_version_with_griffe = _extract_version_with_griffe

__all__ = [
    "extract_version_with_griffe",
    "get_pkg_version",
]
