"""Pytest global configuration and fixtures.

Adjusts process file descriptor limits for concurrent test executions and
configures environment defaults.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types
from collections.abc import Iterator, Mapping

import pytest

from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget


class FakePackageBuilder:
    """Typed builder registering fake Python modules and package versions via `monkeypatch`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize the builder with a scoped `pytest.MonkeyPatch` instance.

        Args:
            monkeypatch: Pytest `MonkeyPatch` fixture restoring `sys.modules` and attributes on teardown.
        """
        self._mp = monkeypatch

    def build_module(
        self,
        name: str,
        version: str | None = None,
        members: Mapping[str, LivePythonTarget] | None = None,
        submodules: Mapping[str, types.ModuleType] | None = None,
        register: bool = True,
    ) -> types.ModuleType:
        """Construct a typed `types.ModuleType` and optionally register it in `sys.modules`.

        Args:
            name: Fully qualified module name (e.g. `'cupy'` or `'triton.language'`).
            version: Optional `__version__` string attribute to attach.
            members: Optional mapping of attribute names to values attached to the module.
            submodules: Optional mapping of child attribute names to child `ModuleType` objects.
            register: If True, register the module (and dotted submodules) in `sys.modules`.

        Returns:
            Constructed `types.ModuleType` instance.
        """
        mod = types.ModuleType(name)
        if version is not None:
            self._mp.setattr(mod, "__version__", version, raising=False)
        if members is not None:
            for attr_name, attr_val in members.items():
                self._mp.setattr(mod, attr_name, attr_val, raising=False)
        if submodules is not None:
            for sub_attr, sub_mod in submodules.items():
                self._mp.setattr(mod, sub_attr, sub_mod, raising=False)
                if register:
                    self._mp.setitem(sys.modules, f"{name}.{sub_attr}", sub_mod)
        if register:
            self._mp.setitem(sys.modules, name, mod)
        return mod

    def block_imports(self, *module_names: str) -> None:
        """Block imports for each specified module name via `monkeypatch.setitem(sys.modules, name, None)`.

        Args:
            *module_names: Module names whose imports should raise `ModuleNotFoundError`.
        """
        for mod_name in module_names:
            self._mp.setitem(sys.modules, mod_name, None)

    def set_metadata_versions(self, versions: Mapping[str, str]) -> None:
        """Configure `importlib.metadata.version` to serve `versions` and raise `PackageNotFoundError` otherwise.

        Args:
            versions: Mapping from distribution name to version string.
        """
        version_table = dict(versions)

        def _lookup_version(dist_name: str) -> str:
            """Return configured distribution version or raise `PackageNotFoundError`.

            Args:
                dist_name: Distribution package name.

            Returns:
                Configured version string.

            Raises:
                importlib.metadata.PackageNotFoundError: If `dist_name` is absent from `version_table`.
            """
            if dist_name in version_table:
                return version_table[dist_name]
            raise importlib.metadata.PackageNotFoundError(dist_name)

        self._mp.setattr(importlib.metadata, "version", _lookup_version)


@pytest.fixture
def fake_package_builder(monkeypatch: pytest.MonkeyPatch) -> FakePackageBuilder:
    """Provide a `monkeypatch`-scoped `FakePackageBuilder` for constructing fake Python packages.

    Args:
        monkeypatch: Pytest `MonkeyPatch` fixture.

    Returns:
        Scoped `FakePackageBuilder` instance.
    """
    return FakePackageBuilder(monkeypatch)


@pytest.fixture
def mock_torch_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Provide real `torch` or a `monkeypatch`-scoped mock `torch` module in `sys.modules`.

    Args:
        monkeypatch: Pytest `MonkeyPatch` fixture restoring `sys.modules` on teardown.

    Returns:
        Active `torch` module instance.
    """
    from torch_mock import mock_torch_module as _install_scoped_torch

    return _install_scoped_torch(monkeypatch)


def _seed_default_griffe_cache() -> None:
    """Populate default `None` sentinels in `DEFAULT_GRIFFE_CACHE` for heavy ML frameworks."""
    import griffe

    from ml_ecosystem_snapshots.introspection.static_parsers import DEFAULT_GRIFFE_CACHE

    DEFAULT_GRIFFE_CACHE.put("__loader__", griffe.load)
    for pkg in (
        "torch",
        "scipy",
        "awkward",
        "numba",
        "pyarrow",
        "tensorflow",
        "jax",
        "flax",
        "optax",
        "orbax",
        "keras",
        "dask",
        "cupy",
        "sparse",
        "dpnp",
        "bohrium",
        "safetensors",
        "torchvision",
        "torchaudio",
        "transformers",
        "deepspeed",
        "onnx",
        "onnxruntime",
        "triton",
    ):
        if not DEFAULT_GRIFFE_CACHE.contains(pkg):
            DEFAULT_GRIFFE_CACHE.put(pkg, None)


@pytest.fixture(autouse=True)
def _reset_griffe_cache_between_tests() -> Iterator[None]:
    """Reset `_GRIFFE_CACHE` after every test to prevent cross-worker state leakage.

    Yields:
        Control to the active test function before restoring default cache entries.
    """
    from ml_ecosystem_snapshots.introspection.static_parsers import clear_griffe_cache

    yield
    clear_griffe_cache()
    _seed_default_griffe_cache()


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment and raise open file limits if possible.

    Args:
        config: Pytest configuration object.
    """
    del config
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(4096, hard) if hard > 0 else 4096
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, OSError, ValueError) as exc:
        _ = str(exc)

    _seed_default_griffe_cache()
