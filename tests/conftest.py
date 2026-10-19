"""Pytest global configuration and fixtures.

Adjusts process file descriptor limits for concurrent test executions and
configures environment defaults.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment and raise open file limits if possible.

    Args:
        config: Pytest configuration object.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(4096, hard) if hard > 0 else 4096
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, OSError, ValueError):
        pass

    import griffe
    from ml_ecosystem_snapshots.introspection.static_parsers import _GRIFFE_CACHE

    _GRIFFE_CACHE["__loader__"] = griffe.load
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
        _GRIFFE_CACHE.setdefault(pkg, None)
