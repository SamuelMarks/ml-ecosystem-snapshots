"""Unit test suite for backward-compatibility shim layer.

Verifies that legacy imports from `ml_framework_snapshots` map transparently
and identically to `ml_ecosystem_snapshots`.
"""

from __future__ import annotations

import sys

from ml_ecosystem_snapshots._alias import (
    AliasFinder,
    AliasLoader,
    register_alias_finder,
)


def test_top_level_shim_import() -> None:
    """Verify that importing ml_framework_snapshots re-exports canonical attributes."""
    import ml_framework_snapshots
    import ml_ecosystem_snapshots

    assert hasattr(ml_framework_snapshots, "__version__")
    assert hasattr(ml_framework_snapshots, "extract_snapshot")
    assert hasattr(ml_framework_snapshots, "extract_all_snapshots")
    assert hasattr(ml_framework_snapshots, "write_snapshot")
    assert ml_framework_snapshots.__version__ == ml_ecosystem_snapshots.__version__


def test_submodule_shim_import() -> None:
    """Verify that submodules under ml_framework_snapshots resolve correctly."""
    import ml_framework_snapshots.api as fw_api
    import ml_ecosystem_snapshots.api as eco_api

    assert fw_api.extract_snapshot is eco_api.extract_snapshot

    import ml_framework_snapshots.models as fw_models
    import ml_ecosystem_snapshots.models as eco_models

    assert fw_models.GhostRef is eco_models.GhostRef
    assert fw_models.GhostInspector is eco_models.GhostInspector


def test_framework_submodules_shim_import() -> None:
    """Verify that framework collectors can be imported via legacy path."""
    import ml_framework_snapshots.frameworks.torch as fw_torch
    import ml_ecosystem_snapshots.frameworks.torch as eco_torch

    assert fw_torch.collect_api is eco_torch.collect_api

    import ml_framework_snapshots.frameworks.numpy as fw_numpy
    import ml_ecosystem_snapshots.frameworks.numpy as eco_numpy

    assert fw_numpy.collect_api is eco_numpy.collect_api


def test_alias_finder_and_loader_mechanisms() -> None:
    """Verify AliasFinder and AliasLoader branches and error handling."""
    finder = AliasFinder()

    # Non-matching spec returns None
    assert finder.find_spec("unrelated_module") is None

    # Matching spec returns ModuleSpec with AliasLoader
    spec = finder.find_spec("ml_framework_snapshots.utils")
    assert spec is not None
    assert spec.loader is not None
    assert isinstance(spec.loader, AliasLoader)

    # Loader create_module returns the target module
    target_mod = spec.loader.create_module(spec)
    assert target_mod is not None

    # exec_module is a safe no-op
    spec.loader.exec_module(target_mod)

    # Branch where target module has no __spec__ (original_spec is None)
    import types

    dummy_mod = types.ModuleType("dummy_module")
    loader_no_spec = AliasLoader(dummy_mod)
    assert loader_no_spec.original_spec is None
    loader_no_spec.exec_module(dummy_mod)

    # Branch where parent_name is not in sys.modules
    sys.modules.pop("ml_framework_snapshots", None)
    spec2 = finder.find_spec("ml_framework_snapshots.models")
    assert spec2 is not None

    # Idempotent registration
    register_alias_finder()
    register_alias_finder()


def test_all_shim_files_execution() -> None:
    """Verify that all physical backward-compatibility shim files execute cleanly and export modules."""
    import os
    import runpy

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shim_dir = os.path.join(repo_root, "src", "ml_framework_snapshots")
    assert os.path.isdir(shim_dir)
    executed_count: int = 0
    for root, _, files in os.walk(shim_dir):
        for f in sorted(files):
            if f.endswith(".py"):
                fpath = os.path.join(root, f)
                res = runpy.run_path(fpath)
                assert "_orig_mod" in res
                executed_count += 1
    assert executed_count == 74
