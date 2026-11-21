"""Tests for total `Result`-returning `resolve_snapshot_path`."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo_ir.typeclasses.outcome import Err, Ok

from ml_ecosystem_snapshots.cli import resolve_snapshot_path
from ml_ecosystem_snapshots.exceptions import SnapshotNotFoundError


def test_resolve_snapshot_path_existing_file(tmp_path: Path) -> None:
    """Verify resolve snapshot path existing file behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    f = Path(os.path.join(tmp_path, "test.json"))
    f.touch()
    assert resolve_snapshot_path(str(f)) == Ok(str(f))


def test_resolve_snapshot_path_append_json(tmp_path: Path) -> None:
    """Verify resolve snapshot path append json behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    f = Path(os.path.join(tmp_path, "test.json"))
    f.touch()
    path_without_ext = str(Path(os.path.join(tmp_path, "test")))
    assert resolve_snapshot_path(path_without_ext) == Ok(str(f))


def test_resolve_snapshot_path_fallback_to_repo(tmp_path: Path) -> None:
    """Verify resolve snapshot path fallback to repo behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots"))
    pkg_dir.mkdir(parents=True)

    snapshots_dir = Path(os.path.join(tmp_path, "snapshots"))
    snapshots_dir.mkdir()

    target_file = Path(os.path.join(snapshots_dir, "jax_v0.4.30.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        assert resolve_snapshot_path("jax_v0.4.30.json") == Ok(str(target_file))
        assert resolve_snapshot_path("./some_other_dir/jax_v0.4.30.json") == Ok(str(target_file))
        assert resolve_snapshot_path("jax_v0.4.30") == Ok(str(target_file))
        assert resolve_snapshot_path("jax") == Ok(str(target_file))


def test_resolve_snapshot_path_not_found(tmp_path: Path) -> None:
    """Verify resolve snapshot path returns Err(SnapshotNotFoundError) when missing.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    del tmp_path
    with patch("ml_ecosystem_snapshots.cli.__file__", "/tmp/does_not_exist/cli.py"):
        resolved = resolve_snapshot_path("missing_framework")
        assert isinstance(resolved, Err)
        assert isinstance(resolved.error, SnapshotNotFoundError)


def test_resolve_multiple_matches(tmp_path: Path) -> None:
    """Verify resolve multiple matches behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots"))
    pkg_dir.mkdir(parents=True)

    snapshots_dir = Path(os.path.join(tmp_path, "snapshots"))
    snapshots_dir.mkdir()

    Path(os.path.join(snapshots_dir, "torch-1.12.json")).touch()
    Path(os.path.join(snapshots_dir, "torch-2.0.json")).touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        res = resolve_snapshot_path("torch")
        assert res == Ok(str(Path(os.path.join(snapshots_dir, "torch-2.0.json"))))


def test_resolve_no_matches_in_existing_dir(tmp_path: Path) -> None:
    """Test when snapshots_dir exists but no glob matches are found.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots"))
    pkg_dir.mkdir(parents=True)

    snapshots_dir = Path(os.path.join(tmp_path, "snapshots"))
    snapshots_dir.mkdir()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        res = resolve_snapshot_path("nonexistent_framework")
        assert isinstance(res, Err)
        assert isinstance(res.error, SnapshotNotFoundError)


def test_resolve_snapshot_path_from_pkg_snapshots(tmp_path: Path) -> None:
    """Test resolving a snapshot bundled directly inside the package snapshots folder.

    Args:
        tmp_path: Temporary directory fixture.
    """
    pkg_dir = Path(os.path.join(Path(os.path.join(tmp_path, "site-packages")), "ml_ecosystem_snapshots"))
    snapshots_dir = Path(os.path.join(pkg_dir, "snapshots"))
    snapshots_dir.mkdir(parents=True)

    target_file = Path(os.path.join(snapshots_dir, "flax_v0.8.0.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        assert resolve_snapshot_path("flax_v0.8.0.json") == Ok(str(target_file))
        assert resolve_snapshot_path("flax_v0.8.0") == Ok(str(target_file))
        assert resolve_snapshot_path("flax") == Ok(str(target_file))


def test_resolve_snapshot_path_from_pkg_frameworks(tmp_path: Path) -> None:
    """Test resolving a snapshot from the package frameworks folder.

    Args:
        tmp_path: Temporary directory fixture.
    """
    pkg_dir = Path(os.path.join(Path(os.path.join(tmp_path, "site-packages")), "ml_ecosystem_snapshots"))
    frameworks_dir = Path(os.path.join(pkg_dir, "frameworks"))
    frameworks_dir.mkdir(parents=True)

    target_file = Path(os.path.join(frameworks_dir, "amd_rdna_exhaustive.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        assert resolve_snapshot_path("amd_rdna_exhaustive.json") == Ok(str(target_file))


def test_resolve_snapshot_path_from_cwd_snapshots(tmp_path: Path) -> None:
    """Test resolving a snapshot from the current working directory snapshots folder.

    Args:
        tmp_path: Temporary directory fixture.
    """
    cwd_snapshots = Path(os.path.join(tmp_path, "snapshots"))
    cwd_snapshots.mkdir(parents=True)

    target_file = Path(os.path.join(cwd_snapshots, "custom_v1.0.0.json"))
    target_file.touch()

    with (
        patch("os.getcwd", return_value=str(tmp_path)),
        patch(
            "ml_ecosystem_snapshots.cli.__file__",
            "/nonexistent/cli.py",
        ),
        patch(
            "ml_ecosystem_snapshots.index.get_cache_dir",
            return_value=str(tmp_path / "empty_cache"),
        ),
        patch(
            "ml_ecosystem_snapshots.network_env.get_custom_snapshots_paths",
            return_value=(),
        ),
    ):
        assert resolve_snapshot_path("custom_v1.0.0.json") == Ok(str(target_file))
        assert resolve_snapshot_path("custom") == Ok(str(target_file))
