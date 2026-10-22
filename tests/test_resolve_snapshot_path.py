"""Unit tests and verification suite for resolve snapshot path."""

from __future__ import annotations

from pathlib import Path
import os

from unittest.mock import patch
from ml_ecosystem_snapshots.cli import resolve_snapshot_path


def test_resolve_snapshot_path_existing_file(tmp_path: Path) -> None:
    """Verify resolve snapshot path existing file behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    file_path = Path(os.path.join(tmp_path, "my_snapshot.json"))
    file_path.touch()

    resolved = resolve_snapshot_path(str(file_path))
    assert resolved == str(file_path)


def test_resolve_snapshot_path_append_json(tmp_path: Path) -> None:
    """Verify resolve snapshot path append json behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    file_path = Path(os.path.join(tmp_path, "my_snapshot.json"))
    file_path.touch()

    # Try resolving without .json extension
    base_path = str(Path(os.path.join(tmp_path, "my_snapshot")))
    resolved = resolve_snapshot_path(base_path)
    assert resolved == str(file_path)


def test_resolve_snapshot_path_fallback_to_repo(tmp_path: Path) -> None:
    """Verify resolve snapshot path fallback to repo behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    # Mock __file__ so repo root points to our temp directory
    pkg_dir = Path(
        os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots")
    )
    pkg_dir.mkdir(parents=True)

    snapshots_dir = Path(os.path.join(tmp_path, "snapshots"))
    snapshots_dir.mkdir()

    target_file = Path(os.path.join(snapshots_dir, "jax_v0.4.30.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        # Providing just the basename
        resolved = resolve_snapshot_path("jax_v0.4.30.json")
        assert resolved == str(target_file)

        # Providing a path that doesn't exist locally, but basename matches
        resolved = resolve_snapshot_path("./some_other_dir/jax_v0.4.30.json")
        assert resolved == str(target_file)

        # Providing basename without .json
        resolved = resolve_snapshot_path("jax_v0.4.30")
        assert resolved == str(target_file)

        # Providing just prefix
        resolved = resolve_snapshot_path("jax")
        assert resolved == str(target_file)


def test_resolve_snapshot_path_not_found(tmp_path: Path) -> None:
    """Verify resolve snapshot path not found behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    # Should just return the input path if nothing works
    with patch("ml_ecosystem_snapshots.cli.__file__", "/tmp/does_not_exist/cli.py"):
        resolved = resolve_snapshot_path("missing_framework")
        assert resolved == "missing_framework"


def test_resolve_multiple_matches(tmp_path: Path) -> None:
    """Verify resolve multiple matches behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(
        os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots")
    )
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
        assert res == str(Path(os.path.join(snapshots_dir, "torch-2.0.json")))


def test_resolve_no_matches_in_existing_dir(tmp_path: Path) -> None:
    """Test when snapshots_dir exists but no glob matches are found.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(
        os.path.join(Path(os.path.join(tmp_path, "src")), "ml_ecosystem_snapshots")
    )
    pkg_dir.mkdir(parents=True)

    snapshots_dir = Path(os.path.join(tmp_path, "snapshots"))
    snapshots_dir.mkdir()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        res = resolve_snapshot_path("nonexistent_framework")
        assert res == "nonexistent_framework"


def test_resolve_snapshot_path_from_pkg_snapshots(tmp_path: Path) -> None:
    """Test resolving a snapshot bundled directly inside the package snapshots folder.

    Args:
        tmp_path: Temporary directory fixture.
    """
    pkg_dir = Path(
        os.path.join(
            Path(os.path.join(tmp_path, "site-packages")), "ml_ecosystem_snapshots"
        )
    )
    snapshots_dir = Path(os.path.join(pkg_dir, "snapshots"))
    snapshots_dir.mkdir(parents=True)

    target_file = Path(os.path.join(snapshots_dir, "flax_v0.8.0.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        # Exact match with json extension
        resolved = resolve_snapshot_path("flax_v0.8.0.json")
        assert resolved == str(target_file)

        # Match without json extension
        resolved = resolve_snapshot_path("flax_v0.8.0")
        assert resolved == str(target_file)

        # Prefix match
        resolved = resolve_snapshot_path("flax")
        assert resolved == str(target_file)


def test_resolve_snapshot_path_from_pkg_frameworks(tmp_path: Path) -> None:
    """Test resolving a snapshot from the package frameworks folder.

    Args:
        tmp_path: Temporary directory fixture.
    """
    pkg_dir = Path(
        os.path.join(
            Path(os.path.join(tmp_path, "site-packages")), "ml_ecosystem_snapshots"
        )
    )
    frameworks_dir = Path(os.path.join(pkg_dir, "frameworks"))
    frameworks_dir.mkdir(parents=True)

    target_file = Path(os.path.join(frameworks_dir, "amd_rdna_exhaustive.json"))
    target_file.touch()

    with patch(
        "ml_ecosystem_snapshots.cli.__file__",
        str(Path(os.path.join(pkg_dir, "cli.py"))),
    ):
        resolved = resolve_snapshot_path("amd_rdna_exhaustive.json")
        assert resolved == str(target_file)


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
            return_value=[],
        ),
    ):
        resolved = resolve_snapshot_path("custom_v1.0.0.json")
        assert resolved == str(target_file)
        resolved_prefix = resolve_snapshot_path("custom")
        assert resolved_prefix == str(target_file)
