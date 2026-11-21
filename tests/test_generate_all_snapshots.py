"""Tests for the generate_all_snapshots.py script."""

from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
import sys

import pytest
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

import scripts.generate_all_snapshots as generate_all_snapshots
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import JsonValue


def _get_script_path() -> str:
    """Resolve the filesystem path to scripts/generate_all_snapshots.py without dunder access.

    Returns:
        Absolute filesystem path string.
    """
    spec = importlib.util.find_spec("scripts.generate_all_snapshots")
    assert spec is not None and spec.origin is not None
    return spec.origin


def test_main_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test the main function for successful snapshot generation.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    makedirs_calls: list[tuple[str, bool]] = []
    extract_calls: list[str] = []
    write_calls: list[tuple[str, dict[str, JsonValue], str]] = []

    def fake_makedirs(path: str, exist_ok: bool = False) -> None:
        """Record os.makedirs invocation.

        Args:
            path: Target directory path.
            exist_ok: Whether existing directory is allowed.
        """
        makedirs_calls.append((path, exist_ok))

    def fake_extract(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Record extract_snapshot invocation.

        Args:
            fw: Framework identifier.

        Returns:
            `Ok` wrapping dummy snapshot dictionary.
        """
        extract_calls.append(fw)
        return Ok({"mock": "data"})

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> Result[SnapshotError, str]:
        """Record write_snapshot invocation.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory path.

        Returns:
            `Ok` wrapping dummy written path.
        """
        write_calls.append((fw, data, out_dir))
        return Ok(os.path.join(out_dir, f"{fw}.json"))

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py"])
    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot", fake_extract)
    monkeypatch.setattr(generate_all_snapshots, "write_snapshot", fake_write)

    generate_all_snapshots.main()

    assert makedirs_calls == [(os.path.join("src", "ml_ecosystem_snapshots", "snapshots"), True)]

    frameworks = generate_all_snapshots.all_frameworks
    assert extract_calls == list(frameworks)
    assert len(write_calls) == len(frameworks)

    captured = capsys.readouterr()
    for fw in frameworks:
        assert (
            fw,
            {"mock": "data"},
            os.path.join("src", "ml_ecosystem_snapshots", "snapshots"),
        ) in write_calls
        assert f"Building snapshot for {fw}..." in captured.out
    assert "  -> Saved" in captured.out


def test_main_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test the main function when snapshot extraction or write fails via Err and exception.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    extract_calls: list[str] = []
    write_calls: list[str] = []

    def fake_makedirs(path: str, exist_ok: bool = False) -> None:
        """No-op directory creation.

        Args:
            path: Target directory path.
            exist_ok: Whether existing directory is allowed.
        """

    def fake_extract(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Fail via Err, write Err, or raised RuntimeError depending on fw.

        Args:
            fw: Framework identifier.

        Raises:
            RuntimeError: Simulated extraction failure on first framework.

        Returns:
            `Err` or `Ok` outcome.
        """
        extract_calls.append(fw)
        if len(extract_calls) == 1:
            raise RuntimeError("Mock exception")
        if len(extract_calls) == 2:
            return Err(SnapshotError("Mock extract Err"))
        return Ok({"data": 1})

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> Result[SnapshotError, str]:
        """Return Err on write to exercise write failure branch.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            `Err(SnapshotError)`.
        """
        del data, out_dir
        write_calls.append(fw)
        return Err(SnapshotError("Mock write Err"))

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py", "torch", "jax", "flax"])
    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot", fake_extract)
    monkeypatch.setattr(generate_all_snapshots, "write_snapshot", fake_write)

    with pytest.raises(SystemExit):
        generate_all_snapshots.main()

    assert extract_calls == ["torch", "jax", "flax"]
    assert write_calls == ["flax"]

    captured = capsys.readouterr()
    assert "  -> Failed: Mock exception" in captured.out
    assert "  -> Failed: Mock extract Err" in captured.out
    assert "  -> Failed: Mock write Err" in captured.out


def test_sys_path_insert() -> None:
    """Test module reload when _src_dir is not in sys.path."""
    script_path = _get_script_path()
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(script_path), "..", "src"))
    orig_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p != _src_dir]
        importlib.reload(generate_all_snapshots)
        assert _src_dir in sys.path
    finally:
        sys.path = orig_path


def test_main_custom_args_and_isolated_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test main function with CLI arguments and successful isolated extraction.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    iso_calls: list[str] = []
    write_calls: list[str] = []

    def fake_makedirs(path: str, exist_ok: bool = False) -> None:
        """No-op directory creation.

        Args:
            path: Target directory path.
            exist_ok: Whether existing directory is allowed.
        """

    def fake_extract_iso(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Record isolated extraction call.

        Args:
            fw: Framework identifier.

        Returns:
            `Ok` wrapping dummy snapshot payload.
        """
        iso_calls.append(fw)
        return Ok({"custom": "snap"})

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> Result[SnapshotError, str]:
        """Record write call.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            `Ok` wrapping dummy path.
        """
        del data, out_dir
        write_calls.append(fw)
        return Ok("")

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py", "torch", "jax"])
    monkeypatch.setenv("ISOLATE_EXTRACTION", "1")
    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot_isolated", fake_extract_iso)
    monkeypatch.setattr(generate_all_snapshots, "write_snapshot", fake_write)

    generate_all_snapshots.main()

    assert iso_calls == ["torch", "jax"]
    assert write_calls == ["torch", "jax"]


def test_main_isolated_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test main function when isolated extraction returns Err and falls back to in-process.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    iso_calls: list[str] = []
    extract_calls: list[str] = []
    write_calls: list[str] = []

    def fake_makedirs(path: str, exist_ok: bool = False) -> None:
        """No-op directory creation.

        Args:
            path: Target directory path.
            exist_ok: Whether existing directory is allowed.
        """

    def fake_extract_iso(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Return Err to trigger fallback.

        Args:
            fw: Framework identifier.

        Returns:
            `Err(SnapshotError)`.
        """
        iso_calls.append(fw)
        return Err(SnapshotError("iso failed"))

    def fake_extract(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Return fallback snapshot payload.

        Args:
            fw: Framework identifier.

        Returns:
            `Ok` wrapping fallback payload.
        """
        extract_calls.append(fw)
        return Ok({"fallback": "data"})

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> Result[SnapshotError, str]:
        """Record write call.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            `Ok` wrapping dummy path.
        """
        del data, out_dir
        write_calls.append(fw)
        return Ok("")

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py", "torch"])
    monkeypatch.setenv("ISOLATE_EXTRACTION", "1")
    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot_isolated", fake_extract_iso)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot", fake_extract)
    monkeypatch.setattr(generate_all_snapshots, "write_snapshot", fake_write)

    generate_all_snapshots.main()

    assert iso_calls == ["torch"]
    assert extract_calls == ["torch"]
    assert write_calls == ["torch"]


def test_generate_all_snapshots_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test running scripts/generate_all_snapshots.py as __main__.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import ml_ecosystem_snapshots.api as api_mod

    extract_calls: list[str] = []
    write_calls: list[str] = []

    def fake_extract(fw: str) -> Result[SnapshotError, dict[str, JsonValue]]:
        """Record extract_snapshot call from __main__.

        Args:
            fw: Framework identifier.

        Returns:
            `Ok` wrapping empty categories dict.
        """
        extract_calls.append(fw)
        return Ok({"categories": {}})

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> Result[SnapshotError, str]:
        """Record write_snapshot call from __main__.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            `Ok` wrapping dummy path.
        """
        del data, out_dir
        write_calls.append(fw)
        return Ok("")

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py", "tikz"])
    monkeypatch.setattr(api_mod, "extract_snapshot", fake_extract)
    monkeypatch.setattr(api_mod, "write_snapshot", fake_write)

    runpy.run_path(_get_script_path(), run_name="__main__")
    assert extract_calls == ["tikz"]
    assert write_calls == ["tikz"]
