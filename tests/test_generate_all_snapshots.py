"""Tests for the generate_all_snapshots.py script."""

from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
import sys

import pytest
from ml_ecosystem_snapshots.models import JsonValue

import scripts.generate_all_snapshots as generate_all_snapshots


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

    def fake_extract(fw: str) -> dict[str, JsonValue]:
        """Record extract_snapshot invocation.

        Args:
            fw: Framework identifier.

        Returns:
            Dummy snapshot dictionary.
        """
        extract_calls.append(fw)
        return {"mock": "data"}

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> str:
        """Record write_snapshot invocation.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory path.

        Returns:
            Dummy written path.
        """
        write_calls.append((fw, data, out_dir))
        return os.path.join(out_dir, f"{fw}.json")

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
    """Test the main function when snapshot extraction fails.

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

    def fake_extract(fw: str) -> dict[str, JsonValue]:
        """Raise an error during snapshot extraction.

        Args:
            fw: Framework identifier.

        Raises:
            RuntimeError: Simulated extraction failure.
        """
        extract_calls.append(fw)
        raise RuntimeError("Mock exception")

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> str:
        """Record unexpected write calls.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            Dummy path.
        """
        write_calls.append(fw)
        return ""

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py"])
    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    monkeypatch.setattr(generate_all_snapshots, "extract_snapshot", fake_extract)
    monkeypatch.setattr(generate_all_snapshots, "write_snapshot", fake_write)

    with pytest.raises(SystemExit):
        generate_all_snapshots.main()

    frameworks = generate_all_snapshots.all_frameworks
    assert extract_calls == list(frameworks)
    assert write_calls == []

    captured = capsys.readouterr()
    for fw in frameworks:
        assert f"Building snapshot for {fw}..." in captured.out
    assert "  -> Failed: Mock exception" in captured.out


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

    def fake_extract_iso(fw: str) -> dict[str, JsonValue]:
        """Record isolated extraction call.

        Args:
            fw: Framework identifier.

        Returns:
            Dummy snapshot payload.
        """
        iso_calls.append(fw)
        return {"custom": "snap"}

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> str:
        """Record write call.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            Dummy path.
        """
        write_calls.append(fw)
        return ""

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
    """Test main function when isolated extraction returns empty snapshot and falls back to in-process.

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

    def fake_extract_iso(fw: str) -> dict[str, JsonValue]:
        """Return empty dict to trigger fallback.

        Args:
            fw: Framework identifier.

        Returns:
            Empty dict.
        """
        iso_calls.append(fw)
        return {}

    def fake_extract(fw: str) -> dict[str, JsonValue]:
        """Return fallback snapshot payload.

        Args:
            fw: Framework identifier.

        Returns:
            Fallback payload.
        """
        extract_calls.append(fw)
        return {"fallback": "data"}

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> str:
        """Record write call.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            Dummy path.
        """
        write_calls.append(fw)
        return ""

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

    def fake_extract(fw: str) -> dict[str, JsonValue]:
        """Record extract_snapshot call from __main__.

        Args:
            fw: Framework identifier.

        Returns:
            Empty categories dict.
        """
        extract_calls.append(fw)
        return {"categories": {}}

    def fake_write(fw: str, data: dict[str, JsonValue], out_dir: str) -> str:
        """Record write_snapshot call from __main__.

        Args:
            fw: Framework identifier.
            data: Snapshot payload.
            out_dir: Output directory.

        Returns:
            Dummy path.
        """
        write_calls.append(fw)
        return ""

    monkeypatch.setattr(sys, "argv", ["generate_all_snapshots.py", "tikz"])
    monkeypatch.setattr(api_mod, "extract_snapshot", fake_extract)
    monkeypatch.setattr(api_mod, "write_snapshot", fake_write)

    runpy.run_path(_get_script_path(), run_name="__main__")
    assert extract_calls == ["tikz"]
    assert write_calls == ["tikz"]
