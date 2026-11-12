"""End-to-end integration tests for the `ml-snapshots` CLI against real `tmp_path` files."""

from __future__ import annotations

import argparse
import ast
import gzip
import importlib.util
import json
import os
import runpy
import typing
from pathlib import Path

import pytest
import pytest_mock
from conftest import FakePackageBuilder

from ml_ecosystem_snapshots.cli import (
    cmd_capture,
    cmd_download_all,
    cmd_export,
    cmd_index,
    cmd_list_snapshots,
    cmd_pull,
    cmd_verify_local_cache,
    main,
)
from ml_ecosystem_snapshots.models import JsonValue


def _sample_snapshot_dict() -> dict[str, JsonValue]:
    """Build a realistic v2 snapshot envelope dictionary for CLI integration tests.

    Returns:
        Snapshot envelope dictionary with class and function symbols.
    """
    return {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "2.4.0",
        "categories": {
            "layers": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "docstring": "Apply an affine linear transformation.",
                    "params": [
                        {"name": "in_features", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "int"},
                        {"name": "out_features", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "int"},
                        {
                            "name": "bias",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "bool",
                            "default": "True",
                        },
                    ],
                }
            ],
            "ops": [
                {
                    "name": "relu",
                    "api_path": "torch.nn.functional.relu",
                    "kind": "function",
                    "docstring": "Apply rectified linear unit element-wise.",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Tensor"},
                        {
                            "name": "inplace",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "bool",
                            "default": "False",
                        },
                    ],
                    "returns_type": "Tensor",
                }
            ],
        },
    }


def test_cli_capture_real_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_package_builder: FakePackageBuilder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `capture` extracts real snapshots to `tmp_path` and handles missing/unsupported targets.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        fake_package_builder: Typed fake-package builder fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    from ml_ecosystem_snapshots.api import FRAMEWORK_COLLECTORS

    fake_package_builder.block_imports("jax")
    fake_package_builder.set_metadata_versions({})

    out_explicit = tmp_path / "captured_explicit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml-snapshots",
            "capture",
            "html_dsl",
            "unsupported_fw",
            "jax",
            "--out-dir",
            str(out_explicit),
        ],
    )
    main()
    out_text = capsys.readouterr().out
    assert "Warning: The following frameworks are unsupported and will be skipped: unsupported_fw" in out_text
    assert "Skipping jax" in out_text

    written_html = out_explicit / "html_dsl_v0.0.2.json"
    assert written_html.is_file()
    parsed_html = json.loads(written_html.read_text(encoding="utf-8"))
    assert parsed_html["target"] == "html_dsl"
    assert len(parsed_html["categories"]["elements"]) > 0

    out_valid_only = tmp_path / "captured_valid_only"
    cmd_capture(
        argparse.Namespace(
            frameworks=["html_dsl"],
            include_nonpublic=False,
            out_dir=str(out_valid_only),
        )
    )
    assert (out_valid_only / "html_dsl_v0.0.2.json").is_file()
    _ = capsys.readouterr()

    scoped_collectors = {
        "html_dsl": FRAMEWORK_COLLECTORS["html_dsl"],
        "jax": lambda _cat, _inc=False: [],
    }
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.api.get_available_frameworks",
        lambda: scoped_collectors,
    )

    out_wildcard = tmp_path / "captured_wildcard"
    cmd_capture(
        argparse.Namespace(
            frameworks=["*"],
            include_nonpublic=False,
            out_dir=str(out_wildcard),
        )
    )
    wildcard_out = capsys.readouterr().out
    assert "Saved snapshot to" in wildcard_out
    assert "Skipping jax" in wildcard_out
    assert (out_wildcard / "html_dsl_v0.0.2.json").is_file()

    out_default_all = tmp_path / "captured_all"
    cmd_capture(
        argparse.Namespace(
            frameworks=[],
            include_nonpublic=False,
            out_dir=str(out_default_all),
        )
    )
    all_out = capsys.readouterr().out
    assert "Saved snapshot to" in all_out
    assert (out_default_all / "html_dsl_v0.0.2.json").is_file()


def test_cli_diff_and_changelog_real_json_and_gzip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `diff` and `diff --changelog` compare real `.json` and `.json.gz` files on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    snap_v1 = {
        "target": "demo",
        "version": "1.0.0",
        "categories": {
            "ops": [
                {
                    "name": "legacy_op",
                    "api_path": "demo.legacy_op",
                    "kind": "function",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                },
                {
                    "name": "compute",
                    "api_path": "demo.compute",
                    "kind": "function",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                },
            ]
        },
    }
    snap_v2 = {
        "target": "demo",
        "version": "2.0.0",
        "categories": {
            "ops": [
                {
                    "name": "modern_op",
                    "api_path": "demo.modern_op",
                    "kind": "function",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                },
                {
                    "name": "compute",
                    "api_path": "demo.compute",
                    "kind": "function",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "scale", "kind": "POSITIONAL_OR_KEYWORD"},
                    ],
                },
            ]
        },
    }

    p_v1 = tmp_path / "demo_v1.json"
    p_v1.write_text(json.dumps(snap_v1), encoding="utf-8")

    p_v2_gz = tmp_path / "demo_v2.json.gz"
    with gzip.open(p_v2_gz, "wt", encoding="utf-8") as gf:
        json.dump(snap_v2, gf)

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "diff", str(p_v1), str(p_v2_gz)])
    main()
    diff_out = capsys.readouterr().out
    assert "ADDED: 1" in diff_out
    assert "+ demo.modern_op" in diff_out
    assert "REMOVED: 1" in diff_out
    assert "- demo.legacy_op" in diff_out
    assert "SIGNATURE CHANGED: 1" in diff_out
    assert "~ demo.compute" in diff_out

    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "diff", str(p_v1), str(p_v2_gz), "--changelog"],
    )
    main()
    changelog_out = capsys.readouterr().out
    assert "demo.modern_op" in changelog_out
    assert "demo.legacy_op" in changelog_out
    assert "demo.compute" in changelog_out


def test_cli_generate_stubs_real_gzip_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `generate-stubs` emits valid `.pyi` stub files on disk from a `.json.gz` snapshot.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    snap_gz = tmp_path / "torch_v2.4.0.json.gz"
    with gzip.open(snap_gz, "wt", encoding="utf-8") as gf:
        json.dump(_sample_snapshot_dict(), gf)

    stubs_dir = tmp_path / "stubs_out"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "generate-stubs", "--input", str(snap_gz), "--out-dir", str(stubs_dir)],
    )
    main()
    out = capsys.readouterr().out
    assert f"Stubs generated in {stubs_dir}" in out

    pyi_files = sorted(stubs_dir.rglob("*.pyi"))
    assert len(pyi_files) >= 1
    pyi_parts: list[str] = []
    for pyi_file in pyi_files:
        text = pyi_file.read_text(encoding="utf-8")
        ast.parse(text)
        pyi_parts.append(text)
    combined_pyi = " ".join(pyi_parts)
    assert "class Linear" in combined_pyi
    assert "def relu" in combined_pyi


def test_cli_export_schema_and_code_formats_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `export` writes real OpenAPI, JSON Schema, Pydantic, Protobuf, C++, and TypeScript files.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    snap_json = tmp_path / "torch_snap.json"
    snap_json.write_text(json.dumps(_sample_snapshot_dict()), encoding="utf-8")

    snap_gz = tmp_path / "torch_snap.json.gz"
    with gzip.open(snap_gz, "wt", encoding="utf-8") as gf:
        json.dump(_sample_snapshot_dict(), gf)

    openapi_dir = tmp_path / "out_openapi"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "export", "--input", str(snap_gz), "--out-dir", str(openapi_dir), "--format", "openapi"],
    )
    main()
    assert "Exported OpenAPI spec" in capsys.readouterr().out
    openapi_doc = json.loads((openapi_dir / "openapi.json").read_text(encoding="utf-8"))
    assert str(openapi_doc["openapi"]).startswith("3.")
    assert len(openapi_doc["paths"]) >= 2

    schema_dir = tmp_path / "out_json_schema"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml-snapshots",
            "export",
            "--input",
            str(snap_json),
            "--out-dir",
            str(schema_dir),
            "--format",
            "json_schema",
        ],
    )
    main()
    assert "Exported 2 JSON schemas" in capsys.readouterr().out
    linear_schema = json.loads((schema_dir / "Linear.schema.json").read_text(encoding="utf-8"))
    assert "in_features" in linear_schema["properties"]

    pyd_dir = tmp_path / "out_pydantic"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "export", "--input", str(snap_json), "--out-dir", str(pyd_dir), "--format", "pydantic"],
    )
    main()
    assert "Exported 2 Pydantic models" in capsys.readouterr().out
    pyd_code = (pyd_dir / "linear.py").read_text(encoding="utf-8")
    ast.parse(pyd_code)
    assert "BaseModel" in pyd_code

    proto_dir = tmp_path / "out_protobuf"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "export", "--input", str(snap_json), "--out-dir", str(proto_dir), "--format", "protobuf"],
    )
    main()
    assert "Exported 2 Protobuf definitions" in capsys.readouterr().out
    proto_code = (proto_dir / "linear.proto").read_text(encoding="utf-8")
    assert 'syntax = "proto3";' in proto_code
    assert "message Linear" in proto_code

    cpp_dir = tmp_path / "out_cpp"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "export", "--input", str(snap_json), "--out-dir", str(cpp_dir), "--format", "cpp_header"],
    )
    main()
    assert "Exported 2 C++ headers" in capsys.readouterr().out
    assert "#pragma once" in (cpp_dir / "linear.hpp").read_text(encoding="utf-8")

    ts_dir = tmp_path / "out_ts"
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "export", "--input", str(snap_json), "--out-dir", str(ts_dir), "--format", "typescript"],
    )
    main()
    assert "Exported 2 TypeScript interfaces" in capsys.readouterr().out
    assert "export interface" in (ts_dir / "interfaces.d.ts").read_text(encoding="utf-8")


def test_cli_export_prompt_formats_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `export` writes real Markdown prompt context files for all prompt formats.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    snap_json = tmp_path / "torch_snap.json"
    snap_json.write_text(json.dumps(_sample_snapshot_dict()), encoding="utf-8")

    prompt_expected_files = {
        "llm_prompt": "llm_context.md",
        "sass_prompt": "sass_context.md",
        "mlir_prompt": "mlir_context.md",
        "rdna_prompt": "rdna_context.md",
        "ptx_prompt": "ptx_context.md",
        "stablehlo_prompt": "stablehlo_context.md",
        "scoped_prompt": "scoped_context.md",
    }
    for fmt, expected_fname in prompt_expected_files.items():
        p_dir = tmp_path / f"out_{fmt}"
        monkeypatch.setattr(
            "sys.argv",
            ["ml-snapshots", "export", "--input", str(snap_json), "--out-dir", str(p_dir), "--format", fmt],
        )
        main()
        assert "Exported" in capsys.readouterr().out
        written_md = p_dir / expected_fname
        assert written_md.is_file()
        assert len(written_md.read_text(encoding="utf-8")) > 0


def test_cli_export_edge_inputs_and_unknown_format(tmp_path: Path) -> None:
    """Verify `cmd_export` handles raw list snapshots, scalar JSON payloads, and unknown format errors.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    list_path = tmp_path / "raw_list.json"
    list_path.write_text(
        json.dumps([{"name": "test_op", "api_path": "pkg.test_op", "kind": "function", "params": []}]),
        encoding="utf-8",
    )
    list_out = tmp_path / "out_list"
    cmd_export(argparse.Namespace(input=str(list_path), out_dir=str(list_out), format="openapi"))
    list_openapi = json.loads((list_out / "openapi.json").read_text(encoding="utf-8"))
    assert "/pkg/test_op" in list_openapi["paths"]

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("123", encoding="utf-8")
    scalar_out = tmp_path / "out_scalar"
    cmd_export(argparse.Namespace(input=str(scalar_path), out_dir=str(scalar_out), format="openapi"))
    scalar_openapi = json.loads((scalar_out / "openapi.json").read_text(encoding="utf-8"))
    assert scalar_openapi["paths"] == {}

    with pytest.raises(ValueError, match="Unknown format"):
        cmd_export(argparse.Namespace(input=str(list_path), out_dir=str(tmp_path / "bad"), format="unknown"))


def test_cli_index_lifecycle_real_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `index` (`--status`, `--rebuild`, `--clear`) operates against a real SQLite DB in `tmp_path`.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    cache_dir = tmp_path / "cache_home"
    snaps_dir = tmp_path / "custom_snaps"
    snaps_dir.mkdir(parents=True)
    (snaps_dir / "torch_v2.4.0.json").write_text(json.dumps(_sample_snapshot_dict()), encoding="utf-8")

    empty_pkg_dir = tmp_path / "empty_pkg"
    empty_pkg_dir.mkdir(parents=True)
    monkeypatch.setenv("ML_ECOSYSTEM_SNAPSHOTS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ML_SNAPSHOTS_PATH", str(snaps_dir))
    monkeypatch.setattr("ml_ecosystem_snapshots.index.os.path.dirname", lambda _p: str(empty_pkg_dir))

    # 1. Status before index exists
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "index", "--status"])
    main()
    status_before = capsys.readouterr().out
    assert "Index database does not exist yet" in status_before

    from ml_ecosystem_snapshots.index import get_index_db_path

    # 2. Rebuild index over real snapshot JSON
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "index", "--rebuild"])
    main()
    rebuild_out = capsys.readouterr().out
    assert "Rebuilt index with 2 symbols" in rebuild_out
    db_path = Path(get_index_db_path())
    assert db_path.is_file()

    # 3. Status after rebuild
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "index", "--status"])
    main()
    status_after = capsys.readouterr().out
    assert "Index database:" in status_after
    assert "Indexed snapshots: 1" in status_after
    assert "Total symbols:     2" in status_after

    # 4. Clear index via CLI
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "index", "--clear"])
    main()
    clear_out = capsys.readouterr().out
    assert "Cleared index database" in clear_out
    assert not db_path.exists()

    # 5. Clear failure message branch when clear_index returns False
    monkeypatch.setattr("ml_ecosystem_snapshots.index.clear_index", lambda: False)
    cmd_index(argparse.Namespace(clear=True, rebuild=False, status=False))
    assert "Failed to clear index database" in capsys.readouterr().out


def test_cli_list_snapshots_and_verify_local_cache_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `list-snapshots` and `verify-local-cache` over real `.json` and `.json.gz` files.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    empty_pkg = tmp_path / "pkg_root"
    (empty_pkg / "snapshots").mkdir(parents=True)
    (empty_pkg / "frameworks").mkdir(parents=True)
    empty_cache = tmp_path / "cache_root"
    empty_cache.mkdir(parents=True)

    monkeypatch.setenv("ML_ECOSYSTEM_SNAPSHOTS_CACHE_DIR", str(empty_cache))
    monkeypatch.delenv("ML_SNAPSHOTS_PATH", raising=False)
    monkeypatch.setattr("ml_ecosystem_snapshots.cli.commands.capture._get_pkg_dir", lambda: str(empty_pkg))

    # 1. Empty list-snapshots and verify-local-cache
    cmd_list_snapshots(argparse.Namespace())
    assert "No snapshot files found." in capsys.readouterr().out

    cmd_verify_local_cache(argparse.Namespace(cache_dir=str(empty_cache)))
    assert "No snapshot files discovered" in capsys.readouterr().out

    # 2. Populate custom + cache snapshots with duplicate filename to test deduplication
    custom_dir = tmp_path / "custom_dir"
    custom_dir.mkdir()
    (custom_dir / "dup_snap.json").write_text(json.dumps([{"name": "op1"}]), encoding="utf-8")
    cache_snaps = empty_cache / "snapshots"
    cache_snaps.mkdir()
    (cache_snaps / "dup_snap.json").write_text(json.dumps([{"name": "op1"}]), encoding="utf-8")
    (cache_snaps / "ignore.txt").write_text("ignored", encoding="utf-8")
    gz_snap = cache_snaps / "compressed.json.gz"
    with gzip.open(gz_snap, "wt", encoding="utf-8") as gf:
        json.dump(_sample_snapshot_dict(), gf)

    monkeypatch.setenv("ML_SNAPSHOTS_PATH", str(custom_dir))
    cmd_list_snapshots(argparse.Namespace())
    list_out = capsys.readouterr().out
    assert "Available Snapshots:" in list_out
    assert list_out.count("dup_snap.json") == 1

    # Duplicate directory in search_dirs (`empty_pkg / "frameworks"`) + valid json & json.gz verification
    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "verify-local-cache", "--cache-dir", str(empty_cache)],
    )
    main()
    verify_out = capsys.readouterr().out
    assert "Cache verification passed: 2 snapshots verified." in verify_out

    fw_dir = empty_pkg / "frameworks"
    (fw_dir / "bundled.json").write_text(json.dumps([{"name": "b_op"}]), encoding="utf-8")
    cmd_verify_local_cache(argparse.Namespace(cache_dir=str(fw_dir)))
    assert "Cache verification passed" in capsys.readouterr().out

    # 3. Corrupted JSON triggers exit code 1
    corrupt_dir = tmp_path / "corrupt_cache"
    corrupt_dir.mkdir()
    (corrupt_dir / "bad.json").write_text("{invalid_json", encoding="utf-8")
    (fw_dir / "bundled.json").unlink()
    with pytest.raises(SystemExit) as exc_info:
        cmd_verify_local_cache(argparse.Namespace(cache_dir=str(corrupt_dir)))
    assert exc_info.value.code == 1
    assert "Cache verification failed" in capsys.readouterr().out


def test_cli_pull_and_download_all_real_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `pull` and `download-all` write and copy snapshot files into `tmp_path`.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    monkeypatch.setenv("ML_ECOSYSTEM_SNAPSHOTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ML_SNAPSHOTS_OFFLINE", raising=False)

    def _fake_urlretrieve(url: str, filename: str) -> tuple[str, None]:
        """Write a valid minimal JSON snapshot file to `filename` instead of performing network I/O.

        Args:
            url: Source download URL.
            filename: Local destination file path.

        Returns:
            Tuple of `(filename, None)`.
        """
        Path(filename).write_text(json.dumps({"url": url, "categories": {}}), encoding="utf-8")
        return filename, None

    def _fail_urlretrieve(_url: str, _filename: str) -> typing.NoReturn:
        """Raise `OSError` to simulate network unavailability.

        Args:
            _url: Ignored URL.
            _filename: Ignored destination filename.

        Raises:
            OSError: Always raised.
        """
        raise OSError("Network error")

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)
    pulled_dir = tmp_path / "pulled"
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "pull", "torch@2.4.0", "--out-dir", str(pulled_dir)])
    main()
    assert "Pulling snapshot for torch (2.4.0)" in capsys.readouterr().out
    assert (pulled_dir / "torch_v2.4.0.json").is_file()

    monkeypatch.setattr("urllib.request.urlretrieve", _fail_urlretrieve)
    cmd_pull(argparse.Namespace(target="jax", out_dir=None))
    fail_pull_out = capsys.readouterr().out
    assert "Pulling snapshot for jax (latest)" in fail_pull_out
    assert "Failed to download snapshot: Network error" in fail_pull_out

    with pytest.raises(SystemExit) as off_pull:
        cmd_pull(argparse.Namespace(target="torch", out_dir=None, offline=True))
    assert off_pull.value.code == 1
    assert "Network access disabled" in capsys.readouterr().out

    dl_dir = tmp_path / "dl_cache"
    with pytest.raises(SystemExit):
        cmd_download_all(argparse.Namespace(offline=True, cache_dir=str(dl_dir)))
    assert "Network access disabled" in capsys.readouterr().out

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "download-all", "--cache-dir", str(dl_dir)])
    main()
    assert "Downloaded torch" in capsys.readouterr().out
    assert (dl_dir / "torch_exhaustive.json").is_file()

    fake_pkg = tmp_path / "fake_pkg"
    fake_fw = fake_pkg / "frameworks"
    fake_fw.mkdir(parents=True)
    (fake_fw / "torch_exhaustive.json").write_text('{"categories": {}}', encoding="utf-8")
    monkeypatch.setattr("ml_ecosystem_snapshots.cli.commands.capture._get_pkg_dir", lambda: str(fake_pkg))
    monkeypatch.setattr("urllib.request.urlretrieve", _fail_urlretrieve)

    fallback_dl_dir = tmp_path / "dl_fallback"
    cmd_download_all(argparse.Namespace(offline=False, cache_dir=str(fallback_dl_dir)))
    fb_out = capsys.readouterr().out
    assert "Copied bundled torch" in fb_out
    assert "Failed to download jax" in fb_out
    assert (fallback_dl_dir / "torch_exhaustive.json").is_file()


def test_cli_mcp_offline_and_dunder_main(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `mcp`, `--offline` environment flag propagation, and `__main__` execution.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    monkeypatch.delenv("ML_SNAPSHOTS_OFFLINE", raising=False)
    mock_mcp = mocker.patch("ml_ecosystem_snapshots.mcp_server.run_mcp_server")
    with monkeypatch.context() as mp:
        mp.setattr("sys.argv", ["ml-snapshots", "--offline", "mcp"])
        main()
        assert os.environ.get("ML_SNAPSHOTS_OFFLINE") == "1"
    os.environ.pop("ML_SNAPSHOTS_OFFLINE", None)
    mock_mcp.assert_called_once()

    cli_spec = importlib.util.find_spec("ml_ecosystem_snapshots.cli")
    assert cli_spec is not None and cli_spec.origin is not None
    monkeypatch.setattr("sys.argv", ["ml-snapshots", "list-snapshots"])
    ns = runpy.run_path(cli_spec.origin, run_name="__main__")
    assert "main" in ns
    assert isinstance(capsys.readouterr().out, str)


def test_cli_check_commands_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `check-ptx`, `check-onnx`, `check-mlir`, and `check-sass` CLI subcommands end-to-end.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    from ml_ecosystem_snapshots.mcp.models import IsaCheckResult

    monkeypatch.setattr(
        "sys.argv",
        ["ml-snapshots", "check-ptx", "add", "--types", ".s32", "--sm-arch", "sm_80"],
    )
    main()
    assert "PTX Instruction 'add' is valid." in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-ptx", "fma"])
    main()
    fma_out = capsys.readouterr().out
    assert "PTX Instruction 'fma' is valid." in fma_out
    assert "Supported Types:" in fma_out

    with monkeypatch.context() as mp:
        mp.setattr(
            "ml_ecosystem_snapshots.cli.commands.verify_isa.verify_ptx_instruction",
            lambda *_a, **_kw: IsaCheckResult(is_valid=True, min_sm=None, supported_types=[], errors=[]),
        )
        mp.setattr("sys.argv", ["ml-snapshots", "check-ptx", "dummy_valid"])
        main()
        dummy_out = capsys.readouterr().out
        assert "PTX Instruction 'dummy_valid' is valid." in dummy_out
        assert "Minimum SM Architecture:" not in dummy_out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-ptx", "nonexistent_ptx_mnemonic"])
    with pytest.raises(SystemExit):
        main()
    assert "PTX Instruction 'nonexistent_ptx_mnemonic' Invalid:" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-ptx"])
    with pytest.raises(SystemExit):
        main()
    assert "Error: Instruction mnemonic must be specified." in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-onnx", "Add", "--inputs", "2"])
    main()
    assert "Add" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-mlir", "arith.addi", "--operands", "2"])
    main()
    assert "arith.addi" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["ml-snapshots", "check-sass", "FADD"])
    main()
    assert "FADD" in capsys.readouterr().out
