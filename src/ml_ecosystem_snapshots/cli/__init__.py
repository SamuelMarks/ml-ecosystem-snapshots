"""Command Line Interface package for ML Framework Snapshots.

Provides the entrypoint, argument parser, path resolver, and subcommand handlers
for generating, diffing, exporting, indexing, and verifying framework snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ml_ecosystem_snapshots.api import extract_snapshot, write_snapshot
from ml_ecosystem_snapshots.cli.commands import (
    cmd_capture,
    cmd_check,
    cmd_check_array_api,
    cmd_check_aten,
    cmd_check_cpp,
    cmd_check_flash_attention,
    cmd_check_metal,
    cmd_check_mlir,
    cmd_check_nccl,
    cmd_check_onnx,
    cmd_check_ptx,
    cmd_check_rdna,
    cmd_check_safetensors,
    cmd_check_sass,
    cmd_check_scipy,
    cmd_check_stablehlo,
    cmd_check_torchaudio,
    cmd_check_torchvision,
    cmd_check_wasm,
    cmd_check_webgl,
    cmd_check_wgsl,
    cmd_diff,
    cmd_download_all,
    cmd_export,
    cmd_generate_stubs,
    cmd_index,
    cmd_index_cache,
    cmd_list_snapshots,
    cmd_mcp,
    cmd_pull,
    cmd_stubs,
    cmd_verify_local_cache,
)
from ml_ecosystem_snapshots.cli.parser import build_parser
from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_ecosystem_snapshots.diff import diff_snapshots, generate_changelog
from ml_ecosystem_snapshots.index import get_cache_dir
from ml_ecosystem_snapshots.models import GhostInspector
from ml_ecosystem_snapshots.stubs import generate_stubs
from ml_ecosystem_snapshots.utils import get_custom_snapshots_paths, is_offline_mode
from ml_switcheroo_ir.schema.ghost import GhostRef


def main() -> None:
    """Parse CLI arguments and dispatch to the selected subcommand handler."""
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "offline", False):
        os.environ["ML_SNAPSHOTS_OFFLINE"] = "1"
    args.func(args)


__all__ = [
    "GhostInspector",
    "GhostRef",
    "argparse",
    "build_parser",
    "cmd_capture",
    "cmd_check",
    "cmd_check_array_api",
    "cmd_check_aten",
    "cmd_check_cpp",
    "cmd_check_flash_attention",
    "cmd_check_metal",
    "cmd_check_mlir",
    "cmd_check_nccl",
    "cmd_check_onnx",
    "cmd_check_ptx",
    "cmd_check_rdna",
    "cmd_check_safetensors",
    "cmd_check_sass",
    "cmd_check_scipy",
    "cmd_check_stablehlo",
    "cmd_check_torchaudio",
    "cmd_check_torchvision",
    "cmd_check_wasm",
    "cmd_check_webgl",
    "cmd_check_wgsl",
    "cmd_diff",
    "cmd_download_all",
    "cmd_export",
    "cmd_generate_stubs",
    "cmd_index",
    "cmd_index_cache",
    "cmd_list_snapshots",
    "cmd_mcp",
    "cmd_pull",
    "cmd_stubs",
    "cmd_verify_local_cache",
    "diff_snapshots",
    "extract_snapshot",
    "generate_changelog",
    "generate_stubs",
    "get_cache_dir",
    "get_custom_snapshots_paths",
    "is_offline_mode",
    "json",
    "main",
    "os",
    "resolve_snapshot_path",
    "sys",
    "write_snapshot",
]

if __name__ == "__main__":
    main()
