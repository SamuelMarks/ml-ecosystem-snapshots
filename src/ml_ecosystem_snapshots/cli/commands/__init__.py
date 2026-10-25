"""CLI subcommand handlers subpackage for ml_ecosystem_snapshots."""

from __future__ import annotations

from ml_ecosystem_snapshots.cli.commands.capture import (
    cmd_capture,
    cmd_download_all,
    cmd_list_snapshots,
    cmd_pull,
    cmd_verify_local_cache,
)
from ml_ecosystem_snapshots.cli.commands.check import (
    cmd_check,
    cmd_check_array_api,
    cmd_check_aten,
    cmd_check_chex,
    cmd_check_cpp,
    cmd_check_flash_attention,
    cmd_check_grain,
    cmd_check_metal,
    cmd_check_mlir,
    cmd_check_nccl,
    cmd_check_onnx,
    cmd_check_optax,
    cmd_check_orbax,
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
)
from ml_ecosystem_snapshots.cli.commands.diff import cmd_diff
from ml_ecosystem_snapshots.cli.commands.export import (
    cmd_export,
    cmd_stubs,
)
from ml_ecosystem_snapshots.cli.commands.index import cmd_index
from ml_ecosystem_snapshots.cli.commands.mcp import cmd_mcp

__all__ = [
    "cmd_capture",
    "cmd_check",
    "cmd_check_array_api",
    "cmd_check_aten",
    "cmd_check_chex",
    "cmd_check_cpp",
    "cmd_check_flash_attention",
    "cmd_check_grain",
    "cmd_check_metal",
    "cmd_check_mlir",
    "cmd_check_nccl",
    "cmd_check_onnx",
    "cmd_check_optax",
    "cmd_check_orbax",
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
    "cmd_index",
    "cmd_list_snapshots",
    "cmd_mcp",
    "cmd_pull",
    "cmd_stubs",
    "cmd_verify_local_cache",
]
