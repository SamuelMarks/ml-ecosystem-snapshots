"""CLI command handlers for compliance, ISA, IR, and specification verification."""

from __future__ import annotations

from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
    _check_assembly_or_mlir_target as _check_assembly_or_mlir_target,
)
from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
    _format_missing_ref_row as _format_missing_ref_row,
)
from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
    _print_missing_apis_table as _print_missing_apis_table,
)
from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
    _print_module_breakdown as _print_module_breakdown,
)
from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
    cmd_check,
)
from ml_ecosystem_snapshots.cli.commands.verify_domains import (
    DomainCheckCallable,
    cmd_check_array_api,
    cmd_check_aten,
    cmd_check_chex,
    cmd_check_cpp,
    cmd_check_flash_attention,
    cmd_check_grain,
    cmd_check_metal,
    cmd_check_nccl,
    cmd_check_onnx,
    cmd_check_optax,
    cmd_check_orbax,
    cmd_check_safetensors,
    cmd_check_scipy,
    cmd_check_torchaudio,
    cmd_check_torchvision,
    cmd_check_wasm,
    cmd_check_webgl,
)
from ml_ecosystem_snapshots.cli.commands.verify_domains import (
    _read_namespace_attr as _read_namespace_attr,
)
from ml_ecosystem_snapshots.cli.commands.verify_domains import (
    _run_domain_op_check_cli as _run_domain_op_check_cli,
)
from ml_ecosystem_snapshots.cli.commands.verify_ir import (
    cmd_check_mlir,
    cmd_check_stablehlo,
    cmd_check_wgsl,
)
from ml_ecosystem_snapshots.cli.commands.verify_isa import (
    cmd_check_ptx,
    cmd_check_rdna,
    cmd_check_sass,
)

__all__ = [
    "DomainCheckCallable",
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
]
