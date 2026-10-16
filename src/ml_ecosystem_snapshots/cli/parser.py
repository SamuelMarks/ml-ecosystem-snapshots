"""Argument parser construction for the ML Ecosystem Snapshots CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

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
    cmd_index,
    cmd_list_snapshots,
    cmd_mcp,
    cmd_pull,
    cmd_stubs,
    cmd_verify_local_cache,
)


@dataclass(frozen=True)
class SubparserCheckSpec:
    """Declarative specification for a `check-*` domain CLI subcommand."""

    command: str
    help_text: str
    handler: Callable[[argparse.Namespace], None]
    pos_arg: str = "op_name"
    pos_help: str = "Operation name"
    count_flag: str = "--inputs-count"
    count_help: str = "Expected number of input operands"
    attr_help: str = "Comma-separated list of expected keyword argument names"
    extra_flag: Optional[Tuple[str, Any, str]] = None


DOMAIN_CHECK_SUBPARSERS: List[SubparserCheckSpec] = [
    SubparserCheckSpec(
        command="check-onnx",
        help_text="Validate ONNX operation against official specification",
        handler=cmd_check_onnx,
        pos_help="Name of the ONNX operator (e.g., 'Conv', 'Add')",
        attr_help="Comma-separated list of expected attribute names",
        extra_flag=("--domain", "", "Optional domain (e.g., 'ai.onnx.ml')"),
    ),
    SubparserCheckSpec(
        command="check-metal",
        help_text="Validate Apple Metal MSL compute operation against specification",
        handler=cmd_check_metal,
        pos_help="Name of the Metal MSL operation (e.g., 'simdgroup_multiply_accumulate')",
        attr_help="Comma-separated list of expected attribute names",
        extra_flag=(
            "--address-space",
            None,
            "Expected address space (e.g., 'device', 'threadgroup', 'thread')",
        ),
    ),
    SubparserCheckSpec(
        command="check-wasm",
        help_text="Validate WebAssembly 2.0 / SIMD instruction against specification",
        handler=cmd_check_wasm,
        pos_arg="mnemonic",
        pos_help="Mnemonic of the WASM instruction (e.g., 'f32x4.add', 'v128.load')",
        count_flag="--operands-count",
        count_help="Expected number of stack operands",
        attr_help="Comma-separated list of expected attribute/immediate names",
    ),
    SubparserCheckSpec(
        command="check-webgl",
        help_text="Validate WebGL 2.0 / GLSL ES 3.00 operation against specification",
        handler=cmd_check_webgl,
        pos_help="Name of the WebGL operation (e.g., 'texelFetch', 'matrixCompMult')",
        attr_help="Comma-separated list of expected attribute names",
    ),
    SubparserCheckSpec(
        command="check-cpp",
        help_text="Validate C++17 / PyBind11 operation against specification",
        handler=cmd_check_cpp,
        pos_help="Name of the C++ operation (e.g., 'clamp', 'from_blob', 'sqrt')",
        attr_help="Comma-separated list of expected attribute names",
    ),
    SubparserCheckSpec(
        command="check-array-api",
        help_text="Validate Python Array API Standard operation against specification",
        handler=cmd_check_array_api,
        pos_help="Name of the Array API operation (e.g., 'add', 'matmul', 'sum')",
    ),
    SubparserCheckSpec(
        command="check-scipy",
        help_text="Validate SciPy operation against specification",
        handler=cmd_check_scipy,
        pos_help="Name of the SciPy operation (e.g., 'erf', 'inv', 'convolve')",
    ),
    SubparserCheckSpec(
        command="check-torchvision",
        help_text="Validate TorchVision operation against specification",
        handler=cmd_check_torchvision,
        pos_help="Name of the TorchVision operation (e.g., 'nms', 'roi_align', 'box_iou')",
    ),
    SubparserCheckSpec(
        command="check-torchaudio",
        help_text="Validate TorchAudio operation against specification",
        handler=cmd_check_torchaudio,
        pos_help="Name of the TorchAudio operation (e.g., 'melscale_fbanks', 'spectrogram', 'MFCC')",
    ),
    SubparserCheckSpec(
        command="check-safetensors",
        help_text="Validate SafeTensors operation against specification",
        handler=cmd_check_safetensors,
        pos_help="Name of the SafeTensors operation (e.g., 'save_file', 'load_file', 'safe_open')",
    ),
    SubparserCheckSpec(
        command="check-aten",
        help_text="Validate PyTorch ATen operation against specification",
        handler=cmd_check_aten,
        pos_help="Name of the ATen operation (e.g., 'add', 'matmul', 'relu')",
    ),
    SubparserCheckSpec(
        command="check-nccl",
        help_text="Validate NCCL/RCCL collective operation against specification",
        handler=cmd_check_nccl,
        pos_help="Name of the collective operation (e.g., 'all_reduce', 'all_gather', 'reduce_scatter')",
    ),
    SubparserCheckSpec(
        command="check-flash-attention",
        help_text="Validate FlashAttention operation against specification",
        handler=cmd_check_flash_attention,
        pos_help="Name of the attention operation (e.g., 'flash_attn_func', 'paged_attention_v1', 'FlashAttention')",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the top-level CLI `ArgumentParser` with all subcommands.

    Returns:
        Configured `argparse.ArgumentParser` instance with subcommand handlers bound via `set_defaults(func=...)`.
    """
    parser = argparse.ArgumentParser(description="ML Framework Snapshots CLI")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enforce offline mode (disables network operations and live framework imports)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # capture
    parser_capture = subparsers.add_parser("capture", help="Capture snapshots")
    parser_capture.add_argument(
        "frameworks",
        nargs="*",
        default=["all"],
        help="Frameworks to capture, or 'all' for all supported frameworks",
    )
    parser_capture.add_argument(
        "--out-dir", type=str, default="snapshots", help="Output directory"
    )
    parser_capture.add_argument(
        "--include-nonpublic", action="store_true", help="Include non-public APIs"
    )
    parser_capture.set_defaults(func=cmd_capture)

    # diff
    parser_diff = subparsers.add_parser("diff", help="Diff two snapshots")
    parser_diff.add_argument("json1", type=str, help="First snapshot JSON")
    parser_diff.add_argument("json2", type=str, help="Second snapshot JSON")
    parser_diff.add_argument(
        "--changelog", action="store_true", help="Generate a markdown changelog"
    )
    parser_diff.set_defaults(func=cmd_diff)

    # generate-stubs
    parser_stubs = subparsers.add_parser("generate-stubs", help="Generate .pyi stubs")
    parser_stubs.add_argument(
        "--input", type=str, required=True, help="Input JSON snapshot"
    )
    parser_stubs.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for stubs"
    )
    parser_stubs.add_argument(
        "--include-nonpublic",
        action="store_true",
        help="Include non-public APIs in stubs",
    )
    parser_stubs.set_defaults(func=cmd_stubs)

    # export
    parser_export = subparsers.add_parser(
        "export", help="Export to JSON Schema, OpenAPI, Pydantic, or Protobuf"
    )
    parser_export.add_argument(
        "--input", type=str, required=True, help="Input JSON snapshot"
    )
    parser_export.add_argument(
        "--out-dir", type=str, required=True, help="Output directory"
    )
    parser_export.add_argument(
        "--format",
        choices=[
            "openapi",
            "json_schema",
            "pydantic",
            "protobuf",
            "llm_prompt",
            "sass_prompt",
            "mlir_prompt",
            "rdna_prompt",
            "ptx_prompt",
            "stablehlo_prompt",
            "scoped_prompt",
            "cpp_header",
            "typescript",
        ],
        required=True,
        help="Format to export",
    )
    parser_export.set_defaults(func=cmd_export)

    # mcp
    parser_mcp = subparsers.add_parser(
        "mcp", help="Start Model Context Protocol (MCP) JSON-RPC tool server"
    )
    parser_mcp.set_defaults(func=cmd_mcp)

    # check
    parser_check = subparsers.add_parser(
        "check", help="Check compliance of a target implementation against a snapshot"
    )
    parser_check.add_argument(
        "snapshot_json", type=str, help="Reference snapshot JSON file"
    )
    parser_check.add_argument(
        "target_path",
        type=str,
        nargs="+",
        help="Target implementation file or directory",
    )
    parser_check.add_argument(
        "--target-prefix",
        type=str,
        required=True,
        help="Module prefix in target (e.g., ml_switcheroo.jax)",
    )
    parser_check.add_argument(
        "--reference-prefix",
        type=str,
        required=True,
        help="Module prefix in reference (e.g., jax)",
    )
    parser_check.set_defaults(func=cmd_check)

    # list-snapshots
    parser_list = subparsers.add_parser(
        "list-snapshots", help="List available pre-bundled snapshots"
    )
    parser_list.set_defaults(func=cmd_list_snapshots)

    # pull
    parser_pull = subparsers.add_parser(
        "pull",
        help="Download a framework snapshot from GitHub Releases into local cache",
    )
    parser_pull.add_argument(
        "target",
        type=str,
        help="Target framework and version (e.g. torch@2.4.0 or jax@0.4.30)",
    )
    parser_pull.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom output directory for downloaded snapshot",
    )
    parser_pull.set_defaults(func=cmd_pull)

    # download-all
    parser_download_all = subparsers.add_parser(
        "download-all",
        help="Download all canonical pre-compiled snapshot assets into cache",
    )
    parser_download_all.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom destination directory for downloaded snapshots",
    )
    parser_download_all.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode",
    )
    parser_download_all.set_defaults(func=cmd_download_all)

    # verify-local-cache
    parser_verify_cache = subparsers.add_parser(
        "verify-local-cache",
        help="Validate SHA256 checksums and integrity of cached snapshots",
    )
    parser_verify_cache.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory to inspect and verify",
    )
    parser_verify_cache.set_defaults(func=cmd_verify_local_cache)

    # index / index-cache
    for idx_cmd, idx_help in (
        ("index", "Manage ephemeral local SQLite search index"),
        ("index-cache", "Manage and clean ephemeral local SQLite search index"),
    ):
        p_idx = subparsers.add_parser(idx_cmd, help=idx_help)
        p_idx.add_argument(
            "--rebuild",
            action="store_true",
            help="Rebuild index from all available snapshots",
        )
        p_idx.add_argument(
            "--clean",
            "--clear",
            dest="clean",
            action="store_true",
            help="Clear the local index database",
        )
        p_idx.add_argument(
            "--status", action="store_true", help="Display index statistics"
        )
        p_idx.set_defaults(func=cmd_index)

    # check-sass
    parser_check_sass = subparsers.add_parser(
        "check-sass",
        help="Validate NVIDIA SASS assembly instruction or snippet",
    )
    parser_check_sass.add_argument(
        "mnemonic",
        type=str,
        nargs="?",
        default=None,
        help="SASS mnemonic to validate (e.g. FADD, WGMMA)",
    )
    parser_check_sass.add_argument(
        "--operands",
        type=str,
        default=None,
        help="Comma-separated operands (e.g. 'R0,R1,R2')",
    )
    parser_check_sass.add_argument(
        "--modifiers",
        type=str,
        default=None,
        help="Comma-separated modifiers (e.g. '.SAT,.FTZ')",
    )
    parser_check_sass.add_argument(
        "--sm-arch",
        type=str,
        default=None,
        help="Target SM architecture (e.g. 'sm_80', 'sm_90')",
    )
    parser_check_sass.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to SASS assembly file to validate",
    )
    parser_check_sass.set_defaults(func=cmd_check_sass)

    # check-rdna
    parser_check_rdna = subparsers.add_parser(
        "check-rdna",
        help="Validate AMD RDNA assembly instruction or snippet",
    )
    parser_check_rdna.add_argument(
        "mnemonic",
        type=str,
        nargs="?",
        default=None,
        help="RDNA mnemonic to validate (e.g. v_add_f32, v_dual_fma_f32)",
    )
    parser_check_rdna.add_argument(
        "--operands",
        type=str,
        default=None,
        help="Comma-separated operands (e.g. 'v0,v1,v2')",
    )
    parser_check_rdna.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Expected encoding profile (e.g. VOP2, VOP3)",
    )
    parser_check_rdna.add_argument(
        "--modifiers",
        type=str,
        default=None,
        help="Comma-separated modifiers (e.g. '-src,clamp')",
    )
    parser_check_rdna.add_argument(
        "--gfx-arch",
        type=str,
        default=None,
        help="Target GFX architecture (e.g. 'GFX11/RDNA3', 'GFX9/CDNA')",
    )
    parser_check_rdna.add_argument(
        "--wave-size",
        type=int,
        default=None,
        help="Wavefront execution size (32 or 64)",
    )
    parser_check_rdna.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to RDNA assembly file to validate",
    )
    parser_check_rdna.set_defaults(func=cmd_check_rdna)

    # check-ptx
    parser_check_ptx = subparsers.add_parser(
        "check-ptx",
        help="Validate NVIDIA PTX instruction mnemonics, operands, and SM requirements",
    )
    parser_check_ptx.add_argument(
        "mnemonic",
        type=str,
        nargs="?",
        default=None,
        help="PTX mnemonic to validate (e.g. add, wgmma.mma_async)",
    )
    parser_check_ptx.add_argument(
        "--mnemonic",
        dest="opt_mnemonic",
        type=str,
        default=None,
        help="PTX mnemonic to validate (alternative to positional argument)",
    )
    parser_check_ptx.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated PTX type qualifiers (e.g. '.f32,.f16')",
    )
    parser_check_ptx.add_argument(
        "--operands",
        type=str,
        default=None,
        help="Comma-separated operands (e.g. '%r0,%r1,%r2')",
    )
    parser_check_ptx.add_argument(
        "--state-space",
        type=str,
        default=None,
        help="Memory state space qualifier (e.g. '.global', '.shared')",
    )
    parser_check_ptx.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Memory visibility scope (e.g. '.cta', '.gpu', '.sys')",
    )
    parser_check_ptx.add_argument(
        "--vector-width",
        type=str,
        default=None,
        help="Vector width qualifier (e.g. '.v2', '.v4')",
    )
    parser_check_ptx.add_argument(
        "--sm-arch",
        type=str,
        default=None,
        help="Target SM architecture (e.g. 'sm_80', 'sm_90')",
    )
    parser_check_ptx.set_defaults(func=cmd_check_ptx)

    # check-mlir / check-stablehlo / check-wgsl (file + op_name capable)
    for ir_cmd, ir_help, ir_pos_help, ir_cnt_flag, ir_fn in (
        (
            "check-mlir",
            "Validate MLIR text or operation",
            "Qualified MLIR op name (e.g. arith.addf)",
            "--operands-count",
            cmd_check_mlir,
        ),
        (
            "check-stablehlo",
            "Validate StableHLO text or operation",
            "Qualified StableHLO op name (e.g. stablehlo.dot_general)",
            "--operands-count",
            cmd_check_stablehlo,
        ),
        (
            "check-wgsl",
            "Validate WGSL shader or operation against specification",
            "Name of the WGSL operation (e.g. storageStore, atomicAdd)",
            "--inputs-count",
            cmd_check_wgsl,
        ),
    ):
        p_ir = subparsers.add_parser(ir_cmd, help=ir_help)
        p_ir.add_argument(
            "op_name", type=str, nargs="?", default=None, help=ir_pos_help
        )
        p_ir.add_argument(
            ir_cnt_flag,
            type=int,
            default=None,
            help="Expected number of operands",
        )
        p_ir.add_argument(
            "--attributes",
            type=str,
            default=None,
            help="Comma-separated attribute names",
        )
        p_ir.add_argument(
            "--file",
            type=str,
            default=None,
            help="Path to source file to validate",
        )
        p_ir.set_defaults(func=ir_fn)

    # Declarative domain check subparsers
    for spec in DOMAIN_CHECK_SUBPARSERS:
        p_dom = subparsers.add_parser(spec.command, help=spec.help_text)
        p_dom.add_argument(spec.pos_arg, help=spec.pos_help)
        if spec.extra_flag is not None:
            flag_name, flag_default, flag_help = spec.extra_flag
            p_dom.add_argument(flag_name, default=flag_default, help=flag_help)
        p_dom.add_argument(
            spec.count_flag, type=int, default=None, help=spec.count_help
        )
        p_dom.add_argument("--attributes", default=None, help=spec.attr_help)
        p_dom.set_defaults(func=spec.handler)

    return parser


__all__ = ["DOMAIN_CHECK_SUBPARSERS", "SubparserCheckSpec", "build_parser"]
