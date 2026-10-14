"""Argument parser construction for the ML Ecosystem Snapshots CLI."""

from __future__ import annotations

import argparse

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

    # index
    parser_index = subparsers.add_parser(
        "index", help="Manage ephemeral local SQLite search index"
    )
    parser_index.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from all available snapshots",
    )
    parser_index.add_argument(
        "--clean",
        "--clear",
        dest="clean",
        action="store_true",
        help="Clear the local index database",
    )
    parser_index.add_argument(
        "--status", action="store_true", help="Display index statistics"
    )
    parser_index.set_defaults(func=cmd_index)

    # index-cache
    parser_index_cache = subparsers.add_parser(
        "index-cache", help="Manage and clean ephemeral local SQLite search index"
    )
    parser_index_cache.add_argument(
        "--clean",
        "--clear",
        dest="clean",
        action="store_true",
        help="Clear the local index cache",
    )
    parser_index_cache.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from all available snapshots",
    )
    parser_index_cache.add_argument(
        "--status", action="store_true", help="Display index statistics"
    )
    parser_index_cache.set_defaults(func=cmd_index)

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

    # check-mlir
    parser_check_mlir = subparsers.add_parser(
        "check-mlir",
        help="Validate MLIR text or operation",
    )
    parser_check_mlir.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Qualified MLIR op name (e.g. arith.addf)",
    )
    parser_check_mlir.add_argument(
        "--operands-count",
        type=int,
        default=None,
        help="Expected number of SSA operands",
    )
    parser_check_mlir.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names",
    )
    parser_check_mlir.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to MLIR text file to validate",
    )
    parser_check_mlir.set_defaults(func=cmd_check_mlir)

    # check-stablehlo
    parser_check_stablehlo = subparsers.add_parser(
        "check-stablehlo",
        help="Validate StableHLO text or operation",
    )
    parser_check_stablehlo.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Qualified StableHLO op name (e.g. stablehlo.dot_general)",
    )
    parser_check_stablehlo.add_argument(
        "--operands-count",
        type=int,
        default=None,
        help="Expected number of SSA operands",
    )
    parser_check_stablehlo.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names",
    )
    parser_check_stablehlo.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to StableHLO text file to validate",
    )
    parser_check_stablehlo.set_defaults(func=cmd_check_stablehlo)

    # check-wgsl
    parser_check_wgsl = subparsers.add_parser(
        "check-wgsl",
        help="Validate WGSL shader or operation against specification",
    )
    parser_check_wgsl.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the WGSL operation (e.g. storageStore, atomicAdd)",
    )
    parser_check_wgsl.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_wgsl.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names",
    )
    parser_check_wgsl.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to WGSL shader file to validate",
    )
    parser_check_wgsl.set_defaults(func=cmd_check_wgsl)

    parser_check_onnx = subparsers.add_parser(
        "check-onnx",
        help="Validate ONNX operation against official specification",
    )
    parser_check_onnx.add_argument(
        "op_name",
        help="Name of the ONNX operator (e.g., 'Conv', 'Add')",
    )
    parser_check_onnx.add_argument(
        "--domain",
        default="",
        help="Optional domain (e.g., 'ai.onnx.ml')",
    )
    parser_check_onnx.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_onnx.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected attribute names",
    )
    parser_check_onnx.set_defaults(func=cmd_check_onnx)

    parser_check_metal = subparsers.add_parser(
        "check-metal",
        help="Validate Apple Metal MSL compute operation against specification",
    )
    parser_check_metal.add_argument(
        "op_name",
        help="Name of the Metal MSL operation (e.g., 'simdgroup_multiply_accumulate')",
    )
    parser_check_metal.add_argument(
        "--address-space",
        default=None,
        help="Expected address space (e.g., 'device', 'threadgroup', 'thread')",
    )
    parser_check_metal.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_metal.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected attribute names",
    )
    parser_check_metal.set_defaults(func=cmd_check_metal)

    parser_check_wasm = subparsers.add_parser(
        "check-wasm",
        help="Validate WebAssembly 2.0 / SIMD instruction against specification",
    )
    parser_check_wasm.add_argument(
        "mnemonic",
        help="Mnemonic of the WASM instruction (e.g., 'f32x4.add', 'v128.load')",
    )
    parser_check_wasm.add_argument(
        "--operands-count",
        type=int,
        default=None,
        help="Expected number of stack operands",
    )
    parser_check_wasm.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected attribute/immediate names",
    )
    parser_check_wasm.set_defaults(func=cmd_check_wasm)

    parser_check_webgl = subparsers.add_parser(
        "check-webgl",
        help="Validate WebGL 2.0 / GLSL ES 3.00 operation against specification",
    )
    parser_check_webgl.add_argument(
        "op_name",
        help="Name of the WebGL operation (e.g., 'texelFetch', 'matrixCompMult')",
    )
    parser_check_webgl.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_webgl.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected attribute names",
    )
    parser_check_webgl.set_defaults(func=cmd_check_webgl)

    parser_check_cpp = subparsers.add_parser(
        "check-cpp",
        help="Validate C++17 / PyBind11 operation against specification",
    )
    parser_check_cpp.add_argument(
        "op_name",
        help="Name of the C++ operation (e.g., 'clamp', 'from_blob', 'sqrt')",
    )
    parser_check_cpp.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_cpp.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected attribute names",
    )
    parser_check_cpp.set_defaults(func=cmd_check_cpp)

    parser_check_array_api = subparsers.add_parser(
        "check-array-api",
        help="Validate Python Array API Standard operation against specification",
    )
    parser_check_array_api.add_argument(
        "op_name",
        help="Name of the Array API operation (e.g., 'add', 'matmul', 'sum')",
    )
    parser_check_array_api.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_array_api.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_array_api.set_defaults(func=cmd_check_array_api)

    parser_check_scipy = subparsers.add_parser(
        "check-scipy",
        help="Validate SciPy operation against specification",
    )
    parser_check_scipy.add_argument(
        "op_name",
        help="Name of the SciPy operation (e.g., 'erf', 'inv', 'convolve')",
    )
    parser_check_scipy.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_scipy.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_scipy.set_defaults(func=cmd_check_scipy)

    parser_check_torchvision = subparsers.add_parser(
        "check-torchvision",
        help="Validate TorchVision operation against specification",
    )
    parser_check_torchvision.add_argument(
        "op_name",
        help="Name of the TorchVision operation (e.g., 'nms', 'roi_align', 'box_iou')",
    )
    parser_check_torchvision.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_torchvision.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_torchvision.set_defaults(func=cmd_check_torchvision)

    parser_check_torchaudio = subparsers.add_parser(
        "check-torchaudio",
        help="Validate TorchAudio operation against specification",
    )
    parser_check_torchaudio.add_argument(
        "op_name",
        help="Name of the TorchAudio operation (e.g., 'melscale_fbanks', 'spectrogram', 'MFCC')",
    )
    parser_check_torchaudio.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_torchaudio.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_torchaudio.set_defaults(func=cmd_check_torchaudio)

    parser_check_safetensors = subparsers.add_parser(
        "check-safetensors",
        help="Validate SafeTensors operation against specification",
    )
    parser_check_safetensors.add_argument(
        "op_name",
        help="Name of the SafeTensors operation (e.g., 'save_file', 'load_file', 'safe_open')",
    )
    parser_check_safetensors.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_safetensors.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_safetensors.set_defaults(func=cmd_check_safetensors)

    parser_check_aten = subparsers.add_parser(
        "check-aten",
        help="Validate PyTorch ATen operation against specification",
    )
    parser_check_aten.add_argument(
        "op_name",
        help="Name of the ATen operation (e.g., 'add', 'matmul', 'relu')",
    )
    parser_check_aten.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_aten.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_aten.set_defaults(func=cmd_check_aten)

    parser_check_nccl = subparsers.add_parser(
        "check-nccl",
        help="Validate NCCL/RCCL collective operation against specification",
    )
    parser_check_nccl.add_argument(
        "op_name",
        help="Name of the collective operation (e.g., 'all_reduce', 'all_gather', 'reduce_scatter')",
    )
    parser_check_nccl.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_nccl.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_nccl.set_defaults(func=cmd_check_nccl)

    parser_check_flash_attention = subparsers.add_parser(
        "check-flash-attention",
        help="Validate FlashAttention operation against specification",
    )
    parser_check_flash_attention.add_argument(
        "op_name",
        help="Name of the attention operation (e.g., 'flash_attn_func', 'paged_attention_v1', 'FlashAttention')",
    )
    parser_check_flash_attention.add_argument(
        "--inputs-count",
        type=int,
        default=None,
        help="Expected number of input operands",
    )
    parser_check_flash_attention.add_argument(
        "--attributes",
        default=None,
        help="Comma-separated list of expected keyword argument names",
    )
    parser_check_flash_attention.set_defaults(func=cmd_check_flash_attention)

    return parser


__all__ = ["build_parser"]
