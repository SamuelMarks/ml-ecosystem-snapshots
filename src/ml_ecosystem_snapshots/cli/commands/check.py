"""CLI command handlers for compliance, ISA, IR, and specification verification."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Callable, Dict, Mapping


def _run_domain_op_check_cli(
    args: argparse.Namespace,
    check_fn: Callable[..., Mapping[str, Any]],
    label: str,
    op_attr: str = "op_name",
    count_attr: str = "inputs_count",
    count_param: str = "inputs_count",
    invalid_format: str = "Invalid:",
    extra_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Execute a domain op verification check and format CLI stdout/exit codes.

    Args:
        args: Parsed CLI `Namespace` object.
        check_fn: Underlying MCP verification callable.
        label: Human-readable domain operation prefix (e.g. 'ONNX Operation', 'WASM Instruction').
        op_attr: Attribute name on `args` holding the operation or mnemonic string.
        count_attr: Attribute name on `args` holding the expected operand/input count.
        count_param: Keyword parameter name accepted by `check_fn` for operand/input count.
        invalid_format: Suffix string when reporting an invalid operation ('Invalid:' or 'is invalid:').
        extra_kwargs: Optional additional keyword arguments to forward to `check_fn`.
    """
    raw_attrs = getattr(args, "attributes", None)
    attrs = (
        [a.strip() for a in raw_attrs.split(",") if a.strip()] if raw_attrs else None
    )
    target = getattr(args, op_attr)
    call_kwargs: Dict[str, Any] = {
        op_attr: target,
        count_param: getattr(args, count_attr, None),
        "attributes": attrs,
    }
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)

    res = check_fn(**call_kwargs)
    if not res.get("is_valid"):
        print(f"{label} '{target}' {invalid_format}")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"{label} '{target}' is valid.")
    print(f"  Inputs:     {len(res.get('expected_inputs', []))}")
    print(f"  Attributes: {', '.join(res.get('expected_attributes', []))}")


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the check command for compliance checking.

    Args:
        args: Parsed command line arguments containing snapshot_json, target_path,
              target_prefix, and reference_prefix.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.compliance import extract_target_refs, score_compliance
    from ml_switcheroo_ir.schema.ghost import GhostRef

    path = resolve_snapshot_path(args.snapshot_json)
    with open(path, "r", encoding="utf-8") as f:
        reference_snapshot = json.load(f)

    target_path = getattr(args, "target_path", "")
    target_ext = os.path.splitext(target_path)[1].lower() if target_path else ""

    if target_ext in (".sass", ".s"):
        from ml_ecosystem_snapshots.compliance import (
            check_rdna_assembly_compliance,
            check_sass_assembly_compliance,
        )

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        target = reference_snapshot.get("target", "").lower()
        if "rdna" in target or "amd" in target:
            res = check_rdna_assembly_compliance(content)
            print(
                f"AMD RDNA snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
            )
        else:
            res = check_sass_assembly_compliance(content)
            print(
                f"NVIDIA SASS snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
            )
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return

    if target_ext == ".mlir":
        from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        print(
            f"MLIR snippet compliance: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return

    print(f"Extracting target APIs from {args.target_path}...")
    target_refs = extract_target_refs(
        args.target_path, args.target_prefix, args.reference_prefix
    )

    print("Scoring compliance...")
    results = score_compliance(reference_snapshot, target_refs)

    score = results.get("score_percentage", 0.0)
    print("\n--- Compliance Report ---")
    print(f"Overall Compliance: {score}%\n")

    print("Breakdown by Module:")
    matched_by_mod = collections.defaultdict(list)
    missing_by_mod = collections.defaultdict(list)

    for api_path in results.get("matched", []):
        mod = ".".join(api_path.split(".")[:-1])
        matched_by_mod[mod].append(api_path)

    for api_path in results.get("missing", []):
        mod = ".".join(api_path.split(".")[:-1])
        missing_by_mod[mod].append(api_path)

    all_mods = set(list(matched_by_mod.keys()) + list(missing_by_mod.keys()))
    for mod in sorted(all_mods):
        m = len(matched_by_mod[mod])
        total = m + len(missing_by_mod[mod])
        mod_score = (m / total) * 100 if total > 0 else 0
        print(f"  - {mod}: {mod_score:.1f}% ({m}/{total})")

    missing = results.get("missing", [])
    if missing:
        print(f"\nMissing APIs ({len(missing)}):\n")

        ref_map: Dict[str, GhostRef] = {}
        for _cat, items in reference_snapshot.get("categories", {}).items():
            for item in items:
                ref_item = GhostRef.model_validate(item)
                ref_map[ref_item.api_path] = ref_item
                for alias in ref_item.aliases or []:
                    ref_map[alias] = ref_item

        print("|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |")
        print("|---|---|---|---|---|---|---|")

        for fqn in sorted(missing):
            ref_match = ref_map.get(fqn)
            if not ref_match:
                continue
            ref = ref_match

            parts = fqn.split(".")
            framework = parts[0]
            symbol = parts[-1]
            namespace = ".".join(parts[:-1])

            if ref.kind == "MODULE":
                sig = "module"
            else:
                sig_parts = []
                for p in ref.params:
                    p_str = p.name
                    if p.annotation:
                        p_str += f": {p.annotation}"
                    if p.default:
                        p_str += f"={p.default}"
                    sig_parts.append(p_str)
                sig = f"({', '.join(sig_parts)})"
                if ref.returns_type:
                    sig += f" -> {ref.returns_type}"

            doc = ref.docstring or ""
            if doc:
                doc = (
                    doc.strip().split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
                )
                if len(doc) > 100:
                    doc = doc[:97] + "..."
            else:
                doc = "No docstring available."

            print(
                f"| [ ] | {framework} | {namespace} | {symbol} | {fqn} | `{sig}` | {doc} |"
            )

    mismatched = results.get("mismatched", [])
    if mismatched:
        print(f"\nMismatched APIs ({len(mismatched)}):")
        for item in sorted(mismatched, key=lambda x: x["api_path"])[:20]:
            print(
                f"  ~ {item['api_path']}\n    Expected: {item.get('expected')}\n    Actual:   {item.get('actual')}"
            )
        if len(mismatched) > 20:
            print(f"  ~ ... and {len(mismatched) - 20} more")


def cmd_check_sass(args: argparse.Namespace) -> None:
    """Validate NVIDIA SASS assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, modifiers, sm_arch, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_sass_assembly_compliance
    from ml_ecosystem_snapshots.mcp_server import check_sass_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: SASS assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_sass_assembly_compliance(content, sm_arch=args.sm_arch)
        if not res.get("is_compliant"):
            print(
                f"SASS Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"SASS snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = (
        [op.strip() for op in args.operands.split(",") if op.strip()]
        if args.operands
        else None
    )
    modifiers = (
        [mod.strip() for mod in args.modifiers.split(",") if mod.strip()]
        if args.modifiers
        else None
    )

    inst_res = check_sass_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        modifiers=modifiers,
        sm_arch=args.sm_arch,
    )

    if not inst_res.get("is_valid"):
        print(f"SASS Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"SASS Instruction '{args.mnemonic}' is valid.")
        print(
            f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}"
        )
        if inst_res.get("valid_modifiers"):
            print(
                f"  Recognized Modifiers:    {', '.join(inst_res.get('valid_modifiers', []))}"
            )


def cmd_check_rdna(args: argparse.Namespace) -> None:
    """Validate AMD RDNA assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, encoding, gfx_arch, modifiers, wave_size, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_rdna_assembly_compliance
    from ml_ecosystem_snapshots.mcp_server import check_rdna_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: RDNA assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_rdna_assembly_compliance(content, gfx_arch=args.gfx_arch)
        if not res.get("is_compliant"):
            print(
                f"RDNA Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"RDNA snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = (
        [op.strip() for op in args.operands.split(",") if op.strip()]
        if args.operands
        else None
    )
    modifiers = (
        [mod.strip() for mod in args.modifiers.split(",") if mod.strip()]
        if args.modifiers
        else None
    )

    inst_res = check_rdna_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        encoding=args.encoding,
        gfx_arch=args.gfx_arch,
        modifiers=modifiers,
        wave_size=args.wave_size,
    )

    if not inst_res.get("is_valid"):
        print(f"RDNA Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"RDNA Instruction '{args.mnemonic}' is valid.")
        print(
            f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}"
        )
        if inst_res.get("encoding"):
            print(f"  Encoding:                {inst_res.get('encoding')}")


def cmd_check_ptx(args: argparse.Namespace) -> None:
    """Validate NVIDIA PTX instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, types, operands, state_space, scope, vector_width, sm_arch.
    """
    from ml_ecosystem_snapshots.mcp_server import check_ptx_instruction

    mnemonic = getattr(args, "opt_mnemonic", None) or getattr(args, "mnemonic", None)
    if not mnemonic:
        print("Error: Instruction mnemonic must be specified.")
        sys.exit(1)

    types = (
        [t.strip() for t in args.types.split(",") if t.strip()]
        if getattr(args, "types", None)
        else None
    )
    operands = (
        [op.strip() for op in args.operands.split(",") if op.strip()]
        if getattr(args, "operands", None)
        else None
    )

    res = check_ptx_instruction(
        mnemonic=mnemonic,
        types=types,
        operands=operands,
        state_space=getattr(args, "state_space", None),
        scope=getattr(args, "scope", None),
        vector_width=getattr(args, "vector_width", None),
        sm_arch=getattr(args, "sm_arch", None),
    )

    if not res.get("is_valid"):
        print(f"PTX Instruction '{mnemonic}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PTX Instruction '{mnemonic}' is valid.")
        if res.get("min_sm"):
            print(f"  Minimum SM Architecture: {res.get('min_sm')}")
        if res.get("supported_types"):
            print(
                f"  Supported Types:         {', '.join(res.get('supported_types', []))}"
            )


def cmd_check_mlir(args: argparse.Namespace) -> None:
    """Validate MLIR text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance
    from ml_ecosystem_snapshots.mcp_server import check_mlir_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: MLIR file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(
                f"MLIR Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"MLIR snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = (
        [a.strip() for a in args.attributes.split(",") if a.strip()]
        if args.attributes
        else None
    )

    op_res = check_mlir_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.get("is_valid"):
        print(f"MLIR Operation '{args.op_name}' Invalid:")
        for err in op_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"MLIR Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(op_res.get('expected_attributes', []))}")


def cmd_check_stablehlo(args: argparse.Namespace) -> None:
    """Validate StableHLO text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance
    from ml_ecosystem_snapshots.mcp_server import check_stablehlo_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: StableHLO file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(
                f"StableHLO Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"StableHLO snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid."
        )
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = (
        [a.strip() for a in args.attributes.split(",") if a.strip()]
        if args.attributes
        else None
    )

    op_res = check_stablehlo_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.get("is_valid"):
        print(f"StableHLO Operation '{args.op_name}' Invalid:")
        for err in op_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"StableHLO Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(op_res.get('expected_attributes', []))}")


def cmd_check_wgsl(args: argparse.Namespace) -> None:
    """Validate WebGPU WGSL shader code or individual operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, inputs_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_wgsl_shader_compliance
    from ml_ecosystem_snapshots.mcp_server import check_wgsl_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: WGSL file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = check_wgsl_shader_compliance(content)
        if not res.get("is_compliant"):
            print(
                f"WGSL Compliance Check Failed ({len(res.get('errors', []))} errors):"
            )
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"WGSL shader verified compliant: {res.get('verified_ops', 0)} operations recognized."
        )
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    _run_domain_op_check_cli(args, check_wgsl_op, "WGSL Operation")


def cmd_check_onnx(args: argparse.Namespace) -> None:
    """Validate ONNX operation against specification definitions.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_onnx_op

    _run_domain_op_check_cli(
        args,
        check_onnx_op,
        "ONNX Operation",
        extra_kwargs={"domain": getattr(args, "domain", "") or ""},
    )


def cmd_check_metal(args: argparse.Namespace) -> None:
    """Validate Apple Metal MSL compute operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_metal_op

    _run_domain_op_check_cli(
        args,
        check_metal_op,
        "Metal Operation",
        extra_kwargs={"address_space": getattr(args, "address_space", None)},
    )


def cmd_check_wasm(args: argparse.Namespace) -> None:
    """Validate WebAssembly 2.0 / SIMD instruction against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wasm_instruction

    _run_domain_op_check_cli(
        args,
        check_wasm_instruction,
        "WASM Instruction",
        op_attr="mnemonic",
        count_attr="operands_count",
        count_param="operands_count",
    )


def cmd_check_webgl(args: argparse.Namespace) -> None:
    """Validate WebGL 2.0 / GLSL ES 3.00 operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_webgl_op

    _run_domain_op_check_cli(args, check_webgl_op, "WebGL Operation")


def cmd_check_cpp(args: argparse.Namespace) -> None:
    """Validate C++17 / PyBind11 operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_cpp_op

    _run_domain_op_check_cli(args, check_cpp_op, "C++ Operation")


def cmd_check_array_api(args: argparse.Namespace) -> None:
    """Validate Python Array API Standard operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_array_api_op

    _run_domain_op_check_cli(
        args,
        check_array_api_op,
        "Array API Operation",
        invalid_format="is invalid:",
    )


def cmd_check_scipy(args: argparse.Namespace) -> None:
    """Validate SciPy operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_scipy_op

    _run_domain_op_check_cli(
        args, check_scipy_op, "SciPy Operation", invalid_format="is invalid:"
    )


def cmd_check_torchvision(args: argparse.Namespace) -> None:
    """Validate TorchVision operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_torchvision_op

    _run_domain_op_check_cli(
        args,
        check_torchvision_op,
        "TorchVision Operation",
        invalid_format="is invalid:",
    )


def cmd_check_torchaudio(args: argparse.Namespace) -> None:
    """Validate TorchAudio operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_torchaudio_op

    _run_domain_op_check_cli(
        args,
        check_torchaudio_op,
        "TorchAudio Operation",
        invalid_format="is invalid:",
    )


def cmd_check_safetensors(args: argparse.Namespace) -> None:
    """Validate SafeTensors operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_safetensors_op

    _run_domain_op_check_cli(
        args,
        check_safetensors_op,
        "SafeTensors Operation",
        invalid_format="is invalid:",
    )


def cmd_check_aten(args: argparse.Namespace) -> None:
    """Validate PyTorch ATen operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_aten_op

    _run_domain_op_check_cli(
        args, check_aten_op, "ATen Operation", invalid_format="is invalid:"
    )


def cmd_check_nccl(args: argparse.Namespace) -> None:
    """Validate NCCL/RCCL collective operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_nccl_op

    _run_domain_op_check_cli(
        args, check_nccl_op, "NCCL Operation", invalid_format="is invalid:"
    )


def cmd_check_flash_attention(args: argparse.Namespace) -> None:
    """Validate FlashAttention operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_flash_attention_op

    _run_domain_op_check_cli(
        args,
        check_flash_attention_op,
        "Attention Operation",
        invalid_format="is invalid:",
    )


__all__ = [
    "_run_domain_op_check_cli",
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
]
