"""CLI command handlers for hardware ISA verification (`check-sass`, `check-rdna`, `check-ptx`)."""

from __future__ import annotations

import argparse
import os
import sys

from ml_ecosystem_snapshots.cli.commands.verify_domains import _read_namespace_attr


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
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_sass_assembly_compliance(content, sm_arch=args.sm_arch)
        if not res.get("is_compliant"):
            print(f"SASS Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"SASS snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None

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
        print(f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}")
        if inst_res.get("valid_modifiers"):
            print(f"  Recognized Modifiers:    {', '.join(inst_res.get('valid_modifiers', []))}")


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
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_rdna_assembly_compliance(content, gfx_arch=args.gfx_arch)
        if not res.get("is_compliant"):
            print(f"RDNA Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"RDNA snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None

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
        print(f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}")
        if inst_res.get("encoding"):
            print(f"  Encoding:                {inst_res.get('encoding')}")


def cmd_check_ptx(args: argparse.Namespace) -> None:
    """Validate NVIDIA PTX instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, types, operands, state_space, scope, vector_width, sm_arch.
    """
    from ml_ecosystem_snapshots.mcp_server import check_ptx_instruction

    mnemonic = _read_namespace_attr(args, "mnemonic")
    if not mnemonic:
        print("Error: Instruction mnemonic must be specified.")
        sys.exit(1)

    raw_types = _read_namespace_attr(args, "types")
    types = [t.strip() for t in str(raw_types).split(",") if t.strip()] if raw_types else None
    raw_operands = _read_namespace_attr(args, "operands")
    operands = [op.strip() for op in str(raw_operands).split(",") if op.strip()] if raw_operands else None

    state_space = _read_namespace_attr(args, "state_space")
    scope = _read_namespace_attr(args, "scope")
    vector_width = _read_namespace_attr(args, "vector_width")
    sm_arch = _read_namespace_attr(args, "sm_arch")

    res = check_ptx_instruction(
        mnemonic=str(mnemonic),
        types=types,
        operands=operands,
        state_space=str(state_space) if state_space is not None else None,
        scope=str(scope) if scope is not None else None,
        vector_width=str(vector_width) if vector_width is not None else None,
        sm_arch=str(sm_arch) if sm_arch is not None else None,
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
            print(f"  Supported Types:         {', '.join(res.get('supported_types', []))}")


__all__ = [
    "cmd_check_ptx",
    "cmd_check_rdna",
    "cmd_check_sass",
]
