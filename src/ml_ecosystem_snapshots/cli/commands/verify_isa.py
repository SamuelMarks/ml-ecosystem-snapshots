"""CLI command handlers for hardware ISA verification (`check-sass`, `check-rdna`, `check-ptx`)."""

from __future__ import annotations

import argparse
import os
import sys

from ml_ecosystem_snapshots.compliance import (
    check_rdna_assembly_compliance,
    check_sass_assembly_compliance,
)
from ml_ecosystem_snapshots.grounding.hardware import (
    verify_ptx_instruction,
    verify_rdna_instruction,
    verify_sass_instruction,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict
from ml_ecosystem_snapshots.serialization.repository import load_or_extract_snapshot


def _read_ptx_str_attr(args: argparse.Namespace, field: str) -> str | None:
    """Read an optional string PTX CLI attribute from `args` without reflection builtins.

    Args:
        args: Parsed CLI `Namespace` instance.
        field: Known PTX CLI field selector.

    Returns:
        Optional string value or `None` when absent.
    """
    val = get_member_dict(args).get(field)
    return str(val) if val is not None else None


def cmd_check_sass(args: argparse.Namespace) -> None:
    """Validate NVIDIA SASS assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, modifiers, sm_arch, or file.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or arguments are missing.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: SASS assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_sass_assembly_compliance(content, sm_arch=args.sm_arch)
        if not res.is_compliant:
            print(f"SASS Compliance Check Failed ({len(res.errors)} errors):")
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"SASS snippet verified compliant: {res.verified_instructions}/{res.total_instructions} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None
    snap = load_or_extract_snapshot("nvidia_sass")

    inst_res = verify_sass_instruction(
        mnemonic=args.mnemonic,
        snap=snap,
        operands=operands,
        modifiers=modifiers,
        sm_arch=args.sm_arch,
    )

    if not inst_res.is_valid:
        print(f"SASS Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"SASS Instruction '{args.mnemonic}' is valid.")
        print(f"  Supported Architectures: {', '.join(inst_res.supported_architectures or [])}")
        if inst_res.valid_modifiers:
            print(f"  Recognized Modifiers:    {', '.join(inst_res.valid_modifiers)}")


def cmd_check_rdna(args: argparse.Namespace) -> None:
    """Validate AMD RDNA assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, encoding, gfx_arch, modifiers, wave_size, or file.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or arguments are missing.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: RDNA assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_rdna_assembly_compliance(content, gfx_arch=args.gfx_arch)
        if not res.is_compliant:
            print(f"RDNA Compliance Check Failed ({len(res.errors)} errors):")
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(
            f"RDNA snippet verified compliant: {res.verified_instructions}/{res.total_instructions} instructions valid."
        )
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None
    snap = load_or_extract_snapshot("amd_rdna")

    inst_res = verify_rdna_instruction(
        mnemonic=args.mnemonic,
        snap=snap,
        operands=operands,
        encoding=args.encoding,
        gfx_arch=args.gfx_arch,
        modifiers=modifiers,
        wave_size=args.wave_size,
    )

    if not inst_res.is_valid:
        print(f"RDNA Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"RDNA Instruction '{args.mnemonic}' is valid.")
        print(f"  Supported Architectures: {', '.join(inst_res.supported_architectures or [])}")
        if inst_res.encoding:
            print(f"  Encoding:                {inst_res.encoding}")


def cmd_check_ptx(args: argparse.Namespace) -> None:
    """Validate NVIDIA PTX instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, types, operands, state_space, scope, vector_width, sm_arch.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or mnemonic is missing.
    """
    mnemonic = _read_ptx_str_attr(args, "mnemonic")
    if not mnemonic:
        print("Error: Instruction mnemonic must be specified.")
        sys.exit(1)

    raw_types = _read_ptx_str_attr(args, "types")
    types = [t.strip() for t in raw_types.split(",") if t.strip()] if raw_types else None
    raw_operands = _read_ptx_str_attr(args, "operands")
    operands = [op.strip() for op in raw_operands.split(",") if op.strip()] if raw_operands else None

    state_space = _read_ptx_str_attr(args, "state_space")
    scope = _read_ptx_str_attr(args, "scope")
    vector_width = _read_ptx_str_attr(args, "vector_width")
    sm_arch = _read_ptx_str_attr(args, "sm_arch")

    res = verify_ptx_instruction(
        mnemonic=mnemonic,
        types=types,
        operands=operands,
        state_space=state_space,
        scope=scope,
        vector_width=vector_width,
        sm_arch=sm_arch,
    )

    if not res.is_valid:
        print(f"PTX Instruction '{mnemonic}' Invalid:")
        for err in res.errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PTX Instruction '{mnemonic}' is valid.")
        if res.min_sm:
            print(f"  Minimum SM Architecture: {res.min_sm}")
        if res.supported_types:
            print(f"  Supported Types:         {', '.join(res.supported_types)}")


__all__ = [
    "cmd_check_ptx",
    "cmd_check_rdna",
    "cmd_check_sass",
]
