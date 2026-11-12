"""CLI command handlers for compiler IR & shader verification (`check-mlir`, `check-stablehlo`, `check-wgsl`)."""

from __future__ import annotations

import argparse
import os
import sys

from ml_ecosystem_snapshots.cli.commands.verify_domains import (
    _parse_cli_attributes,
    _parse_inputs_count,
    _report_domain_check_result,
)
from ml_ecosystem_snapshots.compliance import (
    check_mlir_text_compliance,
    check_wgsl_shader_compliance,
)
from ml_ecosystem_snapshots.grounding.compiler import verify_mlir_op
from ml_ecosystem_snapshots.grounding.domain_ops import check_domain_op
from ml_ecosystem_snapshots.serialization.repository import load_or_extract_snapshot


def cmd_check_mlir(args: argparse.Namespace) -> None:
    """Validate MLIR text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or arguments are missing.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: MLIR file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.is_compliant:
            print(f"MLIR Compliance Check Failed ({len(res.errors)} errors):")
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"MLIR snippet verified compliant: {res.verified_ops}/{res.total_ops} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None
    fw = "stablehlo" if str(args.op_name).startswith("stablehlo.") else "mlir"
    snap = load_or_extract_snapshot(fw)
    op_res = verify_mlir_op(
        op_name=args.op_name,
        snap=snap,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.is_valid:
        print(f"MLIR Operation '{args.op_name}' Invalid:")
        for err in op_res.errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"MLIR Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.expected_operands or [])}")
    print(f"  Attributes: {', '.join(op_res.expected_attributes or [])}")


def cmd_check_stablehlo(args: argparse.Namespace) -> None:
    """Validate StableHLO text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or arguments are missing.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: StableHLO file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.is_compliant:
            print(f"StableHLO Compliance Check Failed ({len(res.errors)} errors):")
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"StableHLO snippet verified compliant: {res.verified_ops}/{res.total_ops} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None
    clean_name = args.op_name if str(args.op_name).startswith("stablehlo.") else f"stablehlo.{args.op_name}"
    snap = load_or_extract_snapshot("stablehlo")
    op_res = verify_mlir_op(
        op_name=clean_name,
        snap=snap,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.is_valid:
        print(f"StableHLO Operation '{args.op_name}' Invalid:")
        for err in op_res.errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"StableHLO Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.expected_operands or [])}")
    print(f"  Attributes: {', '.join(op_res.expected_attributes or [])}")


def cmd_check_wgsl(args: argparse.Namespace) -> None:
    """Validate WebGPU WGSL shader code or individual operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, inputs_count, attributes, or file.

    Raises:
        SystemExit: Exits with status code `1` when validation fails or arguments are missing.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: WGSL file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_wgsl_shader_compliance(content)
        if not res.is_compliant:
            print(f"WGSL Compliance Check Failed ({len(res.errors)} errors):")
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        print(f"WGSL shader verified compliant: {res.verified_ops} operations recognized.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    target = str(args.op_name)
    wgsl_res = check_domain_op(
        "wgsl",
        op_name=target,
        inputs_count=_parse_inputs_count(args),
        attributes=_parse_cli_attributes(args),
    )
    _report_domain_check_result("WGSL Operation", target, wgsl_res)


__all__ = [
    "cmd_check_mlir",
    "cmd_check_stablehlo",
    "cmd_check_wgsl",
]
