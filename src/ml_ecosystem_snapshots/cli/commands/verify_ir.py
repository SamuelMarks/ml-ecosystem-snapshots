"""CLI command handlers for compiler IR & shader verification (`check-mlir`, `check-stablehlo`, `check-wgsl`)."""

from __future__ import annotations

import argparse
import os
import sys

from ml_ecosystem_snapshots.cli.commands.verify_domains import _run_domain_op_check_cli


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
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(f"MLIR Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"MLIR snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None

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
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(f"StableHLO Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"StableHLO snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None

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
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_wgsl_shader_compliance(content)
        if not res.get("is_compliant"):
            print(f"WGSL Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"WGSL shader verified compliant: {res.get('verified_ops', 0)} operations recognized.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    _run_domain_op_check_cli(args, check_wgsl_op, "WGSL Operation")


__all__ = [
    "cmd_check_mlir",
    "cmd_check_stablehlo",
    "cmd_check_wgsl",
]
