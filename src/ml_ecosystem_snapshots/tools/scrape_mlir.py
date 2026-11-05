"""Script to generate an exhaustive JSON dump of MLIR operations.

Extracts operations across core ML dialects (arith, math, tensor, linalg, scf, func,
memref, gpu, vector) via TableGen ODS parsing, live Python dialect inspection,
or official MLIR documentation.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import typing
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ml_ecosystem_snapshots.models import TableGenRecordModel  # noqa: E402
from ml_ecosystem_snapshots.network_env import fetch_url  # noqa: E402
from ml_ecosystem_snapshots.tools.build_stablehlo_snapshot import (  # noqa: E402
    extract_ops,
)
from ml_ecosystem_snapshots.tools.mlir_tablegen import (  # noqa: E402
    TableGenASTParser,
    dump_ast_via_llvm_tooling,
    parse_llvm_tblgen_json,
    parse_mlir_tablegen,
    parse_tablegen_args,
    parse_tablegen_regions,
    parse_tablegen_results,
    strip_tablegen_comments,
    validate_dialect_ops_ods,
)
from ml_ecosystem_snapshots.utils import (  # noqa: E402
    extract_tablegen_traits,
)

MLIR_DOCS_URL = "https://mlir.llvm.org/docs/Dialects/"
CORE_MLIR_DIALECTS: list[str] = [
    "arith",
    "math",
    "tensor",
    "linalg",
    "scf",
    "func",
    "memref",
    "gpu",
    "vector",
    "llvm",
    "nvvm",
    "rocdl",
]


@typing.runtime_checkable
class _MlirOpClassProtocol(typing.Protocol):
    """Protocol for MLIR Python dialect operation classes exposing `OPERATION_NAME`."""

    OPERATION_NAME: str


def inspect_mlir_python_module(
    dialects: list[str] | None = None,
) -> list[TableGenRecordModel]:
    """Inspect installed mlir.dialects Python modules for operation classes.

    Args:
        dialects: Optional list of dialect names to inspect.

    Returns:
        List of extracted operation dictionaries.
    """
    target_dialects = dialects or CORE_MLIR_DIALECTS
    ops: list[TableGenRecordModel] = []

    for dialect in target_dialects:
        try:
            mod = importlib.import_module(f"mlir.dialects.{dialect}")
            for attr_name, cls in inspect.getmembers(mod):
                if attr_name.endswith("Op") and not attr_name.startswith("_"):
                    op_mnemonic = cls.OPERATION_NAME if isinstance(cls, _MlirOpClassProtocol) else f"{dialect}.{attr_name[:-2].lower()}"
                    ops.append(
                        TableGenRecordModel(
                            {
                                "api_path": str(op_mnemonic),
                                "dialect": dialect,
                                "class_name": attr_name,
                                "operands": [],
                                "attributes": [],
                                "description": inspect.getdoc(cls) or f"MLIR {attr_name} operation.",
                            }
                        )
                    )
        except ImportError:
            continue

    return ops


def _append_unique_ops(
    ops: typing.Sequence[TableGenRecordModel],
    all_ops: list[TableGenRecordModel],
    seen_paths: set[str],
) -> None:
    """Append operations with previously unseen `api_path` keys into `all_ops`.

    Args:
        ops: Sequence of candidate `TableGenRecordModel` operation records.
        all_ops: Destination list accumulating unique operation records.
        seen_paths: Set of `api_path` strings already added to `all_ops`.
    """
    for op in ops:
        api_p = str(op.get("api_path", ""))
        if api_p not in seen_paths:
            seen_paths.add(api_p)
            all_ops.append(op)


def _fetch_upstream_dialect_ops(
    tag_ref: str,
    all_ops: list[TableGenRecordModel],
    seen_paths: set[str],
) -> None:
    """Fetch and parse upstream LLVM MLIR TableGen definitions across core dialects.

    Args:
        tag_ref: Git tag or commit reference in `llvm/llvm-project`.
        all_ops: Destination list accumulating unique operation records.
        seen_paths: Set of `api_path` strings already added to `all_ops`.
    """
    dialect_map = {
        "arith": ("Arith", ["ArithOps.td"]),
        "math": ("Math", ["MathOps.td"]),
        "tensor": ("Tensor", ["TensorOps.td"]),
        "linalg": ("Linalg", ["LinalgOps.td", "LinalgStructuredOps.td"]),
        "scf": ("SCF", ["SCFOps.td"]),
        "func": ("Func", ["FuncOps.td"]),
        "memref": ("MemRef", ["MemRefOps.td"]),
        "gpu": ("GPU", ["GPUOps.td"]),
        "vector": ("Vector", ["VectorOps.td"]),
        "llvm": ("LLVMIR", ["LLVMOps.td"]),
    }

    for dialect_name, (d_cap, files) in dialect_map.items():
        subpath = f"{d_cap}" if d_cap == "LLVMIR" else f"{d_cap}/IR"
        for td_file in files:
            url = f"https://raw.githubusercontent.com/llvm/llvm-project/{tag_ref}/mlir/include/mlir/Dialect/{subpath}/{td_file}"
            td_content = fetch_url(url)
            if td_content:
                parsed = parse_llvm_tblgen_json(td_content, dialect_name) if td_content.strip().startswith("{") else parse_mlir_tablegen(td_content, dialect_name)
                _append_unique_ops(parsed, all_ops, seen_paths)


def main(argv: list[str] | None = None) -> None:
    """Run the scraper and output enveloped mlir_exhaustive.json.

    Args:
        argv: Optional list of command-line arguments.
    """
    nl = chr(10)
    parser = argparse.ArgumentParser(description="Scrape exhaustive MLIR dialect operations.")
    parser.add_argument(
        "--tag",
        type=str,
        default="llvmorg-19.1.0",
        help="Git tag or release tag for upstream LLVM.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="19.1.0",
        help="SemVer version for the MLIR dialect snapshot.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Target output path for mlir_exhaustive.json.",
    )
    args, _ = parser.parse_known_args(argv)

    output_path = args.output_path or os.path.join(os.path.dirname(__file__), "..", "frameworks", "mlir_exhaustive.json")

    all_ops: list[TableGenRecordModel] = []
    seen_paths: set[str] = set()

    _fetch_upstream_dialect_ops(args.tag, all_ops, seen_paths)
    _append_unique_ops(inspect_mlir_python_module(), all_ops, seen_paths)

    try:
        stablehlo_ops = extract_ops()
    except RuntimeError:
        stablehlo_ops = []
    _append_unique_ops(stablehlo_ops, all_ops, seen_paths)

    exhaustive_data = {
        "schema_version": "2.0.0",
        "target": "mlir",
        "version": args.version,
        "upstream_commit": args.tag,
        "upstream_version": args.version,
        "source_type": "tablegen",
        "operations": all_ops,
    }

    with open(output_path, "w", encoding="utf-8", newline=nl) as f:
        json.dump(exhaustive_data, f, indent=2, sort_keys=True)
        f.write(nl)
    print(f"Dumped exhaustive MLIR operations to {output_path}")


__all__ = [
    "CORE_MLIR_DIALECTS",
    "MLIR_DOCS_URL",
    "TableGenASTParser",
    "dump_ast_via_llvm_tooling",
    "extract_ops",
    "extract_tablegen_traits",
    "fetch_url",
    "inspect_mlir_python_module",
    "main",
    "parse_llvm_tblgen_json",
    "parse_mlir_tablegen",
    "parse_tablegen_args",
    "parse_tablegen_regions",
    "parse_tablegen_results",
    "strip_tablegen_comments",
    "validate_dialect_ops_ods",
]

if __name__ == "__main__":
    main()
