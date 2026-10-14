"""Script to generate an exhaustive JSON dump of MLIR operations.

Extracts operations across core ML dialects (arith, math, tensor, linalg, scf, func,
memref, gpu, vector) via TableGen ODS parsing, live Python dialect inspection,
or official MLIR documentation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ml_ecosystem_snapshots.tools.mlir_html import (  # noqa: E402
    fetch_html,
    parse_dialect_page,
    parse_table,
    scrape_stablehlo,
)
from ml_ecosystem_snapshots.tools.mlir_tablegen import (  # noqa: E402
    TableGenASTParser,
    dump_ast_via_llvm_tooling,
    parse_llvm_tblgen_json,
    parse_mlir_tablegen,
    parse_tablegen_args,
    parse_tablegen_regions,
    parse_tablegen_results,
    parse_tblgen_json_dump,
    strip_tablegen_comments,
    validate_dialect_ops_ods,
)
from ml_ecosystem_snapshots.utils import (  # noqa: E402
    extract_tablegen_traits,
    fetch_url,
)

MLIR_DOCS_URL = "https://mlir.llvm.org/docs/Dialects/"
CORE_MLIR_DIALECTS: List[str] = [
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


def inspect_mlir_python_module(
    dialects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Inspect installed mlir.dialects Python modules for operation classes.

    Args:
        dialects: Optional list of dialect names to inspect.

    Returns:
        List of extracted operation dictionaries.
    """
    target_dialects = dialects or CORE_MLIR_DIALECTS
    ops: List[Dict[str, Any]] = []

    for dialect in target_dialects:
        try:
            mod = importlib.import_module(f"mlir.dialects.{dialect}")
            for attr_name in dir(mod):
                if attr_name.endswith("Op") and not attr_name.startswith("_"):
                    cls = getattr(mod, attr_name)
                    op_mnemonic = getattr(
                        cls, "OPERATION_NAME", f"{dialect}.{attr_name[:-2].lower()}"
                    )
                    ops.append(
                        {
                            "api_path": str(op_mnemonic),
                            "dialect": dialect,
                            "class_name": attr_name,
                            "operands": [],
                            "attributes": [],
                            "description": cls.__doc__
                            or f"MLIR {attr_name} operation.",
                        }
                    )
        except ImportError:
            continue

    return ops


def main(argv: Optional[List[str]] = None) -> None:
    """Run the scraper and output enveloped mlir_exhaustive.json.

    Args:
        argv: Optional list of command-line arguments.
    """
    nl = chr(10)
    parser = argparse.ArgumentParser(
        description="Scrape exhaustive MLIR dialect operations."
    )
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

    output_path = args.output_path or os.path.join(
        os.path.dirname(__file__), "..", "frameworks", "mlir_exhaustive.json"
    )

    all_ops: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    dialect_map = {
        "arith": ["ArithOps.td"],
        "math": ["MathOps.td"],
        "tensor": ["TensorOps.td"],
        "linalg": ["LinalgOps.td", "LinalgStructuredOps.td"],
        "scf": ["SCFOps.td"],
        "func": ["FuncOps.td"],
        "memref": ["MemRefOps.td"],
        "gpu": ["GPUOps.td"],
        "vector": ["VectorOps.td"],
        "llvm": ["LLVMOps.td"],
    }
    capital_map = {
        "arith": "Arith",
        "math": "Math",
        "tensor": "Tensor",
        "linalg": "Linalg",
        "scf": "SCF",
        "func": "Func",
        "memref": "MemRef",
        "gpu": "GPU",
        "vector": "Vector",
        "llvm": "LLVMIR",
    }

    tag_ref = args.tag
    for d, files in dialect_map.items():
        d_cap = capital_map[d]
        for td_file in files:
            url = (
                f"https://raw.githubusercontent.com/llvm/llvm-project/{tag_ref}/mlir/include/mlir/Dialect/{d_cap}/{td_file}"
                if d_cap == "LLVMIR"
                else f"https://raw.githubusercontent.com/llvm/llvm-project/{tag_ref}/mlir/include/mlir/Dialect/{d_cap}/IR/{td_file}"
            )
            td_content = fetch_html(url)
            if td_content:
                parsed = parse_mlir_tablegen(td_content, d)
                for op in parsed:
                    if op["api_path"] not in seen_paths:
                        seen_paths.add(op["api_path"])
                        all_ops.append(op)

    live_ops = inspect_mlir_python_module()
    for op in live_ops:
        if op["api_path"] not in seen_paths:
            seen_paths.add(op["api_path"])
            all_ops.append(op)

    if not all_ops:
        print(f"Fetching dialects from {MLIR_DOCS_URL}...")
        index_html = fetch_html(MLIR_DOCS_URL)
        dialect_links = re.findall(
            r"""<a href="([^"]+)">'([a-zA-Z0-9_]+)' Dialect</a>""", index_html
        )
        if not dialect_links:
            dialect_links = re.findall(
                r"""<a href=([^>]+)>'([a-zA-Z0-9_]+)' Dialect</a>""",
                index_html,
            )

        seen_urls = set()
        for link, dialect_name in dialect_links:
            dialect_url = (
                f"https://mlir.llvm.org{link}" if link.startswith("/") else link
            )
            if dialect_url in seen_urls:
                continue
            seen_urls.add(dialect_url)

            ops = parse_dialect_page(dialect_url, dialect_name)
            for op in ops:
                if op["api_path"] not in seen_paths:
                    seen_paths.add(op["api_path"])
                    all_ops.append(op)

    stablehlo_ops = scrape_stablehlo()
    for op in stablehlo_ops:
        if op["api_path"] not in seen_paths:
            seen_paths.add(op["api_path"])
            all_ops.append(op)

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
    "extract_tablegen_traits",
    "fetch_html",
    "fetch_url",
    "inspect_mlir_python_module",
    "main",
    "parse_dialect_page",
    "parse_llvm_tblgen_json",
    "parse_mlir_tablegen",
    "parse_table",
    "parse_tablegen_args",
    "parse_tablegen_regions",
    "parse_tablegen_results",
    "parse_tblgen_json_dump",
    "scrape_stablehlo",
    "strip_tablegen_comments",
    "validate_dialect_ops_ods",
]

if __name__ == "__main__":
    main()
