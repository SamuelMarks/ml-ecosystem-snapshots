#!/usr/bin/env python3
"""Build StableHLO Ground Truth Snapshot.

Parses the official openxla/stablehlo TableGen ODS definitions (StableHloOps.td)
or specification spec.md to extract operations, SSA operands, attributes, regions,
traits, and return types, outputting a standard GhostRef JSON snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import typing

_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ml_ecosystem_snapshots.models import JsonValue, TableGenRecordModel  # noqa: E402
from ml_ecosystem_snapshots.tools.stablehlo_tablegen import (  # noqa: E402
    TableGenASTParser,
    build_verification_and_region_metadata,
    is_attribute,
    parse_stablehlo_llvm_tblgen_json,
    parse_stablehlo_tablegen,
    strip_tablegen_comments,
)
from ml_ecosystem_snapshots.utils import extract_tablegen_traits  # noqa: E402

SPEC_URL = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
TABLEGEN_URL = "https://raw.githubusercontent.com/openxla/stablehlo/main/stablehlo/dialect/StablehloOps.td"


def extract_ops(
    source_content: str | typing.Mapping[str, JsonValue] | None = None,
) -> list[TableGenRecordModel]:
    """Download and parse StableHLO TableGen or spec.md to extract ops.

    Args:
        source_content: Optional raw string content (TableGen, JSON, or markdown spec) or parsed dict.

    Returns:
        List of parsed operations as dictionary objects.
    """
    nl = chr(10)
    content = source_content
    if content is None:
        from ml_ecosystem_snapshots.network_env import fetch_url

        content = fetch_url(TABLEGEN_URL, timeout=10)
        if content is None:
            content = fetch_url(SPEC_URL)
            if content is None:
                raise RuntimeError("Failed to fetch StableHLO sources")

    assert content is not None

    if isinstance(content, typing.Mapping):
        return parse_stablehlo_llvm_tblgen_json(content)

    stripped = content.strip()
    if stripped.startswith("{") and (
        "!instanceof" in stripped or "StableHLO" in stripped
    ):
        return parse_stablehlo_llvm_tblgen_json(stripped)

    if "StableHLO_" in content:
        return parse_stablehlo_tablegen(content)

    ops_inputs: dict[str, list[tuple[str, str]]] = {}
    ops_outputs: dict[str, list[tuple[str, str]]] = {}
    ops_semantics: dict[str, list[str]] = {}
    current_op: str | None = None
    current_section: str | None = None

    for line in content.split(nl):
        if line.startswith("### "):
            op_name = line.strip()[4:]
            if op_name.replace("_", "").isalnum():
                current_op = op_name
                ops_inputs[current_op] = []
                ops_outputs[current_op] = []
                ops_semantics[current_op] = []
                current_section = None

        elif line.startswith("#### "):
            section_name = line.strip()[5:]
            if section_name.lower() in ("inputs", "outputs", "semantics"):
                current_section = section_name.lower()
            else:
                current_section = None
        elif current_op and current_section == "semantics":
            if line.strip() and not line.startswith("####"):
                ops_semantics[current_op].append(line)
        elif (
            line.startswith("|")
            and current_section in ("inputs", "outputs")
            and current_op
        ):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if (
                len(parts) >= 2
                and "---" not in parts[0]
                and "Label" not in parts[0]
                and "Name" not in parts[0]
            ):
                if current_section == "inputs":
                    if len(parts) >= 3 and parts[0].startswith("("):
                        name = parts[1]
                        type_str = parts[2]
                    else:
                        name = parts[0]
                        type_str = parts[1]
                else:
                    name = parts[0]
                    type_str = parts[1] if len(parts) > 1 else ""

                name = name.replace("`", "")
                if name:
                    if current_section == "inputs":
                        ops_inputs[current_op].append((name, type_str))
                    else:
                        ops_outputs[current_op].append((name, type_str))

    ghost_refs: list[TableGenRecordModel] = []

    for op_name in ops_inputs:
        in_list = ops_inputs[op_name]
        out_list = ops_outputs[op_name]
        sem_list = ops_semantics[op_name]
        if not in_list and not out_list:
            continue

        params: list[JsonValue] = []
        for name, type_str in in_list:
            kind = "KEYWORD_ONLY" if is_attribute(type_str) else "POSITIONAL_OR_KEYWORD"
            params.append(
                {
                    "name": name,
                    "annotation": type_str,
                    "default": None,
                    "kind": kind,
                    "description": None,
                    "standardized_name": None,
                }
            )

        returns_type: str | None = None
        if out_list:
            returns_type = ", ".join(t for _, t in out_list if t)
            if len(out_list) > 1:
                returns_type = f"tuple[{returns_type}]"

        docstring = nl.join(sem_list).strip()

        ghost_refs.append(
            TableGenRecordModel(
                {
                    "api_path": f"stablehlo.{op_name}",
                    "name": op_name,
                    "kind": "function",
                    "is_public": True,
                    "has_varargs": False,
                    "environment_tags": [],
                    "aliases": [],
                    "overloads": [],
                    "params": params,
                    "returns_type": returns_type,
                    "returns_description": None,
                    "raises": [],
                    "docstring": docstring if docstring else None,
                }
            )
        )

    return ghost_refs


def main(argv: list[str] | None = None) -> None:
    """Main entrypoint for StableHLO snapshot generation.

    Args:
        argv: Optional list of command-line arguments.
    """
    nl = chr(10)
    parser = argparse.ArgumentParser(
        description="Build StableHLO Ground Truth Snapshot."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v1.9.0",
        help="Git tag or commit hash for upstream StableHLO.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.9.0",
        help="SemVer version string for the snapshot.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for versioned snapshot JSON.",
    )
    parser.add_argument(
        "--exhaustive-path",
        type=str,
        default=None,
        help="Output path for exhaustive framework JSON.",
    )
    args, _ = parser.parse_known_args(argv)

    ghost_refs = extract_ops()

    snapshot_data = {
        "schema_version": "2.0.0",
        "target": "stablehlo",
        "version": args.version,
        "upstream_commit": args.tag,
        "upstream_version": args.version,
        "source_type": "tablegen",
        "categories": {"stablehlo_op": ghost_refs},
    }

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "snapshots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"stablehlo_v{args.version}.json"))

    with open(out_path, "w", encoding="utf-8", newline=nl) as f:
        json.dump(snapshot_data, f, indent=2, sort_keys=True)
        f.write(nl)

    exhaustive_path = args.exhaustive_path or os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "frameworks",
            "stablehlo_exhaustive.json",
        )
    )
    exhaustive_data = {
        "schema_version": "2.0.0",
        "target": "stablehlo",
        "version": args.version,
        "upstream_commit": args.tag,
        "upstream_version": args.version,
        "source_type": "tablegen",
        "operations": ghost_refs,
    }
    with open(exhaustive_path, "w", encoding="utf-8", newline=nl) as f:
        json.dump(exhaustive_data, f, indent=2, sort_keys=True)
        f.write(nl)

    print(f"Wrote {len(ghost_refs)} ops to {out_path} and {exhaustive_path}")


__all__ = [
    "SPEC_URL",
    "TABLEGEN_URL",
    "TableGenASTParser",
    "build_verification_and_region_metadata",
    "extract_ops",
    "extract_tablegen_traits",
    "is_attribute",
    "main",
    "parse_stablehlo_llvm_tblgen_json",
    "parse_stablehlo_tablegen",
    "strip_tablegen_comments",
]

if __name__ == "__main__":
    main()
