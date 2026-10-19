"""CLI command handlers for exporting snapshots and generating .pyi stubs."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from ml_ecosystem_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef


def cmd_stubs(args: argparse.Namespace) -> None:
    """Handle the generate-stubs command to emit `.pyi` stub files.

    Args:
        args: Parsed command line arguments containing input, out_dir, and include_nonpublic.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.stubs import generate_stubs

    path = resolve_snapshot_path(args.input)
    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    generate_stubs(snap, args.out_dir, include_nonpublic=args.include_nonpublic)
    print(f"Stubs generated in {args.out_dir}")


cmd_generate_stubs = cmd_stubs


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export command for schemas, models, headers, and prompt contexts.

    Args:
        args: Parsed command line arguments containing input, out_dir, and format.

    Raises:
        ValueError: If an unsupported export format is requested.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path

    path = resolve_snapshot_path(args.input)
    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    refs: List[GhostRef] = []
    if isinstance(snap, list):
        for item in snap:
            refs.append(GhostInspector.hydrate(item))
    elif isinstance(snap, dict):
        for _cat, items in snap.get("categories", {}).items():
            for item in items:
                refs.append(GhostInspector.hydrate(item))

    os.makedirs(args.out_dir, exist_ok=True)

    if args.format == "openapi":
        from ml_ecosystem_snapshots.export import to_openapi

        openapi_spec = to_openapi(refs)
        out_path = os.path.join(args.out_dir, "openapi.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Exported OpenAPI spec to {out_path}")
    elif args.format == "json_schema":
        from ml_ecosystem_snapshots.export import to_json_schema

        for ref in refs:
            schema = to_json_schema(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name}.schema.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, sort_keys=True)
                f.write("\n")
        print(f"Exported {len(refs)} JSON schemas to {args.out_dir}")
    elif args.format == "pydantic":
        from ml_ecosystem_snapshots.export import to_pydantic

        for ref in refs:
            code = to_pydantic(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name.lower()}.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Pydantic models to {args.out_dir}")
    elif args.format == "protobuf":
        from ml_ecosystem_snapshots.export import to_protobuf

        for ref in refs:
            code = to_protobuf(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name.lower()}.proto")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Protobuf definitions to {args.out_dir}")
    elif args.format == "llm_prompt":
        from ml_ecosystem_snapshots.export import export_llm_prompt_context

        content = export_llm_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "llm_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported LLM prompt context to {out_path}")
    elif args.format == "sass_prompt":
        from ml_ecosystem_snapshots.export import export_sass_prompt_context

        content = export_sass_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "sass_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported SASS prompt context to {out_path}")
    elif args.format == "mlir_prompt":
        from ml_ecosystem_snapshots.export import export_mlir_prompt_context

        content = export_mlir_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "mlir_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported MLIR prompt context to {out_path}")
    elif args.format == "rdna_prompt":
        from ml_ecosystem_snapshots.export import export_rdna_prompt_context

        content = export_rdna_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "rdna_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported RDNA prompt context to {out_path}")
    elif args.format == "ptx_prompt":
        from ml_ecosystem_snapshots.export import export_ptx_prompt_context

        content = export_ptx_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "ptx_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported PTX prompt context to {out_path}")
    elif args.format == "stablehlo_prompt":
        from ml_ecosystem_snapshots.export import export_stablehlo_prompt_context

        content = export_stablehlo_prompt_context(refs)
        out_path = os.path.join(args.out_dir, "stablehlo_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported StableHLO prompt context to {out_path}")
    elif args.format == "scoped_prompt":
        from ml_ecosystem_snapshots.export import export_scoped_prompt_context

        fw_name = (
            (snap.get("target") if isinstance(snap, dict) else None)
            or (getattr(refs[0], "framework", None) if refs else None)
            or "unknown"
        )
        content = export_scoped_prompt_context(str(fw_name))
        out_path = os.path.join(args.out_dir, "scoped_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported Scoped prompt context to {out_path}")
    elif args.format == "cpp_header":
        from ml_ecosystem_snapshots.export import to_cpp_header

        for ref in refs:
            code = to_cpp_header(ref)
            out_path = os.path.join(args.out_dir, f"{ref.name.lower()}.hpp")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} C++ headers to {args.out_dir}")
    elif args.format == "typescript":
        from ml_ecosystem_snapshots.export import to_typescript_interface

        ts_defs = [to_typescript_interface(ref) for ref in refs]
        out_path = os.path.join(args.out_dir, "interfaces.d.ts")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(ts_defs) + "\n")
        print(f"Exported {len(refs)} TypeScript interfaces to {out_path}")
    else:
        raise ValueError(f"Unknown format: {args.format}")


__all__ = ["cmd_export", "cmd_generate_stubs", "cmd_stubs"]
