"""CLI command handlers for exporting snapshots and generating .pyi stubs."""

from __future__ import annotations

import argparse
import json
import os
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list


def cmd_stubs(args: argparse.Namespace) -> None:
    """Handle the generate-stubs command to emit `.pyi` stub files.

    Args:
        args: Parsed command line arguments containing input, out_dir, and include_nonpublic.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.network_env import load_json_data
    from ml_ecosystem_snapshots.stubs import generate_stubs

    path = resolve_snapshot_path(args.input)
    snap = ensure_json_dict(load_json_data(path), "snapshot")
    generate_stubs(snap, args.out_dir, include_nonpublic=args.include_nonpublic)
    print(f"Stubs generated in {args.out_dir}")


def _hydrate_snapshot_refs(snap: JsonValue) -> list[GhostRef]:
    """Hydrate a loaded snapshot JSON list or dictionary into `GhostRef` instances.

    Args:
        snap: Loaded JSON snapshot payload (`list` or `dict`).

    Returns:
        List of hydrated `GhostRef` objects.
    """
    refs: list[GhostRef] = []
    if isinstance(snap, list):
        for item in snap:
            refs.append(GhostInspector.hydrate(ensure_json_dict(item, "snapshot item")))
    elif isinstance(snap, dict):
        cats = ensure_json_dict(snap.get("categories", {}), "snapshot categories")
        for _cat, items in cats.items():
            for item in ensure_json_list(items, "category items"):
                refs.append(GhostInspector.hydrate(ensure_json_dict(item, "snapshot item")))
    return refs


def _export_prompt_format(
    fmt: str,
    out_dir: str,
    refs: list[GhostRef],
    snap: JsonValue,
) -> bool:
    """Export prompt context formats if `fmt` matches a prompt format identifier.

    Args:
        fmt: Requested export format string.
        out_dir: Destination directory path.
        refs: Hydrated `GhostRef` list.
        snap: Raw loaded snapshot JSON payload.

    Returns:
        True if `fmt` was handled as a prompt format, False otherwise.
    """
    from ml_ecosystem_snapshots.export import (
        export_llm_prompt_context,
        export_mlir_prompt_context,
        export_ptx_prompt_context,
        export_rdna_prompt_context,
        export_sass_prompt_context,
        export_scoped_prompt_context,
        export_stablehlo_prompt_context,
    )

    prompt_dispatch: dict[str, tuple[typing.Callable[[typing.Sequence[GhostRef]], str], str, str]] = {
        "llm_prompt": (export_llm_prompt_context, "llm_context.md", "LLM"),
        "sass_prompt": (export_sass_prompt_context, "sass_context.md", "SASS"),
        "mlir_prompt": (export_mlir_prompt_context, "mlir_context.md", "MLIR"),
        "rdna_prompt": (export_rdna_prompt_context, "rdna_context.md", "RDNA"),
        "ptx_prompt": (export_ptx_prompt_context, "ptx_context.md", "PTX"),
        "stablehlo_prompt": (export_stablehlo_prompt_context, "stablehlo_context.md", "StableHLO"),
    }
    if fmt in prompt_dispatch:
        fn, filename, label = prompt_dispatch[fmt]
        content = fn(refs)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported {label} prompt context to {out_path}")
        return True
    if fmt == "scoped_prompt":
        fw_name = (
            (snap.get("target") if isinstance(snap, dict) else None)
            or (refs[0].model_dump().get("framework") if refs else None)
            or "unknown"
        )
        content = export_scoped_prompt_context(str(fw_name))
        out_path = os.path.join(out_dir, "scoped_context.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported Scoped prompt context to {out_path}")
        return True
    return False


def _export_per_ref_code(
    fmt: str,
    out_dir: str,
    refs: list[GhostRef],
) -> bool:
    """Export per-symbol code files (`pydantic`, `protobuf`, `cpp_header`) if `fmt` matches.

    Args:
        fmt: Requested export format string.
        out_dir: Destination directory path.
        refs: Hydrated `GhostRef` list.

    Returns:
        True if `fmt` was handled as a per-ref code format, False otherwise.
    """
    from ml_ecosystem_snapshots.export import to_cpp_header, to_protobuf, to_pydantic

    if fmt == "pydantic":
        for ref in refs:
            code = to_pydantic(ref)
            out_path = os.path.join(out_dir, f"{ref.name.lower()}.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Pydantic models to {out_dir}")
        return True
    if fmt == "protobuf":
        for ref in refs:
            code = to_protobuf(ref)
            out_path = os.path.join(out_dir, f"{ref.name.lower()}.proto")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} Protobuf definitions to {out_dir}")
        return True
    if fmt == "cpp_header":
        for ref in refs:
            code = to_cpp_header(ref)
            out_path = os.path.join(out_dir, f"{ref.name.lower()}.hpp")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
        print(f"Exported {len(refs)} C++ headers to {out_dir}")
        return True
    return False


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export command for schemas, models, headers, and prompt contexts.

    Args:
        args: Parsed command line arguments containing input, out_dir, and format.

    Raises:
        ValueError: If an unsupported export format is requested.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.network_env import load_json_data

    path = resolve_snapshot_path(args.input)
    snap: JsonValue = load_json_data(path)

    refs = _hydrate_snapshot_refs(snap)
    os.makedirs(args.out_dir, exist_ok=True)

    if _export_prompt_format(args.format, args.out_dir, refs, snap):
        return
    if _export_per_ref_code(args.format, args.out_dir, refs):
        return

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
    elif args.format == "typescript":
        from ml_ecosystem_snapshots.export import to_typescript_interface

        ts_defs = [to_typescript_interface(ref) for ref in refs]
        out_path = os.path.join(args.out_dir, "interfaces.d.ts")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(ts_defs) + "\n")
        print(f"Exported {len(refs)} TypeScript interfaces to {out_path}")
    else:
        raise ValueError(f"Unknown format: {args.format}")


__all__ = ["cmd_export", "cmd_stubs"]
