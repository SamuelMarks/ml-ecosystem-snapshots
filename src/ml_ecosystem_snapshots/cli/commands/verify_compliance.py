"""CLI command handlers for Python/assembly/MLIR target compliance checking (`check`)."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.compliance.models import NamespaceAlignmentRule
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list

ZERO_ECOSYSTEM_ALIGNMENT_RULE = NamespaceAlignmentRule(
    prefix_map={
        "zero_jax": "jax",
        "zero_optax": "optax",
        "zero_chex": "chex",
        "zero_orbax": "orbax",
        "zero_grain": "grain",
    },
    subpackage_roots=("jax", "optax"),
    node_prefix_rewrites={"zero_jax.": "{target_prefix}.jax."},
    path_rewrites=(
        (".activation", ""),
        (".nn.nn", ".nn"),
        (".initializers.initializers", ".initializers"),
    ),
)

CLI_ALIGNMENT_PRESETS: dict[str, NamespaceAlignmentRule] = {
    "default": NamespaceAlignmentRule(),
    "zero": ZERO_ECOSYSTEM_ALIGNMENT_RULE,
}


def resolve_alignment_preset(preset_name: str | None, target_prefix: str = "") -> NamespaceAlignmentRule:
    """Resolve a `NamespaceAlignmentRule` from an optional CLI preset name or target prefix.

    Args:
        preset_name: Optional CLI preset identifier (`'default'` or `'zero'`).
        target_prefix: Target namespace prefix string.

    Returns:
        Matching `NamespaceAlignmentRule` instance.
    """
    if preset_name and preset_name in CLI_ALIGNMENT_PRESETS:
        return CLI_ALIGNMENT_PRESETS[preset_name]
    if target_prefix.startswith("zero"):
        return ZERO_ECOSYSTEM_ALIGNMENT_RULE
    return CLI_ALIGNMENT_PRESETS["default"]


def _read_target_path(args: argparse.Namespace) -> str:
    """Extract `args.target_path` as a string or empty string when absent.

    Args:
        args: Parsed CLI `Namespace` instance.

    Returns:
        Target path string or `""`.
    """
    raw_tp = get_member_dict(args).get("target_path")
    return str(raw_tp or "")


def _check_assembly_or_mlir_target(
    target_path: str,
    target_ext: str,
    reference_snapshot: typing.Mapping[str, JsonValue],
) -> bool:
    """Run assembly or MLIR snippet compliance verification when the target file extension matches.

    Args:
        target_path: Path to the target source file.
        target_ext: Lowercase file extension of `target_path`.
        reference_snapshot: Loaded reference snapshot dictionary.

    Returns:
        True if `target_ext` was handled as an assembly or MLIR file, False otherwise.
    """
    if target_ext in (".sass", ".s"):
        from ml_ecosystem_snapshots.compliance import (
            check_rdna_assembly_compliance,
            check_sass_assembly_compliance,
        )

        with open(target_path, encoding="utf-8") as f:
            content = f.read()
        target = str(reference_snapshot.get("target", "")).lower()
        if "rdna" in target or "amd" in target:
            res = check_rdna_assembly_compliance(content)
            print(
                f"AMD RDNA snippet compliance: {res.verified_instructions}/{res.total_instructions} instructions valid."
            )
        else:
            res = check_sass_assembly_compliance(content)
            print(
                f"NVIDIA SASS snippet compliance: {res.verified_instructions}/{res.total_instructions} instructions valid."
            )
        if not res.is_compliant:
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        return True

    if target_ext == ".mlir":
        from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance

        with open(target_path, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        print(f"MLIR snippet compliance: {res.verified_ops}/{res.total_ops} operations valid.")
        if not res.is_compliant:
            for err in res.errors:
                print(f"  - {err}")
            sys.exit(1)
        return True

    return False


def _print_module_breakdown(matched: list[str], missing: list[str]) -> None:
    """Print per-module compliance percentage breakdown.

    Args:
        matched: List of matched API paths.
        missing: List of missing API paths.
    """
    print("Breakdown by Module:")
    matched_by_mod: dict[str, list[str]] = collections.defaultdict(list)
    missing_by_mod: dict[str, list[str]] = collections.defaultdict(list)

    for api_path in matched:
        mod = ".".join(api_path.split(".")[:-1])
        matched_by_mod[mod].append(api_path)

    for api_path in missing:
        mod = ".".join(api_path.split(".")[:-1])
        missing_by_mod[mod].append(api_path)

    all_mods = set(list(matched_by_mod.keys()) + list(missing_by_mod.keys()))
    for mod in sorted(all_mods):
        m = len(matched_by_mod[mod])
        total = m + len(missing_by_mod[mod])
        mod_score = (m / total) * 100 if total > 0 else 0
        print(f"  - {mod}: {mod_score:.1f}% ({m}/{total})")


def _format_missing_ref_row(fqn: str, ref: GhostRef) -> str:
    """Format a Markdown table row describing a missing API endpoint.

    Args:
        fqn: Fully qualified API path of the missing symbol.
        ref: Reference `GhostRef` definition for the missing symbol.

    Returns:
        Formatted Markdown table row string.
    """
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
        doc = doc.strip().split("\n\n")[0].replace("\n", " ").replace("|", r"\|")
        if len(doc) > 100:
            doc = doc[:97] + "..."
    else:
        doc = "No docstring available."

    return f"| [ ] | {framework} | {namespace} | {symbol} | {fqn} | `{sig}` | {doc} |"


def _print_missing_apis_table(
    missing: list[str],
    reference_snapshot: typing.Mapping[str, JsonValue],
) -> None:
    """Print the Markdown table of missing APIs from a compliance check.

    Args:
        missing: List of missing fully qualified API paths.
        reference_snapshot: Reference snapshot dictionary containing categories.
    """
    print(f"\nMissing APIs ({len(missing)}):\n")

    ref_map: dict[str, GhostRef] = {}
    cats = ensure_json_dict(reference_snapshot.get("categories", {}), "compliance categories")
    for _cat, raw_items in cats.items():
        for item in ensure_json_list(raw_items, "category items"):
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
        print(_format_missing_ref_row(fqn, ref_match))


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the check command for compliance checking.

    Args:
        args: Parsed command line arguments containing snapshot_json, target_path,
              target_prefix, and reference_prefix.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.compliance import extract_target_refs, score_compliance

    path = resolve_snapshot_path(args.snapshot_json)
    with open(path, encoding="utf-8") as f:
        reference_snapshot = json.load(f)

    target_path = _read_target_path(args)
    target_ext = os.path.splitext(target_path)[1].lower() if target_path else ""

    if _check_assembly_or_mlir_target(target_path, target_ext, reference_snapshot):
        return

    raw_preset = get_member_dict(args).get("alignment_preset")
    preset_name = str(raw_preset) if isinstance(raw_preset, str) else None
    alignment_rule = resolve_alignment_preset(preset_name, str(args.target_prefix))

    print(f"Extracting target APIs from {args.target_path}...")
    target_refs = extract_target_refs(
        args.target_path,
        args.target_prefix,
        args.reference_prefix,
        alignment_rule=alignment_rule,
    )

    print("Scoring compliance...")
    results = score_compliance(reference_snapshot, target_refs)

    score = results.score_percentage
    print("\n--- Compliance Report ---")
    print(f"Overall Compliance: {score}%\n")

    matched = results.matched
    missing = results.missing
    _print_module_breakdown(matched, missing)

    if missing:
        _print_missing_apis_table(missing, reference_snapshot)

    mismatched = results.mismatched
    if mismatched:
        print(f"\nMismatched APIs ({len(mismatched)}):")
        for item in sorted(mismatched, key=lambda x: x.api_path)[:20]:
            print(f"  ~ {item.api_path}\n    Expected: {item.expected}\n    Actual:   {item.actual}")
        if len(mismatched) > 20:
            print(f"  ~ ... and {len(mismatched) - 20} more")


__all__ = [
    "CLI_ALIGNMENT_PRESETS",
    "ZERO_ECOSYSTEM_ALIGNMENT_RULE",
    "cmd_check",
    "resolve_alignment_preset",
]
