"""Snapshot Diffing Module.

Provides functions to diff two snapshots and detect additions, removals,
and signature changes.
"""

from __future__ import annotations

import typing

import pydantic
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope

from ml_ecosystem_snapshots.compliance import ParamSignatureTuple
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


class DiffResult(pydantic.BaseModel):
    """Result of diffing two snapshots."""

    model_config = pydantic.ConfigDict(extra="forbid")

    added: list[str] = pydantic.Field(default_factory=list, description="List of added api_paths.")
    removed: list[str] = pydantic.Field(default_factory=list, description="List of removed api_paths.")
    signature_changed: list[str] = pydantic.Field(default_factory=list, description="List of api_paths with changed signatures.")
    breaking_signature_changed: list[str] = pydantic.Field(
        default_factory=list,
        description="List of api_paths with breaking signature changes.",
    )
    non_breaking_signature_changed: list[str] = pydantic.Field(
        default_factory=list,
        description="List of api_paths with non-breaking signature changes.",
    )
    metadata_changed: dict[str, tuple[str | None, str | None]] = pydantic.Field(
        default_factory=dict,
        description="List of top-level metadata differences: (old_value, new_value).",
    )
    deprecated_capabilities: list[str] = pydantic.Field(
        default_factory=list,
        description="List of deprecated or removed microarchitecture capabilities.",
    )
    overloads_changed: list[str] = pydantic.Field(
        default_factory=list,
        description="List of api_paths with modified, added, or removed overloads.",
    )


def _extract_string_set(meta: dict[str, JsonValue], key: str) -> set[str]:
    """Extract a set of strings from a metadata list field.

    Args:
        meta: Domain metadata dictionary.
        key: Key inside `meta` holding a list of items.

    Returns:
        Set of stringified elements, or an empty set if absent/non-list.
    """
    raw = meta.get(key)
    return {str(x) for x in raw} if isinstance(raw, list) else set()


def _has_breaking_metadata_or_return_change(
    meta1: dict[str, JsonValue] | None,
    meta2: dict[str, JsonValue] | None,
    ret1: JsonValue,
    ret2: JsonValue,
    ov1: list[dict[str, JsonValue]] | None,
    ov2: list[dict[str, JsonValue]] | None,
) -> bool:
    """Check whether domain metadata, return type, or overload counts changed incompatibly.

    Args:
        meta1: Optional old domain metadata dictionary.
        meta2: Optional new domain metadata dictionary.
        ret1: Optional old return type annotation.
        ret2: Optional new return type annotation.
        ov1: Optional old list of overload signatures.
        ov2: Optional new list of overload signatures.

    Returns:
        True if metadata, return type, or overload count indicates a breaking change.
    """
    if meta1 and meta2:
        if _extract_string_set(meta1, "modifiers") - _extract_string_set(meta2, "modifiers"):
            return True
        if _extract_string_set(meta1, "valid_architectures") - _extract_string_set(meta2, "valid_architectures"):
            return True

    if ret1 is not None and ret2 is not None and ret1 != ret2:
        if str(ret1).strip() not in ("", "Any", "None"):
            return True

    return bool(ov1 is not None and ov2 is not None and len(ov1) > len(ov2))


def _has_breaking_positional_order_change(
    p1_list: list[dict[str, JsonValue]],
    p2_list: list[dict[str, JsonValue]],
    p1_map: dict[str, dict[str, JsonValue]],
    p2_map: dict[str, dict[str, JsonValue]],
) -> bool:
    """Check if any old parameter was removed or positional ordering shifted.

    Args:
        p1_list: Old parameter dictionary list.
        p2_list: New parameter dictionary list.
        p1_map: Mapping of old parameter names to definitions.
        p2_map: Mapping of new parameter names to definitions.

    Returns:
        True if a parameter was removed or positional parameters were reordered.
    """
    if any(name not in p2_map for name in p1_map):
        return True

    pos_kinds = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    pos1_names = [str(p.get("name")) for p in p1_list if p.get("kind") in pos_kinds]
    pos2_names = [str(p.get("name")) for p in p2_list if p.get("kind") in pos_kinds]
    common_pos1 = [n for n in pos1_names if n in p2_map]
    common_pos2 = [n for n in pos2_names if n in p1_map]
    if common_pos1 != common_pos2:
        return True

    return any(idx >= len(pos2_names) or pos2_names[idx] != name for idx, name in enumerate(common_pos1))


def _is_param_breaking_change(
    p1: dict[str, JsonValue] | None,
    p2: dict[str, JsonValue],
) -> bool:
    """Check whether a single new or modified parameter introduces a breaking change.

    Args:
        p1: Previous parameter definition if it existed, or None for newly added parameters.
        p2: New parameter definition.

    Returns:
        True if the parameter addition or modification is backwards-incompatible.
    """
    if p1 is None:
        return p2.get("default") is None and p2.get("kind") not in (
            "VAR_POSITIONAL",
            "VAR_KEYWORD",
        )

    k1 = p1.get("kind")
    k2 = p2.get("kind")
    if k1 != k2:
        if k2 in ("POSITIONAL_ONLY", "KEYWORD_ONLY") and k1 == "POSITIONAL_OR_KEYWORD":
            return True
        if ("VAR" in str(k1)) != ("VAR" in str(k2)):
            return True

    if p1.get("default") is not None and p2.get("default") is None:
        return True

    return p1.get("default") != p2.get("default")


def _is_breaking_change(
    p1_list: list[dict[str, JsonValue]],
    p2_list: list[dict[str, JsonValue]],
    meta1: dict[str, JsonValue] | None = None,
    meta2: dict[str, JsonValue] | None = None,
    ret1: JsonValue = None,
    ret2: JsonValue = None,
    ov1: list[dict[str, JsonValue]] | None = None,
    ov2: list[dict[str, JsonValue]] | None = None,
) -> bool:
    """Determine if a signature change is backwards-incompatible.

    Args:
        p1_list: Old parameter list.
        p2_list: New parameter list.
        meta1: Optional old domain metadata (modifiers, architectures).
        meta2: Optional new domain metadata.
        ret1: Optional old return type annotation.
        ret2: Optional new return type annotation.
        ov1: Optional old list of overload signatures.
        ov2: Optional new list of overload signatures.

    Returns:
        True if breaking, False otherwise.
    """
    if _has_breaking_metadata_or_return_change(meta1, meta2, ret1, ret2, ov1, ov2):
        return True

    p1_map = {str(p.get("name")): p for p in p1_list}
    p2_map = {str(p.get("name")): p for p in p2_list}

    if _has_breaking_positional_order_change(p1_list, p2_list, p1_map, p2_map):
        return True

    return any(_is_param_breaking_change(p1_map.get(name), p2) for name, p2 in p2_map.items())


def _flatten_snapshot(
    snap: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]] | ScalarValue),
) -> dict[str, dict[str, JsonValue]]:
    """Flatten a snapshot dictionary or list into a mapping of API paths to items.

    Args:
        snap: The snapshot dictionary, envelope, or list.

    Returns:
        Mapping of API path or mnemonic strings to symbol item dictionaries.
    """
    flattened: dict[str, dict[str, JsonValue]] = {}
    categories_dict: dict[str, list[dict[str, JsonValue]]] = {}
    if isinstance(snap, list):
        categories_dict = {"all": [dict(x) for x in snap if isinstance(x, dict)]}
    elif isinstance(snap, dict):
        raw_cats = snap.get("categories", {})
        assert isinstance(raw_cats, dict)
        for cat_k, cat_v in raw_cats.items():
            assert isinstance(cat_v, list)
            categories_dict[str(cat_k)] = [dict(x) for x in cat_v if isinstance(x, dict)]

    for items in categories_dict.values():
        for item in items:
            key = item.get("api_path") or item.get("name") or item.get("mnemonic")
            if key:
                flattened[str(key)] = item
    return flattened


def _param_sig_tuple(p: dict[str, JsonValue]) -> ParamSignatureTuple:
    """Convert a parameter dictionary to a tuple for signature comparison.

    Args:
        p: The parameter dictionary.

    Returns:
        A 4-tuple representation `(name, kind, default, annotation)`.
    """
    raw_def = p.get("default")
    raw_ann = p.get("annotation")
    return (
        str(p.get("name") or ""),
        str(p.get("kind") or ""),
        str(raw_def) if raw_def is not None else None,
        str(raw_ann) if raw_ann is not None else None,
    )


def _extract_dict_list(raw: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract a list of dictionaries from a raw JSON value.

    Args:
        raw: Raw JSON field value expected to be a list of dicts.

    Returns:
        Filtered list of dictionaries, or empty list when not a list.
    """
    return [dict(x) for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _compare_existing_symbol(
    display_path: str,
    item1: dict[str, JsonValue],
    item2: dict[str, JsonValue],
    signature_changed: list[str],
    breaking_signature_changed: list[str],
    non_breaking_signature_changed: list[str],
    overloads_changed: list[str],
) -> None:
    """Compare two versions of a symbol and record signature or overload changes.

    Args:
        display_path: Formatted symbol path for reporting.
        item1: Older symbol dictionary.
        item2: Newer symbol dictionary.
        signature_changed: Accumulator for changed signatures.
        breaking_signature_changed: Accumulator for breaking signature changes.
        non_breaking_signature_changed: Accumulator for non-breaking signature changes.
        overloads_changed: Accumulator for changed overloads.
    """
    p1 = _extract_dict_list(item1.get("params"))
    p2 = _extract_dict_list(item2.get("params"))
    sig1 = [_param_sig_tuple(p) for p in p1]
    sig2 = [_param_sig_tuple(p) for p in p2]

    raw_m1 = item1.get("domain_metadata")
    raw_m2 = item2.get("domain_metadata")
    meta1 = dict(raw_m1) if isinstance(raw_m1, dict) else None
    meta2 = dict(raw_m2) if isinstance(raw_m2, dict) else None
    ret1 = item1.get("returns_type") or item1.get("returns")
    ret2 = item2.get("returns_type") or item2.get("returns")
    ov1 = _extract_dict_list(item1.get("overloads"))
    ov2 = _extract_dict_list(item2.get("overloads"))

    is_sig_diff = sig1 != sig2
    is_meta_diff = bool(meta1 and meta2 and meta1 != meta2)
    is_ret_diff = ret1 != ret2 and (ret1 is not None or ret2 is not None)
    is_ov_diff = ov1 != ov2

    if is_ov_diff:
        overloads_changed.append(display_path)

    if is_sig_diff or is_meta_diff or is_ret_diff or is_ov_diff:
        signature_changed.append(display_path)
        target_list = breaking_signature_changed if _is_breaking_change(p1, p2, meta1, meta2, ret1, ret2, ov1, ov2) else non_breaking_signature_changed
        target_list.append(display_path)


def _diff_envelope_metadata(
    snap1: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]] | ScalarValue),
    snap2: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]] | ScalarValue),
) -> tuple[dict[str, tuple[str | None, str | None]], list[str]]:
    """Compare top-level metadata fields and environment capability tags between two snapshots.

    Args:
        snap1: Older snapshot data.
        snap2: Newer snapshot data.

    Returns:
        Tuple of `(metadata_changed, deprecated_capabilities)`.
    """
    metadata_changed: dict[str, tuple[str | None, str | None]] = {}
    deprecated_capabilities: list[str] = []

    if isinstance(snap1, dict) and isinstance(snap2, dict):
        for meta_k in (
            "schema_version",
            "version",
            "upstream_commit",
            "source_type",
            "target",
        ):
            v1 = snap1.get(meta_k)
            v2 = snap2.get(meta_k)
            if v1 != v2 and (v1 is not None or v2 is not None):
                metadata_changed[meta_k] = (
                    str(v1) if v1 is not None else None,
                    str(v2) if v2 is not None else None,
                )

        env1 = _extract_string_set(snap1, "environment_tags")
        env2 = _extract_string_set(snap2, "environment_tags")
        dep_envs = env1 - env2
        if dep_envs:
            deprecated_capabilities.extend(sorted(dep_envs))

    return metadata_changed, deprecated_capabilities


def diff_snapshots(
    snap1: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]] | ScalarValue),
    snap2: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]] | ScalarValue),
) -> DiffResult:
    """Diff two snapshots to find changes.

    Args:
        snap1: The older snapshot data.
        snap2: The newer snapshot data.

    Returns:
        A DiffResult containing the differences.
    """
    flat1 = _flatten_snapshot(snap1)
    flat2 = _flatten_snapshot(snap2)

    added: list[str] = []
    removed: list[str] = []
    signature_changed: list[str] = []
    breaking_signature_changed: list[str] = []
    non_breaking_signature_changed: list[str] = []
    overloads_changed: list[str] = []

    for path, item2 in flat2.items():
        display_path = path if item2.get("is_public", True) else f"{path} (non-public)"
        if path not in flat1:
            added.append(display_path)
        else:
            _compare_existing_symbol(
                display_path,
                flat1[path],
                item2,
                signature_changed,
                breaking_signature_changed,
                non_breaking_signature_changed,
                overloads_changed,
            )

    for path, item1 in flat1.items():
        if path not in flat2:
            display_path = path if item1.get("is_public", True) else f"{path} (non-public)"
            removed.append(display_path)

    metadata_changed, deprecated_capabilities = _diff_envelope_metadata(snap1, snap2)

    return DiffResult(
        added=sorted(added),
        removed=sorted(removed),
        signature_changed=sorted(signature_changed),
        breaking_signature_changed=sorted(breaking_signature_changed),
        non_breaking_signature_changed=sorted(non_breaking_signature_changed),
        metadata_changed=metadata_changed,
        deprecated_capabilities=deprecated_capabilities,
        overloads_changed=sorted(overloads_changed),
    )


def _append_changelog_section(lines: list[str], heading: str, items: list[str]) -> None:
    """Append a markdown bullet list section to `lines` when `items` is non-empty.

    Args:
        lines: Mutable list of markdown lines.
        heading: Section heading title (without `## ` prefix).
        items: List of code-formatted item strings to render.
    """
    if items:
        lines.append(f"## {heading}")
        for item in items:
            lines.append(f"- `{item}`")
        lines.append("")


def generate_changelog(diff: DiffResult) -> str:
    """Generate a markdown changelog from a DiffResult.

    Args:
        diff: The DiffResult to generate a changelog for.

    Returns:
        A markdown formatted changelog string.
    """
    lines = ["# Changelog Report", ""]

    has_changes = bool(diff.added or diff.removed or diff.signature_changed or diff.metadata_changed or diff.deprecated_capabilities or diff.overloads_changed)
    if not has_changes:
        lines.append("No changes detected.")
        return "\n".join(lines)

    if diff.metadata_changed:
        lines.append("## Metadata Changes")
        for k, (old_v, new_v) in sorted(diff.metadata_changed.items()):
            lines.append(f"- **{k}**: `{old_v}` -> `{new_v}`")
        lines.append("")

    sections: list[tuple[str, list[str]]] = [
        ("Deprecated Capabilities", diff.deprecated_capabilities),
        ("Added", diff.added),
        ("Removed (Breaking)", diff.removed),
        ("Breaking Signature Changes", diff.breaking_signature_changed),
        ("Non-Breaking Signature Changes", diff.non_breaking_signature_changed),
        ("Overload Changes", diff.overloads_changed),
    ]
    for heading, items in sections:
        _append_changelog_section(lines, heading, items)

    return "\n".join(lines).strip() + "\n"
