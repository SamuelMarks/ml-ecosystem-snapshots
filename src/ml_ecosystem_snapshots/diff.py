"""Snapshot Diffing Module.

Provides functions to diff two snapshots and detect additions, removals,
and signature changes.
"""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope
import pydantic

from ml_ecosystem_snapshots.compliance import ParamSignatureTuple
from ml_ecosystem_snapshots.models import JsonValue


class DiffResult(pydantic.BaseModel):
    """Result of diffing two snapshots."""

    model_config = pydantic.ConfigDict(extra="forbid")

    added: list[str] = pydantic.Field(
        default_factory=list, description="List of added api_paths."
    )
    removed: list[str] = pydantic.Field(
        default_factory=list, description="List of removed api_paths."
    )
    signature_changed: list[str] = pydantic.Field(
        default_factory=list, description="List of api_paths with changed signatures."
    )
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
    if meta1 and meta2:
        raw_mods1 = meta1.get("modifiers")
        raw_mods2 = meta2.get("modifiers")
        m1_mods = {str(x) for x in raw_mods1} if isinstance(raw_mods1, list) else set()
        m2_mods = {str(x) for x in raw_mods2} if isinstance(raw_mods2, list) else set()
        if m1_mods - m2_mods:
            return True

        raw_archs1 = meta1.get("valid_architectures")
        raw_archs2 = meta2.get("valid_architectures")
        m1_archs = (
            {str(x) for x in raw_archs1} if isinstance(raw_archs1, list) else set()
        )
        m2_archs = (
            {str(x) for x in raw_archs2} if isinstance(raw_archs2, list) else set()
        )
        if m1_archs - m2_archs:
            return True

    if ret1 is not None and ret2 is not None and ret1 != ret2:
        clean_ret1 = str(ret1).strip()
        if clean_ret1 not in ("", "Any", "None"):
            return True

    if ov1 is not None and ov2 is not None and len(ov1) > len(ov2):
        return True

    p1_map = {str(p.get("name")): p for p in p1_list}
    p2_map = {str(p.get("name")): p for p in p2_list}

    for name in p1_map:
        if name not in p2_map:
            return True

    pos1_names = [
        str(p.get("name"))
        for p in p1_list
        if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    ]
    pos2_names = [
        str(p.get("name"))
        for p in p2_list
        if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    ]
    common_pos1 = [n for n in pos1_names if n in p2_map]
    common_pos2 = [n for n in pos2_names if n in p1_map]
    if common_pos1 != common_pos2:
        return True

    for idx, name in enumerate(common_pos1):
        if idx >= len(pos2_names) or pos2_names[idx] != name:
            return True

    for name, p2 in p2_map.items():
        if name not in p1_map:
            if p2.get("default") is None and p2.get("kind") not in (
                "VAR_POSITIONAL",
                "VAR_KEYWORD",
            ):
                return True
        else:
            p1 = p1_map[name]
            k1 = p1.get("kind")
            k2 = p2.get("kind")

            if k1 != k2:
                if (
                    k2 in ("POSITIONAL_ONLY", "KEYWORD_ONLY")
                    and k1 == "POSITIONAL_OR_KEYWORD"
                ):
                    return True
                if ("VAR" in str(k1) and "VAR" not in str(k2)) or (
                    "VAR" not in str(k1) and "VAR" in str(k2)
                ):
                    return True

            if p1.get("default") is not None and p2.get("default") is None:
                return True

            if p1.get("default") != p2.get("default"):
                return True

    return False


def diff_snapshots(
    snap1: (
        SnapshotEnvelope
        | typing.Mapping[str, JsonValue]
        | typing.Sequence[typing.Mapping[str, JsonValue]]
        | object
    ),
    snap2: (
        SnapshotEnvelope
        | typing.Mapping[str, JsonValue]
        | typing.Sequence[typing.Mapping[str, JsonValue]]
        | object
    ),
) -> DiffResult:
    """Diff two snapshots to find changes.

    Args:
        snap1: The older snapshot data.
        snap2: The newer snapshot data.

    Returns:
        A DiffResult containing the differences.
    """

    def _flatten(snap: object) -> dict[str, dict[str, JsonValue]]:
        """Flatten a snapshot dictionary or list into a mapping of API paths to items.

        Args:
            snap: The snapshot dictionary or list.

        Returns:
            The flattened dict.
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
                categories_dict[str(cat_k)] = [
                    dict(x) for x in cat_v if isinstance(x, dict)
                ]

        for _cat, items in categories_dict.items():
            for item in items:
                key = item.get("api_path") or item.get("name") or item.get("mnemonic")
                if key:
                    flattened[str(key)] = item
        return flattened

    flat1 = _flatten(snap1)
    flat2 = _flatten(snap2)

    added: list[str] = []
    removed: list[str] = []
    signature_changed: list[str] = []
    breaking_signature_changed: list[str] = []
    non_breaking_signature_changed: list[str] = []
    overloads_changed: list[str] = []

    for path, item2 in flat2.items():
        is_public = item2.get("is_public", True)
        display_path = path if is_public else f"{path} (non-public)"
        if path not in flat1:
            added.append(display_path)
        else:
            item1 = flat1[path]
            raw_p1 = item1.get("params")
            raw_p2 = item2.get("params")
            p1: list[dict[str, JsonValue]] = (
                [dict(x) for x in raw_p1 if isinstance(x, dict)]
                if isinstance(raw_p1, list)
                else []
            )
            p2: list[dict[str, JsonValue]] = (
                [dict(x) for x in raw_p2 if isinstance(x, dict)]
                if isinstance(raw_p2, list)
                else []
            )

            def sig_tuple(p: dict[str, JsonValue]) -> ParamSignatureTuple:
                """Convert a parameter dictionary to a tuple for easy comparison.

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

            sig1 = [sig_tuple(p) for p in p1]
            sig2 = [sig_tuple(p) for p in p2]
            raw_m1 = item1.get("domain_metadata")
            raw_m2 = item2.get("domain_metadata")
            meta1 = dict(raw_m1) if isinstance(raw_m1, dict) else None
            meta2 = dict(raw_m2) if isinstance(raw_m2, dict) else None
            ret1 = item1.get("returns_type") or item1.get("returns")
            ret2 = item2.get("returns_type") or item2.get("returns")
            raw_ov1 = item1.get("overloads")
            raw_ov2 = item2.get("overloads")
            ov1: list[dict[str, JsonValue]] = (
                [dict(x) for x in raw_ov1 if isinstance(x, dict)]
                if isinstance(raw_ov1, list)
                else []
            )
            ov2: list[dict[str, JsonValue]] = (
                [dict(x) for x in raw_ov2 if isinstance(x, dict)]
                if isinstance(raw_ov2, list)
                else []
            )

            is_sig_diff = sig1 != sig2
            is_meta_diff = bool(meta1 and meta2 and meta1 != meta2)
            is_ret_diff = ret1 != ret2 and (ret1 is not None or ret2 is not None)
            is_ov_diff = ov1 != ov2

            if is_ov_diff:
                overloads_changed.append(display_path)

            if is_sig_diff or is_meta_diff or is_ret_diff or is_ov_diff:
                signature_changed.append(display_path)
                if _is_breaking_change(p1, p2, meta1, meta2, ret1, ret2, ov1, ov2):
                    breaking_signature_changed.append(display_path)
                else:
                    non_breaking_signature_changed.append(display_path)

    for path, item1 in flat1.items():
        if path not in flat2:
            is_public = item1.get("is_public", True)
            display_path = path if is_public else f"{path} (non-public)"
            removed.append(display_path)

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

        raw_env1 = snap1.get("environment_tags")
        raw_env2 = snap2.get("environment_tags")
        env1 = {str(x) for x in raw_env1} if isinstance(raw_env1, list) else set()
        env2 = {str(x) for x in raw_env2} if isinstance(raw_env2, list) else set()
        dep_envs = env1 - env2
        if dep_envs:
            deprecated_capabilities.extend(sorted(dep_envs))

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


def generate_changelog(diff: DiffResult) -> str:
    """Generate a markdown changelog from a DiffResult.

    Args:
        diff: The DiffResult to generate a changelog for.

    Returns:
        A markdown formatted changelog string.
    """
    lines = ["# Changelog Report", ""]

    if (
        not diff.added
        and not diff.removed
        and not diff.signature_changed
        and not diff.metadata_changed
        and not diff.deprecated_capabilities
        and not diff.overloads_changed
    ):
        lines.append("No changes detected.")
        return "\n".join(lines)

    if diff.metadata_changed:
        lines.append("## Metadata Changes")
        for k, (old_v, new_v) in sorted(diff.metadata_changed.items()):
            lines.append(f"- **{k}**: `{old_v}` -> `{new_v}`")
        lines.append("")

    if diff.deprecated_capabilities:
        lines.append("## Deprecated Capabilities")
        for cap in diff.deprecated_capabilities:
            lines.append(f"- `{cap}`")
        lines.append("")

    if diff.added:
        lines.append("## Added")
        for path in diff.added:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.removed:
        lines.append("## Removed (Breaking)")
        for path in diff.removed:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.breaking_signature_changed:
        lines.append("## Breaking Signature Changes")
        for path in diff.breaking_signature_changed:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.non_breaking_signature_changed:
        lines.append("## Non-Breaking Signature Changes")
        for path in diff.non_breaking_signature_changed:
            lines.append(f"- `{path}`")
        lines.append("")

    if diff.overloads_changed:
        lines.append("## Overload Changes")
        for path in diff.overloads_changed:
            lines.append(f"- `{path}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
