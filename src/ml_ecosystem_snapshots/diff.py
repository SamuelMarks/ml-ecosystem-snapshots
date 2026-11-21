"""Pure snapshot diffing module with closed signature-change ADTs and immutable results."""

from __future__ import annotations

import enum
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope

from ml_ecosystem_snapshots.compliance.models import ParamSignatureTuple
from ml_ecosystem_snapshots.models import (
    JsonValue,
    ScalarValue,
    WireSerializableProtocol,
    json_dict_or_empty,
    json_tuple_or_empty,
)

_NL: typing.Final[str] = chr(10)


class SignatureChangeKind(str, enum.Enum):
    """Closed classification of symbol signature evolution between two snapshots."""

    UNCHANGED = "unchanged"
    NON_BREAKING = "non_breaking"
    BREAKING = "breaking"


class DiffResult(pydantic.BaseModel):
    """Deeply immutable result of diffing two snapshots."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    added: tuple[str, ...] = pydantic.Field(default=(), description="Tuple of added api_paths.")
    removed: tuple[str, ...] = pydantic.Field(default=(), description="Tuple of removed api_paths.")
    signature_changed: tuple[str, ...] = pydantic.Field(
        default=(), description="Tuple of api_paths with changed signatures."
    )
    breaking_signature_changed: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Tuple of api_paths with breaking signature changes.",
    )
    non_breaking_signature_changed: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Tuple of api_paths with non-breaking signature changes.",
    )
    metadata_changed: dict[str, tuple[str | None, str | None]] = pydantic.Field(
        default_factory=dict,
        description="Mapping of top-level metadata differences: (old_value, new_value).",
    )
    deprecated_capabilities: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Tuple of deprecated or removed microarchitecture capabilities.",
    )
    overloads_changed: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Tuple of api_paths with modified, added, or removed overloads.",
    )


DiffableSnapshotInput: typing_extensions.TypeAlias = (
    "SnapshotEnvelope"
    " | WireSerializableProtocol"
    " | typing.Mapping[str, JsonValue]"
    " | typing.Sequence[typing.Mapping[str, JsonValue]]"
    " | ScalarValue"
)


def _extract_string_set(meta: typing.Mapping[str, JsonValue], key: str) -> frozenset[str]:
    """Extract an immutable frozenset of strings from a metadata sequence field.

    Args:
        meta: Domain metadata dictionary.
        key: Key inside `meta` holding a list or tuple of items.

    Returns:
        Frozenset of stringified elements, or empty frozenset if absent/non-sequence.
    """
    raw = meta.get(key)
    return frozenset(str(x) for x in raw) if isinstance(raw, (list, tuple)) else frozenset()


def _has_breaking_metadata_or_return_change(
    meta1: dict[str, JsonValue] | None,
    meta2: dict[str, JsonValue] | None,
    ret1: JsonValue,
    ret2: JsonValue,
    ov1: typing.Sequence[dict[str, JsonValue]] | None,
    ov2: typing.Sequence[dict[str, JsonValue]] | None,
) -> bool:
    """Check whether domain metadata, return type, or overload counts changed incompatibly.

    Args:
        meta1: Optional old domain metadata dictionary.
        meta2: Optional new domain metadata dictionary.
        ret1: Optional old return type annotation.
        ret2: Optional new return type annotation.
        ov1: Optional old sequence of overload signatures.
        ov2: Optional new sequence of overload signatures.

    Returns:
        True if metadata, return type, or overload count indicates a breaking change.
    """
    if meta1 and meta2:
        if _extract_string_set(meta1, "modifiers") - _extract_string_set(meta2, "modifiers"):
            return True
        if _extract_string_set(meta1, "valid_architectures") - _extract_string_set(meta2, "valid_architectures"):
            return True

    if ret1 is not None and ret2 is not None and ret1 != ret2:
        if str(ret1).strip() not in ("", "Any", "untyped", "Incomplete", "None"):
            return True

    return bool(ov1 is not None and ov2 is not None and len(ov1) > len(ov2))


def _has_breaking_positional_order_change(
    p1_list: typing.Sequence[dict[str, JsonValue]],
    p2_list: typing.Sequence[dict[str, JsonValue]],
    p1_map: typing.Mapping[str, dict[str, JsonValue]],
    p2_map: typing.Mapping[str, dict[str, JsonValue]],
) -> bool:
    """Check if any old parameter was removed or positional ordering shifted.

    Args:
        p1_list: Old parameter dictionary sequence.
        p2_list: New parameter dictionary sequence.
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
    p1_list: typing.Sequence[dict[str, JsonValue]],
    p2_list: typing.Sequence[dict[str, JsonValue]],
    meta1: dict[str, JsonValue] | None = None,
    meta2: dict[str, JsonValue] | None = None,
    ret1: JsonValue = None,
    ret2: JsonValue = None,
    ov1: typing.Sequence[dict[str, JsonValue]] | None = None,
    ov2: typing.Sequence[dict[str, JsonValue]] | None = None,
) -> bool:
    """Determine if a signature change is backwards-incompatible.

    Args:
        p1_list: Old parameter sequence.
        p2_list: New parameter sequence.
        meta1: Optional old domain metadata (modifiers, architectures).
        meta2: Optional new domain metadata.
        ret1: Optional old return type annotation.
        ret2: Optional new return type annotation.
        ov1: Optional old sequence of overload signatures.
        ov2: Optional new sequence of overload signatures.

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


def _project_diff_mapping(snap: DiffableSnapshotInput) -> dict[str, JsonValue] | None:
    """Project a `SnapshotEnvelope`, `WireSerializableProtocol`, or `Mapping` into a plain dict.

    Args:
        snap: Input snapshot value.

    Returns:
        Dictionary representation if mapping-like, otherwise `None`.
    """
    if isinstance(snap, SnapshotEnvelope):
        dumped: dict[str, JsonValue] = snap.model_dump()
        return dumped
    if isinstance(snap, WireSerializableProtocol):
        return snap.to_wire_dict()
    if isinstance(snap, typing.Mapping):
        return dict(snap)
    return None


def _flatten_snapshot(snap: DiffableSnapshotInput) -> dict[str, dict[str, JsonValue]]:
    """Flatten a snapshot dictionary, envelope, `ProvenSnapshot`, or sequence into `api_path -> item`.

    Args:
        snap: Snapshot envelope, `ProvenSnapshot`, dictionary, or sequence of items.

    Returns:
        Mapping of API path or mnemonic strings to symbol item dictionaries.
    """
    flattened: dict[str, dict[str, JsonValue]] = {}
    categories_dict: dict[str, list[dict[str, JsonValue]]] = {}

    projected = _project_diff_mapping(snap)
    if projected is not None:
        raw_cats = json_dict_or_empty(projected.get("categories", {}))
        for cat_k, cat_v in raw_cats.items():
            categories_dict[str(cat_k)] = [dict(x) for x in json_tuple_or_empty(cat_v) if isinstance(x, dict)]
    elif isinstance(snap, (list, tuple)):
        categories_dict = {"all": [dict(x) for x in snap if isinstance(x, typing.Mapping)]}

    for items in categories_dict.values():
        for item in items:
            key = item.get("api_path") or item.get("name") or item.get("mnemonic")
            if key:
                flattened[str(key)] = item
    return flattened


def _param_sig_tuple(p: typing.Mapping[str, JsonValue]) -> ParamSignatureTuple:
    """Convert a parameter dictionary to a tuple for signature comparison.

    Args:
        p: Parameter dictionary mapping.

    Returns:
        4-tuple representation `(name, kind, default, annotation)`.
    """
    raw_def = p.get("default")
    raw_ann = p.get("annotation")
    return (
        str(p.get("name") or ""),
        str(p.get("kind") or ""),
        str(raw_def) if raw_def is not None else None,
        str(raw_ann) if raw_ann is not None else None,
    )


def _extract_dict_tuple(raw: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    """Extract an immutable tuple of dictionaries from a raw JSON value.

    Args:
        raw: Raw JSON field value expected to be a sequence of dicts.

    Returns:
        Filtered tuple of dictionaries, or empty tuple when not a sequence.
    """
    return tuple(dict(x) for x in raw if isinstance(x, dict)) if isinstance(raw, (list, tuple)) else ()


def _classify_symbol_change(
    item1: typing.Mapping[str, JsonValue],
    item2: typing.Mapping[str, JsonValue],
) -> tuple[SignatureChangeKind, bool]:
    """Purely classify signature evolution and overload delta between two symbol versions.

    Args:
        item1: Older symbol dictionary.
        item2: Newer symbol dictionary.

    Returns:
        Tuple of `(SignatureChangeKind, has_overload_change)`.
    """
    p1 = _extract_dict_tuple(item1.get("params"))
    p2 = _extract_dict_tuple(item2.get("params"))
    sig1 = tuple(_param_sig_tuple(p) for p in p1)
    sig2 = tuple(_param_sig_tuple(p) for p in p2)

    raw_m1 = item1.get("domain_metadata")
    raw_m2 = item2.get("domain_metadata")
    meta1 = dict(raw_m1) if isinstance(raw_m1, dict) else None
    meta2 = dict(raw_m2) if isinstance(raw_m2, dict) else None
    ret1 = item1.get("returns_type") or item1.get("returns")
    ret2 = item2.get("returns_type") or item2.get("returns")
    ov1 = _extract_dict_tuple(item1.get("overloads"))
    ov2 = _extract_dict_tuple(item2.get("overloads"))

    is_sig_diff = sig1 != sig2
    is_meta_diff = bool(meta1 and meta2 and meta1 != meta2)
    is_ret_diff = ret1 != ret2 and (ret1 is not None or ret2 is not None)
    is_ov_diff = ov1 != ov2

    if not (is_sig_diff or is_meta_diff or is_ret_diff or is_ov_diff):
        return SignatureChangeKind.UNCHANGED, False

    if _is_breaking_change(p1, p2, meta1, meta2, ret1, ret2, ov1, ov2):
        return SignatureChangeKind.BREAKING, is_ov_diff
    return SignatureChangeKind.NON_BREAKING, is_ov_diff


def _diff_envelope_metadata(
    snap1: DiffableSnapshotInput,
    snap2: DiffableSnapshotInput,
) -> tuple[dict[str, tuple[str | None, str | None]], tuple[str, ...]]:
    """Compare top-level metadata fields and environment capability tags between two snapshots.

    Args:
        snap1: Older snapshot data.
        snap2: Newer snapshot data.

    Returns:
        Tuple of `(metadata_changed, deprecated_capabilities)`.
    """
    metadata_changed: dict[str, tuple[str | None, str | None]] = {}
    deprecated_capabilities: tuple[str, ...] = ()

    dict1 = _project_diff_mapping(snap1)
    dict2 = _project_diff_mapping(snap2)
    if dict1 is not None and dict2 is not None:
        for meta_k in (
            "schema_version",
            "version",
            "upstream_commit",
            "source_type",
            "target",
        ):
            v1 = dict1.get(meta_k)
            v2 = dict2.get(meta_k)
            if v1 != v2 and (v1 is not None or v2 is not None):
                metadata_changed[meta_k] = (
                    str(v1) if v1 is not None else None,
                    str(v2) if v2 is not None else None,
                )

        env1 = _extract_string_set(dict1, "environment_tags")
        env2 = _extract_string_set(dict2, "environment_tags")
        dep_envs = env1 - env2
        if dep_envs:
            deprecated_capabilities = tuple(sorted(dep_envs))

    return metadata_changed, deprecated_capabilities


def diff_snapshots(
    snap1: DiffableSnapshotInput,
    snap2: DiffableSnapshotInput,
) -> DiffResult:
    """Purely diff two snapshots into a frozen `DiffResult`.

    Args:
        snap1: Older snapshot envelope, `ProvenSnapshot`, dict, or list.
        snap2: Newer snapshot envelope, `ProvenSnapshot`, dict, or list.

    Returns:
        Frozen `DiffResult` containing sorted immutable change tuples.
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
            continue

        change_kind, has_ov_diff = _classify_symbol_change(flat1[path], item2)
        if has_ov_diff:
            overloads_changed.append(display_path)

        if change_kind is SignatureChangeKind.UNCHANGED:
            pass
        elif change_kind is SignatureChangeKind.BREAKING:
            signature_changed.append(display_path)
            breaking_signature_changed.append(display_path)
        elif change_kind is SignatureChangeKind.NON_BREAKING:
            signature_changed.append(display_path)
            non_breaking_signature_changed.append(display_path)
        else:
            typing_extensions.assert_never(change_kind)

    for path, item1 in flat1.items():
        if path not in flat2:
            display_path = path if item1.get("is_public", True) else f"{path} (non-public)"
            removed.append(display_path)

    metadata_changed, deprecated_capabilities = _diff_envelope_metadata(snap1, snap2)

    return DiffResult(
        added=tuple(sorted(added)),
        removed=tuple(sorted(removed)),
        signature_changed=tuple(sorted(signature_changed)),
        breaking_signature_changed=tuple(sorted(breaking_signature_changed)),
        non_breaking_signature_changed=tuple(sorted(non_breaking_signature_changed)),
        metadata_changed=metadata_changed,
        deprecated_capabilities=deprecated_capabilities,
        overloads_changed=tuple(sorted(overloads_changed)),
    )


def _render_changelog_section(heading: str, items: typing.Sequence[str]) -> tuple[str, ...]:
    """Purely render a markdown bullet list section when `items` is non-empty.

    Args:
        heading: Section heading title (without `## ` prefix).
        items: Sequence of code-formatted item strings to render.

    Returns:
        Immutable tuple of markdown lines (empty when `items` is empty).
    """
    if not items:
        return ()
    return (f"## {heading}", *(f"- `{item}`" for item in items), "")


def generate_changelog(diff: DiffResult) -> str:
    """Purely render a markdown changelog string from a `DiffResult`.

    Args:
        diff: Frozen `DiffResult` instance to format.

    Returns:
        Markdown-formatted changelog report.
    """
    header_lines: tuple[str, ...] = ("# Changelog Report", "")

    has_changes = bool(
        diff.added
        or diff.removed
        or diff.signature_changed
        or diff.metadata_changed
        or diff.deprecated_capabilities
        or diff.overloads_changed
    )
    if not has_changes:
        return _NL.join((*header_lines, "No changes detected."))

    meta_lines: tuple[str, ...] = (
        (
            "## Metadata Changes",
            *(f"- **{k}**: `{old_v}` -> `{new_v}`" for k, (old_v, new_v) in sorted(diff.metadata_changed.items())),
            "",
        )
        if diff.metadata_changed
        else ()
    )

    sections: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Deprecated Capabilities", diff.deprecated_capabilities),
        ("Added", diff.added),
        ("Removed (Breaking)", diff.removed),
        ("Breaking Signature Changes", diff.breaking_signature_changed),
        ("Non-Breaking Signature Changes", diff.non_breaking_signature_changed),
        ("Overload Changes", diff.overloads_changed),
    )
    section_lines = tuple(line for heading, items in sections for line in _render_changelog_section(heading, items))

    return _NL.join((*header_lines, *meta_lines, *section_lines)).strip() + _NL


__all__ = [
    "DiffResult",
    "DiffableSnapshotInput",
    "SignatureChangeKind",
    "diff_snapshots",
    "generate_changelog",
]
