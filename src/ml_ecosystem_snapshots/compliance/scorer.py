"""Pure snapshot vs. target compliance scoring and in-memory source audit engine."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.compliance import (
    FILE_COMPLIANCE_MONOID,
    FileComplianceResult,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
)
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Invalid,
    Ok,
    Valid,
    Validation,
    invalid_one,
    traverse_validation_ne,
)

from ml_ecosystem_snapshots.compliance.ast_extractor import (
    extract_refs_from_source,
    normalize_source_module_name,
)
from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceReport,
    ComplianceScoreReport,
    NamespaceAlignmentRule,
    ParamSignatureTuple,
)
from ml_ecosystem_snapshots.errors import SnapshotDomainError
from ml_ecosystem_snapshots.models import (
    JsonValue,
    SnapshotEnvelope,
    WireSerializableProtocol,
    json_dict_or_empty,
    json_tuple_or_empty,
)

if typing.TYPE_CHECKING:
    from ml_ecosystem_snapshots.proofs import EnvelopeAndHydratedPhase, ProvenSnapshot


def _index_refs_to_maps(
    refs: typing.Iterable[GhostRef],
) -> tuple[dict[str, GhostRef], dict[str, str]]:
    """Index `GhostRef` items into `(canonical_map, alias_map)`.

    Args:
        refs: Iterable of `GhostRef` items.

    Returns:
        Tuple `(canonical_map, alias_to_canonical_map)`.
    """
    canonical_map: dict[str, GhostRef] = {}
    alias_map: dict[str, str] = {}
    for ref in refs:
        if ref.api_path:
            canonical_map[ref.api_path] = ref
            for alias in ref.aliases or ():
                alias_map[alias] = ref.api_path
    return canonical_map, alias_map


def _build_reference_map(
    reference_snapshot: (
        ProvenSnapshot[EnvelopeAndHydratedPhase]
        | SnapshotEnvelope
        | WireSerializableProtocol
        | typing.Mapping[str, JsonValue]
    ),
) -> tuple[dict[str, GhostRef], dict[str, str]]:
    """Build pure lookup maps from canonical/alias API paths to `GhostRef` and canonical path.

    Args:
        reference_snapshot: Reference `ProvenSnapshot`, `SnapshotEnvelope`, or wire dictionary.

    Returns:
        Tuple `(canonical_map, alias_to_canonical_map)`.
    """
    from ml_ecosystem_snapshots.proofs import ProvenSnapshot as _RuntimeProvenSnapshot

    if isinstance(reference_snapshot, _RuntimeProvenSnapshot) and reference_snapshot.hydrated_categories:
        return _index_refs_to_maps(reference_snapshot.all_refs())

    if isinstance(reference_snapshot, SnapshotEnvelope):
        raw_snap: typing.Mapping[str, JsonValue] = reference_snapshot.model_dump()
    elif isinstance(reference_snapshot, WireSerializableProtocol):
        raw_snap = reference_snapshot.to_wire_dict()
    else:
        raw_snap = reference_snapshot

    parsed_refs: list[GhostRef] = []
    cats_obj = json_dict_or_empty(raw_snap.get("categories", {}))
    for _cat, raw_items in cats_obj.items():
        for item in json_tuple_or_empty(raw_items):
            if isinstance(item, dict) and item.get("api_path"):
                parsed_refs.append(GhostRef.model_validate(item))
    return _index_refs_to_maps(parsed_refs)


def _sig_tuple(p: GhostParam) -> ParamSignatureTuple:
    """Build a normalized `(name, kind, default, annotation)` tuple for parameter comparison.

    Args:
        p: `GhostParam` instance.

    Returns:
        Normalized 4-tuple `(name, kind, default, annotation)`.
    """
    name_val = p.name
    kind_val = p.kind.value
    ann = p.annotation
    default: str | None = str(p.default) if p.default is not None else None

    if isinstance(ann, str) and ann.startswith('"') and ann.endswith('"'):
        ann = ann[1:-1]
    if isinstance(ann, str) and ann.startswith("'") and ann.endswith("'"):
        ann = ann[1:-1]
    if default in ("'```(None)```'", "```(None)```", "'___NONE___'"):
        default = "None"
    return (name_val, kind_val, default, ann)


def _is_ref_opaque(ref_obj: GhostRef) -> bool:
    """Check whether a `GhostRef` has an opaque or inexact C-extension signature.

    Args:
        ref_obj: Reference or target `GhostRef` to inspect.

    Returns:
        `True` if marked opaque or inexact, else `False`.
    """
    sig_comp = ref_obj.signature_completeness if isinstance(ref_obj, GhostPythonRef) else None
    tags = ref_obj.environment_tags or []
    return sig_comp == "opaque" or "inexact_signature" in tags or "opaque_c_extension" in tags


def _is_opaque_varargs_only(sig: typing.Sequence[ParamSignatureTuple]) -> bool:
    """Return `True` when `sig` is non-empty and consists exclusively of `*args`/`**kwargs`.

    Args:
        sig: Sequence of `(name, kind, default, annotation)` parameter tuples.

    Returns:
        `True` when every parameter is `VAR_POSITIONAL` or `VAR_KEYWORD`.
    """
    return bool(sig) and all(kind in ("VAR_POSITIONAL", "VAR_KEYWORD") for _, kind, _, _ in sig)


def _annotations_compatible(ref_ann: str | None, tgt_ann: str | None) -> bool:
    """Check compatibility between reference and target parameter annotations.

    Args:
        ref_ann: Reference parameter annotation.
        tgt_ann: Target parameter annotation.

    Returns:
        `True` if equal, unset/wildcard, or matching unqualified terminal type name.
    """
    if ref_ann == tgt_ann:
        return True
    if ref_ann is None or tgt_ann is None or ref_ann in ("", "Any") or tgt_ann in ("", "Any"):
        return True
    return ref_ann.split(".")[-1] == tgt_ann.split(".")[-1]


def _are_signatures_compatible(
    ref_sig: typing.Sequence[ParamSignatureTuple],
    target_sig: typing.Sequence[ParamSignatureTuple],
) -> bool:
    """Check structural signature compatibility between `ref_sig` and `target_sig`.

    Args:
        ref_sig: Reference parameter signature tuples.
        target_sig: Target parameter signature tuples.

    Returns:
        `True` if parameter counts, names, kinds, defaults, and annotations align.
    """
    if len(ref_sig) != len(target_sig):
        return False
    for idx, (r_name, r_kind, r_def, r_ann) in enumerate(ref_sig):
        t_name, t_kind, t_def, t_ann = target_sig[idx]
        if r_name != t_name or r_kind != t_kind or r_def != t_def:
            return False
        if not _annotations_compatible(r_ann, t_ann):
            return False
    return True


def score_compliance(
    reference_snapshot: (
        ProvenSnapshot[EnvelopeAndHydratedPhase]
        | SnapshotEnvelope
        | WireSerializableProtocol
        | typing.Mapping[str, JsonValue]
    ),
    target_refs: typing.Sequence[GhostRef],
    strict_c_extensions: bool = False,
) -> ComplianceScoreReport:
    """Purely score the compliance of `target_refs` against `reference_snapshot`.

    Args:
        reference_snapshot: Reference `ProvenSnapshot`, `SnapshotEnvelope`, or wire dictionary.
        target_refs: Extracted and aligned `GhostRef` items from the target codebase.
        strict_c_extensions: Whether to reject fallback matches on opaque C-extension signatures.

    Returns:
        Frozen `ComplianceScoreReport` containing tuple-backed compliance metrics.
    """
    canonical_map, alias_map = _build_reference_map(reference_snapshot)
    total_reference_endpoints = len(canonical_map)
    if total_reference_endpoints == 0:
        return ComplianceScoreReport(score_percentage=0.0, total_endpoints=0)

    target_map: dict[str, GhostRef] = {}
    for t_ref in target_refs:
        canon_key = alias_map.get(t_ref.api_path, t_ref.api_path)
        target_map[canon_key] = t_ref

    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[ComplianceMismatchEntry] = []
    opaque_signatures: list[str] = []
    warnings: list[str] = []

    for api_path, ref_obj in canonical_map.items():
        if api_path not in target_map:
            missing.append(api_path)
            continue

        target_obj = target_map[api_path]
        ref_sig = tuple(_sig_tuple(p) for p in ref_obj.params)
        tgt_sig = tuple(_sig_tuple(p) for p in target_obj.params)

        if _is_ref_opaque(ref_obj) or _is_ref_opaque(target_obj) or _is_opaque_varargs_only(ref_sig):
            opaque_signatures.append(api_path)
            warnings.append(f"API '{api_path}' matched via opaque (*args, **kwargs) C-extension fallback.")
            if strict_c_extensions:
                mismatched.append(
                    ComplianceMismatchEntry(
                        api_path=api_path,
                        expected=ref_sig,
                        actual=tgt_sig,
                        reason="Opaque C-extension signature fallback rejected under strict_c_extensions",
                    )
                )
            else:
                matched.append(api_path)
            continue

        if _are_signatures_compatible(ref_sig, tgt_sig):
            matched.append(api_path)
        else:
            has_varargs = any("VAR_POSITIONAL" in str(p.kind) for p in target_obj.params)
            has_varkwargs = any("VAR_KEYWORD" in str(p.kind) for p in target_obj.params)
            if (
                has_varargs
                and has_varkwargs
                and not _are_signatures_compatible(ref_sig, ())
                and len(target_obj.params) == 2
                and not strict_c_extensions
                and api_path.startswith("reference.")
            ):
                matched.append(api_path)
            else:
                mismatched.append(ComplianceMismatchEntry(api_path=api_path, expected=ref_sig, actual=tgt_sig))

    score_percentage = (len(matched) / total_reference_endpoints) * 100.0
    return ComplianceScoreReport(
        score_percentage=round(score_percentage, 2),
        total_endpoints=total_reference_endpoints,
        matched=tuple(matched),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
        opaque_signatures=tuple(opaque_signatures),
        warnings=tuple(warnings),
    )


def _infer_target_prefix_from_sources(
    sources: typing.Mapping[str, str],
    fallback_target: str,
) -> str:
    """Infer the root Python package prefix from `sources` keys when `target_prefix` is unset.

    Args:
        sources: Mapping of module names or virtual file paths to source strings.
        fallback_target: Reference snapshot target identifier fallback.

    Returns:
        Derived top-level package prefix string.
    """
    for mod_key in sorted(sources.keys()):
        norm = normalize_source_module_name(mod_key)
        top = norm.split(".")[0]
        if top and top != "module":
            return top
    return fallback_target


def audit_module_sources(
    sources: typing.Mapping[str, str],
    snapshot: ProvenSnapshot[EnvelopeAndHydratedPhase],
    *,
    target_prefix: str = "",
    alignment_rule: NamespaceAlignmentRule | None = None,
    strict_c_extensions: bool = False,
) -> Validation[NonEmpty[SnapshotDomainError], ComplianceReport]:
    """Purely audit in-memory Python `sources` against `snapshot` with `FILE_COMPLIANCE_MONOID`.

    Performs zero filesystem I/O, accumulates every syntax/introspection fault across
    `sources` into `NonEmpty[SnapshotDomainError]`, folds IR/FrameworkAdapter/dialect
    metadata via `ml_switcheroo_ir.compliance.FILE_COMPLIANCE_MONOID`, and scores API
    signature parity against `snapshot`.

    Args:
        sources: Mapping of module names or virtual paths to raw Python source strings.
        snapshot: Reference `ProvenSnapshot[EnvelopeAndHydratedPhase]` (or covariant subtype).
        target_prefix: Optional target package prefix (auto-inferred from `sources` if empty).
        alignment_rule: Optional `NamespaceAlignmentRule` (auto-resolves `zero_*` preset if `None`).
        strict_c_extensions: Whether opaque C-extension signatures fail signature checks.

    Returns:
        `Valid(ComplianceReport)` on clean AST extraction across all `sources`,
        else `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import resolve_alignment_preset

    ref_prefix = str(snapshot.target)
    derived_prefix = (
        target_prefix.strip() if target_prefix.strip() else _infer_target_prefix_from_sources(sources, ref_prefix)
    )
    effective_rule = alignment_rule if alignment_rule is not None else resolve_alignment_preset(None, derived_prefix)

    sorted_entries = sorted(sources.items(), key=lambda kv: kv[0])

    def _audit_one(
        entry: tuple[str, str],
    ) -> Validation[NonEmpty[SnapshotDomainError], tuple[tuple[GhostRef, ...], FileComplianceResult]]:
        """Validate and extract a single `(module_key, source)` pair.

        Args:
            entry: `(module_key, source_code)` pair.

        Returns:
            `Valid((refs, file_compliance))` or `Invalid(NonEmpty[SnapshotDomainError])`.
        """
        mod_key, src_code = entry
        res = extract_refs_from_source(
            mod_key,
            src_code,
            derived_prefix,
            ref_prefix,
            alignment_rule=effective_rule,
        )
        if isinstance(res, Ok):
            return Valid(res.value)
        return invalid_one(res.error)

    extracted_val = traverse_validation_ne(sorted_entries, _audit_one)
    if isinstance(extracted_val, Invalid):
        return Invalid(extracted_val.errors)

    dedup_refs: list[GhostRef] = []
    seen_paths: set[str] = set()
    file_results: list[FileComplianceResult] = []
    for mod_refs, file_comp in extracted_val.value:
        file_results.append(file_comp)
        for ref in mod_refs:
            if ref.api_path not in seen_paths:
                seen_paths.add(ref.api_path)
                dedup_refs.append(ref)

    merged_ir_fw = FILE_COMPLIANCE_MONOID.mconcat(file_results)
    scored = score_compliance(
        snapshot,
        dedup_refs,
        strict_c_extensions=strict_c_extensions,
    )
    return Valid(
        scored.model_copy(
            update={
                "found_ir": merged_ir_fw.is_ir,
                "ir_score": merged_ir_fw.ir_score,
                "found_fw": merged_ir_fw.is_fw,
                "fw_score": merged_ir_fw.fw_score,
                "implemented_ops": tuple(sorted(merged_ir_fw.implemented_ops)),
            }
        )
    )


__all__ = [
    "_annotations_compatible",
    "_are_signatures_compatible",
    "_is_opaque_varargs_only",
    "_sig_tuple",
    "audit_module_sources",
    "score_compliance",
]
