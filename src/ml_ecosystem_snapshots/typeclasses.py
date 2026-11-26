"""Explicit Haskell-style typeclass witness instances for `ml_ecosystem_snapshots`.

Re-exports fundamental algebraic, codec, fold, and outcome primitives from
`ml_switcheroo_ir.typeclasses` and defines domain-specific frozen witness
dictionaries (`Monoid`, `Semigroup`, `Eq`, `HashableWitness`, `JsonCodec`,
`Pretty`, and `Endo`) over snapshot diffs, compliance/grounding reports,
`GhostRef` alias consolidation, envelopes, and `.pyi` stubs.
"""

from __future__ import annotations

import json
import typing

from ml_switcheroo_ir.compliance import (
    FILE_COMPLIANCE_MONOID as FILE_COMPLIANCE_MONOID,
)
from ml_switcheroo_ir.compliance import (
    FileComplianceResult as FileComplianceResult,
)
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    SnapshotEnvelope,
)
from ml_switcheroo_ir.typeclasses import (
    BOOL_ALL_MONOID as BOOL_ALL_MONOID,
)
from ml_switcheroo_ir.typeclasses import (
    BOOL_ANY_MONOID as BOOL_ANY_MONOID,
)
from ml_switcheroo_ir.typeclasses import (
    INT_SUM_MONOID as INT_SUM_MONOID,
)
from ml_switcheroo_ir.typeclasses import (
    BoundedLattice as BoundedLattice,
)
from ml_switcheroo_ir.typeclasses import (
    Codec as Codec,
)
from ml_switcheroo_ir.typeclasses import (
    CodecError as CodecError,
)
from ml_switcheroo_ir.typeclasses import (
    Endo as Endo,
)
from ml_switcheroo_ir.typeclasses import (
    EndoRewrite as EndoRewrite,
)
from ml_switcheroo_ir.typeclasses import (
    Eq as Eq,
)
from ml_switcheroo_ir.typeclasses import (
    Err as Err,
)
from ml_switcheroo_ir.typeclasses import (
    HashableWitness as HashableWitness,
)
from ml_switcheroo_ir.typeclasses import (
    Invalid as Invalid,
)
from ml_switcheroo_ir.typeclasses import (
    JsonCodec as JsonCodec,
)
from ml_switcheroo_ir.typeclasses import (
    Monoid as Monoid,
)
from ml_switcheroo_ir.typeclasses import (
    NonEmpty as NonEmpty,
)
from ml_switcheroo_ir.typeclasses import (
    Ok as Ok,
)
from ml_switcheroo_ir.typeclasses import (
    Ord as Ord,
)
from ml_switcheroo_ir.typeclasses import (
    Ordering as Ordering,
)
from ml_switcheroo_ir.typeclasses import (
    Pretty as Pretty,
)
from ml_switcheroo_ir.typeclasses import (
    RecursiveFold as RecursiveFold,
)
from ml_switcheroo_ir.typeclasses import (
    Result as Result,
)
from ml_switcheroo_ir.typeclasses import (
    Semigroup as Semigroup,
)
from ml_switcheroo_ir.typeclasses import (
    Valid as Valid,
)
from ml_switcheroo_ir.typeclasses import (
    Validation as Validation,
)
from ml_switcheroo_ir.typeclasses import (
    and_then as and_then,
)
from ml_switcheroo_ir.typeclasses import (
    ap_validation as ap_validation,
)
from ml_switcheroo_ir.typeclasses import (
    collect_unit_validations_ne as collect_unit_validations_ne,
)
from ml_switcheroo_ir.typeclasses import (
    endo_monoid as endo_monoid,
)
from ml_switcheroo_ir.typeclasses import (
    fold_result as fold_result,
)
from ml_switcheroo_ir.typeclasses import (
    fold_validation as fold_validation,
)
from ml_switcheroo_ir.typeclasses import (
    from_error_sequence as from_error_sequence,
)
from ml_switcheroo_ir.typeclasses import (
    frozenset_union_monoid as frozenset_union_monoid,
)
from ml_switcheroo_ir.typeclasses import (
    invalid_one as invalid_one,
)
from ml_switcheroo_ir.typeclasses import (
    map_err as map_err,
)
from ml_switcheroo_ir.typeclasses import (
    map_result as map_result,
)
from ml_switcheroo_ir.typeclasses import (
    map_validation as map_validation,
)
from ml_switcheroo_ir.typeclasses import (
    map_validation_err as map_validation_err,
)
from ml_switcheroo_ir.typeclasses import (
    nonempty_eq as nonempty_eq,
)
from ml_switcheroo_ir.typeclasses import (
    nonempty_hash as nonempty_hash,
)
from ml_switcheroo_ir.typeclasses import (
    nonempty_ord as nonempty_ord,
)
from ml_switcheroo_ir.typeclasses import (
    nonempty_pretty as nonempty_pretty,
)
from ml_switcheroo_ir.typeclasses import (
    nonempty_semigroup as nonempty_semigroup,
)
from ml_switcheroo_ir.typeclasses import (
    partition_results as partition_results,
)
from ml_switcheroo_ir.typeclasses import (
    raise_on_err as raise_on_err,
)
from ml_switcheroo_ir.typeclasses import (
    result_to_validation as result_to_validation,
)
from ml_switcheroo_ir.typeclasses import (
    traverse_result as traverse_result,
)
from ml_switcheroo_ir.typeclasses import (
    traverse_validation as traverse_validation,
)
from ml_switcheroo_ir.typeclasses import (
    traverse_validation_ne as traverse_validation_ne,
)
from ml_switcheroo_ir.typeclasses import (
    tuple_monoid as tuple_monoid,
)
from ml_switcheroo_ir.typeclasses import (
    validation_ne_to_tuple as validation_ne_to_tuple,
)
from ml_switcheroo_ir.typeclasses import (
    validation_to_result as validation_to_result,
)
from ml_switcheroo_ir.typeclasses import (
    validation_tuple_to_ne as validation_tuple_to_ne,
)
from ml_switcheroo_ir.typeclasses import (
    value_or as value_or,
)
from ml_switcheroo_ir.types import JsonValue as IrJsonValue

from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceScoreReport,
    NamespaceAlignmentRule,
    SnippetComplianceReport,
)
from ml_ecosystem_snapshots.diff import DiffResult, generate_changelog
from ml_ecosystem_snapshots.errors import format_snapshot_error
from ml_ecosystem_snapshots.grounding.models import CodeBlockCheckResult, CodeBlockFinding
from ml_ecosystem_snapshots.identifiers import API_PATH_LEN_LEX_ORD, ApiPath
from ml_ecosystem_snapshots.introspection.sanitize import (
    TYPE_STR_SANITIZE_ENDO as TYPE_STR_SANITIZE_ENDO,
)
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref
from ml_ecosystem_snapshots.serialization.io import validate_snapshot_envelope
from ml_ecosystem_snapshots.stubs import render_stub_modules

_NL: typing.Final[str] = chr(10)
_STR_TUPLE_MONOID: typing.Final[Monoid[tuple[str, ...]]] = tuple_monoid()
_MISMATCH_TUPLE_MONOID: typing.Final[Monoid[tuple[ComplianceMismatchEntry, ...]]] = tuple_monoid()
_FINDING_TUPLE_MONOID: typing.Final[Monoid[tuple[CodeBlockFinding, ...]]] = tuple_monoid()
_STR_FROZENSET_MONOID: typing.Final[Monoid[frozenset[str]]] = frozenset_union_monoid()


def _merge_sorted_unique(lhs: tuple[str, ...], rhs: tuple[str, ...]) -> tuple[str, ...]:
    """Combine two string tuples via associative sorted deduplicated set union.

    Args:
        lhs: Left-hand string tuple.
        rhs: Right-hand string tuple.

    Returns:
        Sorted deduplicated string tuple.
    """
    return tuple(sorted(_STR_FROZENSET_MONOID.combine(frozenset(lhs), frozenset(rhs))))


def _combine_diff_results(lhs: DiffResult, rhs: DiffResult) -> DiffResult:
    """Associatively combine two `DiffResult` values.

    Metadata transitions `(old_a, new_a)` and `(old_b, new_b)` combine via the
    rectangular-band semigroup `(old_a, new_b)` (`first x last`), guaranteeing
    strict associativity and left/right identity with `DiffResult()`.

    Args:
        lhs: First `DiffResult`.
        rhs: Second `DiffResult`.

    Returns:
        Combined frozen `DiffResult`.
    """
    merged_meta: dict[str, tuple[str | None, str | None]] = dict(lhs.metadata_changed)
    for k, (old_r, new_r) in rhs.metadata_changed.items():
        if k in merged_meta:
            old_l, _mid = merged_meta[k]
            merged_meta[k] = (old_l, new_r)
        else:
            merged_meta[k] = (old_r, new_r)

    return DiffResult(
        added=_merge_sorted_unique(lhs.added, rhs.added),
        removed=_merge_sorted_unique(lhs.removed, rhs.removed),
        signature_changed=_merge_sorted_unique(lhs.signature_changed, rhs.signature_changed),
        breaking_signature_changed=_merge_sorted_unique(
            lhs.breaking_signature_changed,
            rhs.breaking_signature_changed,
        ),
        non_breaking_signature_changed=_merge_sorted_unique(
            lhs.non_breaking_signature_changed,
            rhs.non_breaking_signature_changed,
        ),
        metadata_changed=merged_meta,
        deprecated_capabilities=_merge_sorted_unique(lhs.deprecated_capabilities, rhs.deprecated_capabilities),
        overloads_changed=_merge_sorted_unique(lhs.overloads_changed, rhs.overloads_changed),
    )


DIFF_RESULT_EQ: typing.Final[Eq[DiffResult]] = Eq(eq=lambda a, b: a == b)
DIFF_RESULT_SEMIGROUP: typing.Final[Semigroup[DiffResult]] = Semigroup(combine=_combine_diff_results)
DIFF_RESULT_MONOID: typing.Final[Monoid[DiffResult]] = Monoid(
    empty=DiffResult,
    semigroup=DIFF_RESULT_SEMIGROUP,
)


def _combine_compliance_reports(
    lhs: ComplianceScoreReport,
    rhs: ComplianceScoreReport,
) -> ComplianceScoreReport:
    """Associatively combine two `ComplianceScoreReport` instances.

    Args:
        lhs: First `ComplianceScoreReport`.
        rhs: Second `ComplianceScoreReport`.

    Returns:
        Combined `ComplianceScoreReport` with recomputed weighted `score_percentage`.
    """
    total = INT_SUM_MONOID.combine(lhs.total_endpoints, rhs.total_endpoints)
    matched = _STR_TUPLE_MONOID.combine(lhs.matched, rhs.matched)
    missing = _STR_TUPLE_MONOID.combine(lhs.missing, rhs.missing)
    mismatched = _MISMATCH_TUPLE_MONOID.combine(lhs.mismatched, rhs.mismatched)
    opaque = _STR_TUPLE_MONOID.combine(lhs.opaque_signatures, rhs.opaque_signatures)
    warnings = _STR_TUPLE_MONOID.combine(lhs.warnings, rhs.warnings)
    score = round((len(matched) / total) * 100.0, 2) if total > 0 else 0.0
    combined_ir_fw = FILE_COMPLIANCE_MONOID.combine(
        FileComplianceResult(
            is_ir=lhs.found_ir,
            is_fw=lhs.found_fw,
            implemented_ops=set(lhs.implemented_ops),
            ir_score=lhs.ir_score,
            fw_score=lhs.fw_score,
        ),
        FileComplianceResult(
            is_ir=rhs.found_ir,
            is_fw=rhs.found_fw,
            implemented_ops=set(rhs.implemented_ops),
            ir_score=rhs.ir_score,
            fw_score=rhs.fw_score,
        ),
    )
    return ComplianceScoreReport(
        score_percentage=score,
        total_endpoints=total,
        matched=matched,
        missing=missing,
        mismatched=mismatched,
        opaque_signatures=opaque,
        warnings=warnings,
        found_ir=combined_ir_fw.is_ir,
        ir_score=combined_ir_fw.ir_score,
        found_fw=combined_ir_fw.is_fw,
        fw_score=combined_ir_fw.fw_score,
        implemented_ops=tuple(sorted(combined_ir_fw.implemented_ops)),
    )


COMPLIANCE_REPORT_EQ: typing.Final[Eq[ComplianceScoreReport]] = Eq(eq=lambda a, b: a == b)
COMPLIANCE_REPORT_MONOID: typing.Final[Monoid[ComplianceScoreReport]] = Monoid(
    empty=ComplianceScoreReport,
    semigroup=Semigroup(combine=_combine_compliance_reports),
)


def _empty_snippet_report() -> SnippetComplianceReport:
    """Return neutral identity element (`is_compliant=True`) for `SNIPPET_REPORT_MONOID`.

    Returns:
        Neutral `SnippetComplianceReport` identity value.
    """
    return SnippetComplianceReport(is_compliant=True)


def _combine_snippet_reports(
    lhs: SnippetComplianceReport,
    rhs: SnippetComplianceReport,
) -> SnippetComplianceReport:
    """Associatively combine two `SnippetComplianceReport` values.

    Args:
        lhs: First `SnippetComplianceReport`.
        rhs: Second `SnippetComplianceReport`.

    Returns:
        Combined `SnippetComplianceReport`.
    """
    return SnippetComplianceReport(
        is_compliant=BOOL_ALL_MONOID.combine(lhs.is_compliant, rhs.is_compliant),
        total_ops=INT_SUM_MONOID.combine(lhs.total_ops, rhs.total_ops),
        verified_ops=INT_SUM_MONOID.combine(lhs.verified_ops, rhs.verified_ops),
        total_instructions=INT_SUM_MONOID.combine(lhs.total_instructions, rhs.total_instructions),
        verified_instructions=INT_SUM_MONOID.combine(lhs.verified_instructions, rhs.verified_instructions),
        errors=_STR_TUPLE_MONOID.combine(lhs.errors, rhs.errors),
    )


SNIPPET_REPORT_EQ: typing.Final[Eq[SnippetComplianceReport]] = Eq(eq=lambda a, b: a == b)
SNIPPET_REPORT_MONOID: typing.Final[Monoid[SnippetComplianceReport]] = Monoid(
    empty=_empty_snippet_report,
    semigroup=Semigroup(combine=_combine_snippet_reports),
)


def _empty_code_block_result() -> CodeBlockCheckResult:
    """Return neutral identity element (`is_valid=True`) for `CODE_BLOCK_RESULT_MONOID`.

    Returns:
        Neutral `CodeBlockCheckResult` identity value.
    """
    return CodeBlockCheckResult(total_analyzed=0, hallucinations_detected=0, is_valid=True, findings=())


def _combine_code_block_results(
    lhs: CodeBlockCheckResult,
    rhs: CodeBlockCheckResult,
) -> CodeBlockCheckResult:
    """Associatively combine two `CodeBlockCheckResult` values.

    Args:
        lhs: First `CodeBlockCheckResult`.
        rhs: Second `CodeBlockCheckResult`.

    Returns:
        Combined `CodeBlockCheckResult`.
    """
    return CodeBlockCheckResult(
        total_analyzed=INT_SUM_MONOID.combine(lhs.total_analyzed, rhs.total_analyzed),
        hallucinations_detected=INT_SUM_MONOID.combine(lhs.hallucinations_detected, rhs.hallucinations_detected),
        is_valid=BOOL_ALL_MONOID.combine(lhs.is_valid, rhs.is_valid),
        findings=_FINDING_TUPLE_MONOID.combine(lhs.findings, rhs.findings),
    )


CODE_BLOCK_RESULT_EQ: typing.Final[Eq[CodeBlockCheckResult]] = Eq(eq=lambda a, b: a == b)
CODE_BLOCK_RESULT_MONOID: typing.Final[Monoid[CodeBlockCheckResult]] = Monoid(
    empty=_empty_code_block_result,
    semigroup=Semigroup(combine=_combine_code_block_results),
)


def _param_canonical_key(param: GhostParam) -> tuple[str, ...]:
    """Compute a hashable structural key tuple for a `GhostParam`.

    Args:
        param: `GhostParam` or `ExtendedGhostParam` instance.

    Returns:
        Hashable string tuple representing parameter structure.
    """
    allowed_dtypes: tuple[str, ...] = ()
    rank_str = ""
    dir_str = ""
    role_str = ""
    if isinstance(param, ExtendedGhostParam):
        if param.allowed_dtypes:
            allowed_dtypes = tuple(str(dt) for dt in param.allowed_dtypes)
        if param.rank_constraint is not None:
            rank_str = str(param.rank_constraint)
        if param.direction is not None:
            dir_str = str(param.direction)
        if param.role is not None:
            role_str = str(param.role)
    return (
        param.name,
        param.kind.value,
        str(param.default) if param.default is not None else "",
        str(param.annotation) if param.annotation is not None else "",
        dir_str,
        role_str,
        ",".join(allowed_dtypes),
        rank_str,
    )


def ghost_ref_canonical_key(ref: GhostRef) -> tuple[str, ...]:
    """Compute a pure structural signature key for `GhostRef` alias consolidation.

    Args:
        ref: `GhostRef` or `ExtendedGhostRef` instance.

    Returns:
        Deterministic hashable key tuple ignoring `api_path`, `aliases`, and `is_public`.
    """
    domain_json = ""
    overloads_count = "0"
    returns_sig = ""
    domain_type = ""
    if isinstance(
        ref,
        (
            GhostPythonRef,
            GhostMlirRef,
            GhostOperationRef,
            GhostIsaRef,
            GhostInstructionRef,
        ),
    ):
        domain_type = str(ref.domain_type)
    if isinstance(ref, ExtendedGhostRef):
        if ref.domain_metadata:
            domain_json = json.dumps(ref.domain_metadata, sort_keys=True, default=str)
        if ref.overloads:
            overloads_count = str(len(ref.overloads))
        if ref.returns:
            returns_sig = ";".join(f"{r.name}:{r.type}" for r in ref.returns)
    param_flat = tuple(part for p in ref.params for part in _param_canonical_key(p))
    return (
        ref.name,
        ref.kind,
        domain_type,
        ref.docstring or "",
        str(ref.has_varargs),
        ref.returns_type or "",
        returns_sig,
        domain_json,
        overloads_count,
        *param_flat,
    )


GHOST_REF_CANONICAL_EQ: typing.Final[Eq[GhostRef]] = Eq(
    eq=lambda lhs, rhs: ghost_ref_canonical_key(lhs) == ghost_ref_canonical_key(rhs)
)
GHOST_REF_CANONICAL_HASH: typing.Final[HashableWitness[GhostRef]] = HashableWitness(
    eq_witness=GHOST_REF_CANONICAL_EQ,
    hash_value=lambda ref: hash(ghost_ref_canonical_key(ref)),
)


def _combine_consolidated_ghost_refs(lhs: GhostRef, rhs: GhostRef) -> GhostRef:
    """Purely consolidate two structurally equivalent `GhostRef` entries without mutation.

    Selects the canonical `GhostRef` with the shorter (then lexicographically earlier)
    `api_path`, unites all alternative paths into sorted `aliases`, and combines
    `is_public` via `BOOL_ANY_MONOID`.

    Args:
        lhs: First `GhostRef`.
        rhs: Second `GhostRef`.

    Returns:
        Fresh consolidated `GhostRef` clone (`lhs` and `rhs` are never mutated).
    """
    lhs_path = ApiPath(lhs.api_path)
    rhs_path = ApiPath(rhs.api_path)
    primary, secondary = (lhs, rhs) if API_PATH_LEN_LEX_ORD.le(lhs_path, rhs_path) else (rhs, lhs)

    lhs_aliases = frozenset(lhs.aliases) if lhs.aliases else frozenset()
    rhs_aliases = frozenset(rhs.aliases) if rhs.aliases else frozenset()
    all_paths = _STR_FROZENSET_MONOID.mconcat(
        (
            lhs_aliases,
            rhs_aliases,
            frozenset({secondary.api_path}),
        )
    )
    merged_aliases = sorted(p for p in all_paths if p != primary.api_path)

    merged_public: bool | None
    if lhs.is_public is None and rhs.is_public is None:
        merged_public = None
    else:
        merged_public = BOOL_ANY_MONOID.combine(bool(lhs.is_public), bool(rhs.is_public))

    return primary.model_copy(
        update={"aliases": merged_aliases, "is_public": merged_public},
        deep=True,
    )


GHOST_REF_CONSOLIDATION_SEMIGROUP: typing.Final[Semigroup[GhostRef]] = Semigroup(
    combine=_combine_consolidated_ghost_refs
)


def _encode_envelope_json(env: SnapshotEnvelope) -> IrJsonValue:
    """Encode a `SnapshotEnvelope` into a JSON-compatible dictionary value.

    Args:
        env: Validated `SnapshotEnvelope`.

    Returns:
        JSON-serializable dictionary payload.
    """
    dumped: IrJsonValue = json.loads(env.model_dump_json())
    return dumped


def _decode_envelope_json(raw: IrJsonValue) -> Result[CodecError, SnapshotEnvelope]:
    """Decode a raw JSON value into `Result[CodecError, SnapshotEnvelope]`.

    Args:
        raw: Candidate JSON value.

    Returns:
        `Ok(SnapshotEnvelope)` or `Err(CodecError)`.
    """
    if not isinstance(raw, dict):
        return Err(
            CodecError(
                path=(),
                expected="JSON object",
                actual_description=str(type(raw)),
                message="SnapshotEnvelope JSON payload must be an object.",
            )
        )
    return map_err(
        validate_snapshot_envelope(raw),
        lambda err: CodecError(
            path=(),
            expected="SnapshotEnvelope(v2.0.0)",
            actual_description="invalid_envelope",
            message=format_snapshot_error(err),
        ),
    )


SNAPSHOT_ENVELOPE_JSON_CODEC: typing.Final[JsonCodec[SnapshotEnvelope]] = Codec(
    encode=_encode_envelope_json,
    decode=_decode_envelope_json,
)


def _encode_ghost_ref_json(ref: ExtendedGhostRef) -> IrJsonValue:
    """Encode an `ExtendedGhostRef` into a JSON-compatible dictionary value.

    Args:
        ref: Validated `ExtendedGhostRef`.

    Returns:
        JSON-serializable dictionary payload.
    """
    dumped: IrJsonValue = json.loads(ref.model_dump_json())
    return dumped


def _decode_ghost_ref_json(raw: IrJsonValue) -> Result[CodecError, ExtendedGhostRef]:
    """Decode a raw JSON value into `Result[CodecError, ExtendedGhostRef]`.

    Args:
        raw: Candidate JSON value.

    Returns:
        `Ok(ExtendedGhostRef)` or `Err(CodecError)`.
    """
    if not isinstance(raw, dict):
        return Err(
            CodecError(
                path=(),
                expected="JSON object",
                actual_description=str(type(raw)),
                message="ExtendedGhostRef JSON payload must be an object.",
            )
        )
    return map_err(
        hydrate_ghost_ref(raw),
        lambda err: CodecError(
            path=(),
            expected="ExtendedGhostRef(v2.0.0)",
            actual_description="invalid_ghost_ref",
            message=format_snapshot_error(err),
        ),
    )


GHOST_REF_JSON_CODEC: typing.Final[JsonCodec[ExtendedGhostRef]] = Codec(
    encode=_encode_ghost_ref_json,
    decode=_decode_ghost_ref_json,
)

DIFF_CHANGELOG_PRETTY: typing.Final[Pretty[DiffResult]] = Pretty(render=generate_changelog)


def _render_envelope_stubs(env: SnapshotEnvelope) -> str:
    """Render all `.pyi` stub modules in `env` into a concatenated pretty string.

    Args:
        env: `SnapshotEnvelope` to render as `.pyi` text.

    Returns:
        Deterministically ordered `.pyi` module text blocks.
    """
    res = render_stub_modules(env)
    if isinstance(res, Err):
        return f"# stub error: {format_snapshot_error(res.error)}"
    return (_NL * 2).join(f"# module: {mod_name}{_NL}{src.rstrip()}" for mod_name, src in res.value)


STUB_MODULE_PRETTY: typing.Final[Pretty[SnapshotEnvelope]] = Pretty(render=_render_envelope_stubs)


def _apply_path_rewrites(path: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    """Apply sequential substring rewrites from `NamespaceAlignmentRule.path_rewrites`.

    Args:
        path: Candidate symbol path string.
        rewrites: Tuple of `(old_fragment, new_fragment)` pairs.

    Returns:
        Rewritten symbol path string.
    """
    out = path
    for src_frag, dst_frag in rewrites:
        out = out.replace(src_frag, dst_frag)
    return out


def _match_rule_prefix(
    api_path: str,
    target_prefix: str,
    rule: NamespaceAlignmentRule,
) -> str | None:
    """Match `api_path` against `rule.subpackage_roots` or `rule.prefix_map` if applicable.

    Args:
        api_path: Input symbol API path.
        target_prefix: Target package prefix.
        rule: Active `NamespaceAlignmentRule`.

    Returns:
        Aligned path if matched by `rule`, else `None`.
    """
    for sub_target in rule.subpackage_roots:
        candidate_prefix = f"{target_prefix}.{sub_target}."
        if api_path.startswith(candidate_prefix):
            return api_path.replace(candidate_prefix, f"{sub_target}.", 1)

    for src_pref, ref_pref in rule.prefix_map:
        if api_path == src_pref:
            return ref_pref
        if api_path.startswith(f"{src_pref}."):
            return _apply_path_rewrites(
                api_path.replace(f"{src_pref}.", f"{ref_pref}.", 1),
                rule.path_rewrites,
            )
    return None


def make_namespace_align_endo(
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> Endo[str]:
    """Construct a composable `Endo[str]` rewriting target symbol paths to reference paths.

    Args:
        target_prefix: Target package prefix (e.g. `'zero_jax'`).
        reference_prefix: Canonical reference prefix (e.g. `'jax'`).
        alignment_rule: Optional `NamespaceAlignmentRule` table.

    Returns:
        Pure `Endo[str]` path normalizer.
    """
    rule = alignment_rule or NamespaceAlignmentRule()

    def _align(api_path: str) -> str:
        """Apply subpackage roots, prefix lookup, target->reference substitution, and path rewrites.

        Args:
            api_path: Input symbol API path.

        Returns:
            Aligned canonical API path string.
        """
        if api_path == target_prefix and reference_prefix:
            return reference_prefix
        matched_rule = _match_rule_prefix(api_path, target_prefix, rule)
        if matched_rule is not None:
            return matched_rule
        aligned = (
            f"{reference_prefix}.{api_path[len(target_prefix) + 1 :]}"
            if target_prefix and reference_prefix and api_path.startswith(f"{target_prefix}.")
            else api_path
        )
        return _apply_path_rewrites(aligned, rule.path_rewrites)

    return Endo(app=_align)


__all__ = [
    "BOOL_ALL_MONOID",
    "BOOL_ANY_MONOID",
    "BoundedLattice",
    "CODE_BLOCK_RESULT_EQ",
    "CODE_BLOCK_RESULT_MONOID",
    "COMPLIANCE_REPORT_EQ",
    "COMPLIANCE_REPORT_MONOID",
    "Codec",
    "CodecError",
    "DIFF_CHANGELOG_PRETTY",
    "DIFF_RESULT_EQ",
    "DIFF_RESULT_MONOID",
    "DIFF_RESULT_SEMIGROUP",
    "Endo",
    "EndoRewrite",
    "Eq",
    "Err",
    "FILE_COMPLIANCE_MONOID",
    "FileComplianceResult",
    "GHOST_REF_CANONICAL_EQ",
    "GHOST_REF_CANONICAL_HASH",
    "GHOST_REF_CONSOLIDATION_SEMIGROUP",
    "GHOST_REF_JSON_CODEC",
    "HashableWitness",
    "INT_SUM_MONOID",
    "Invalid",
    "JsonCodec",
    "Monoid",
    "NonEmpty",
    "Ok",
    "Ord",
    "Ordering",
    "Pretty",
    "RecursiveFold",
    "Result",
    "SNAPSHOT_ENVELOPE_JSON_CODEC",
    "SNIPPET_REPORT_EQ",
    "SNIPPET_REPORT_MONOID",
    "STUB_MODULE_PRETTY",
    "Semigroup",
    "TYPE_STR_SANITIZE_ENDO",
    "Valid",
    "Validation",
    "and_then",
    "ap_validation",
    "collect_unit_validations_ne",
    "endo_monoid",
    "fold_result",
    "fold_validation",
    "from_error_sequence",
    "frozenset_union_monoid",
    "ghost_ref_canonical_key",
    "invalid_one",
    "make_namespace_align_endo",
    "map_err",
    "map_result",
    "map_validation",
    "map_validation_err",
    "nonempty_eq",
    "nonempty_hash",
    "nonempty_ord",
    "nonempty_pretty",
    "nonempty_semigroup",
    "partition_results",
    "raise_on_err",
    "result_to_validation",
    "traverse_result",
    "traverse_validation",
    "traverse_validation_ne",
    "tuple_monoid",
    "validation_ne_to_tuple",
    "validation_to_result",
    "validation_tuple_to_ne",
    "value_or",
]
