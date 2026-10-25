"""Snapshot vs. target compliance scoring engine."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
)

from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceScoreReport,
    ParamSignatureTuple,
)
from ml_ecosystem_snapshots.models import JsonValue, SnapshotEnvelope


def score_compliance(
    reference_snapshot: SnapshotEnvelope | typing.Mapping[str, JsonValue],
    target_refs: typing.Sequence[GhostRef],
    strict_c_extensions: bool = False,
) -> ComplianceScoreReport:
    """Score the compliance of target refs against a reference snapshot.

    Args:
        reference_snapshot: The reference snapshot dictionary or envelope containing categories.
        target_refs: The extracted and aligned GhostRefs from the target.
        strict_c_extensions: Whether to reject fallback matches with opaque C-extension signatures.

    Returns:
        A `ComplianceScoreReport` containing compliance metrics.
    """
    reference_map: dict[str, GhostRef] = {}
    raw_snap = (
        reference_snapshot.model_dump()
        if isinstance(reference_snapshot, SnapshotEnvelope)
        else reference_snapshot
    )
    cats_obj: JsonValue = raw_snap.get("categories", {})
    assert isinstance(cats_obj, dict)
    for _cat, items in cats_obj.items():
        assert isinstance(items, list)
        for item in items:
            ref = GhostRef.model_validate(item)
            reference_map[ref.api_path] = ref
            for alias in ref.aliases or []:
                reference_map[alias] = ref

    target_map = {ref.api_path: ref for ref in target_refs}

    total_reference_endpoints = len({ref.api_path for ref in reference_map.values()})
    if total_reference_endpoints == 0:
        return ComplianceScoreReport(
            score_percentage=0.0,
            total_endpoints=0,
            matched=[],
            missing=[],
            mismatched=[],
            opaque_signatures=[],
            warnings=[],
        )

    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[ComplianceMismatchEntry] = []
    opaque_signatures: list[str] = []
    warnings: list[str] = []

    def sig_tuple(p: GhostParam) -> ParamSignatureTuple:
        """Build a normalized `(name, kind, default, annotation)` tuple for parameter comparison.

        Args:
            p: A `GhostParam` with `name`, `kind`, `default`, and `annotation`.

        Returns:
            Normalized 4-tuple `(name, kind, default, annotation)`.
        """
        name_val = p.name
        kind_val = str(p.kind)
        ann = p.annotation
        default: str | None = str(p.default) if p.default is not None else None

        if isinstance(ann, str) and ann.startswith('"') and ann.endswith('"'):
            ann = ann[1:-1]
        if isinstance(ann, str) and ann.startswith("'") and ann.endswith("'"):
            ann = ann[1:-1]
        if default in ("'```(None)```'", "```(None)```", "'___NONE___'"):
            default = "None"
        return (name_val, kind_val, default, ann)

    for api_path, ref_obj in reference_map.items():
        if api_path not in target_map:
            missing.append(api_path)
            continue

        target_obj = target_map[api_path]

        ref_sig = [sig_tuple(p) for p in ref_obj.params]
        tgt_sig = [sig_tuple(p) for p in target_obj.params]

        ref_sig_comp = (
            ref_obj.signature_completeness
            if isinstance(ref_obj, GhostPythonRef)
            else None
        )
        tgt_sig_comp = (
            target_obj.signature_completeness
            if isinstance(target_obj, GhostPythonRef)
            else None
        )

        ref_is_opaque = (
            ref_sig_comp == "opaque"
            or "inexact_signature" in (ref_obj.environment_tags or [])
            or "opaque_c_extension" in (ref_obj.environment_tags or [])
        )
        tgt_is_opaque = (
            tgt_sig_comp == "opaque"
            or "inexact_signature" in (target_obj.environment_tags or [])
            or "opaque_c_extension" in (target_obj.environment_tags or [])
        )

        if ref_is_opaque or tgt_is_opaque:
            opaque_signatures.append(api_path)
            warnings.append(
                f"API '{api_path}' matched via opaque (*args, **kwargs) C-extension fallback."
            )
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

        if ref_sig == tgt_sig:
            matched.append(api_path)
        else:
            has_varargs = any("VAR_POSITIONAL" in p.kind for p in target_obj.params)
            has_varkwargs = any("VAR_KEYWORD" in p.kind for p in target_obj.params)

            if has_varargs and has_varkwargs:
                matched.append(api_path)
            else:
                mismatched.append(
                    ComplianceMismatchEntry(
                        api_path=api_path, expected=ref_sig, actual=tgt_sig
                    )
                )

    score_percentage = (len(matched) / total_reference_endpoints) * 100.0

    return ComplianceScoreReport(
        score_percentage=round(score_percentage, 2),
        total_endpoints=total_reference_endpoints,
        matched=matched,
        missing=missing,
        mismatched=mismatched,
        opaque_signatures=opaque_signatures,
        warnings=warnings,
    )


__all__ = ["score_compliance"]
