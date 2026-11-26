"""Exhaustiveness, equality, pretty-printing, and boundary tests for `SnapshotDomainError`."""

from __future__ import annotations

import pytest
import typing_extensions
from ml_switcheroo_ir.errors import ShapeUnificationError
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import Err, Invalid, Ok, Valid
from ml_switcheroo_ir.types import TensorShape

from ml_ecosystem_snapshots.errors import (
    SNAPSHOT_DOMAIN_ERROR_EQ,
    SNAPSHOT_DOMAIN_ERROR_PRETTY,
    CExtensionParseFault,
    ComplianceThresholdFault,
    DeterminismLeakFault,
    GroundingBatchFault,
    HydrationBatchFault,
    IdentifierValidationFault,
    IntrospectionFault,
    OfflineNetworkFault,
    SnapshotDomainError,
    SnapshotEnvelopeFault,
    SnapshotNotFoundFault,
    UpstreamIrFault,
    format_snapshot_error,
)
from ml_ecosystem_snapshots.exceptions import (
    SnapshotError,
    raise_snapshot_boundary,
    raise_validation_boundary,
    to_snapshot_exception,
)


def _classify_primary_fault(
    err: (
        SnapshotNotFoundFault
        | SnapshotEnvelopeFault
        | IdentifierValidationFault
        | HydrationBatchFault
        | DeterminismLeakFault
    ),
) -> str:
    """Classify primary snapshot lifecycle faults via `assert_never`.

    Args:
        err: Closed primary fault variant.

    Returns:
        Tag name identifying the matched variant.
    """
    if isinstance(err, SnapshotNotFoundFault):
        return "not_found"
    if isinstance(err, SnapshotEnvelopeFault):
        return "envelope"
    if isinstance(err, IdentifierValidationFault):
        return "identifier"
    if isinstance(err, HydrationBatchFault):
        return "hydration"
    if isinstance(err, DeterminismLeakFault):
        return "determinism"
    typing_extensions.assert_never(err)


def _classify_snapshot_domain_error(err: SnapshotDomainError) -> str:
    """Exhaustively classify all 11 `SnapshotDomainError` variants via `assert_never`.

    Args:
        err: Closed `SnapshotDomainError` instance.

    Returns:
        Tag name identifying the matched variant.
    """
    tag = ""
    if isinstance(
        err,
        (
            SnapshotNotFoundFault,
            SnapshotEnvelopeFault,
            IdentifierValidationFault,
            HydrationBatchFault,
            DeterminismLeakFault,
        ),
    ):
        tag = _classify_primary_fault(err)
    elif isinstance(err, IntrospectionFault):
        tag = "introspection"
    elif isinstance(err, CExtensionParseFault):
        tag = "c_extension"
    elif isinstance(err, ComplianceThresholdFault):
        tag = "compliance"
    elif isinstance(err, GroundingBatchFault):
        tag = "grounding"
    elif isinstance(err, OfflineNetworkFault):
        tag = "offline_network"
    elif isinstance(err, UpstreamIrFault):
        tag = "upstream_ir"
    else:
        typing_extensions.assert_never(err)
    return tag


def _all_eleven_variants() -> tuple[SnapshotDomainError, ...]:
    """Construct representative instances of all 11 `SnapshotDomainError` variants.

    Returns:
        11-tuple covering every variant constructor of `SnapshotDomainError`.
    """
    return (
        SnapshotNotFoundFault(target="torch", version="2.4.0", searched_paths=("/tmp/snaps",)),
        SnapshotEnvelopeFault(reason="bad schema", schema_version="1.0.0", details=("missing target",)),
        IdentifierValidationFault(field="FrameworkId", raw_value="bad!", reason="invalid char"),
        HydrationBatchFault(category="neural", reasons=NonEmpty(head="bad param", tail=("bad kind",))),
        DeterminismLeakFault(
            leaked_addresses=("<object at 0x1234>",),
            stub_syntax_errors=("SyntaxError line 3",),
        ),
        IntrospectionFault(module_path="pkg.mod", symbol_name="fn", reason="SyntaxError"),
        CExtensionParseFault(signature_text="add(??)", offset=4, reason="unexpected '?'"),
        ComplianceThresholdFault(actual_score=75.0, required_score=100.0, missing_symbols=("torch.add",)),
        GroundingBatchFault(ungrounded_ops=NonEmpty.singleton("torch.hallucinated")),
        OfflineNetworkFault(url="https://example.com/spec.json", operation="offline_blocked"),
        UpstreamIrFault(
            ir_error=ShapeUnificationError(
                lhs=TensorShape([2, 3]),
                rhs=TensorShape([4, 5]),
                reason="matmul",
            )
        ),
    )


def test_assert_never_exhaustiveness_and_eq_pretty_laws() -> None:
    """Verify `assert_never` dispatch, reflexivity, symmetry, and non-empty rendering across all 11 variants."""
    variants = _all_eleven_variants()
    assert len(variants) == 11

    seen_tags: list[str] = []
    for idx, var in enumerate(variants):
        tag = _classify_snapshot_domain_error(var)
        seen_tags.append(tag)
        rendered = format_snapshot_error(var)
        assert rendered
        assert SNAPSHOT_DOMAIN_ERROR_PRETTY.render(var) == rendered
        assert SNAPSHOT_DOMAIN_ERROR_EQ.eq(var, var)
        for other_idx, other_var in enumerate(variants):
            if idx != other_idx:
                assert SNAPSHOT_DOMAIN_ERROR_EQ.neq(var, other_var)

    assert len(set(seen_tags)) == 11


def test_format_snapshot_error_optional_field_branches() -> None:
    """Verify formatting branches when optional fault fields are omitted or empty."""
    bare_nf = SnapshotNotFoundFault(target="jax")
    assert format_snapshot_error(bare_nf) == "Snapshot not found: jax"

    bare_env = SnapshotEnvelopeFault(reason="not a dict")
    assert format_snapshot_error(bare_env) == "Snapshot envelope error: not a dict"

    empty_det = DeterminismLeakFault(leaked_addresses=(), stub_syntax_errors=())
    assert "determinism verification failed" in format_snapshot_error(empty_det)

    leak_only_det = DeterminismLeakFault(leaked_addresses=("leak1",), stub_syntax_errors=())
    assert "leaked addresses: leak1" in format_snapshot_error(leak_only_det)

    stub_only_det = DeterminismLeakFault(leaked_addresses=(), stub_syntax_errors=("err1",))
    assert "stub syntax errors: err1" in format_snapshot_error(stub_only_det)

    bare_intro = IntrospectionFault(module_path="mod.py", symbol_name=None, reason="missing")
    assert format_snapshot_error(bare_intro) == "Introspection failed for 'mod.py': missing"

    bare_cext = CExtensionParseFault(signature_text="", offset=None, reason="no doc")
    assert "at offset" not in format_snapshot_error(bare_cext)

    bare_comp = ComplianceThresholdFault(actual_score=50.0, required_score=90.0)
    assert "50.00% below required 90.00%" in format_snapshot_error(bare_comp)


def test_boundary_exception_wrappers() -> None:
    """Verify `SnapshotError`, `to_snapshot_exception`, `raise_snapshot_boundary`, and `raise_validation_boundary`."""
    fault = OfflineNetworkFault(url="https://example.org", operation="offline_blocked")
    exc = to_snapshot_exception(fault)
    assert isinstance(exc, SnapshotError)
    assert exc.domain_error == fault
    assert "offline_blocked" in str(exc)

    assert raise_snapshot_boundary(Ok(42)) == 42
    with pytest.raises(SnapshotError) as exc_info:
        raise_snapshot_boundary(Err(fault))
    assert exc_info.value.domain_error == fault

    assert raise_validation_boundary(Valid("certified")) == "certified"
    with pytest.raises(SnapshotError) as val_exc:
        raise_validation_boundary(Invalid(NonEmpty.singleton(fault)))
    assert val_exc.value.domain_error == fault
