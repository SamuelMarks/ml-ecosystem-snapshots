"""Closed algebraic `SnapshotDomainError` sum type and exhaustive diagnostics witnesses.

Replaces open OOP exception hierarchies across the pure functional core of
`ml_ecosystem_snapshots` with an 11-variant frozen `@dataclass(frozen=True)`
sum type (`SnapshotDomainError`), paired with explicit `Eq`, `Pretty`, and
`assert_never`-verified formatting.
"""

from __future__ import annotations

from dataclasses import dataclass

import typing_extensions
from ml_switcheroo_ir.errors import (
    IR_DOMAIN_ERROR_EQ,
    IR_DOMAIN_ERROR_PRETTY,
    IrDomainError,
)
from ml_switcheroo_ir.typeclasses import Pretty
from ml_switcheroo_ir.typeclasses.algebra import (
    STR_EQ,
    Eq,
    NonEmpty,
    nonempty_eq,
)

_NONEMPTY_STR_EQ: Eq[NonEmpty[str]] = nonempty_eq(STR_EQ)


@dataclass(frozen=True)
class SnapshotNotFoundFault:
    """Missing snapshot file or empty repository lookup variant of `SnapshotDomainError`.

    Attributes:
        target: Requested framework, path, or snapshot identifier.
        version: Optional version tag requested by the caller.
        searched_paths: Ordered tuple of candidate directories inspected.
    """

    target: str
    version: str | None = None
    searched_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotEnvelopeFault:
    """Envelope, JSON codec, compression, or wire schema failure variant of `SnapshotDomainError`.

    Attributes:
        reason: Human-readable explanation of the envelope or codec failure.
        schema_version: Optional encountered schema version tag.
        details: Optional tuple of structured sub-error detail strings.
    """

    reason: str
    schema_version: str | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class IdentifierValidationFault:
    """Domain `NewType` identifier smart-constructor failure variant of `SnapshotDomainError`.

    Attributes:
        field: Domain identifier type or field name (`FrameworkId`, `SnapshotVersion`, etc.).
        raw_value: Raw unvalidated string rejected by the smart constructor.
        reason: Human-readable validation failure description.
    """

    field: str
    raw_value: str
    reason: str


@dataclass(frozen=True)
class HydrationBatchFault:
    """Accumulated `GhostRef` hydration failure batch variant of `SnapshotDomainError`.

    Attributes:
        category: Category key under which hydration failed.
        reasons: Non-empty batch of individual hydration error messages.
    """

    category: str
    reasons: NonEmpty[str]


@dataclass(frozen=True)
class DeterminismLeakFault:
    """Memory-address leak or `.pyi` stub syntax failure variant of `SnapshotDomainError`.

    Attributes:
        leaked_addresses: Tuple of leaked hex memory address diagnostics.
        stub_syntax_errors: Tuple of `.pyi` stub AST syntax diagnostics.
    """

    leaked_addresses: tuple[str, ...]
    stub_syntax_errors: tuple[str, ...]


@dataclass(frozen=True)
class IntrospectionFault:
    """Live or static Python AST/module introspection failure variant of `SnapshotDomainError`.

    Attributes:
        module_path: Module path, file path, or target representation inspected.
        symbol_name: Optional unqualified or dotted symbol name being inspected.
        reason: Human-readable introspection failure reason.
    """

    module_path: str
    symbol_name: str | None
    reason: str


@dataclass(frozen=True)
class CExtensionParseFault:
    """C-extension, PyBind11, or ATen docstring signature parse failure variant of `SnapshotDomainError`.

    Attributes:
        signature_text: Raw docstring or signature line that failed parsing.
        offset: Optional character offset where parsing failed.
        reason: Human-readable parse failure reason.
    """

    signature_text: str
    offset: int | None
    reason: str


@dataclass(frozen=True)
class ComplianceThresholdFault:
    """Snapshot compliance threshold shortfall variant of `SnapshotDomainError`.

    Attributes:
        actual_score: Computed compliance percentage (`0.0` to `100.0`).
        required_score: Minimum required compliance percentage threshold.
        missing_symbols: Tuple of missing or mismatched API paths.
    """

    actual_score: float
    required_score: float
    missing_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingBatchFault:
    """Accumulated anti-hallucination grounding failure batch variant of `SnapshotDomainError`.

    Attributes:
        ungrounded_ops: Non-empty batch of ungrounded operation diagnostic strings.
    """

    ungrounded_ops: NonEmpty[str]


@dataclass(frozen=True)
class OfflineNetworkFault:
    """Blocked or failed network I/O variant of `SnapshotDomainError`.

    Attributes:
        url: Target URL for the network operation.
        operation: Network operation label or failure reason (`offline_blocked`, `fetch`, etc.).
    """

    url: str
    operation: str


@dataclass(frozen=True)
class UpstreamIrFault:
    """Compositionally embedded Tier-0 `IrDomainError` variant of `SnapshotDomainError`.

    Attributes:
        ir_error: Underlying `ml_switcheroo_ir.errors.IrDomainError` variant.
    """

    ir_error: IrDomainError


SnapshotDomainError: typing_extensions.TypeAlias = (
    "SnapshotNotFoundFault"
    " | SnapshotEnvelopeFault"
    " | IdentifierValidationFault"
    " | HydrationBatchFault"
    " | DeterminismLeakFault"
    " | IntrospectionFault"
    " | CExtensionParseFault"
    " | ComplianceThresholdFault"
    " | GroundingBatchFault"
    " | OfflineNetworkFault"
    " | UpstreamIrFault"
)


_CoreSnapshotFault: typing_extensions.TypeAlias = (
    "SnapshotNotFoundFault"
    " | SnapshotEnvelopeFault"
    " | IdentifierValidationFault"
    " | HydrationBatchFault"
    " | DeterminismLeakFault"
)

_CORE_FAULT_TYPES = (
    SnapshotNotFoundFault,
    SnapshotEnvelopeFault,
    IdentifierValidationFault,
    HydrationBatchFault,
    DeterminismLeakFault,
)


def _format_determinism_leak_fault(err: DeterminismLeakFault) -> str:
    """Format a `DeterminismLeakFault` into a deterministic diagnostic string.

    Args:
        err: `DeterminismLeakFault` instance.

    Returns:
        Formatted determinism failure string.
    """
    parts: list[str] = []
    if err.leaked_addresses:
        parts.append(f"leaked addresses: {', '.join(err.leaked_addresses)}")
    if err.stub_syntax_errors:
        parts.append(f"stub syntax errors: {', '.join(err.stub_syntax_errors)}")
    summary = "; ".join(parts) if parts else "determinism verification failed"
    return f"Determinism verification failed: {summary}"


def _format_core_snapshot_fault(err: _CoreSnapshotFault) -> str:
    """Format any `_CoreSnapshotFault` variant with `assert_never` exhaustiveness.

    Args:
        err: Closed `_CoreSnapshotFault` variant.

    Returns:
        Deterministic diagnostic string.
    """
    if isinstance(err, SnapshotNotFoundFault):
        ver_suffix = f"@{err.version}" if err.version is not None else ""
        searched_suffix = f" (searched: {', '.join(err.searched_paths)})" if err.searched_paths else ""
        return f"Snapshot not found: {err.target}{ver_suffix}{searched_suffix}"
    if isinstance(err, SnapshotEnvelopeFault):
        ver_part = f" [schema_version={err.schema_version}]" if err.schema_version is not None else ""
        detail_part = f" ({'; '.join(err.details)})" if err.details else ""
        return f"Snapshot envelope error{ver_part}: {err.reason}{detail_part}"
    if isinstance(err, IdentifierValidationFault):
        return f"Invalid {err.field} {err.raw_value!r}: {err.reason}"
    if isinstance(err, HydrationBatchFault):
        joined = "; ".join(err.reasons.to_tuple())
        return f"Hydration batch error in category '{err.category}' ({len(err.reasons)} error(s)): {joined}"
    if isinstance(err, DeterminismLeakFault):
        return _format_determinism_leak_fault(err)
    typing_extensions.assert_never(err)


def _format_extended_snapshot_fault(
    err: (
        IntrospectionFault
        | CExtensionParseFault
        | ComplianceThresholdFault
        | GroundingBatchFault
        | OfflineNetworkFault
        | UpstreamIrFault
    ),
) -> str:
    """Format an extended `SnapshotDomainError` variant with `assert_never` exhaustiveness.

    Args:
        err: Closed non-core `SnapshotDomainError` variant.

    Returns:
        Deterministic human-readable error diagnostic message.
    """
    if isinstance(err, IntrospectionFault):
        sym_part = f" ({err.symbol_name})" if err.symbol_name is not None else ""
        return f"Introspection failed for '{err.module_path}'{sym_part}: {err.reason}"
    if isinstance(err, CExtensionParseFault):
        off_part = f" at offset {err.offset}" if err.offset is not None else ""
        return f"C-extension signature parse error{off_part} in {err.signature_text!r}: {err.reason}"
    if isinstance(err, ComplianceThresholdFault):
        miss_part = f" (missing/mismatched: {', '.join(err.missing_symbols)})" if err.missing_symbols else ""
        return f"Compliance score {err.actual_score:.2f}% below required {err.required_score:.2f}%{miss_part}"
    if isinstance(err, GroundingBatchFault):
        joined_ops = "; ".join(err.ungrounded_ops.to_tuple())
        return f"Grounding audit failed ({len(err.ungrounded_ops)} ungrounded op(s)): {joined_ops}"
    if isinstance(err, OfflineNetworkFault):
        return f"Network fault ({err.operation}) for '{err.url}'"
    if isinstance(err, UpstreamIrFault):
        return f"Upstream IR error: {IR_DOMAIN_ERROR_PRETTY.render(err.ir_error)}"
    typing_extensions.assert_never(err)


def format_snapshot_error(err: SnapshotDomainError) -> str:
    """Render any `SnapshotDomainError` variant into a deterministic diagnostic string.

    Verified for exhaustiveness across all 11 closed variants via `assert_never`.

    Args:
        err: Closed `SnapshotDomainError` variant instance.

    Returns:
        Deterministic human-readable error diagnostic message.
    """
    if isinstance(err, _CORE_FAULT_TYPES):
        return _format_core_snapshot_fault(err)
    return _format_extended_snapshot_fault(err)


def _eq_core_snapshot_fault(lhs: _CoreSnapshotFault, rhs: SnapshotDomainError) -> bool:
    """Compare a `_CoreSnapshotFault` against `rhs` structurally via `assert_never`.

    Args:
        lhs: Left `_CoreSnapshotFault` value.
        rhs: Right `SnapshotDomainError` value.

    Returns:
        True when both variants share the same constructor and equal payloads.
    """
    if isinstance(lhs, SnapshotNotFoundFault):
        return isinstance(rhs, SnapshotNotFoundFault) and lhs == rhs
    if isinstance(lhs, SnapshotEnvelopeFault):
        return isinstance(rhs, SnapshotEnvelopeFault) and lhs == rhs
    if isinstance(lhs, IdentifierValidationFault):
        return isinstance(rhs, IdentifierValidationFault) and lhs == rhs
    if isinstance(lhs, HydrationBatchFault):
        return (
            isinstance(rhs, HydrationBatchFault)
            and lhs.category == rhs.category
            and _NONEMPTY_STR_EQ.eq(lhs.reasons, rhs.reasons)
        )
    if isinstance(lhs, DeterminismLeakFault):
        return isinstance(rhs, DeterminismLeakFault) and lhs == rhs
    typing_extensions.assert_never(lhs)


def _eq_snapshot_domain_error(lhs: SnapshotDomainError, rhs: SnapshotDomainError) -> bool:
    """Compare two `SnapshotDomainError` variants structurally with `assert_never` dispatch.

    Args:
        lhs: Left `SnapshotDomainError` value.
        rhs: Right `SnapshotDomainError` value.

    Returns:
        True when both variants are of the same constructor and carry equal payloads.
    """
    result = False
    if isinstance(lhs, _CORE_FAULT_TYPES):
        result = _eq_core_snapshot_fault(lhs, rhs)
    elif isinstance(lhs, IntrospectionFault):
        result = isinstance(rhs, IntrospectionFault) and lhs == rhs
    elif isinstance(lhs, CExtensionParseFault):
        result = isinstance(rhs, CExtensionParseFault) and lhs == rhs
    elif isinstance(lhs, ComplianceThresholdFault):
        result = isinstance(rhs, ComplianceThresholdFault) and lhs == rhs
    elif isinstance(lhs, GroundingBatchFault):
        result = isinstance(rhs, GroundingBatchFault) and _NONEMPTY_STR_EQ.eq(lhs.ungrounded_ops, rhs.ungrounded_ops)
    elif isinstance(lhs, OfflineNetworkFault):
        result = isinstance(rhs, OfflineNetworkFault) and lhs == rhs
    elif isinstance(lhs, UpstreamIrFault):
        result = isinstance(rhs, UpstreamIrFault) and IR_DOMAIN_ERROR_EQ.eq(lhs.ir_error, rhs.ir_error)
    else:
        typing_extensions.assert_never(lhs)
    return result


SNAPSHOT_DOMAIN_ERROR_EQ: Eq[SnapshotDomainError] = Eq(eq=_eq_snapshot_domain_error)
SNAPSHOT_DOMAIN_ERROR_PRETTY: Pretty[SnapshotDomainError] = Pretty(render=format_snapshot_error)

__all__ = [
    "CExtensionParseFault",
    "ComplianceThresholdFault",
    "DeterminismLeakFault",
    "GroundingBatchFault",
    "HydrationBatchFault",
    "IdentifierValidationFault",
    "IntrospectionFault",
    "OfflineNetworkFault",
    "SNAPSHOT_DOMAIN_ERROR_EQ",
    "SNAPSHOT_DOMAIN_ERROR_PRETTY",
    "SnapshotDomainError",
    "SnapshotEnvelopeFault",
    "SnapshotNotFoundFault",
    "UpstreamIrFault",
    "format_snapshot_error",
]
