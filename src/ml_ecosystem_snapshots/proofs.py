"""Phantom-typed proof-carrying snapshot witnesses (`ProvenSnapshot[PhaseT]`).

Models verified snapshot invariants as static phantom phases mirrored after
`ml_switcheroo_ir.proofs.ProvenGraph[PhaseT]`:

1. `EnvelopeValidPhase` — validated v2.0.0 `SnapshotEnvelope`, `FrameworkId`, and `SnapshotVersion`.
2. `HydratedRefsPhase` — every category entry hydrated into immutable `ExtendedGhostRef` tuples.
3. `DeterminismVerifiedPhase` — zero leaked hex memory addresses (`0x...`) and valid `.pyi` ASTs.
4. `ComplianceVerifiedPhase` — target `GhostRef` compliance score meets or exceeds threshold.
"""

from __future__ import annotations

import re
import typing

import pydantic
from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef, GhostRef, SnapshotEnvelope
from ml_switcheroo_ir.typeclasses.algebra import Monoid, tuple_monoid
from ml_switcheroo_ir.typeclasses.outcome import (
    Err,
    Invalid,
    Ok,
    Result,
    Valid,
    Validation,
    traverse_validation,
)

from ml_ecosystem_snapshots.compliance.scorer import score_compliance
from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
from ml_ecosystem_snapshots.identifiers import (
    CategoryName,
    FrameworkId,
    SnapshotVersion,
    parse_category_name,
    parse_framework_id,
    parse_snapshot_version,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue
from ml_ecosystem_snapshots.serialization.hydration import HydratableInput, hydrate_ghost_refs
from ml_ecosystem_snapshots.serialization.io import validate_snapshot_envelope
from ml_ecosystem_snapshots.stubs import render_stub_modules

_SnapshotPhaseT = typing.TypeVar("_SnapshotPhaseT")
_HEX_ADDR_RE: typing.Final[re.Pattern[str]] = re.compile(r"\b0x[0-9a-fA-F]{4,}\b")


class EnvelopeValidPhase:
    """Phantom phase tag witnessing schema v2.0.0 envelope validity and parsed target/version IDs."""


class HydratedRefsPhase:
    """Phantom phase tag witnessing full `ExtendedGhostRef` hydration across every category."""


class DeterminismVerifiedPhase:
    """Phantom phase tag witnessing zero leaked memory addresses (`0x...`) and valid `.pyi` stub ASTs."""


class ComplianceVerifiedPhase:
    """Phantom phase tag witnessing compliance score satisfaction against target `GhostRef`s."""


class ProvenSnapshot(pydantic.BaseModel, typing.Generic[_SnapshotPhaseT]):
    """Immutable proof-carrying snapshot wrapper parameterized by `_SnapshotPhaseT`.

    Attributes:
        target: Validated `FrameworkId` identifier.
        version: Validated `SnapshotVersion` tag.
        envelope: Deep-cloned `SnapshotEnvelope` captured at proof-minting time.
        hydrated_categories: Immutable `(CategoryName, tuple[ExtendedGhostRef, ...])` pairs.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    target: FrameworkId
    version: SnapshotVersion
    envelope: SnapshotEnvelope
    hydrated_categories: tuple[tuple[CategoryName, tuple[ExtendedGhostRef, ...]], ...] = ()

    def category_dict(self) -> dict[str, tuple[ExtendedGhostRef, ...]]:
        """Project `hydrated_categories` into a fresh dictionary keyed by category string.

        Returns:
            Mapping from category string to immutable `ExtendedGhostRef` tuple.
        """
        return {str(cat): refs for cat, refs in self.hydrated_categories}

    def all_refs(self) -> tuple[ExtendedGhostRef, ...]:
        """Flatten all hydrated category tuples in deterministic order.

        Returns:
            Immutable tuple of every hydrated `ExtendedGhostRef` in the snapshot.
        """
        return tuple(ref for _cat, refs in self.hydrated_categories for ref in refs)

    def to_wire_dict(self) -> dict[str, JsonValue]:
        """Project the underlying `SnapshotEnvelope` into a fresh JSON-serializable wire dict.

        Returns:
            Fresh JSON-serializable snapshot dictionary excluding unset `None` fields.
        """
        dumped: dict[str, JsonValue] = self.envelope.model_dump(exclude_none=True)
        return dumped


def prove_envelope_valid(
    raw: SnapshotEnvelope | typing.Mapping[str, JsonValue] | ScalarValue,
) -> Result[SnapshotSchemaError, ProvenSnapshot[EnvelopeValidPhase]]:
    """Validate `raw` and mint a `ProvenSnapshot[EnvelopeValidPhase]` witness on success.

    Deep-copies `SnapshotEnvelope` so subsequent caller mutations cannot invalidate
    the returned proof witness.

    Args:
        raw: Candidate `SnapshotEnvelope` or raw snapshot dictionary.

    Returns:
        `Ok(ProvenSnapshot[EnvelopeValidPhase])` if valid, else `Err(SnapshotSchemaError)`.
    """
    env_res = validate_snapshot_envelope(raw)
    if isinstance(env_res, Err):
        return env_res
    env = env_res.value.model_copy(deep=True)

    fw_res = parse_framework_id(env.target)
    if isinstance(fw_res, Err):
        return Err(SnapshotSchemaError(fw_res.error))

    ver_res = parse_snapshot_version(env.version or "unknown")
    if isinstance(ver_res, Err):
        return Err(SnapshotSchemaError(ver_res.error))

    return Ok(
        ProvenSnapshot(
            target=fw_res.value,
            version=ver_res.value,
            envelope=env,
            hydrated_categories=(),
        )
    )


_PROOF_ERR_MONOID: typing.Final[Monoid[tuple[SnapshotSchemaError, ...]]] = tuple_monoid()


def _hydrate_category_pair(
    pair: tuple[str, list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef]],
) -> Validation[tuple[SnapshotSchemaError, ...], tuple[CategoryName, tuple[ExtendedGhostRef, ...]]]:
    """Validate a single `(category_key, raw_items)` entry via applicative `Validation`.

    Args:
        pair: Category key and list of raw dictionaries or `GhostRef`s.

    Returns:
        `Valid((CategoryName, tuple[ExtendedGhostRef, ...]))` or `Invalid(errors)`.
    """
    raw_cat, raw_items = pair
    cat_res = parse_category_name(raw_cat)
    cat_errs: list[SnapshotSchemaError] = [SnapshotSchemaError(cat_res.error)] if isinstance(cat_res, Err) else []

    coerced_inputs: list[HydratableInput] = [
        item.model_dump() if isinstance(item, GhostRef) else item for item in raw_items
    ]
    refs_val = hydrate_ghost_refs(coerced_inputs)

    if isinstance(refs_val, Invalid):
        return Invalid((*cat_errs, *refs_val.errors))
    if isinstance(cat_res, Err):
        return Invalid(tuple(cat_errs))

    return Valid((cat_res.value, refs_val.value))


def prove_hydrated_refs(
    proven: ProvenSnapshot[EnvelopeValidPhase],
) -> Validation[tuple[SnapshotSchemaError, ...], ProvenSnapshot[HydratedRefsPhase]]:
    """Hydrate every category in `proven`, accumulating all schema errors via `Validation`.

    Args:
        proven: Envelope-validated `ProvenSnapshot[EnvelopeValidPhase]`.

    Returns:
        `Valid(ProvenSnapshot[HydratedRefsPhase])` if all symbols hydrate cleanly,
        or `Invalid(tuple[SnapshotSchemaError, ...])` collecting every failure.
    """
    sorted_pairs = sorted(proven.envelope.categories.items(), key=lambda kv: kv[0])
    hydrated_val: Validation[
        tuple[SnapshotSchemaError, ...],
        tuple[tuple[CategoryName, tuple[ExtendedGhostRef, ...]], ...],
    ] = traverse_validation(_PROOF_ERR_MONOID.semigroup, sorted_pairs, _hydrate_category_pair)
    if isinstance(hydrated_val, Invalid):
        return Invalid(hydrated_val.errors)
    return Valid(
        ProvenSnapshot(
            target=proven.target,
            version=proven.version,
            envelope=proven.envelope.model_copy(deep=True),
            hydrated_categories=hydrated_val.value,
        )
    )


def _collect_ref_address_leaks(ref: ExtendedGhostRef) -> tuple[str, ...]:
    """Scan a hydrated `ExtendedGhostRef` for leaked hex memory addresses (`0x...`).

    Args:
        ref: Hydrated `ExtendedGhostRef` to inspect.

    Returns:
        Tuple of human-readable diagnostic messages (empty if deterministic).
    """
    diagnostics: list[str] = []
    if ref.docstring and _HEX_ADDR_RE.search(ref.docstring):
        diagnostics.append(f"{ref.api_path}: leaked hex address in docstring.")
    if ref.returns_type and _HEX_ADDR_RE.search(ref.returns_type):
        diagnostics.append(f"{ref.api_path}: leaked hex address in returns_type '{ref.returns_type}'.")
    for param in ref.params:
        if param.default and _HEX_ADDR_RE.search(str(param.default)):
            diagnostics.append(f"{ref.api_path}.{param.name}: leaked hex address in default '{param.default}'.")
        if param.annotation and _HEX_ADDR_RE.search(param.annotation):
            diagnostics.append(f"{ref.api_path}.{param.name}: leaked hex address in annotation '{param.annotation}'.")
    return tuple(diagnostics)


def prove_determinism_verified(
    proven: ProvenSnapshot[HydratedRefsPhase],
) -> Validation[tuple[str, ...], ProvenSnapshot[DeterminismVerifiedPhase]]:
    """Verify zero leaked runtime addresses (`0x...`) and valid `.pyi` stub AST compilation.

    Accumulates both address-leak diagnostics and `.pyi` stub syntax errors before
    failing so callers see every determinism violation in a single pass.

    Args:
        proven: Hydrated `ProvenSnapshot[HydratedRefsPhase]`.

    Returns:
        `Valid(ProvenSnapshot[DeterminismVerifiedPhase])` when deterministic,
        else `Invalid(tuple[str, ...])` listing all violations.
    """
    errors: list[str] = []
    for ref in proven.all_refs():
        errors.extend(_collect_ref_address_leaks(ref))

    stub_res = render_stub_modules(proven.envelope)
    if isinstance(stub_res, Err):
        errors.append(str(stub_res.error))

    if errors:
        return Invalid(tuple(errors))

    return Valid(
        ProvenSnapshot(
            target=proven.target,
            version=proven.version,
            envelope=proven.envelope.model_copy(deep=True),
            hydrated_categories=proven.hydrated_categories,
        )
    )


def prove_compliance_verified(
    proven: ProvenSnapshot[HydratedRefsPhase],
    target_refs: typing.Sequence[GhostRef],
    min_score: float = 100.0,
    strict_c_extensions: bool = False,
) -> Validation[tuple[str, ...], ProvenSnapshot[ComplianceVerifiedPhase]]:
    """Verify target `GhostRef` compliance against `proven` and mint `ComplianceVerifiedPhase`.

    Args:
        proven: Hydrated reference `ProvenSnapshot[HydratedRefsPhase]`.
        target_refs: Extracted and aligned target `GhostRef` sequence.
        min_score: Minimum acceptable `score_percentage` (default `100.0`).
        strict_c_extensions: Whether opaque C-extension fallbacks count as mismatches.

    Returns:
        `Valid(ProvenSnapshot[ComplianceVerifiedPhase])` if `score_percentage >= min_score`,
        else `Invalid(tuple[str, ...])` accumulating score, missing, and mismatch diagnostics.
    """
    report = score_compliance(
        proven.envelope,
        target_refs,
        strict_c_extensions=strict_c_extensions,
    )
    if report.score_percentage >= min_score:
        return Valid(
            ProvenSnapshot(
                target=proven.target,
                version=proven.version,
                envelope=proven.envelope.model_copy(deep=True),
                hydrated_categories=proven.hydrated_categories,
            )
        )

    reasons: list[str] = [f"Compliance score {report.score_percentage:.2f}% is below required {min_score:.2f}%."]
    for missing_path in report.missing:
        reasons.append(f"Missing endpoint: {missing_path}")
    for mismatch in report.mismatched:
        detail = f" ({mismatch.reason})" if mismatch.reason else ""
        reasons.append(f"Signature mismatch: {mismatch.api_path}{detail}")
    return Invalid(tuple(reasons))


__all__ = [
    "ComplianceVerifiedPhase",
    "DeterminismVerifiedPhase",
    "EnvelopeValidPhase",
    "HydratedRefsPhase",
    "ProvenSnapshot",
    "prove_compliance_verified",
    "prove_determinism_verified",
    "prove_envelope_valid",
    "prove_hydrated_refs",
]
