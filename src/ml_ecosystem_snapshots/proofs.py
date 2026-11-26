"""Covariant cumulative proof-carrying snapshot witnesses (`ProvenSnapshot[_SnapshotPhaseT_co]`).

Models verified snapshot invariants as covariant cumulative phantom phases mirrored
after `ml_switcheroo_ir.proofs.ProvenGraph`:

1. `EnvelopeValidPhase`: Witnessing `schema_version == '2.0.0'`, valid `FrameworkId`,
   and valid `SnapshotVersion`.
2. `EnvelopeAndHydratedPhase(EnvelopeValidPhase, HydratedRefsPhase)`: Accumulating
   envelope validity with full `ExtendedGhostRef` hydration across every category.
3. `DeterministicSnapshotPhase(EnvelopeAndHydratedPhase, DeterminismVerifiedPhase)`:
   Accumulating envelope + hydration guarantees with zero leaked hex memory addresses
   and clean `.pyi` stub AST compilation.
4. `FullyCertifiedSnapshotPhase(DeterministicSnapshotPhase, ComplianceVerifiedPhase)`:
   Accumulating envelope + hydration + determinism guarantees with verified target
   compliance at or above `min_score`.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pydantic
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostRef,
    GhostRef,
    SnapshotEnvelope,
)
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Err,
    Invalid,
    Valid,
    Validation,
    from_error_sequence,
    invalid_one,
    traverse_validation_ne,
)

from ml_ecosystem_snapshots.errors import (
    ComplianceThresholdFault,
    DeterminismLeakFault,
    HydrationBatchFault,
    IntrospectionFault,
    SnapshotDomainError,
    SnapshotEnvelopeFault,
    format_snapshot_error,
)

if typing.TYPE_CHECKING:
    from ml_ecosystem_snapshots.compliance.models import NamespaceAlignmentRule
from ml_ecosystem_snapshots.identifiers import (
    CategoryName,
    FrameworkId,
    SnapshotVersion,
    parse_category_name,
    parse_framework_id,
    parse_snapshot_version,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue
from ml_ecosystem_snapshots.serialization.hydration import (
    HydratableInput,
    hydrate_ghost_refs,
)
from ml_ecosystem_snapshots.serialization.io import validate_snapshot_envelope
from ml_ecosystem_snapshots.stubs import render_stub_modules

_NL_CHAR: typing.Final[str] = chr(10)
_HEX_ADDRESS_LEAK_RE: typing.Final[re.Pattern[str]] = re.compile(
    rf"<[^{_NL_CHAR}>]*\bat\s+0x[0-9a-fA-F]+\b[^{_NL_CHAR}>]*>"
)


class EnvelopeValidPhase:
    """Phantom phase tag witnessing a validated v2.0.0 `SnapshotEnvelope`."""


class HydratedRefsPhase:
    """Phantom phase tag witnessing full `ExtendedGhostRef` hydration across every category."""


class DeterminismVerifiedPhase:
    """Phantom phase tag witnessing zero memory-address leaks and valid `.pyi` stub syntax."""


class ComplianceVerifiedPhase:
    """Phantom phase tag witnessing compliance score at or above a required threshold."""


class EnvelopeAndHydratedPhase(EnvelopeValidPhase, HydratedRefsPhase):
    """Cumulative phase witnessing both `EnvelopeValidPhase` and `HydratedRefsPhase`."""


class DeterministicSnapshotPhase(EnvelopeAndHydratedPhase, DeterminismVerifiedPhase):
    """Cumulative phase witnessing envelope validity, hydrated refs, and determinism."""


class FullyCertifiedSnapshotPhase(DeterministicSnapshotPhase, ComplianceVerifiedPhase):
    """Cumulative phase witnessing envelope, hydration, determinism, and compliance certification."""


_SnapshotPhaseT_co = typing.TypeVar("_SnapshotPhaseT_co", covariant=True)


class ProvenSnapshot(pydantic.BaseModel, typing.Generic[_SnapshotPhaseT_co]):
    """Deeply immutable proof-carrying wrapper around a `SnapshotEnvelope`.

    Covariant in `_SnapshotPhaseT_co` so downstream consumers accepting
    `ProvenSnapshot[EnvelopeAndHydratedPhase]` statically accept
    `ProvenSnapshot[DeterministicSnapshotPhase]` and
    `ProvenSnapshot[FullyCertifiedSnapshotPhase]`.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    target: FrameworkId
    version: SnapshotVersion
    envelope: SnapshotEnvelope
    hydrated_categories: tuple[tuple[CategoryName, tuple[ExtendedGhostRef, ...]], ...] = ()

    def category_dict(self) -> dict[str, tuple[ExtendedGhostRef, ...]]:
        """Project hydrated categories into a fresh dictionary keyed by category string.

        Returns:
            Fresh mapping of category name to immutable tuple of `ExtendedGhostRef`.
        """
        return {str(cat): refs for cat, refs in self.hydrated_categories}

    def all_refs(self) -> tuple[ExtendedGhostRef, ...]:
        """Flatten all hydrated `ExtendedGhostRef` entries in deterministic category order.

        Returns:
            Immutable tuple of every hydrated `ExtendedGhostRef` in the snapshot.
        """
        collected: list[ExtendedGhostRef] = []
        for _cat, refs in self.hydrated_categories:
            collected.extend(refs)
        return tuple(collected)

    def to_wire_dict(self) -> dict[str, JsonValue]:
        """Project the underlying `SnapshotEnvelope` into a JSON-serializable wire dict.

        Returns:
            JSON-serializable dictionary omitting `None` optional metadata fields.
        """
        dumped: dict[str, JsonValue] = self.envelope.model_dump(mode="json", exclude_none=True)
        return dumped


def prove_envelope_valid(
    raw: SnapshotEnvelope | typing.Mapping[str, JsonValue] | ScalarValue,
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[EnvelopeValidPhase]]:
    """Validate `raw` and mint a `ProvenSnapshot[EnvelopeValidPhase]` witness on success.

    Deep-copies `SnapshotEnvelope` so subsequent caller mutations cannot invalidate
    the returned proof witness, and accumulates all identifier errors into `NonEmpty`.

    Args:
        raw: Candidate `SnapshotEnvelope` or raw snapshot dictionary.

    Returns:
        `Valid(ProvenSnapshot[EnvelopeValidPhase])` if valid, else
        `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    env_res = validate_snapshot_envelope(raw)
    if isinstance(env_res, Err):
        return invalid_one(env_res.error)
    env = env_res.value.model_copy(deep=True)

    fw_res = parse_framework_id(env.target)
    ver_res = parse_snapshot_version(env.version or "unknown")

    id_errors: list[SnapshotDomainError] = []
    if isinstance(fw_res, Err):
        id_errors.append(fw_res.error)
    if isinstance(ver_res, Err):
        id_errors.append(ver_res.error)
    if isinstance(fw_res, Err) or isinstance(ver_res, Err):
        return Invalid(NonEmpty(head=id_errors[0], tail=tuple(id_errors[1:])))

    return Valid(
        ProvenSnapshot(
            target=fw_res.value,
            version=ver_res.value,
            envelope=env,
            hydrated_categories=(),
        )
    )


def _to_hydratable_payload(item: ExtendedGhostRef | GhostRef | dict[str, JsonValue]) -> HydratableInput:
    """Convert an envelope category entry into a `HydratableInput` mapping.

    Args:
        item: Either an already-validated `GhostRef` or a raw dict.

    Returns:
        Dictionary suitable for `hydrate_ghost_ref`.
    """
    if isinstance(item, GhostRef):
        dumped: dict[str, JsonValue] = item.model_dump()
        return dumped
    return dict(item)


def _hydrate_category_pair(
    pair: tuple[str, list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef]],
) -> Validation[NonEmpty[SnapshotDomainError], tuple[CategoryName, tuple[ExtendedGhostRef, ...]]]:
    """Validate a single `(category_key, raw_items)` pair with `NonEmpty` error accumulation.

    Args:
        pair: Category key and list of raw/hydrated items from `SnapshotEnvelope.categories`.

    Returns:
        `Valid((CategoryName, hydrated_tuple))` or `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    raw_cat, raw_items = pair
    cat_res = parse_category_name(raw_cat)
    hyd_val = hydrate_ghost_refs([_to_hydratable_payload(it) for it in raw_items])

    batch_errors: list[SnapshotDomainError] = []
    if isinstance(cat_res, Err):
        batch_errors.append(cat_res.error)
    if isinstance(hyd_val, Invalid):
        reason_strs = tuple(format_snapshot_error(e) for e in hyd_val.errors.to_tuple())
        reasons_ne = NonEmpty(head=reason_strs[0], tail=reason_strs[1:])
        batch_errors.append(HydrationBatchFault(category=raw_cat, reasons=reasons_ne))
    if isinstance(cat_res, Err) or isinstance(hyd_val, Invalid):
        return Invalid(NonEmpty(head=batch_errors[0], tail=tuple(batch_errors[1:])))

    return Valid((cat_res.value, hyd_val.value))


def upgrade_to_hydrated(
    proven: ProvenSnapshot[EnvelopeValidPhase],
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[EnvelopeAndHydratedPhase]]:
    """Upgrade `ProvenSnapshot[EnvelopeValidPhase]` to cumulative `EnvelopeAndHydratedPhase`.

    Args:
        proven: Envelope-validated snapshot witness.

    Returns:
        `Valid(ProvenSnapshot[EnvelopeAndHydratedPhase])` when all categories and
        references hydrate cleanly, else `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    sorted_pairs = sorted(proven.envelope.categories.items(), key=lambda kv: kv[0])
    cats_val = traverse_validation_ne(sorted_pairs, _hydrate_category_pair)
    if isinstance(cats_val, Invalid):
        return Invalid(cats_val.errors)

    hydrated_cats = cats_val.value
    updated_env = proven.envelope.model_copy(
        update={
            "categories": {str(cat): list(refs) for cat, refs in hydrated_cats},
        },
        deep=True,
    )
    return Valid(
        ProvenSnapshot(
            target=proven.target,
            version=proven.version,
            envelope=updated_env,
            hydrated_categories=hydrated_cats,
        )
    )


def prove_hydrated_refs(
    proven: ProvenSnapshot[EnvelopeValidPhase],
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[EnvelopeAndHydratedPhase]]:
    """Hydrate every category in `proven`, returning cumulative `EnvelopeAndHydratedPhase`.

    Args:
        proven: Envelope-validated snapshot witness.

    Returns:
        `Valid(ProvenSnapshot[EnvelopeAndHydratedPhase])` or `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    return upgrade_to_hydrated(proven)


def _scan_ref_memory_leaks(ref: ExtendedGhostRef) -> tuple[str, ...]:
    """Scan a hydrated `ExtendedGhostRef` for unstripped runtime hex memory addresses.

    Args:
        ref: Hydrated `ExtendedGhostRef` to inspect.

    Returns:
        Tuple of human-readable leak descriptions (`()` when clean).
    """
    leaks: list[str] = []
    if ref.docstring and _HEX_ADDRESS_LEAK_RE.search(ref.docstring):
        leaks.append(f"Memory address leak in docstring of '{ref.api_path}'.")
    if ref.returns_type and _HEX_ADDRESS_LEAK_RE.search(ref.returns_type):
        leaks.append(f"Memory address leak in returns_type of '{ref.api_path}': {ref.returns_type!r}.")
    for p in ref.params:
        if p.default is not None and _HEX_ADDRESS_LEAK_RE.search(str(p.default)):
            leaks.append(f"Memory address leak in default of '{ref.api_path}.{p.name}': {p.default!r}.")
        if p.annotation and _HEX_ADDRESS_LEAK_RE.search(p.annotation):
            leaks.append(f"Memory address leak in annotation of '{ref.api_path}.{p.name}': {p.annotation!r}.")
    return tuple(leaks)


def upgrade_to_deterministic(
    proven: ProvenSnapshot[EnvelopeAndHydratedPhase],
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[DeterministicSnapshotPhase]]:
    """Upgrade `ProvenSnapshot[EnvelopeAndHydratedPhase]` to cumulative `DeterministicSnapshotPhase`.

    Args:
        proven: Envelope-valid and hydrated snapshot witness.

    Returns:
        `Valid(ProvenSnapshot[DeterministicSnapshotPhase])` when zero memory-address leaks
        exist and rendered `.pyi` stubs compile cleanly, else `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    leaked_addresses: list[str] = []
    for ref in proven.all_refs():
        leaked_addresses.extend(_scan_ref_memory_leaks(ref))

    stub_syntax_errors: list[str] = []
    stub_res = render_stub_modules(proven.envelope, include_nonpublic=True)
    if isinstance(stub_res, Err):
        stub_syntax_errors.append(format_snapshot_error(stub_res.error))

    if leaked_addresses or stub_syntax_errors:
        return invalid_one(
            DeterminismLeakFault(
                leaked_addresses=tuple(leaked_addresses),
                stub_syntax_errors=tuple(stub_syntax_errors),
            )
        )

    return Valid(
        ProvenSnapshot(
            target=proven.target,
            version=proven.version,
            envelope=proven.envelope.model_copy(deep=True),
            hydrated_categories=proven.hydrated_categories,
        )
    )


def prove_determinism_verified(
    proven: ProvenSnapshot[HydratedRefsPhase],
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[DeterministicSnapshotPhase]]:
    """Verify determinism on a hydrated snapshot, returning cumulative `DeterministicSnapshotPhase`.

    Args:
        proven: Hydrated snapshot witness.

    Returns:
        `Valid(ProvenSnapshot[DeterministicSnapshotPhase])` or `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    lifted: ProvenSnapshot[EnvelopeAndHydratedPhase] = ProvenSnapshot(
        target=proven.target,
        version=proven.version,
        envelope=proven.envelope,
        hydrated_categories=proven.hydrated_categories,
    )
    return upgrade_to_deterministic(lifted)


def prove_compliance_verified(
    proven: ProvenSnapshot[HydratedRefsPhase],
    target_refs: typing.Sequence[GhostRef],
    min_score: float = 100.0,
    strict_c_extensions: bool = False,
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[FullyCertifiedSnapshotPhase]]:
    """Verify compliance against `target_refs` at or above `min_score`.

    Args:
        proven: Hydrated snapshot witness serving as the reference contract.
        target_refs: Extracted target `GhostRef` sequence under audit.
        min_score: Minimum acceptable compliance percentage (`0.0` to `100.0`).
        strict_c_extensions: Whether opaque C-extension signatures fail signature checks.

    Returns:
        `Valid(ProvenSnapshot[FullyCertifiedSnapshotPhase])` when `score_percentage >= min_score`,
        else `Invalid(NonEmpty[SnapshotDomainError])` carrying `ComplianceThresholdFault`.
    """
    from ml_ecosystem_snapshots.compliance.scorer import score_compliance

    report = score_compliance(
        proven.envelope,
        target_refs,
        strict_c_extensions=strict_c_extensions,
    )
    if report.score_percentage < min_score:
        missing_and_mismatched = tuple(sorted(set(report.missing) | {m.api_path for m in report.mismatched}))
        return invalid_one(
            ComplianceThresholdFault(
                actual_score=report.score_percentage,
                required_score=min_score,
                missing_symbols=missing_and_mismatched,
            )
        )
    return Valid(
        ProvenSnapshot(
            target=proven.target,
            version=proven.version,
            envelope=proven.envelope.model_copy(deep=True),
            hydrated_categories=proven.hydrated_categories,
        )
    )


def upgrade_to_compliance_certified(
    proven: ProvenSnapshot[DeterministicSnapshotPhase],
    source_dir: Path,
    min_score: float = 100.0,
    *,
    target_prefix: str = "",
    alignment_rule: NamespaceAlignmentRule | None = None,
    strict_c_extensions: bool = False,
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[FullyCertifiedSnapshotPhase]]:
    """Upgrade `ProvenSnapshot[DeterministicSnapshotPhase]` to `FullyCertifiedSnapshotPhase`.

    Extracts static Python AST signatures from `source_dir`, aligns them to
    `proven.target`, and certifies that compliance meets or exceeds `min_score`.

    Args:
        proven: Deterministic snapshot witness serving as the reference contract.
        source_dir: Filesystem path to the target Python package directory or `.py` file.
        min_score: Minimum acceptable compliance percentage (`0.0` to `100.0`).
        target_prefix: Optional explicit target module prefix (derived from `source_dir` if empty).
        alignment_rule: Optional explicit `NamespaceAlignmentRule`.
        strict_c_extensions: Whether opaque C-extension signatures fail signature checks.

    Returns:
        `Valid(ProvenSnapshot[FullyCertifiedSnapshotPhase])` when certified,
        else `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import resolve_alignment_preset
    from ml_ecosystem_snapshots.compliance.ast_extractor import (
        extract_target_refs,
        get_module_info_from_path,
    )

    if not source_dir.exists():
        return invalid_one(
            IntrospectionFault(
                module_path=str(source_dir),
                symbol_name=None,
                reason=f"Target compliance source path does not exist: '{source_dir}'.",
            )
        )

    derived_prefix = target_prefix
    if not derived_prefix:
        mod_res = get_module_info_from_path(str(source_dir))
        if isinstance(mod_res, Err):
            return invalid_one(mod_res.error)
        _search_root, derived_prefix = mod_res.value

    effective_rule = alignment_rule if alignment_rule is not None else resolve_alignment_preset(None, derived_prefix)
    refs_res = extract_target_refs(
        str(source_dir),
        derived_prefix,
        str(proven.target),
        alignment_rule=effective_rule,
    )
    if isinstance(refs_res, Err):
        return invalid_one(refs_res.error)

    return prove_compliance_verified(
        proven,
        refs_res.value,
        min_score=min_score,
        strict_c_extensions=strict_c_extensions,
    )


def certify_snapshot_pipeline(
    raw: SnapshotEnvelope | typing.Mapping[str, JsonValue] | ScalarValue,
    source_dir: Path,
    min_score: float = 100.0,
    *,
    target_prefix: str = "",
    alignment_rule: NamespaceAlignmentRule | None = None,
    strict_c_extensions: bool = False,
) -> Validation[NonEmpty[SnapshotDomainError], ProvenSnapshot[FullyCertifiedSnapshotPhase]]:
    """Execute the full 4-stage snapshot certification pipeline from raw input to `FullyCertifiedSnapshotPhase`.

    Sequentially runs:
    1. `prove_envelope_valid(raw)`
    2. `upgrade_to_hydrated(...)`
    3. `upgrade_to_deterministic(...)`
    4. `upgrade_to_compliance_certified(..., source_dir, min_score)`

    Args:
        raw: Candidate `SnapshotEnvelope` or raw snapshot dictionary.
        source_dir: Filesystem path to the target Python package directory or `.py` file.
        min_score: Minimum acceptable compliance percentage (`0.0` to `100.0`).
        target_prefix: Optional explicit target module prefix.
        alignment_rule: Optional explicit `NamespaceAlignmentRule`.
        strict_c_extensions: Whether opaque C-extension signatures fail signature checks.

    Returns:
        `Valid(ProvenSnapshot[FullyCertifiedSnapshotPhase])` or `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    env_val = prove_envelope_valid(raw)
    if isinstance(env_val, Invalid):
        return Invalid(env_val.errors)

    hyd_val = upgrade_to_hydrated(env_val.value)
    if isinstance(hyd_val, Invalid):
        return Invalid(hyd_val.errors)

    det_val = upgrade_to_deterministic(hyd_val.value)
    if isinstance(det_val, Invalid):
        return Invalid(det_val.errors)

    return upgrade_to_compliance_certified(
        det_val.value,
        source_dir=source_dir,
        min_score=min_score,
        target_prefix=target_prefix,
        alignment_rule=alignment_rule,
        strict_c_extensions=strict_c_extensions,
    )


__all__ = [
    "ComplianceVerifiedPhase",
    "DeterminismVerifiedPhase",
    "DeterministicSnapshotPhase",
    "EnvelopeAndHydratedPhase",
    "EnvelopeValidPhase",
    "FullyCertifiedSnapshotPhase",
    "HydratedRefsPhase",
    "ProvenSnapshot",
    "certify_snapshot_pipeline",
    "prove_compliance_verified",
    "prove_determinism_verified",
    "prove_envelope_valid",
    "prove_hydrated_refs",
    "SnapshotEnvelopeFault",
    "from_error_sequence",
    "upgrade_to_compliance_certified",
    "upgrade_to_deterministic",
    "upgrade_to_hydrated",
]
