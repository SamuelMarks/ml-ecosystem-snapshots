"""ML Framework Snapshots Pure Functional SDK.

Provides total `Result` and `Validation` interfaces for capturing, consolidating,
proving, and saving API signatures from machine learning frameworks and domains.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import os
import subprocess
import sys
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Err,
    Invalid,
    Ok,
    Result,
    Validation,
    from_error_sequence,
)

from ml_ecosystem_snapshots.errors import (
    IntrospectionFault,
    SnapshotDomainError,
    SnapshotEnvelopeFault,
    SnapshotNotFoundFault,
)
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import (
    SCHEMA_VERSION,
    JsonValue,
    SnapshotEnvelope,
    SnapshotSymbolModel,
    ensure_json_dict,
)
from ml_ecosystem_snapshots.network_env import CompressionFormat
from ml_ecosystem_snapshots.proofs import (
    EnvelopeValidPhase,
    ProvenSnapshot,
    prove_envelope_valid,
)
from ml_ecosystem_snapshots.registry import (
    _CANONICAL_COLLECTOR_SPECS as _CANONICAL_COLLECTOR_SPECS,
)
from ml_ecosystem_snapshots.registry import (
    COLLECTOR_REGISTRY,
    DOMAIN_COLLECTORS,
    FRAMEWORK_COLLECTORS,
    FRAMEWORK_PROVENANCE_METADATA,
    CollectorFn,
    ProvenanceMetadata,
    get_available_frameworks,
    make_lazy_collector,
)
from ml_ecosystem_snapshots.serialization.io import (
    validate_snapshot_envelope,
    write_snapshot,
)
from ml_ecosystem_snapshots.typeclasses import (
    GHOST_REF_CONSOLIDATION_SEMIGROUP,
    ghost_ref_canonical_key,
)
from ml_ecosystem_snapshots.version_resolver import (
    _extract_version_with_griffe as _resolve_version_with_griffe,
)
from ml_ecosystem_snapshots.version_resolver import (
    get_pkg_version as _resolve_pkg_version,
)

_LOGGER = logging.getLogger(__name__)
_make_lazy_collector = make_lazy_collector


def extract_version_with_griffe(module_name: str) -> str | None:
    """Extract `__version__` statically from a module using the custom Griffe fork.

    Args:
        module_name: Dotted module name to inspect (e.g. `'keras'`, `'chex'`, `'optax'`).

    Returns:
        Extracted version string with quotes stripped, or `None` if unavailable.
    """
    return _resolve_version_with_griffe(module_name)


_extract_version_with_griffe = extract_version_with_griffe


def get_pkg_version(package_name: str) -> str:
    """Safely retrieve the installed or pinned version of a framework package.

    Args:
        package_name: Name of the framework or pip package.

    Returns:
        Version string, or `'unknown'` if not installed.
    """
    return _resolve_pkg_version(
        package_name,
        base_file=__file__,
        extract_griffe_fn=extract_version_with_griffe,
    )


def consolidate_aliases(refs: typing.Sequence[GhostRef]) -> tuple[GhostRef, ...]:
    """Purely consolidate structurally identical `GhostRef`s via `GHOST_REF_CONSOLIDATION_SEMIGROUP`.

    Never mutates input `GhostRef` instances in-place.

    Args:
        refs: Sequence of `GhostRef` objects.

    Returns:
        Immutable tuple of consolidated `GhostRef` clones.
    """
    consolidated: dict[tuple[str, ...], GhostRef] = {}
    for ref in refs:
        key = ghost_ref_canonical_key(ref)
        if key not in consolidated:
            existing_aliases = sorted(set(ref.aliases)) if ref.aliases else []
            consolidated[key] = ref.model_copy(update={"aliases": existing_aliases}, deep=True)
        else:
            consolidated[key] = GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(consolidated[key], ref)
    return tuple(consolidated.values())


_consolidate_aliases = consolidate_aliases


def _serialize_refs_to_models(refs: typing.Sequence[GhostRef]) -> tuple[SnapshotSymbolModel, ...]:
    """Consolidate aliases, sort, and validate `GhostRef` items into `SnapshotSymbolModel` tuples.

    Args:
        refs: Collected `GhostRef` sequence.

    Returns:
        Sorted tuple of frozen `SnapshotSymbolModel` instances.
    """
    consolidated = sorted(_consolidate_aliases(refs), key=lambda x: x.name)
    dumped_refs: list[SnapshotSymbolModel] = []
    for r in consolidated:
        d: dict[str, JsonValue] = r.model_dump(exclude_unset=True)
        kwargs_list: list[JsonValue] = [
            p.name for p in r.params if p.kind in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD", "VAR_KEYWORD")
        ]
        d["kwargs"] = kwargs_list
        dumped_refs.append(SnapshotSymbolModel.model_validate(d))
    return tuple(dumped_refs)


def _serialize_category_refs(
    collect_func: CollectorFn,
    cat: SemanticTier,
    include_nonpublic: bool,
) -> tuple[str, tuple[SnapshotSymbolModel, ...]]:
    """Collect, consolidate, and validate all API symbols for a semantic tier.

    Args:
        collect_func: Framework category collector callable.
        cat: `SemanticTier` enum category to extract.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        Tuple of `(category_name, validated_snapshot_symbol_models)`.
    """
    try:
        refs = collect_func(cat, include_nonpublic)
        if refs:
            return cat.value, _serialize_refs_to_models(refs)
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        SnapshotError,
    ) as exc:
        _LOGGER.debug("Category %s collection failed: %s", cat.value, exc)
    return cat.value, ()


def _collect_framework_categories(
    framework_name: str,
    include_nonpublic: bool,
) -> dict[str, list[dict[str, JsonValue]]]:
    """Collect and serialize all category symbol lists for `framework_name`.

    Args:
        framework_name: Registered framework or domain identifier.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        Mapping from category name to serialized symbol dictionaries.
    """
    categories_map: dict[str, list[dict[str, JsonValue]]] = {}
    if framework_name in DOMAIN_COLLECTORS:
        domain_collector = DOMAIN_COLLECTORS[framework_name]
        try:
            domain_refs = domain_collector.collect_domain(include_nonpublic)
            if domain_refs:
                symbol_models = _serialize_refs_to_models(domain_refs)
                categories_map[domain_collector.category_key] = [
                    m.model_dump(mode="json", exclude_unset=True) for m in symbol_models
                ]
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            SnapshotError,
        ) as exc:
            _LOGGER.debug("Domain collection failed for %s: %s", framework_name, exc)
        return categories_map

    collect_func = FRAMEWORK_COLLECTORS[framework_name]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_cat = {
            executor.submit(_serialize_category_refs, collect_func, cat, include_nonpublic): cat for cat in SemanticTier
        }
        for future in concurrent.futures.as_completed(future_to_cat):
            cat_val, symbol_models = future.result()
            if symbol_models:
                categories_map[cat_val] = [m.model_dump(mode="json", exclude_unset=True) for m in symbol_models]
    return categories_map


def extract_snapshot(
    framework_name: str,
    include_nonpublic: bool = False,
) -> Result[SnapshotDomainError, ProvenSnapshot[EnvelopeValidPhase]]:
    """Extract and prove an envelope-valid snapshot for `framework_name` totally into `Result`.

    Args:
        framework_name: Framework identifier (e.g. `'torch'`, `'jax'`, `'ir'`).
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        `Ok(ProvenSnapshot[EnvelopeValidPhase])` on success, `Err(SnapshotNotFoundFault)`
        if unknown or uninstalled, or `Err(SnapshotDomainError)` if empty or schema-invalid.
    """
    if framework_name not in FRAMEWORK_COLLECTORS:
        return Err(SnapshotNotFoundFault(target=framework_name, version=None, searched_paths=()))

    version = get_pkg_version(framework_name)
    if version == "unknown":
        return Err(SnapshotNotFoundFault(target=framework_name, version="unknown", searched_paths=()))

    categories_map = _collect_framework_categories(framework_name, include_nonpublic)
    if not categories_map:
        return Err(
            IntrospectionFault(
                module_path=framework_name,
                symbol_name=None,
                reason=f"No API symbols collected for framework {framework_name!r}.",
            )
        )

    raw_envelope: dict[str, JsonValue] = {
        "schema_version": SCHEMA_VERSION,
        "target": framework_name,
        "version": version,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if framework_name in FRAMEWORK_PROVENANCE_METADATA:
        for k, v in FRAMEWORK_PROVENANCE_METADATA[framework_name].items():
            raw_envelope[k] = list(v) if isinstance(v, list) else v
    raw_envelope["categories"] = {cat_name: list(cat_items) for cat_name, cat_items in categories_map.items()}
    validated_envelope = SnapshotEnvelope.model_validate(raw_envelope)
    proven_val = prove_envelope_valid(validated_envelope)
    if isinstance(proven_val, Invalid):
        return Err(proven_val.errors.head)
    return Ok(proven_val.value)


def extract_snapshot_isolated(
    framework: str,
    include_nonpublic: bool = False,
    timeout: int = 300,
) -> Result[SnapshotDomainError, ProvenSnapshot[EnvelopeValidPhase]]:
    """Extract a framework snapshot in an isolated child subprocess totally into `Result`.

    Args:
        framework: Name of the framework to extract.
        include_nonpublic: Whether to include non-public APIs.
        timeout: Subprocess timeout in seconds.

    Returns:
        `Ok(ProvenSnapshot[EnvelopeValidPhase])` on success, else `Err(SnapshotDomainError)`.
    """
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    src_paths = [p for p in sys.path if p]
    if existing_pypath:
        src_paths.append(existing_pypath)
    env["PYTHONPATH"] = os.pathsep.join(src_paths)

    child_code = (
        "import json, sys; "
        "from ml_switcheroo_ir.typeclasses.outcome import Ok; "
        "from ml_ecosystem_snapshots.api import extract_snapshot; "
        f"res = extract_snapshot({repr(framework)}, include_nonpublic={include_nonpublic}); "
        "sys.stdout.write(json.dumps(res.value.to_wire_dict()) if isinstance(res, Ok) else '')"
    )
    cmd = [sys.executable, "-c", child_code]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return Err(
                IntrospectionFault(
                    module_path=framework,
                    symbol_name=None,
                    reason=f"Isolated extraction returned empty/non-zero for {framework!r} (rc={proc.returncode}).",
                )
            )
        parsed_res = ensure_json_dict(json.loads(proc.stdout), "isolated snapshot")
        if isinstance(parsed_res, Err):
            return parsed_res
        if not parsed_res.value:
            return Err(SnapshotEnvelopeFault(reason=f"Isolated extraction returned empty dict for {framework!r}."))
        proven_val = prove_envelope_valid(parsed_res.value)
        if isinstance(proven_val, Invalid):
            return Err(proven_val.errors.head)
        return Ok(proven_val.value)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, SnapshotError) as exc:
        _LOGGER.warning("Subprocess extraction failed for %s: %s", framework, exc)
        return Err(
            IntrospectionFault(
                module_path=framework,
                symbol_name=None,
                reason=f"Subprocess extraction failed for {framework!r}: {exc}",
            )
        )


def extract_all_snapshots(
    include_nonpublic: bool = False,
    isolated: bool = False,
    output_dir: str | None = None,
    frameworks: typing.Sequence[str] | None = None,
    compress: str | CompressionFormat | None = None,
) -> Validation[NonEmpty[SnapshotDomainError], dict[str, ProvenSnapshot[EnvelopeValidPhase]]]:
    """Extract snapshots across frameworks, accumulating errors into `NonEmpty`.

    When `frameworks` is explicitly provided, every extraction or write failure is
    accumulated into `Invalid(NonEmpty[SnapshotDomainError])`. When `frameworks` is
    `None`, uninstalled optional frameworks are skipped while any `output_dir` write
    failure is accumulated into `Invalid`.

    Args:
        include_nonpublic: Whether to include non-public APIs.
        isolated: Whether to execute each framework extraction in an isolated subprocess.
        output_dir: Optional directory to write extracted snapshots to disk.
        frameworks: Optional explicit subset of framework identifiers to extract.
        compress: Optional compression format when `output_dir` is set.

    Returns:
        `Valid(dict[framework, ProvenSnapshot[EnvelopeValidPhase]])` or
        `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    strict_targets = frameworks is not None
    targets = tuple(frameworks) if frameworks is not None else tuple(FRAMEWORK_COLLECTORS.keys())
    results: dict[str, ProvenSnapshot[EnvelopeValidPhase]] = {}
    errors: list[SnapshotDomainError] = []

    for fw in targets:
        res = (
            extract_snapshot_isolated(fw, include_nonpublic=include_nonpublic)
            if isolated
            else extract_snapshot(fw, include_nonpublic=include_nonpublic)
        )
        if isinstance(res, Err):
            if strict_targets:
                errors.append(res.error)
        else:
            proven = res.value
            if output_dir is not None:
                write_res = write_snapshot(fw, proven, output_dir, compress=compress)
                if isinstance(write_res, Err):
                    errors.append(write_res.error)
                else:
                    results[fw] = proven
            else:
                results[fw] = proven

    return from_error_sequence(errors, results)


__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFn",
    "FRAMEWORK_COLLECTORS",
    "FRAMEWORK_PROVENANCE_METADATA",
    "ProvenanceMetadata",
    "consolidate_aliases",
    "extract_all_snapshots",
    "extract_snapshot",
    "extract_snapshot_isolated",
    "extract_version_with_griffe",
    "get_available_frameworks",
    "get_pkg_version",
    "make_lazy_collector",
    "validate_snapshot_envelope",
    "write_snapshot",
]
