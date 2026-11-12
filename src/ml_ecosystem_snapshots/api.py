"""ML Framework Snapshots SDK.

Provides a programmatic interface for capturing and saving API signatures
from various machine learning frameworks.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import os
import subprocess
import sys

import typing_extensions
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostPythonRef,
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import (
    SCHEMA_VERSION,
    JsonValue,
    SnapshotEnvelope,
    SnapshotSymbolModel,
    ensure_json_dict,
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
        module_name: Dotted module name to inspect (e.g. 'keras', 'chex', 'optax').

    Returns:
        Extracted version string with quotes stripped, or `None` if unavailable.
    """
    return _resolve_version_with_griffe(module_name)


_extract_version_with_griffe = extract_version_with_griffe


HashableKey: typing_extensions.TypeAlias = 'str | int | float | bool | None | tuple["HashableKey", ...]'


def get_pkg_version(package_name: str) -> str:
    """Safely retrieves the installed version of a python package.

    Args:
        package_name: The name of the pip package.

    Returns:
        The version string, or "unknown" if not installed.
    """
    return _resolve_pkg_version(
        package_name,
        base_file=__file__,
        extract_griffe_fn=extract_version_with_griffe,
    )


def _domain_meta_to_key(meta: JsonValue) -> HashableKey:
    """Convert domain metadata into a deterministic, hashable key.

    Args:
        meta: Domain metadata value (dict, list, primitive, etc.).

    Returns:
        A hashable representation of the metadata.
    """
    if meta is None:
        return None
    if isinstance(meta, dict):
        return tuple(sorted((str(k), _domain_meta_to_key(v)) for k, v in meta.items()))
    if isinstance(meta, (list, tuple, set)):
        return tuple(_domain_meta_to_key(x) for x in meta)
    return str(meta)


def consolidate_aliases(refs: list[GhostRef]) -> list[GhostRef]:
    """Consolidate identical GhostRefs into a single reference with aliases.

    Args:
        refs: A list of GhostRef objects.

    Returns:
        A consolidated list of GhostRef objects.
    """
    consolidated: dict[HashableKey, GhostRef] = {}
    for ref in refs:
        param_sigs = tuple(
            (
                p.name,
                str(p.kind),
                str(p.default) if p.default is not None else None,
                p.annotation,
                str(p.direction if isinstance(p, ExtendedGhostParam) else None),
                str(p.role if isinstance(p, ExtendedGhostParam) else None),
                tuple(str(dt) for dt in ((p.allowed_dtypes or ()) if isinstance(p, ExtendedGhostParam) else ())),
                str(p.rank_constraint if isinstance(p, ExtendedGhostParam) else None),
            )
            for p in ref.params
        )
        returns = ref.returns if isinstance(ref, ExtendedGhostRef) else None
        returns_sig = tuple((str(r.name), str(r.type)) for r in returns) if returns else None
        domain_type = str(
            ref.domain_type
            if isinstance(
                ref,
                (
                    GhostPythonRef,
                    GhostMlirRef,
                    GhostOperationRef,
                    GhostIsaRef,
                    GhostInstructionRef,
                ),
            )
            else None
        )
        domain_metadata_key = _domain_meta_to_key(ref.domain_metadata if isinstance(ref, ExtendedGhostRef) else None)

        key: HashableKey = (
            ref.name,
            ref.kind,
            domain_type,
            param_sigs,
            returns_sig,
            domain_metadata_key,
            ref.docstring,
        )

        if key not in consolidated:
            consolidated[key] = ref
        else:
            existing = consolidated[key]
            if ref.aliases is None:
                ref.aliases = []
            if existing.aliases is None:
                existing.aliases = []
            if len(ref.api_path) < len(existing.api_path):
                ref.aliases.append(existing.api_path)
                ref.aliases.extend(existing.aliases)
                ref.is_public = ref.is_public or existing.is_public
                consolidated[key] = ref
            else:
                existing.aliases.append(ref.api_path)
                existing.aliases.extend(ref.aliases)
                existing.is_public = existing.is_public or ref.is_public

    for ref in consolidated.values():
        if ref.aliases is not None:
            ref.aliases = sorted(list(set(ref.aliases)))
        else:
            ref.aliases = []

    return list(consolidated.values())


_consolidate_aliases = consolidate_aliases


def _serialize_refs_to_models(refs: list[GhostRef]) -> list[SnapshotSymbolModel]:
    """Consolidate aliases, sort, and validate `GhostRef` items into `SnapshotSymbolModel` records.

    Args:
        refs: Collected `GhostRef` list.

    Returns:
        Sorted list of validated `SnapshotSymbolModel` instances.
    """
    consolidated = _consolidate_aliases(refs)
    consolidated.sort(key=lambda x: x.name)
    dumped_refs: list[SnapshotSymbolModel] = []
    for r in consolidated:
        d: dict[str, JsonValue] = r.model_dump(exclude_unset=True)
        kwargs_list: list[JsonValue] = [
            p.name for p in r.params if p.kind in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD", "VAR_KEYWORD")
        ]
        d["kwargs"] = kwargs_list
        dumped_refs.append(SnapshotSymbolModel.model_validate(d))
    return dumped_refs


def _serialize_category_refs(
    collect_func: CollectorFn,
    cat: SemanticTier,
    include_nonpublic: bool,
) -> tuple[str, list[SnapshotSymbolModel]]:
    """Collect, consolidate, and validate all API symbols for a semantic tier.

    Args:
        collect_func: Framework category collector callable.
        cat: The `SemanticTier` enum category to extract.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A tuple of `(category_name, validated_snapshot_symbol_models)`.
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
    return cat.value, []


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
                    m.model_dump(exclude_unset=True) for m in symbol_models
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
                categories_map[cat_val] = [m.model_dump(exclude_unset=True) for m in symbol_models]
    return categories_map


def extract_snapshot(framework_name: str, include_nonpublic: bool = False) -> dict[str, JsonValue]:
    """Extract a validated snapshot dictionary for a specific framework.

    Args:
        framework_name: The framework identifier (e.g., "torch", "jax").
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A dictionary serialized from a validated `SnapshotEnvelope` containing the version and category signatures.
        Returns an empty dict if the framework is unknown or not installed.
    """
    if framework_name not in FRAMEWORK_COLLECTORS:
        return {}

    version = get_pkg_version(framework_name)
    if version == "unknown":
        return {}

    categories_map = _collect_framework_categories(framework_name, include_nonpublic)
    if not categories_map:
        return {}

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
    return validated_envelope.model_dump(exclude_none=True)


def extract_snapshot_isolated(
    framework: str, include_nonpublic: bool = False, timeout: int = 300
) -> dict[str, JsonValue]:
    """Extract a framework snapshot within an isolated child subprocess.

    Prevents CUDA/Metal memory pre-allocation locks, framework-level C-extension
    conflicts, and memory leaks from polluting the parent process.

    Args:
        framework: Name of the framework to extract.
        include_nonpublic: Whether to include non-public APIs.
        timeout: Subprocess timeout in seconds.

    Returns:
        The extracted snapshot dictionary, or empty dict on failure.
    """
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    src_paths = [p for p in sys.path if p]
    if existing_pypath:
        src_paths.append(existing_pypath)
    env["PYTHONPATH"] = os.pathsep.join(src_paths)

    cmd = [
        sys.executable,
        "-c",
        (
            f"import json, sys; from ml_ecosystem_snapshots.api import extract_snapshot; snap = extract_snapshot({repr(framework)}, include_nonpublic={include_nonpublic}); sys.stdout.write(json.dumps(snap))"
        ),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parsed_snap = ensure_json_dict(json.loads(proc.stdout), "isolated snapshot")
            if not parsed_snap:
                return {}
            return validate_snapshot_envelope(parsed_snap).model_dump(exclude_none=True)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, SnapshotError) as e:
        _LOGGER.warning("Subprocess extraction failed for %s: %s", framework, e)

    return {}


def extract_all_snapshots(include_nonpublic: bool = False, isolated: bool = False) -> dict[str, dict[str, JsonValue]]:
    """Extract snapshots for all supported and installed frameworks.

    Args:
        include_nonpublic: Whether to include non-public APIs.
        isolated: Whether to execute each framework extraction in an isolated child subprocess.

    Returns:
        A dictionary mapping framework identifiers to their snapshot data.
    """
    results: dict[str, dict[str, JsonValue]] = {}
    for fw in FRAMEWORK_COLLECTORS:
        if isolated:
            data = extract_snapshot_isolated(fw, include_nonpublic)
        else:
            data = extract_snapshot(fw, include_nonpublic)
        if data:
            results[fw] = data
    return results


__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFn",
    "FRAMEWORK_COLLECTORS",
    "FRAMEWORK_PROVENANCE_METADATA",
    "HashableKey",
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
