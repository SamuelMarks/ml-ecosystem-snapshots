"""Core snapshot symbol signature lookup and substring/FTS5/fuzzy search."""

from __future__ import annotations

import difflib
import logging
import sqlite3
import typing

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.grounding._concept_catalog import DEFAULT_CONCEPT_MAP
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list

_LOGGER = logging.getLogger(__name__)


def lookup_api_signature_in_snapshot(
    framework: str,
    api_path: str,
    snap: typing.Mapping[str, JsonValue],
    version: str | None = None,
    use_index: bool = True,
) -> dict[str, JsonValue] | None:
    """Look up an API signature in the local SQLite index or a resolved snapshot dictionary.

    Args:
        framework: Target framework name (e.g. torch, jax).
        api_path: Canonical API path or mnemonic (e.g. torch.nn.Linear, FADD).
        snap: Categorized framework snapshot dictionary.
        version: Optional framework version string.
        use_index: Whether to query the local SQLite FTS5 index first.

    Returns:
        Dictionary of the matched GhostRef, or `None` if not found.
    """
    if use_index:
        try:
            from ml_ecosystem_snapshots.index import IndexedSymbolRecord, lookup_symbol

            cached_sym: IndexedSymbolRecord | dict[str, JsonValue] | None = lookup_symbol(
                framework, api_path, version=version
            )
            if cached_sym is not None:
                if isinstance(cached_sym, IndexedSymbolRecord):
                    return dict(cached_sym.data)
                return dict(ensure_json_dict(cached_sym))
        except (
            ImportError,
            OSError,
            sqlite3.Error,
            ValueError,
            TypeError,
            RuntimeError,
            LookupError,
            SnapshotError,
        ) as exc:
            _LOGGER.debug("Index symbol lookup fallback for %s.%s: %s", framework, api_path, exc)

    raw_cats = ensure_json_dict(snap.get("categories") or {})
    for _cat, raw_items in raw_cats.items():
        items = ensure_json_list(raw_items)
        for raw_item in items:
            item = ensure_json_dict(raw_item)
            if item.get("api_path") == api_path or item.get("name") == api_path or item.get("mnemonic") == api_path:
                return dict(item)
            raw_aliases = item.get("aliases")
            if isinstance(raw_aliases, list) and api_path in raw_aliases:
                return dict(item)
    return None


def _collect_concept_alias_targets(
    clean_fw: str,
    q_lower: str,
    concept_map: typing.Mapping[str, JsonValue],
) -> list[str]:
    """Collect framework-specific API paths matching a semantic concept alias query.

    Args:
        clean_fw: Lowercased target framework identifier.
        q_lower: Lowercased query string.
        concept_map: Concept alias map dictionary.

    Returns:
        List of matching API paths.
    """
    concept_targets: list[str] = []
    for concept_key, fw_map in concept_map.items():
        if concept_key != "_parameter_translations" and isinstance(fw_map, dict):
            if q_lower == concept_key or (len(q_lower) >= 3 and q_lower in concept_key):
                raw_targets = ensure_json_list(fw_map.get(clean_fw, []))
                concept_targets.extend([str(t) for t in raw_targets])
    return concept_targets


def _append_fts5_matches(
    matches: list[str],
    query: str,
    framework: str,
    version: str | None,
    limit: int,
) -> bool:
    """Query the local SQLite FTS5 index and append unique matching API paths.

    Args:
        matches: Mutable list of matched API paths accumulated so far.
        query: Search term or prefix.
        framework: Target framework name.
        version: Optional framework version string.
        limit: Maximum number of matches to collect.

    Returns:
        True if `matches` reached `limit`, False otherwise.
    """
    try:
        from ml_ecosystem_snapshots.index import search_index

        index_results = search_index(query, framework=framework, version=version, limit=limit)
        if index_results:
            for item in index_results:
                path = item.api_path or item.data.get("name") or item.data.get("mnemonic")
                if path and str(path) not in matches:
                    matches.append(str(path))
                    if len(matches) >= limit:
                        return True
    except (
        ImportError,
        OSError,
        sqlite3.Error,
        ValueError,
        TypeError,
        RuntimeError,
        LookupError,
        SnapshotError,
    ) as exc:
        _LOGGER.debug("FTS5 index search fallback for %s (%s): %s", framework, query, exc)
    return False


def _scan_snapshot_categories(
    snap: typing.Mapping[str, JsonValue],
    q_lower: str,
    matches: list[str],
    limit: int,
) -> list[str]:
    """Scan categorized snapshot entries for substring matches and collect all paths.

    Args:
        snap: Resolved framework snapshot dictionary.
        q_lower: Lowercased search query string.
        matches: Mutable list of matched API paths accumulated so far.
        limit: Maximum number of matches to collect.

    Returns:
        List of all discovered API paths in the snapshot (for fuzzy fallback).
    """
    all_paths: list[str] = []
    raw_cats = ensure_json_dict(snap.get("categories") or {})
    for _cat, raw_items in raw_cats.items():
        items = ensure_json_list(raw_items)
        for item_val in items:
            raw_item = ensure_json_dict(item_val)
            raw_path = raw_item.get("api_path") or raw_item.get("name") or raw_item.get("mnemonic") or ""
            path_str = str(raw_path)
            if path_str:
                all_paths.append(path_str)
            if q_lower in path_str.lower() and path_str not in matches:
                matches.append(path_str)
                if len(matches) >= limit:
                    return all_paths
    return all_paths


def search_snapshot_apis(
    framework: str,
    query: str,
    snap_loader: typing.Callable[[], typing.Mapping[str, JsonValue]],
    limit: int = 10,
    version: str | None = None,
    concept_map: typing.Mapping[str, JsonValue] | None = None,
) -> list[str]:
    """Search APIs or instructions in a framework snapshot matching a query.

    Args:
        framework: Target framework name.
        query: Search term or prefix.
        snap_loader: Lazy callback loading the target framework snapshot dictionary.
        limit: Maximum number of matches to return.
        version: Optional framework version string.
        concept_map: Optional concept alias mapping override.

    Returns:
        List of matching API paths or mnemonics.
    """
    clean_fw = framework.lower().strip()
    q_lower = query.lower().strip()
    active_concepts = concept_map if concept_map is not None else DEFAULT_CONCEPT_MAP

    concept_targets = _collect_concept_alias_targets(clean_fw, q_lower, active_concepts)
    matches: list[str] = []
    for ct in concept_targets:
        if ct not in matches:
            matches.append(ct)
            if len(matches) >= limit:
                return matches

    if _append_fts5_matches(matches, query, framework, version, limit):
        return matches

    snap = snap_loader()
    all_paths = _scan_snapshot_categories(snap, q_lower, matches, limit)
    if len(matches) >= limit:
        return matches

    if not matches and all_paths:
        matches = difflib.get_close_matches(query, all_paths, n=limit, cutoff=0.4)

    return matches


__all__ = ["lookup_api_signature_in_snapshot", "search_snapshot_apis"]
