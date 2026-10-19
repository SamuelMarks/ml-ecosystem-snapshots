"""MCP tool module for prefix, substring, and FTS5 symbol search."""

from __future__ import annotations

import difflib

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.tools.translation import CONCEPT_ALIAS_MAP


def search_apis(
    framework: str,
    query: str,
    limit: int = 10,
    version: str | None = None,
    repository: SnapshotRepository | None = None,
) -> list[str]:
    """Search available APIs or instructions in a framework snapshot matching a query.

    Supports both substring matching and semantic concept alias resolution
    (e.g., querying 'normalization', 'attention', or 'wgmma_mma_async').

    Args:
        framework: Target framework name.
        query: Search term or prefix.
        limit: Maximum number of matches to return.
        version: Optional framework version string.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        List of matching API paths or mnemonics.
    """
    clean_fw = framework.lower().strip()
    q_lower = query.lower().strip()

    concept_targets: list[str] = []
    for concept_key, fw_map in CONCEPT_ALIAS_MAP.items():
        if concept_key != "_parameter_translations" and isinstance(fw_map, dict):
            if q_lower == concept_key or (len(q_lower) >= 3 and q_lower in concept_key):
                raw_targets = fw_map.get(clean_fw, [])
                assert isinstance(raw_targets, list)
                concept_targets.extend([str(t) for t in raw_targets])

    matches: list[str] = []
    for ct in concept_targets:
        if ct not in matches:
            matches.append(ct)
            if len(matches) >= limit:
                return matches

    try:
        from ml_ecosystem_snapshots.index import search_index

        index_results = search_index(
            query, framework=framework, version=version, limit=limit
        )
        if index_results:
            for item in index_results:
                path = item.get("api_path") or item.get("name") or item.get("mnemonic")
                if path and str(path) not in matches:
                    matches.append(str(path))
                    if len(matches) >= limit:
                        return matches
    except Exception:
        pass

    snap = resolve_snapshot(framework, version=version, repository=repository)
    all_paths: list[str] = []
    raw_cats = snap.get("categories") or {}
    assert isinstance(raw_cats, dict)
    for _cat, items in raw_cats.items():
        assert isinstance(items, list)
        for raw_item in items:
            assert isinstance(raw_item, dict)
            raw_path = (
                raw_item.get("api_path")
                or raw_item.get("name")
                or raw_item.get("mnemonic")
                or ""
            )
            path_str = str(raw_path)
            if path_str:
                all_paths.append(path_str)
            if q_lower in path_str.lower() and path_str not in matches:
                matches.append(path_str)
                if len(matches) >= limit:
                    return matches

    if not matches and all_paths:
        matches = difflib.get_close_matches(query, all_paths, n=limit, cutoff=0.4)

    return matches


__all__ = ["search_apis"]
