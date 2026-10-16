"""MCP tool module for prefix, substring, and FTS5 symbol search."""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Union

from ml_ecosystem_snapshots.mcp.cache import get_framework_snapshot
from ml_ecosystem_snapshots.mcp.tools.translation import (
    CONCEPT_ALIAS_MAP,
    load_concept_map,
)


def search_apis(
    framework: str,
    query: str,
    limit: int = 10,
    version: Optional[str] = None,
    custom_concept_map: Optional[Union[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> List[str]:
    """Search available APIs in a framework snapshot by keyword or concept with fuzzy fallback.

    Args:
        framework: The framework name.
        query: Search substring, concept name (e.g. 'convolution', 'matmul'), or mnemonic.
        limit: Maximum number of matches to return.
        version: Optional framework version string.
        custom_concept_map: Optional custom concept ontology path or dictionary.

    Returns:
        List of matching API paths.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_server

    active_get_snap = getattr(
        _mcp_server, "get_framework_snapshot", get_framework_snapshot
    )
    active_load_map = getattr(_mcp_server, "load_concept_map", load_concept_map)
    default_alias_map = getattr(_mcp_server, "CONCEPT_ALIAS_MAP", CONCEPT_ALIAS_MAP)

    q = query.lower().strip()
    matches: List[str] = []

    active_concept_map: Dict[str, Dict[str, List[str]]]
    if isinstance(custom_concept_map, str):
        active_concept_map = active_load_map(custom_concept_map)
    elif isinstance(custom_concept_map, dict):
        active_concept_map = custom_concept_map
    else:
        active_concept_map = default_alias_map

    if q in active_concept_map:
        for alias in active_concept_map[q].get(framework, []):
            if alias not in matches:
                matches.append(alias)
        if matches:
            return matches[:limit]

    try:
        from ml_ecosystem_snapshots.index import search_index

        index_results = search_index(
            query, framework=framework, version=version, limit=limit
        )
        if index_results:
            for item in index_results:
                path = item.get("api_path") or item.get("name") or item.get("mnemonic")
                if path and path not in matches:
                    matches.append(path)
            if matches:
                return matches[:limit]
    except Exception:
        pass

    snap = active_get_snap(framework, version=version)

    all_paths: List[str] = []
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            path = (
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            name = item.get("name") or item.get("mnemonic") or ""
            if path:
                all_paths.append(path)
            if q in path.lower() or q in name.lower():
                if path not in matches:
                    matches.append(path)
                if len(matches) >= limit:
                    return matches

    if not matches and all_paths:
        matches = difflib.get_close_matches(query, all_paths, n=limit, cutoff=0.4)

    return matches


__all__ = ["search_apis"]
