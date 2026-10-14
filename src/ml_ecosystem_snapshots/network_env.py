"""Environment, network, and JSON streaming utilities for ml_ecosystem_snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


def is_offline_mode() -> bool:
    """Check if offline mode is explicitly enabled via environment variable.

    Returns:
        True if ML_SNAPSHOTS_OFFLINE or ML_FRAMEWORK_SNAPSHOTS_OFFLINE is set to a truthy value.
    """
    for var in ("ML_SNAPSHOTS_OFFLINE", "ML_FRAMEWORK_SNAPSHOTS_OFFLINE"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
    return False


def get_custom_snapshots_paths() -> List[str]:
    """Retrieve custom snapshot directory paths configured via environment variables.

    Returns:
        List of directory paths specified in ML_SNAPSHOTS_PATH or ML_FRAMEWORK_SNAPSHOTS_PATH.
    """
    paths: List[str] = []
    for var in ("ML_SNAPSHOTS_PATH", "ML_FRAMEWORK_SNAPSHOTS_PATH"):
        raw = os.environ.get(var)
        if raw:
            for p in raw.split(os.pathsep):
                p_clean = p.strip()
                if p_clean and os.path.isdir(p_clean) and p_clean not in paths:
                    paths.append(p_clean)
    return paths


def load_json_data(file_path: Union[str, Path]) -> Any:
    """Load JSON data from a plain .json, compressed .json.gz, or .json.zst file.

    Args:
        file_path: Absolute or relative path to the snapshot file.

    Returns:
        Deserialized JSON object (dict or list).
    """
    path_str = str(file_path)
    if path_str.endswith(".gz"):
        import gzip

        with gzip.open(path_str, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif path_str.endswith(".zst"):
        import importlib

        try:
            zstd = importlib.import_module("zstandard")
            dctx = getattr(zstd, "ZstdDecompressor")()
            with open(path_str, "rb") as bf:
                decompressed: bytes = dctx.decompress(bf.read())
            return json.loads(decompressed.decode("utf-8"))
        except Exception:
            with open(path_str, "rb") as bf:
                return json.loads(bf.read().decode("utf-8"))
    else:
        with open(path_str, "r", encoding="utf-8") as f:
            return json.load(f)


def stream_snapshot_items(file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Iteratively yield items from a snapshot manifest to minimize memory overhead.

    Args:
        file_path: Absolute or relative path to the snapshot file.

    Yields:
        Individual item dictionaries parsed from categories or items lists.
    """
    data = load_json_data(file_path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            for _cat, items in categories.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            yield item
        items_list = data.get("items", [])
        if isinstance(items_list, list):
            for item in items_list:
                if isinstance(item, dict):
                    yield item


def fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch content from a URL with timeout.

    Args:
        url: The URL to fetch.
        timeout: Timeout in seconds.

    Returns:
        The decoded string content, or None on failure.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return str(response.read().decode("utf-8"))
    except urllib.error.URLError:
        return None


__all__ = [
    "fetch_url",
    "get_custom_snapshots_paths",
    "is_offline_mode",
    "load_json_data",
    "stream_snapshot_items",
]
