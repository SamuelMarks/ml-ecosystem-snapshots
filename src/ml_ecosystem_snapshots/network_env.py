"""Environment, network, and JSON streaming utilities for ml_ecosystem_snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
import typing

from ml_ecosystem_snapshots.exceptions import OfflineModeNetworkError
from ml_ecosystem_snapshots.models import JsonValue


def is_offline_mode() -> bool:
    """Check if offline mode is explicitly enabled via environment variable.

    Returns:
        True if ML_SNAPSHOTS_OFFLINE is set to a truthy value.
    """
    val = os.environ.get("ML_SNAPSHOTS_OFFLINE", "").strip().lower()
    return val in ("1", "true", "yes", "on", "enable", "enabled")


def get_custom_snapshots_paths() -> list[str]:
    """Retrieve custom snapshot directory paths configured via environment variables.

    Returns:
        List of directory paths specified in ML_SNAPSHOTS_PATH.
    """
    paths: list[str] = []
    raw = os.environ.get("ML_SNAPSHOTS_PATH")
    if raw:
        for p in raw.split(os.pathsep):
            p_clean = p.strip()
            if p_clean and os.path.isdir(p_clean) and p_clean not in paths:
                paths.append(p_clean)
    return paths


def load_json_data(file_path: str | Path) -> JsonValue:
    """Load JSON data from a plain .json, compressed .json.gz, or .json.zst file.

    Args:
        file_path: Absolute or relative path to the snapshot file.

    Returns:
        Deserialized JSON object (dict, list, or primitive).
    """
    path_str = str(file_path)
    if path_str.endswith(".gz"):
        import gzip

        with gzip.open(path_str, "rt", encoding="utf-8") as f:
            res_gz: JsonValue = json.load(f)
            return res_gz
    elif path_str.endswith(".zst"):
        import importlib

        try:
            zstd = importlib.import_module("zstandard")
            dctx = getattr(zstd, "ZstdDecompressor")()
            with open(path_str, "rb") as bf:
                decompressed: bytes = dctx.decompress(bf.read())
            res_zst: JsonValue = json.loads(decompressed.decode("utf-8"))
            return res_zst
        except Exception:
            with open(path_str, "rb") as bf:
                res_fallback: JsonValue = json.loads(bf.read().decode("utf-8"))
                return res_fallback
    else:
        with open(path_str, "r", encoding="utf-8") as f:
            res_plain: JsonValue = json.load(f)
            return res_plain


def stream_snapshot_items(
    file_path: str | Path,
) -> typing.Iterator[dict[str, JsonValue]]:
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


def fetch_url(url: str, timeout: int = 10, strict_offline: bool = False) -> str | None:
    """Fetch content from a URL with timeout.

    Args:
        url: The URL to fetch.
        timeout: Timeout in seconds.
        strict_offline: Whether to raise `OfflineModeNetworkError` if offline mode is active.

    Returns:
        The decoded string content, or None on failure.

    Raises:
        OfflineModeNetworkError: If `strict_offline` is True and `is_offline_mode()` is True.
    """
    import urllib.error
    import urllib.request

    if is_offline_mode():
        if strict_offline:
            raise OfflineModeNetworkError(
                f"Outbound network request to '{url}' blocked in offline mode."
            )
        return None

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
