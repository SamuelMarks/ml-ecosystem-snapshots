"""Environment, network, and JSON streaming utilities for ml_ecosystem_snapshots."""

from __future__ import annotations

import json
import os
import typing
from pathlib import Path

from ml_ecosystem_snapshots.exceptions import OfflineModeNetworkError, SnapshotError
from ml_ecosystem_snapshots.models import JsonValue

ZSTD_MAGIC_HEADER: bytes = b"\x28\xb5\x2f\xfd"


@typing.runtime_checkable
class ZstdCompressorProtocol(typing.Protocol):
    """Protocol for a Zstandard compressor instance."""

    def compress(self, data: bytes) -> bytes:
        """Compress raw bytes.

        Args:
            data: Uncompressed byte buffer.

        Returns:
            Compressed byte buffer.
        """


@typing.runtime_checkable
class ZstdDecompressorProtocol(typing.Protocol):
    """Protocol for a Zstandard decompressor instance."""

    def decompress(self, data: bytes) -> bytes:
        """Decompress zstd-compressed bytes.

        Args:
            data: Compressed byte buffer.

        Returns:
            Decompressed byte buffer.
        """


@typing.runtime_checkable
class ZstdModuleProtocol(typing.Protocol):
    """Protocol for the `zstandard` compression module."""

    def ZstdCompressor(self) -> ZstdCompressorProtocol:
        """Instantiate a Zstandard compressor.

        Returns:
            Compressor instance.
        """

    def ZstdDecompressor(self) -> ZstdDecompressorProtocol:
        """Instantiate a Zstandard decompressor.

        Returns:
            Decompressor instance.
        """


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


def load_zstandard_module() -> ZstdModuleProtocol:
    """Import and validate the optional `zstandard` compression module.

    Returns:
        Validated `ZstdModuleProtocol` module object.

    Raises:
        SnapshotError: If `zstandard` is not installed or lacks compressor/decompressor classes.
    """
    import importlib

    try:
        mod = importlib.import_module("zstandard")
    except ImportError as exc:
        raise SnapshotError("zstandard package is required for .zst snapshot compression.") from exc
    if not isinstance(mod, ZstdModuleProtocol):
        raise SnapshotError("Installed 'zstandard' module does not satisfy ZstdModuleProtocol.")
    return mod


def load_json_data(file_path: str | Path) -> JsonValue:
    """Load JSON data from a plain .json, compressed .json.gz, or .json.zst file.

    Args:
        file_path: Absolute or relative path to the snapshot file.

    Returns:
        Deserialized JSON object (dict, list, or primitive).

    Raises:
        SnapshotError: If a binary `.zst` frame cannot be decompressed due to missing `zstandard`.
    """
    path_str = str(file_path)
    if path_str.endswith(".gz"):
        import gzip

        with gzip.open(path_str, "rt", encoding="utf-8") as f:
            res_gz: JsonValue = json.load(f)
            return res_gz
    if path_str.endswith(".zst"):
        with open(path_str, "rb") as bf:
            raw_bytes = bf.read()
        if not raw_bytes.startswith(ZSTD_MAGIC_HEADER):
            try:
                plain_zst: JsonValue = json.loads(raw_bytes.decode("utf-8"))
                return plain_zst
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        zstd = load_zstandard_module()
        decompressed: bytes = zstd.ZstdDecompressor().decompress(raw_bytes)
        res_zst: JsonValue = json.loads(decompressed.decode("utf-8"))
        return res_zst
    with open(path_str, encoding="utf-8") as f:
        res_plain: JsonValue = json.load(f)
        return res_plain


def _iter_dict_items(items: JsonValue) -> typing.Iterator[dict[str, JsonValue]]:
    """Yield dictionary elements from a JSON list value.

    Args:
        items: JSON value expected to be a list of dictionaries.

    Yields:
        Dictionary items contained in `items`.
    """
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                yield item


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
        yield from _iter_dict_items(data)
    elif isinstance(data, dict):
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            for _cat, items in categories.items():
                yield from _iter_dict_items(items)
        yield from _iter_dict_items(data.get("items", []))


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
            raise OfflineModeNetworkError(f"Outbound network request to '{url}' blocked in offline mode.")
        return None

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return str(response.read().decode("utf-8"))
    except urllib.error.URLError:
        return None


__all__ = [
    "ZSTD_MAGIC_HEADER",
    "ZstdCompressorProtocol",
    "ZstdDecompressorProtocol",
    "ZstdModuleProtocol",
    "fetch_url",
    "get_custom_snapshots_paths",
    "is_offline_mode",
    "load_json_data",
    "load_zstandard_module",
    "stream_snapshot_items",
]
