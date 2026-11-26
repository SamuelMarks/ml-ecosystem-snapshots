"""Environment, network, compression ADTs, and JSON streaming utilities."""

from __future__ import annotations

import enum
import gzip
import json
import os
import types
import typing
from pathlib import Path

from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.errors import (
    OfflineNetworkFault,
    SnapshotDomainError,
    SnapshotEnvelopeFault,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue

ZSTD_MAGIC_HEADER: bytes = b"\x28\xb5\x2f\xfd"


class CompressionFormat(str, enum.Enum):
    """Closed enumeration of supported snapshot compression codecs."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


def parse_compression_format(
    raw: str | CompressionFormat | None,
) -> Result[SnapshotDomainError, CompressionFormat]:
    """Parse an optional compression label into a closed `CompressionFormat` variant.

    Args:
        raw: Optional compression format string (`'none'`, `'gzip'`, `'gz'`, `'zstd'`, `'zst'`) or enum.

    Returns:
        `Ok(CompressionFormat)` if recognized, else `Err(SnapshotEnvelopeFault)`.
    """
    if raw is None:
        return Ok(CompressionFormat.NONE)
    if isinstance(raw, CompressionFormat):
        return Ok(raw)
    cleaned = raw.strip().lower()
    if cleaned in ("", "none"):
        return Ok(CompressionFormat.NONE)
    if cleaned in ("gzip", "gz"):
        return Ok(CompressionFormat.GZIP)
    if cleaned in ("zstd", "zst"):
        return Ok(CompressionFormat.ZSTD)
    return Err(SnapshotEnvelopeFault(reason=f"Unsupported compression format: {raw!r}."))


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
    """Protocol for the `zstandard` compression module.

    Attributes:
        ZstdCompressor: Zero-argument callable constructing a `ZstdCompressorProtocol`.
        ZstdDecompressor: Zero-argument callable constructing a `ZstdDecompressorProtocol`.
    """

    ZstdCompressor: typing.Callable[[], ZstdCompressorProtocol]
    ZstdDecompressor: typing.Callable[[], ZstdDecompressorProtocol]


def is_offline_mode() -> bool:
    """Check if offline mode is explicitly enabled via environment variable.

    Returns:
        True if `ML_SNAPSHOTS_OFFLINE` is set to a truthy value.
    """
    val = os.environ.get("ML_SNAPSHOTS_OFFLINE", "").strip().lower()
    return val in ("1", "true", "yes", "on", "enable", "enabled")


def get_custom_snapshots_paths() -> tuple[str, ...]:
    """Retrieve custom snapshot directory paths configured via environment variables.

    Returns:
        Immutable tuple of directory paths specified in `ML_SNAPSHOTS_PATH`.
    """
    paths: list[str] = []
    raw = os.environ.get("ML_SNAPSHOTS_PATH")
    if raw:
        for p in raw.split(os.pathsep):
            p_clean = p.strip()
            if p_clean and os.path.isdir(p_clean) and p_clean not in paths:
                paths.append(p_clean)
    return tuple(paths)


def _import_optional_module(module_name: str) -> types.ModuleType | ScalarValue:
    """Import an optional runtime module by name.

    Args:
        module_name: Dotted module name to import.

    Returns:
        Imported runtime module object (or custom `sys.modules` entry).
    """
    import importlib

    return importlib.import_module(module_name)


def load_zstandard_module() -> Result[SnapshotDomainError, ZstdModuleProtocol]:
    """Import and validate the optional `zstandard` compression module without raising.

    Returns:
        `Ok(ZstdModuleProtocol)` if available and valid, else `Err(SnapshotEnvelopeFault)`.
    """
    try:
        mod = _import_optional_module("zstandard")
    except ImportError:
        return Err(SnapshotEnvelopeFault(reason="zstandard package is required for .zst snapshot compression."))
    if not isinstance(mod, ZstdModuleProtocol):
        return Err(SnapshotEnvelopeFault(reason="Installed 'zstandard' module does not satisfy ZstdModuleProtocol."))
    return Ok(mod)


def _load_zst_json_data(path_str: str) -> Result[SnapshotDomainError, JsonValue]:
    """Load and decode a `.zst` JSON file, supporting plain-UTF8 fallback when unframed.

    Args:
        path_str: Path to the `.zst` file.

    Returns:
        `Ok(JsonValue)` on success, else `Err(SnapshotDomainError)`.
    """
    with open(path_str, "rb") as bf:
        raw_bytes = bf.read()
    if not raw_bytes.startswith(ZSTD_MAGIC_HEADER):
        try:
            plain_zst: JsonValue = json.loads(raw_bytes.decode("utf-8"))
            return Ok(plain_zst)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    zstd_res = load_zstandard_module()
    if isinstance(zstd_res, Err):
        return zstd_res
    decompressed: bytes = zstd_res.value.ZstdDecompressor().decompress(raw_bytes)
    res_zst: JsonValue = json.loads(decompressed.decode("utf-8"))
    return Ok(res_zst)


def load_json_data(file_path: str | Path) -> Result[SnapshotDomainError, JsonValue]:
    """Load JSON data totally from a plain `.json`, `.json.gz`, or `.json.zst` file.

    Args:
        file_path: Absolute or relative path to the snapshot file.

    Returns:
        `Ok(JsonValue)` on success, or `Err(SnapshotDomainError)` on I/O, compression, or decode failure.
    """
    path_str = str(file_path)
    try:
        if path_str.endswith(".gz"):
            with gzip.open(path_str, "rt", encoding="utf-8") as f:
                res_gz: JsonValue = json.load(f)
                return Ok(res_gz)
        if path_str.endswith(".zst"):
            return _load_zst_json_data(path_str)
        with open(path_str, encoding="utf-8") as f:
            res_plain: JsonValue = json.load(f)
            return Ok(res_plain)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Failed to load JSON data from '{path_str}': {exc}"))


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
        Individual item dictionaries parsed from `categories` or `items` lists.
    """
    data_res = load_json_data(file_path)
    if isinstance(data_res, Err):
        return
    data = data_res.value
    if isinstance(data, list):
        yield from _iter_dict_items(data)
    elif isinstance(data, dict):
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            for _cat, items in categories.items():
                yield from _iter_dict_items(items)
        yield from _iter_dict_items(data.get("items", []))


def fetch_url(url: str, timeout: int = 10, strict_offline: bool = False) -> Result[SnapshotDomainError, str]:
    """Fetch UTF-8 content from a URL totally into a `Result[SnapshotDomainError, str]`.

    Args:
        url: Target HTTP/HTTPS URL to fetch.
        timeout: Timeout in seconds.
        strict_offline: Preserved flag; offline mode always yields `Err(OfflineNetworkFault)`.

    Returns:
        `Ok(decoded_text)` on success, `Err(OfflineNetworkFault)` when offline mode is active
        or when network/decode I/O fails.
    """
    import urllib.error
    import urllib.request

    del strict_offline
    if is_offline_mode():
        return Err(OfflineNetworkFault(url=url, operation="offline_blocked"))

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return Ok(str(response.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, UnicodeDecodeError) as exc:
        return Err(OfflineNetworkFault(url=url, operation=f"fetch_failed: {exc}"))


__all__ = [
    "CompressionFormat",
    "ZSTD_MAGIC_HEADER",
    "ZstdCompressorProtocol",
    "ZstdDecompressorProtocol",
    "ZstdModuleProtocol",
    "fetch_url",
    "get_custom_snapshots_paths",
    "is_offline_mode",
    "load_json_data",
    "load_zstandard_module",
    "parse_compression_format",
    "stream_snapshot_items",
]
