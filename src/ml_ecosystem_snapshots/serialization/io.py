"""Total snapshot envelope validation, serialization, and streaming I/O utilities."""

from __future__ import annotations

import datetime
import gzip
import json
import os
import typing
from pathlib import Path

import typing_extensions
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.errors import SnapshotDomainError, SnapshotEnvelopeFault
from ml_ecosystem_snapshots.models import (
    SCHEMA_VERSION,
    JsonValue,
    ScalarValue,
    SnapshotEnvelope,
    WireSerializableProtocol,
)
from ml_ecosystem_snapshots.network_env import (
    CompressionFormat,
    load_json_data,
    load_zstandard_module,
    parse_compression_format,
    stream_snapshot_items,
)

_NL: typing.Final[str] = chr(10)
_NL_BYTES: typing.Final[bytes] = _NL.encode("utf-8")


def _to_serializable_dict(
    snapshot_data: SnapshotEnvelope | WireSerializableProtocol | typing.Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Project an envelope, `ProvenSnapshot`, or mapping into a JSON-serializable dict.

    Args:
        snapshot_data: Snapshot envelope, `WireSerializableProtocol`, or raw mapping.

    Returns:
        Fresh JSON-serializable dictionary.
    """
    if isinstance(snapshot_data, SnapshotEnvelope):
        dumped: dict[str, JsonValue] = snapshot_data.model_dump()
        return dumped
    if isinstance(snapshot_data, WireSerializableProtocol):
        return snapshot_data.to_wire_dict()
    return dict(snapshot_data)


def write_snapshot(
    framework_name: str,
    snapshot_data: SnapshotEnvelope | WireSerializableProtocol | typing.Mapping[str, JsonValue],
    output_dir: str,
    compress: str | CompressionFormat | None = None,
) -> Result[SnapshotDomainError, str]:
    """Write a snapshot dictionary or envelope to disk totally into `Result[SnapshotDomainError, str]`.

    Args:
        framework_name: Framework identifier used in the output filename.
        snapshot_data: `SnapshotEnvelope`, `ProvenSnapshot`, or raw snapshot mapping.
        output_dir: Directory path where the output file should be written.
        compress: Optional compression algorithm (`'none'`, `'gzip'`, `'gz'`, `'zstd'`, `'zst'`).

    Returns:
        `Ok(written_file_path)` on success, or `Err(SnapshotDomainError)` on invalid compression or I/O failure.
    """
    fmt_res = parse_compression_format(compress)
    if isinstance(fmt_res, Err):
        return fmt_res
    fmt = fmt_res.value

    snap_dict = _to_serializable_dict(snapshot_data)
    raw_version = snap_dict.get("version")
    version = str(raw_version) if raw_version is not None else "unknown"
    safe_ver = version.replace("+", "_").replace(" ", "_")

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if fmt is CompressionFormat.NONE:
            file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json"))
            with open(file_path, "w", encoding="utf-8", newline=_NL) as f:
                json.dump(snap_dict, f, indent=2, sort_keys=True)
                f.write(_NL)
            return Ok(str(file_path))
        if fmt is CompressionFormat.GZIP:
            file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.gz"))
            with gzip.open(file_path, "wt", encoding="utf-8", newline=_NL) as f:
                json.dump(snap_dict, f, indent=2, sort_keys=True)
                f.write(_NL)
            return Ok(str(file_path))
        if fmt is CompressionFormat.ZSTD:
            file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.zst"))
            raw_bytes = json.dumps(snap_dict, indent=2, sort_keys=True).encode("utf-8") + _NL_BYTES
            zstd_res = load_zstandard_module()
            if isinstance(zstd_res, Err):
                return zstd_res
            compressed: bytes = zstd_res.value.ZstdCompressor().compress(raw_bytes)
            with open(file_path, "wb") as f:
                f.write(compressed)
            return Ok(str(file_path))
        typing_extensions.assert_never(fmt)
    except (OSError, TypeError, ValueError) as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Failed to write snapshot for '{framework_name}': {exc}"))


def validate_snapshot_envelope(
    snapshot_dict: SnapshotEnvelope | typing.Mapping[str, JsonValue] | ScalarValue,
) -> Result[SnapshotEnvelopeFault, SnapshotEnvelope]:
    """Validate and normalize a snapshot payload totally into `Result[SnapshotEnvelopeFault, SnapshotEnvelope]`.

    Args:
        snapshot_dict: Candidate `SnapshotEnvelope` or raw snapshot dictionary.

    Returns:
        `Ok(SnapshotEnvelope)` if valid, else `Err(SnapshotEnvelopeFault)`.
    """
    if isinstance(snapshot_dict, SnapshotEnvelope):
        return Ok(snapshot_dict.model_copy(deep=True))
    if not isinstance(snapshot_dict, typing.Mapping):
        return Err(SnapshotEnvelopeFault(reason="Snapshot must be a dictionary."))

    target = snapshot_dict.get("target") or "unknown"
    envelope_data = dict(snapshot_dict)
    envelope_data.setdefault("target", target)
    raw_ver = envelope_data.get("schema_version")
    if raw_ver is None or raw_ver == "1.0.0":
        envelope_data["schema_version"] = SCHEMA_VERSION
    elif raw_ver != SCHEMA_VERSION:
        return Err(
            SnapshotEnvelopeFault(
                reason=f"Unsupported envelope schema_version '{raw_ver}'; expected '{SCHEMA_VERSION}'.",
                schema_version=str(raw_ver),
            )
        )
    if "generated_at" not in envelope_data or not envelope_data["generated_at"]:
        envelope_data["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        return Ok(SnapshotEnvelope.model_validate(envelope_data))
    except (ValueError, TypeError) as exc:
        return Err(
            SnapshotEnvelopeFault(
                reason=f"Invalid snapshot envelope schema: {exc}",
                schema_version=SCHEMA_VERSION,
            )
        )


__all__ = [
    "CompressionFormat",
    "load_json_data",
    "parse_compression_format",
    "stream_snapshot_items",
    "validate_snapshot_envelope",
    "write_snapshot",
]
