"""Snapshot envelope serialization, validation, and streaming I/O utilities."""

from __future__ import annotations

import datetime
import json
import os
import typing
from pathlib import Path

from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_ecosystem_snapshots.models import (
    JsonValue,
    ScalarValue,
    SnapshotEnvelope,
)
from ml_ecosystem_snapshots.network_env import (
    ZstdModuleProtocol,
    load_json_data,
    stream_snapshot_items,
)


def write_snapshot(
    framework_name: str,
    snapshot_data: SnapshotEnvelope | typing.Mapping[str, JsonValue],
    output_dir: str,
    compress: str | None = None,
) -> str:
    """Write a snapshot dictionary to a JSON or compressed file on disk.

    Args:
        framework_name: The framework identifier.
        snapshot_data: The snapshot dictionary or envelope.
        output_dir: The directory path where the file should be written.
        compress: Optional compression algorithm ('gzip', 'gz', 'zstd').

    Returns:
        The path to the generated snapshot file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snap_dict: dict[str, JsonValue] = snapshot_data.model_dump() if isinstance(snapshot_data, SnapshotEnvelope) else dict(snapshot_data)
    raw_version = snap_dict.get("version")
    version = str(raw_version) if raw_version is not None else "unknown"
    safe_ver = version.replace("+", "_").replace(" ", "_")

    if compress in ("gzip", "gz"):
        file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.gz"))
        import gzip

        with gzip.open(file_path, "wt", encoding="utf-8", newline="\n") as f:
            json.dump(snap_dict, f, indent=2, sort_keys=True)
            f.write("\n")
    elif compress == "zstd":
        file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.zst"))
        raw_bytes = json.dumps(snap_dict, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        try:
            import importlib

            zstd: InspectableTarget = importlib.import_module("zstandard")
            assert isinstance(zstd, ZstdModuleProtocol)
            cctx = zstd.ZstdCompressor()
            compressed: bytes = cctx.compress(raw_bytes)
            with open(file_path, "wb") as f:
                f.write(compressed)
        except (ImportError, OSError, ValueError, TypeError, RuntimeError, AssertionError):
            import gzip

            file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json.gz"))
            with gzip.open(file_path, "wt", encoding="utf-8", newline="\n") as f:
                json.dump(snap_dict, f, indent=2, sort_keys=True)
                f.write("\n")
    else:
        file_path = Path(os.path.join(out_path, f"{framework_name}_v{safe_ver}.json"))
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(snap_dict, f, indent=2, sort_keys=True)
            f.write("\n")

    return str(file_path)


def validate_snapshot_envelope(
    snapshot_dict: typing.Mapping[str, JsonValue] | ScalarValue,
) -> SnapshotEnvelope:
    """Validate and normalize a snapshot dictionary into a SnapshotEnvelope.

    Args:
        snapshot_dict: Raw snapshot dictionary.

    Returns:
        Validated SnapshotEnvelope instance.

    Raises:
        SnapshotSchemaError: If required fields are missing or invalid.
    """
    if not isinstance(snapshot_dict, dict):
        raise SnapshotSchemaError("Snapshot must be a dictionary.")
    target = snapshot_dict.get("target") or "unknown"
    envelope_data = dict(snapshot_dict)
    envelope_data.setdefault("target", target)
    envelope_data.setdefault("schema_version", "1.0.0")
    if "generated_at" not in envelope_data or not envelope_data["generated_at"]:
        envelope_data["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        return SnapshotEnvelope.model_validate(envelope_data)
    except (ValueError, TypeError) as exc:
        raise SnapshotSchemaError(f"Invalid snapshot envelope schema: {exc}") from exc


__all__ = [
    "load_json_data",
    "stream_snapshot_items",
    "validate_snapshot_envelope",
    "write_snapshot",
]
