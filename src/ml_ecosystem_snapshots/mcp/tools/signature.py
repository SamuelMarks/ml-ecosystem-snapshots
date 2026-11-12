"""Thin MCP adapter for API signature retrieval (`get_api_signature`)."""

from __future__ import annotations

from ml_ecosystem_snapshots.grounding.lookup import lookup_api_signature_in_snapshot
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.models import JsonValue


def get_api_signature(
    framework: str,
    api_path: str,
    version: str | None = None,
    repository: SnapshotRepository | None = None,
) -> dict[str, JsonValue] | None:
    """Look up the exact ground-truth API signature for a given framework and path.

    Args:
        framework: Target framework name (e.g. torch, jax).
        api_path: Canonical API path or mnemonic (e.g. torch.nn.Linear, FADD).
        version: Optional framework version string (e.g. 2.4.0).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        Dictionary of the matched GhostRef, or `None` if not found.
    """
    fast = lookup_api_signature_in_snapshot(framework, api_path, {"categories": {}}, version=version, use_index=True)
    if fast is not None:
        return fast
    snap = resolve_snapshot(framework, version=version, repository=repository)
    return lookup_api_signature_in_snapshot(framework, api_path, snap, version=version, use_index=False)


__all__ = ["get_api_signature"]
