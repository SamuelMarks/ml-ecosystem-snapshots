"""CLI command handler for diffing two snapshots and generating changelogs."""

from __future__ import annotations

import argparse
import typing

from ml_switcheroo_ir.typeclasses.outcome import Valid, and_then, value_or

from ml_ecosystem_snapshots.exceptions import raise_snapshot_boundary
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict
from ml_ecosystem_snapshots.proofs import (
    EnvelopeAndHydratedPhase,
    ProvenSnapshot,
    prove_envelope_valid,
    upgrade_to_hydrated,
)


def coerce_proven_or_raw_snapshot(
    snap: ProvenSnapshot[EnvelopeAndHydratedPhase] | typing.Mapping[str, JsonValue],
) -> ProvenSnapshot[EnvelopeAndHydratedPhase] | dict[str, JsonValue]:
    """Promote a raw mapping to `ProvenSnapshot[EnvelopeAndHydratedPhase]` when valid.

    Args:
        snap: Hydrated `ProvenSnapshot` witness or raw snapshot dictionary.

    Returns:
        `ProvenSnapshot[EnvelopeAndHydratedPhase]` when envelope and hydration succeed,
        or a shallow dictionary copy for flat non-envelope JSON fixtures.
    """
    if isinstance(snap, ProvenSnapshot):
        return snap
    env_val = prove_envelope_valid(snap)
    if isinstance(env_val, Valid):
        hyd_val = upgrade_to_hydrated(env_val.value)
        if isinstance(hyd_val, Valid):
            return hyd_val.value
    return dict(snap)


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the diff command comparing two snapshot files at the CLI I/O edge.

    Args:
        args: Parsed command line arguments containing json1, json2, and changelog flag.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.diff import diff_snapshots, generate_changelog
    from ml_ecosystem_snapshots.network_env import load_json_data

    path1 = value_or(resolve_snapshot_path(args.json1), args.json1)
    path2 = value_or(resolve_snapshot_path(args.json2), args.json2)

    raw1 = raise_snapshot_boundary(and_then(load_json_data(path1), lambda d: ensure_json_dict(d, "snapshot 1")))
    raw2 = raise_snapshot_boundary(and_then(load_json_data(path2), lambda d: ensure_json_dict(d, "snapshot 2")))

    snap1 = coerce_proven_or_raw_snapshot(raw1)
    snap2 = coerce_proven_or_raw_snapshot(raw2)

    result = diff_snapshots(snap1, snap2)

    if args.changelog:
        print(generate_changelog(result))
    else:
        print(f"ADDED: {len(result.added)}")
        for p in result.added:
            print(f"  + {p}")

        print(f"REMOVED: {len(result.removed)}")
        for p in result.removed:
            print(f"  - {p}")

        print(f"SIGNATURE CHANGED: {len(result.signature_changed)}")
        for p in result.signature_changed:
            print(f"  ~ {p}")


__all__ = ["cmd_diff", "coerce_proven_or_raw_snapshot"]
