"""CLI command handler for diffing two snapshots and generating changelogs."""

from __future__ import annotations

import argparse


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the diff command comparing two snapshot files.

    Args:
        args: Parsed command line arguments containing json1, json2, and changelog flag.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.diff import diff_snapshots, generate_changelog
    from ml_ecosystem_snapshots.models import ensure_json_dict
    from ml_ecosystem_snapshots.network_env import load_json_data

    path1 = resolve_snapshot_path(args.json1)
    path2 = resolve_snapshot_path(args.json2)

    snap1 = ensure_json_dict(load_json_data(path1), "snapshot 1")
    snap2 = ensure_json_dict(load_json_data(path2), "snapshot 2")

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


__all__ = ["cmd_diff"]
