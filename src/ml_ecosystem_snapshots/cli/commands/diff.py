"""CLI command handler for diffing two snapshots and generating changelogs."""

from __future__ import annotations

import argparse
import json


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the diff command comparing two snapshot files.

    Args:
        args: Parsed command line arguments containing json1, json2, and changelog flag.
    """
    import ml_ecosystem_snapshots.cli as cli_pkg

    path1 = cli_pkg.resolve_snapshot_path(args.json1)
    path2 = cli_pkg.resolve_snapshot_path(args.json2)

    with open(path1, "r", encoding="utf-8") as f:
        snap1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        snap2 = json.load(f)

    result = cli_pkg.diff_snapshots(snap1, snap2)

    if args.changelog:
        print(cli_pkg.generate_changelog(result))
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
