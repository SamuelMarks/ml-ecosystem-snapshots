"""CLI command handlers for managing the local SQLite search index."""

from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3


def cmd_index(args: argparse.Namespace) -> None:
    """Manage ephemeral local SQLite search index.

    Args:
        args: Parsed command line arguments containing index action flags.
    """
    from ml_ecosystem_snapshots.index import (
        clear_index,
        ensure_index,
        get_index_db_path,
    )

    db_path = get_index_db_path()

    if getattr(args, "clear", False):
        if clear_index():
            print(f"Cleared index database at {db_path}")
        else:
            print(f"Failed to clear index database at {db_path}")
        return

    if args.rebuild:
        clear_index()
        with contextlib.closing(ensure_index()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM symbols")
            count = cur.fetchone()[0]
        print(f"Rebuilt index with {count} symbols at {db_path}")
        return

    if not os.path.exists(db_path):
        print(f"Index database does not exist yet at {db_path}.")
        return

    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM symbols")
        symbol_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM indexed_files")
        file_count = cur.fetchone()[0]
    size = os.path.getsize(db_path)
    print(f"Index database: {db_path}")
    print(f"  Indexed snapshots: {file_count}")
    print(f"  Total symbols:     {symbol_count}")
    print(f"  Database size:     {size} bytes")


__all__ = ["cmd_index"]
