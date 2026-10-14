"""CLI command handler for launching the Model Context Protocol (MCP) server."""

from __future__ import annotations

import argparse


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle the mcp command to start Model Context Protocol JSON-RPC server.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import run_mcp_server

    run_mcp_server()


__all__ = ["cmd_mcp"]
