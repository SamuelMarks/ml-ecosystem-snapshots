"""MCP tool module for hallucination verification, anti-pattern explanation, and AST code-block checking."""

from __future__ import annotations

from ml_ecosystem_snapshots.mcp_server import (
    check_code_block,
    check_hallucination,
    explain_anti_pattern,
)

__all__ = ["check_code_block", "check_hallucination", "explain_anti_pattern"]
